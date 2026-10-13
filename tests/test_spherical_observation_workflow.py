from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import panorai.data.observations as observation_module

from panorai.data import (
    EquirectangularImage,
    FacePointBatch,
    FaceRegionBatch,
    FaceViewOutput,
    RasterFieldResult,
    RasterFieldSpec,
    SphericalBinaryMask,
    SphericalPointSet,
    SphericalViewSet,
)


def _erp(shape=(32, 64)) -> EquirectangularImage:
    height, width = shape
    y, x = np.indices(shape)
    image = np.stack((x / width, y / height, (x + y) / (width + height)), axis=-1)
    return EquirectangularImage(image.astype(np.float32))


def test_generic_fields_round_trip_and_preserve_zero_validity():
    values = np.zeros((32, 64, 3), dtype=np.float32)
    valid = np.ones((32, 64), dtype=bool)
    valid[10, 20] = False
    spec = RasterFieldSpec.continuous(component_ids=("pipe", "vessel", "valve"))
    erp = _erp().with_field("cam", values, spec=spec, valid=valid)

    views = erp.views("cube", size=24)
    mapped = views.map(lambda value: value, input="cam")
    reconstructed = mapped.reconstruct(modalities="cam")

    assert erp.field_spec("cam") is spec
    assert reconstructed.field("cam").shape == values.shape
    assert np.all(reconstructed.field("cam")[reconstructed.validity("cam")] == 0)
    assert reconstructed.field_spec("cam") == spec

    normalized = _erp().with_field(
        " confidence ",
        np.ones((32, 64), dtype=np.float32),
        spec=RasterFieldSpec.continuous(),
    )
    assert normalized.field("confidence").shape == (32, 64)
    field_only = EquirectangularImage.from_field(
        "gray",
        np.ones((32, 64), dtype=np.float32),
        spec=RasterFieldSpec.continuous(),
    )
    assert field_only.is_multi_channel()
    assert field_only.get_channels() == ["gray"]
    assert field_only.shape == (32, 64)
    with pytest.raises(ValueError, match="non-empty string"):
        _erp().with_field(
            "   ",
            np.ones((32, 64), dtype=np.float32),
            spec=RasterFieldSpec.continuous(),
        )
    with pytest.raises(ValueError, match="non-empty string"):
        EquirectangularImage.from_field(
            7,
            np.ones((32, 64), dtype=np.float32),
            spec=RasterFieldSpec.continuous(),
        )
    invalid = np.ones((32, 64), dtype=np.float32)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        EquirectangularImage.from_field(
            "confidence", invalid, spec=RasterFieldSpec.continuous()
        )
    valid = np.ones((32, 64), dtype=bool)
    valid[0, 0] = False
    accepted = EquirectangularImage.from_field(
        "confidence",
        invalid,
        spec=RasterFieldSpec.continuous(),
        valid=valid,
    )
    assert not accepted.validity("confidence")[0, 0]
    support = np.ones((32, 64), dtype=bool)
    support[0, 0] = False
    attached = _erp().with_field(
        "confidence",
        invalid,
        spec=RasterFieldSpec.continuous(),
        support=support,
    )
    assert not attached.validity("confidence")[0, 0]


def test_new_map_output_requires_spec_and_explicit_support_is_intersected():
    views = _erp().views("cube", size=12)
    with pytest.raises(ValueError, match="requires spec"):
        views.map(lambda image: image[..., 0], output="confidence")

    def operator(image):
        support = np.ones(image.shape[:2], dtype=bool)
        support[:, :2] = False
        return RasterFieldResult(image[..., 0], support=support)

    result = views.map(
        operator,
        output="confidence",
        spec=RasterFieldSpec.continuous(),
    )
    assert not result[0]._workflow_support["confidence"][:, :2].any()
    assert not result[0].validity("confidence")[:, :2].any()
    for bad_name in ("", "   ", 7):
        with pytest.raises(ValueError, match="non-empty string"):
            views.map(
                lambda image: image[..., 0],
                output=bad_name,
                spec=RasterFieldSpec.continuous(),
            )
        with pytest.raises(ValueError, match="non-empty string"):
            _erp().apply_spherical(
                lambda image: image,
                input="image",
                output=bad_name,
                spec=RasterFieldSpec.continuous(),
            )


def test_categorical_fields_reject_unsafe_interpolation_and_float_values():
    with pytest.raises(ValueError, match="nearest"):
        RasterFieldSpec("categorical", interpolation="bilinear")
    with pytest.raises(ValueError, match="bilinear"):
        RasterFieldSpec.continuous(interpolation="nearest")
    with pytest.raises(TypeError, match="integer or boolean"):
        _erp().with_field(
            "classes",
            np.zeros((32, 64), dtype=np.float32),
            spec=RasterFieldSpec.categorical(),
        )
    with pytest.raises(ValueError, match="component_ids"):
        _erp().with_field(
            "cam",
            np.zeros((32, 64, 2), dtype=np.float32),
            spec=RasterFieldSpec.continuous(component_ids=("only-one",)),
        )

    categorical = EquirectangularImage.from_field(
        "classes",
        np.arange(32 * 64, dtype=np.int16).reshape(32, 64) % 5,
        spec=RasterFieldSpec.categorical(),
    ).views("cube", size=16)
    for unsafe in ("average", "gaussian", object()):
        with pytest.raises(ValueError, match="closest blending"):
            categorical.reconstruct(modalities="classes", blend=unsafe)
    reconstructed = categorical.reconstruct(modalities="classes", blend="closest")
    assert reconstructed.field("classes").dtype == np.int16


def test_apply_spherical_matches_direct_gaussian_and_canny():
    from panorai.image_processing import spherical_canny, spherical_gaussian_blur

    source = np.arange(32 * 64, dtype=np.uint8).reshape(32, 64) % 251
    erp = EquirectangularImage.from_field(
        "gray", source, spec=RasterFieldSpec.continuous()
    )
    blurred = erp.apply_spherical(
        lambda value: spherical_gaussian_blur(value, ksize=5, sigma=1.0),
        input="gray",
        output="blur",
        spec=RasterFieldSpec.continuous(),
    )
    edges = erp.apply_spherical(
        lambda value: spherical_canny(value, 40, 120),
        input="gray",
        output="edges",
        spec=RasterFieldSpec.categorical(),
    )
    assert np.array_equal(
        blurred.field("blur"), spherical_gaussian_blur(source, ksize=5, sigma=1.0)
    )
    assert np.array_equal(edges.field("edges"), spherical_canny(source, 40, 120))


def test_point_lifting_records_bearings_erp_pixels_and_deterministic_nms():
    face = _erp().views("cube", size=21)[0]
    batch = FacePointBatch(
        np.array([[10.0, 10.0], [10.1, 10.0], [-3.0, 2.0]]),
        scores=np.array([0.8, 0.9, 1.0]),
        class_ids=np.array([4, 4, 4]),
    )
    points = face.lift_points(batch, source_id="opencv")
    fused = points.angular_nms(1.0)

    assert len(points) == 3
    assert np.allclose(points.observations[0].bearing_xyz, [0.0, 0.0, 1.0])
    assert points.observations[0].source_erp_xy is not None
    assert not points.observations[2].valid
    assert fused.kept_indices == (1,)
    assert fused.suppressed_by == ((0, 1),)
    assert fused.invalid_indices == (2,)
    assert not points.observations[0].bearing_xyz.flags.writeable


def test_point_and_region_lifting_preserve_string_class_ids():
    face = _erp().views("cube", size=21)[0]
    class_ids = np.array(["pipe"], dtype=object)

    point = face.lift_points(
        FacePointBatch(np.array([[10.0, 10.0]]), class_ids=class_ids),
        source_id="string-point-class",
    ).observations[0]
    region = face.lift_regions(
        FaceRegionBatch.from_boxes(
            np.array([[4.0, 4.0, 16.0, 16.0]]), class_ids=class_ids
        ),
        source_id="string-region-class",
    ).observations[0]

    assert point.class_id == "pipe"
    assert region.class_id == "pipe"


def test_angular_nms_zero_threshold_deduplicates_identical_bearings():
    face = _erp().views("cube", size=21)[0]
    points = face.lift_points(
        FacePointBatch(np.array([[0.0, 0.0], [0.0, 0.0]])),
        source_id="duplicates",
    )

    assert points.angular_nms(0.0).kept_indices == (0,)


def test_reduction_ties_use_numeric_input_order_and_preserve_missing_scores():
    face = _erp().views("cube", size=21)[0]
    valid = np.zeros(11, dtype=bool)
    valid[[2, 10]] = True
    repeated_points = np.repeat([[10.0, 10.0]], 11, axis=0)
    point_result = face.lift_points(
        FacePointBatch(repeated_points, valid=valid, scores=np.zeros(11)),
        source_id="numeric-point-tie",
    ).angular_nms(0.0)
    assert point_result.kept_indices == (2,)
    assert point_result.suppressed_by == ((10, 2),)

    repeated_boxes = np.repeat([[4.0, 4.0, 16.0, 16.0]], 11, axis=0)
    regions = face.lift_regions(
        FaceRegionBatch.from_boxes(repeated_boxes, valid=valid, scores=np.zeros(11)),
        source_id="numeric-region-tie",
    )
    region_result = regions.nms(0.5)
    assert region_result.kept_indices == (2,)
    assert region_result.suppressed_by == ((10, 2),)

    unscored = face.lift_regions(
        FaceRegionBatch.from_boxes(
            np.array([[4.0, 4.0, 16.0, 16.0], [5.0, 5.0, 15.0, 15.0]])
        ),
        source_id="unscored-regions",
    ).merge(containment_threshold=0.5)
    assert unscored.observations.observations[0].score is None


def test_point_lifting_covers_face_corners_seam_pole_and_partial_support():
    shape = (128, 256)
    views = _erp(shape).views("cube", size=64)
    pixels = np.array([[31.5, 31.5], [-0.5, -0.5], [63.5, 63.5]])

    back = next(face for face in views if face.face_id == "back")
    back_points = back.lift_points(FacePointBatch(pixels), source_id="seam-points")
    assert all(item.valid for item in back_points.observations)
    assert back_points.observations[0].source_erp_xy[0] == pytest.approx(shape[1] - 0.5)
    assert back_points.observations[1].source_erp_xy[0] > 0.75 * shape[1]
    assert back_points.observations[2].source_erp_xy[0] < 0.25 * shape[1]

    up = next(face for face in views if face.face_id == "up")
    pole = up.lift_points(
        FacePointBatch(pixels[:1]), source_id="pole-point"
    ).observations[0]
    assert pole.valid
    assert pole.source_erp_xy[1] == pytest.approx(-0.5)
    assert np.allclose(pole.bearing_xyz, [0.0, 1.0, 0.0], atol=1e-12)

    source_support = np.ones(shape, dtype=bool)
    source_support[56:72, 120:136] = False
    partial_views = EquirectangularImage(
        np.zeros((*shape, 3), dtype=np.uint8), support_mask=source_support
    ).views("cube", size=64)
    front = next(face for face in partial_views if face.face_id == "front")
    unsupported = front.lift_points(
        FacePointBatch(pixels[:1]), source_id="partial-support"
    ).observations[0]
    assert not unsupported.valid


def test_face_set_map_points_passes_full_face_to_callback():
    views = _erp().views("cube", size=9)
    seen = []

    def detector(face):
        seen.append((face.face_id, face.image.shape, face.spec.output_shape_hw))
        return FacePointBatch(np.array([[4.0, 4.0]]))

    result = views.map_points(detector, source_id="detector")
    assert len(result) == len(views)
    assert len(seen) == len(views)
    assert all(item[1][:2] == item[2] for item in seen)


def test_real_opencv_consumers_cover_points_boxes_labels_depth_and_views():
    cv2 = pytest.importorskip("cv2")
    y, x = np.indices((64, 128))
    pattern = ((((x // 8) + (y // 8)) % 2) * 255).astype(np.uint8)
    rgb = np.stack((pattern, np.roll(pattern, 3, axis=1), pattern), axis=-1)
    views = EquirectangularImage(rgb).views("cube", size=48)

    def points(face):
        gray = cv2.cvtColor(face.image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        detected = cv2.goodFeaturesToTrack(gray, 40, 0.01, 3)
        pixels = (
            np.empty((0, 2), dtype=np.float64)
            if detected is None
            else detected[:, 0].astype(np.float64)
        )
        return FacePointBatch(pixels)

    def boxes(face):
        gray = cv2.cvtColor(face.image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary)
        xyxy = [
            [left, top, left + width, top + height]
            for left, top, width, height, area in stats[1:count]
            if area >= 16
        ]
        values = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
        return FaceRegionBatch.from_boxes(values)

    point_result = views.map_points(points, source_id="opencv-gftt")
    box_result = views.map_regions(boxes, source_id="opencv-components")
    labels = views.map(
        lambda image: cv2.threshold(
            cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY),
            127,
            1,
            cv2.THRESH_BINARY,
        )[1].astype(np.int16),
        output="labels",
    )
    depth = views.map(
        lambda image: cv2.GaussianBlur(
            image.mean(axis=-1, dtype=np.float32), (3, 3), 0
        ),
        output="depth",
        units="relative",
    )
    classified = views.map_views(
        lambda face: FaceViewOutput(
            scores=np.asarray(cv2.mean(face.image)[:3]),
            class_ids=np.asarray([0, 1, 2]),
        ),
        source_id="opencv-mean",
    )

    assert len(point_result) > 0
    assert len(box_result) > 0
    assert labels.reconstruct(modalities="labels").labels.dtype == np.int16
    assert depth.reconstruct(modalities="depth").depth.dtype == np.float32
    assert classified.aggregate("coverage_mean").scores.shape == (3,)


def test_region_lifting_nms_merge_and_class_awareness():
    face = _erp().views("cube", size=48)[0]
    regions = face.lift_regions(
        FaceRegionBatch.from_boxes(
            np.array([[8, 8, 32, 32], [9, 9, 33, 33], [9, 9, 33, 33]]),
            scores=np.array([0.7, 0.9, 0.8]),
            class_ids=np.array([1, 1, 2]),
        ),
        source_id="boxes",
    )
    nms = regions.nms(0.7)
    merged = regions.merge(iou_threshold=0.7)

    assert nms.kept_indices == (1, 2)
    assert nms.suppressed_by == ((0, 1),)
    assert len(merged.observations.observations) == 2
    assert merged.observations.observations[0].source_kind == "merged"
    assert merged.observations.observations[0].to_mask((64, 128)).pixel_count > 0

    clipped = face.lift_regions(
        FaceRegionBatch.from_polygons(
            [np.array([[-8, 10], [12, 10], [12, 20], [-8, 20]], dtype=float)]
        ),
        source_id="clipped",
    ).observations[0]
    outside = face.lift_regions(
        FaceRegionBatch.from_boxes(np.array([[70, 70, 90, 90]], dtype=float)),
        source_id="outside",
    ).observations[0]
    assert clipped.clipped and clipped.valid
    assert outside.clipped and not outside.valid


@pytest.mark.parametrize("iou_threshold", [0.0, 0.25, 0.8])
@pytest.mark.parametrize("class_aware", [False, True])
def test_region_nms_cap_pruning_matches_exhaustive_result(
    monkeypatch, iou_threshold, class_aware
):
    rng = np.random.default_rng(7052)
    shape = (24, 48)
    face = _erp(shape).views("cube", size=16)[0]
    template_set = face.lift_regions(
        FaceRegionBatch.from_boxes(np.array([[2.0, 2.0, 12.0, 12.0]])),
        source_id="cap-equivalence",
    )
    template = template_set.observations[0]
    observations = []
    masks = []
    for index in range(36):
        mask = np.zeros(shape, dtype=bool)
        if index % 13:
            row = int(rng.integers(0, shape[0] - 5))
            column = int(rng.integers(0, shape[1]))
            height = int(rng.integers(2, 7))
            width = int(rng.integers(2, 10))
            columns = np.arange(column, column + width) % shape[1]
            mask[row : min(shape[0], row + height), columns] = True
        if index >= 24 and index % 3 == 0:
            mask = masks[index - 24].copy()
        masks.append(mask)
        center = rng.normal(size=3)
        center /= np.linalg.norm(center)
        observations.append(
            replace(
                template,
                region_id=f"region-{index}",
                face_id=f"face-{index % 6}",
                center_bearing_xyz=center,
                boundary_bearings_xyz=np.empty((0, 3)),
                polygon_xy=None,
                local_mask=None,
                global_mask=SphericalBinaryMask.from_array(mask),
                valid=index % 17 != 0,
                score=float((index * 7) % 11),
                class_id=index % 3,
            )
        )
    regions = replace(template_set, observations=tuple(observations))

    optimized = regions.nms(iou_threshold, class_aware=class_aware)
    monkeypatch.setattr(
        observation_module,
        "_spherical_cap_candidates",
        lambda center, radius, kept_centers, kept_radii: np.ones(
            len(kept_radii), dtype=bool
        ),
    )
    exhaustive = regions.nms(iou_threshold, class_aware=class_aware)

    assert optimized.kept_indices == exhaustive.kept_indices
    assert optimized.suppressed_by == exhaustive.suppressed_by
    assert optimized.invalid_indices == exhaustive.invalid_indices
    assert optimized.groups == exhaustive.groups
    assert optimized.policy == exhaustive.policy
    assert optimized.parameters == exhaustive.parameters
    assert tuple(
        item.region_id for item in optimized.observations.observations
    ) == tuple(item.region_id for item in exhaustive.observations.observations)


def test_region_nms_cap_filter_never_rejects_discrete_mask_overlap():
    rng = np.random.default_rng(7053)
    shape = (32, 64)
    masks = []
    centers = []
    radii = []
    for _ in range(48):
        mask = rng.random(shape) < 0.025
        packed = SphericalBinaryMask.from_array(mask)
        center = rng.normal(size=3)
        center /= np.linalg.norm(center)
        masks.append(mask)
        centers.append(center)
        radii.append(observation_module._mask_enclosing_cap_radius(packed, center))

    center_array = np.asarray(centers)
    radius_array = np.asarray(radii)
    for index, mask in enumerate(masks):
        candidates = observation_module._spherical_cap_candidates(
            center_array[index],
            radius_array[index],
            center_array,
            radius_array,
        )
        overlapping = np.asarray([np.any(mask & other) for other in masks])
        assert np.all(candidates[overlapping])


def test_compact_nms_iou_matches_original_solid_angle_metric_exactly():
    rng = np.random.default_rng(7054)
    shape = (31, 61)
    weights = observation_module._solid_angle_row_weights(shape)
    arrays = [np.zeros(shape, dtype=bool)] + [
        rng.random(shape) < probability for probability in (0.01, 0.1, 0.5, 1.0)
    ]
    masks = [SphericalBinaryMask.from_array(value) for value in arrays]

    for first_index, first in enumerate(masks):
        for second_index, second in enumerate(masks):
            expected, _ = observation_module._compact_mask_metrics(first, second)
            actual = observation_module._compact_mask_iou(
                first,
                second,
                weights,
                observation_module._compact_mask_area(first, weights),
                observation_module._compact_mask_area(second, weights),
            )
            assert actual == pytest.approx(
                expected, rel=0.0, abs=4 * np.finfo(float).eps
            ), (
                first_index,
                second_index,
            )


def test_region_nms_preserves_historical_decision_at_roundoff_threshold():
    rng = np.random.default_rng(88)
    shape = (7, 11)
    for _ in range(2):
        first_array = rng.random(shape) < rng.random()
        second_array = rng.random(shape) < rng.random()
    first = SphericalBinaryMask.from_array(first_array)
    second = SphericalBinaryMask.from_array(second_array)
    historical_iou, _ = observation_module._compact_mask_metrics(first, second)
    assert historical_iou == 0.19131579073965113

    face = _erp(shape).views("cube", size=7)[0]
    template_set = face.lift_regions(
        FaceRegionBatch.from_boxes(np.array([[1.0, 1.0, 5.0, 5.0]])),
        source_id="roundoff-threshold",
    )
    template = template_set.observations[0]
    observations = (
        replace(
            template,
            region_id="roundoff-first",
            global_mask=first,
            local_mask=None,
            polygon_xy=None,
            score=1.0,
            class_id=0,
        ),
        replace(
            template,
            region_id="roundoff-second",
            global_mask=second,
            local_mask=None,
            polygon_xy=None,
            score=0.5,
            class_id=0,
        ),
    )
    regions = replace(template_set, observations=observations)

    result = regions.nms(historical_iou)

    assert result.kept_indices == (0,)
    assert result.suppressed_by == ((1, 0),)


def test_region_nms_caps_scan_large_erp_in_bounded_chunks(monkeypatch):
    shape = (1024, 2048)
    mask = SphericalBinaryMask.from_array(np.ones(shape, dtype=bool))
    face = _erp((24, 48)).views("cube", size=16)[0]
    template_set = face.lift_regions(
        FaceRegionBatch.from_boxes(np.array([[2.0, 2.0, 12.0, 12.0]])),
        source_id="bounded-cap-scan",
        erp_shape_hw=shape,
    )
    observation = replace(
        template_set.observations[0],
        global_mask=mask,
        local_mask=None,
        polygon_xy=None,
    )
    regions = replace(template_set, observations=(observation,))
    original = observation_module.erp_pixels_to_rays
    largest_input = 0

    def recording_erp_pixels_to_rays(pixels_xy, shape_hw):
        nonlocal largest_input
        largest_input = max(
            largest_input, np.asarray(pixels_xy).reshape(-1, 2).shape[0]
        )
        return original(pixels_xy, shape_hw)

    monkeypatch.setattr(
        observation_module, "erp_pixels_to_rays", recording_erp_pixels_to_rays
    )
    result = regions.nms(0.5)

    assert result.kept_indices == (0,)
    assert largest_input <= observation_module._MASK_SCAN_CHUNK_PIXELS


def test_region_merge_uses_keeper_bearing_for_full_sphere_degeneracy():
    shape = (256, 512)
    face = _erp((24, 48)).views("cube", size=16)[0]
    template_set = face.lift_regions(
        FaceRegionBatch.from_boxes(np.array([[2.0, 2.0, 12.0, 12.0]])),
        source_id="full-sphere-merge",
        erp_shape_hw=shape,
    )
    template = template_set.observations[0]
    full_sphere = SphericalBinaryMask.from_array(np.ones(shape, dtype=bool))
    keeper_bearing = np.asarray([0.0, 0.0, 1.0])
    observations = (
        replace(
            template,
            region_id="full-sphere-first",
            center_bearing_xyz=keeper_bearing,
            global_mask=full_sphere,
            local_mask=None,
            polygon_xy=None,
            score=1.0,
        ),
        replace(
            template,
            region_id="full-sphere-second",
            center_bearing_xyz=np.asarray([1.0, 0.0, 0.0]),
            global_mask=full_sphere,
            local_mask=None,
            polygon_xy=None,
            score=0.5,
        ),
    )
    regions = replace(template_set, observations=observations)

    result = regions.merge(iou_threshold=0.5)

    assert len(result.observations.observations) == 1
    assert np.array_equal(
        result.observations.observations[0].center_bearing_xyz, keeper_bearing
    )


def test_batched_region_rasterization_matches_independent_backprojections():
    shape = (48, 96)
    face = _erp(shape).views("cube", size=24)[0]
    boxes = face.lift_regions(
        FaceRegionBatch.from_boxes(
            np.array([[1.0, 2.0, 13.0, 18.0], [8.0, 5.0, 23.0, 22.0]])
        ),
        source_id="batched-raster-boxes",
    )
    local_masks = np.zeros((2, 24, 24), dtype=bool)
    local_masks[0, 3:17, 2:12] = True
    local_masks[1, 8:23, 9:21] = True
    masks = face.lift_regions(
        FaceRegionBatch.from_masks(local_masks),
        source_id="batched-raster-masks",
    )
    combined = boxes.observations + tuple(
        replace(item, source_id=boxes.source_id, global_mask=None)
        for item in masks.observations
    )
    supports = []
    for index, item in enumerate(combined):
        support = np.ones(item.spec.output_shape_hw, dtype=bool)
        support[:, index : index + 2] = False
        supports.append(replace(item, source_support=support))
    observations = tuple(supports)

    actual = observation_module._region_masks(
        observations, tuple(range(len(observations))), shape
    )
    for index, observation in enumerate(observations):
        if observation.local_mask is not None:
            local = observation.local_mask.to_array()
        else:
            local = observation_module._polygon_mask(
                observation.spec.output_shape_hw, observation.polygon_xy
            )
        projector = observation_module.GnomonicProjector(
            observation.spec, interpolation="nearest"
        )
        projected_region = projector.back_project(local, shape)
        projected_support = projector.back_project(observation.source_support, shape)
        expected = (
            np.asarray(projected_region.data, dtype=bool)
            & np.asarray(projected_region.support_mask, dtype=bool)
            & np.asarray(projected_support.data, dtype=bool)
            & np.asarray(projected_support.support_mask, dtype=bool)
        )
        actual_array = actual[index].to_array()
        assert np.array_equal(actual_array, expected), (
            index,
            np.count_nonzero(actual_array ^ expected),
            np.count_nonzero(actual_array),
            np.count_nonzero(expected),
        )


def test_masks_use_nearest_and_round_trip_face_sphere_face():
    face = _erp((256, 512)).views("cube", size=64)[0]
    local = np.zeros((64, 64), dtype=bool)
    local[16:48, 18:46] = True
    regions = face.lift_regions(
        FaceRegionBatch.from_masks(local[None]), source_id="masks"
    )
    spherical = regions.observations[0].to_mask().to_array()

    from panorai.geometry import GnomonicProjector

    returned = GnomonicProjector(face.spec, interpolation="nearest").project(spherical)
    recovered = np.asarray(returned.data, dtype=bool) & np.asarray(
        returned.support_mask, dtype=bool
    )
    intersection = np.count_nonzero(recovered & local)
    union = np.count_nonzero(recovered | local)
    assert intersection / union >= 0.95


def test_region_masks_wrap_at_seam_and_reach_pole_without_discontinuity():
    views = _erp((128, 256)).views("cube", size=64)
    local = np.zeros((64, 64), dtype=bool)
    local[20:44, 20:44] = True
    batch = FaceRegionBatch.from_masks(local[None])

    back = next(face for face in views if face.face_id == "back")
    seam = back.lift_regions(batch, source_id="seam").observations[0]
    seam_mask = seam.to_mask().to_array()
    assert seam_mask[:, 0].any()
    assert seam_mask[:, -1].any()

    up = next(face for face in views if face.face_id == "up")
    pole = up.lift_regions(batch, source_id="pole").observations[0]
    assert pole.to_mask().to_array()[0].any()


def test_region_boxes_polygons_and_masks_cross_seam_and_reduce_deterministically():
    back = next(
        face
        for face in _erp((128, 256)).views("cube", size=64)
        if face.face_id == "back"
    )
    box = np.array([[20.0, 20.0, 44.0, 44.0]])
    polygon = [np.array([[20.0, 20.0], [44.0, 20.0], [44.0, 44.0], [20.0, 44.0]])]
    local_mask = np.zeros((1, 64, 64), dtype=bool)
    local_mask[0, 20:44, 20:44] = True
    batches = (
        FaceRegionBatch.from_boxes(box),
        FaceRegionBatch.from_polygons(polygon),
        FaceRegionBatch.from_masks(local_mask),
    )
    for index, batch in enumerate(batches):
        spherical = (
            back.lift_regions(batch, source_id=f"seam-region-{index}")
            .observations[0]
            .to_mask()
            .to_array()
        )
        assert spherical[:, 0].any()
        assert spherical[:, -1].any()

    tied = back.lift_regions(
        FaceRegionBatch.from_boxes(
            np.array([[18, 18, 46, 46], [18, 18, 46, 46]], dtype=float),
            scores=np.array([0.8, 0.8]),
            class_ids=np.array([3, 3]),
        ),
        source_id="tie",
    ).nms(0.8)
    assert tied.kept_indices == (0,)
    assert tied.suppressed_by == ((1, 0),)
    assert tied.groups == ((0, 1),)

    contained = back.lift_regions(
        FaceRegionBatch.from_boxes(
            np.array([[14, 14, 50, 50], [24, 24, 40, 40]], dtype=float),
            scores=np.array([0.7, 0.9]),
            class_ids=np.array([4, 4]),
        ),
        source_id="containment",
    ).merge(containment_threshold=0.95)
    assert contained.kept_indices == (1,)
    assert contained.suppressed_by == ((0, 1),)
    assert contained.groups == ((0, 1),)


def test_spherical_binary_mask_is_bit_packed_and_solid_angle_weighted():
    from panorai.experimental.deep_learning.segmentation import (
        SphericalBinaryMask as ExperimentalMask,
    )

    first = np.zeros((16, 32), dtype=bool)
    first[4:12, 8:24] = True
    second = first.copy()
    mask = SphericalBinaryMask.from_array(first)
    assert len(mask.packed_bits) == 16 * 32 // 8
    assert mask.solid_angle_iou(SphericalBinaryMask.from_array(second)) == 1.0
    assert not mask.packed_bits.flags.writeable
    assert ExperimentalMask is SphericalBinaryMask

    mutable_shape = [2, 4]
    immutable = SphericalBinaryMask(
        mutable_shape, np.packbits(np.ones(8, dtype=bool), bitorder="little")
    )  # type: ignore[arg-type]
    mutable_shape[0] = 1
    assert immutable.shape_hw == (2, 4)

    with pytest.raises(TypeError, match="boolean dtype"):
        SphericalBinaryMask.from_array(np.ones((2, 4), dtype=np.uint8))
    with pytest.raises(TypeError, match="boolean dtype"):
        SphericalBinaryMask.from_array(np.ones((2, 4), dtype=np.float32))
    with pytest.raises(TypeError, match="boolean dtype"):
        mask.solid_angle_area(np.ones(mask.shape_hw, dtype=np.uint8))
    with pytest.raises(TypeError, match="boolean dtype"):
        FaceRegionBatch.from_masks(np.ones((1, 2, 4), dtype=np.uint8))


def test_observation_shapes_and_validity_require_explicit_integer_boolean_types():
    with pytest.raises(TypeError, match="shape dimensions must be integers"):
        SphericalPointSet("source", (), (16.5, 32))
    with pytest.raises(TypeError, match="boolean dtype"):
        FacePointBatch(np.array([[1.0, 1.0]]), valid=np.array([1], dtype=np.uint8))
    with pytest.raises(TypeError, match="boolean dtype"):
        FaceRegionBatch.from_boxes(
            np.array([[0.0, 0.0, 2.0, 2.0]]),
            valid=np.array([1], dtype=np.uint8),
        )
    with pytest.raises(TypeError, match="boolean dtype"):
        FaceViewOutput(
            scores=np.array([1.0]), class_ids=np.array([1]), valid=np.uint8(1)
        )
    with pytest.raises(TypeError, match="exactly two integers"):
        SphericalBinaryMask((True, 4), np.zeros(1, dtype=np.uint8))

    face = _erp().views("cube", size=16)[0]
    regions = face.lift_regions(
        FaceRegionBatch.from_boxes(np.array([[2.0, 2.0, 8.0, 8.0]])),
        source_id="shape-validation",
    )
    with pytest.raises(TypeError, match="shape dimensions must be integers"):
        regions.nms(0.5, evaluation_shape_hw=(16.5, 32))
    with pytest.raises(TypeError, match="boolean dtype"):
        replace(
            regions.observations[0],
            source_support=np.ones(face.spec.output_shape_hw, dtype=np.uint8),
        )
    view_observation = face.lift_view(
        FaceViewOutput(scores=np.array([1.0]), class_ids=np.array([1])),
        source_id="shape-validation",
    )
    evidence = SphericalViewSet("shape-validation", (view_observation,), (32, 64))
    with pytest.raises(TypeError, match="boolean dtype"):
        replace(
            view_observation,
            support=np.ones(face.spec.output_shape_hw, dtype=np.uint8),
        )
    with pytest.raises(TypeError, match="shape dimensions must be integers"):
        evidence.aggregate("coverage_mean", evaluation_shape_hw=(16, 32.5))


@pytest.mark.parametrize("shape", [(1, 2), (2, 4), (16, 32)])
def test_full_spherical_mask_has_exact_four_pi_solid_angle(shape):
    mask = SphericalBinaryMask.from_array(np.ones(shape, dtype=bool))

    assert mask.solid_angle_area() == pytest.approx(4.0 * np.pi, abs=1e-12)


def test_view_evidence_coverage_mean_and_max_contracts():
    views = _erp().views("cube", size=16)

    scores = views.map_views(
        lambda face: FaceViewOutput(
            scores=np.array([1.0, 0.25]), class_ids=np.array(["pipe", "valve"])
        ),
        source_id="classifier",
    )
    mean = scores.aggregate("coverage_mean")
    maximum = scores.aggregate("max")
    assert np.allclose(mean.scores, [1.0, 0.25])
    assert np.allclose(maximum.scores, [1.0, 0.25])
    assert 0.99 <= mean.supported_solid_angle_fraction <= 1.0
    assert maximum.supported_solid_angle_fraction == pytest.approx(
        mean.supported_solid_angle_fraction
    )

    embedded = views.map_views(
        lambda face: FaceViewOutput(embedding=np.array([3.0, 4.0])),
        source_id="embedding",
    )
    aggregate = embedded.aggregate("coverage_mean", normalize_embeddings=True)
    assert np.allclose(aggregate.embedding, [0.6, 0.8])
    with pytest.raises(ValueError, match="not defined for embeddings"):
        embedded.aggregate("max")

    observation = scores.observations[0]
    values = dict(
        observation_id=observation.observation_id,
        source_id=observation.source_id,
        face_id=observation.face_id,
        spec=observation.spec,
        support=observation.support,
        scores=None,
        class_ids=None,
        embedding=None,
        valid=True,
        attributes={},
    )
    with pytest.raises(ValueError, match="at least one"):
        type(observation)(**values)
    values["class_ids"] = np.array([1])
    with pytest.raises(ValueError, match="require scores"):
        type(observation)(**values)

    aggregate_type = type(mean)
    aggregate_values = dict(
        source_id="classifier",
        policy="coverage_mean",
        scores=np.array([1.0]),
        class_ids=np.array([1]),
        embedding=None,
        supported_solid_angle_fraction=1.0,
        contributing_view_ids=("front",),
    )
    with pytest.raises(ValueError, match="policy"):
        aggregate_type(**{**aggregate_values, "policy": "unknown"})
    with pytest.raises(ValueError, match="finite"):
        aggregate_type(**{**aggregate_values, "scores": np.array([np.nan])})
    with pytest.raises(ValueError, match="align"):
        aggregate_type(**{**aggregate_values, "class_ids": np.array([1, 2])})
    with pytest.raises(ValueError, match="not defined"):
        aggregate_type(
            **{
                **aggregate_values,
                "policy": "max",
                "scores": None,
                "class_ids": None,
                "embedding": np.array([1.0]),
            }
        )


def test_view_coverage_mean_matches_dense_reference_with_unequal_overlap():
    views = _erp().views("cube", size=20, fov=115)
    face_index = {face.face_id: index for index, face in enumerate(views)}
    evidence = views.map_views(
        lambda face: FaceViewOutput(
            scores=np.array([face_index[face.face_id] + 1.0]),
            class_ids=np.array([7]),
            embedding=np.array([1.0, face_index[face.face_id] + 1.0], dtype=np.float64),
        ),
        source_id="unequal-overlap",
    )
    shape = (32, 64)
    result = evidence.aggregate(
        "coverage_mean",
        evaluation_shape_hw=shape,
        normalize_embeddings=False,
    )

    footprints = np.stack(
        [item.footprint(shape).to_array() for item in evidence.observations]
    )
    coverage = footprints.sum(axis=0)
    latitude = np.pi / 2 - (np.arange(shape[0]) + 0.5) * np.pi / shape[0]
    area = np.broadcast_to(np.cos(latitude)[:, None], shape)
    supported = coverage > 0
    score_raster = (
        sum(
            footprint[..., None] * item.scores
            for footprint, item in zip(footprints, evidence.observations, strict=True)
        )
        / np.where(supported, coverage, 1)[..., None]
    )
    embedding_raster = (
        sum(
            footprint[..., None] * item.embedding
            for footprint, item in zip(footprints, evidence.observations, strict=True)
        )
        / np.where(supported, coverage, 1)[..., None]
    )
    expected_scores = (score_raster * area[..., None]).sum(axis=(0, 1)) / area[
        supported
    ].sum()
    expected_embedding = (embedding_raster * area[..., None]).sum(axis=(0, 1)) / area[
        supported
    ].sum()

    assert np.allclose(result.scores, expected_scores, atol=1e-12)
    assert np.allclose(result.embedding, expected_embedding, atol=1e-12)


def test_torch_conversion_is_explicit_for_numpy_observations():
    torch = pytest.importorskip("torch")
    pixels = torch.tensor([[1.0, 2.0]])
    with pytest.raises(TypeError, match="from_torch"):
        FacePointBatch(pixels)
    with pytest.raises(TypeError, match="from_torch"):
        FacePointBatch([[torch.tensor(1.0), torch.tensor(2.0)]])
    with pytest.raises(TypeError, match="from_torch"):
        FaceViewOutput(scores=[torch.tensor(1.0)], class_ids=[1])
    face = _erp().views("cube", size=9)[0]
    point = face.lift_points(
        FacePointBatch(np.array([[4.0, 4.0]])), source_id="torch-scalars"
    ).observations[0]
    with pytest.raises(TypeError, match="from_torch"):
        type(point)(
            observation_id=point.observation_id,
            source_id=point.source_id,
            face_id=point.face_id,
            pixel_xy=point.pixel_xy,
            source_erp_xy=point.source_erp_xy,
            bearing_xyz=point.bearing_xyz,
            valid=torch.tensor(True),
        )
    converted = FacePointBatch.from_torch(pixels, scores=torch.tensor([0.5]))
    assert isinstance(converted.pixels_xy, np.ndarray)
    assert not converted.pixels_xy.flags.writeable
    with pytest.raises(TypeError, match="from_torch"):
        SphericalBinaryMask.from_array(torch.ones((4, 8), dtype=torch.bool))
    converted_mask = SphericalBinaryMask.from_torch(
        torch.ones((4, 8), dtype=torch.bool)
    )
    assert converted_mask.pixel_count == 32
    with pytest.raises(TypeError, match="boolean dtype"):
        SphericalBinaryMask.from_torch(torch.ones((4, 8)))

    polygons = FaceRegionBatch.from_torch(
        "polygon",
        [
            torch.tensor([[1.0, 1.0], [4.0, 1.0], [2.0, 4.0]]),
            torch.tensor([[5.0, 5.0], [8.0, 5.0], [6.0, 8.0]]),
        ],
    )
    assert all(isinstance(item, np.ndarray) for item in polygons.geometries)

    with pytest.raises(TypeError, match="from_torch"):
        FaceViewOutput(
            scores=np.array([0.9]),
            class_ids=np.array([1]),
            valid=torch.tensor(True),
        )
    view_output = FaceViewOutput.from_torch(
        scores=torch.tensor([0.9]),
        class_ids=torch.tensor([1]),
        valid=torch.tensor(True),
    )
    assert isinstance(view_output.valid, bool) and view_output.valid


def test_torch_raster_fields_preserve_backend_dtype_device_and_batch():
    torch = pytest.importorskip("torch")
    numpy_data = np.ones((16, 32), dtype=np.float32)
    torch_support = torch.ones((16, 32), dtype=torch.bool)
    with pytest.raises(TypeError, match="same backend"):
        EquirectangularImage.from_field(
            "confidence",
            numpy_data,
            spec=RasterFieldSpec.continuous(),
            support=torch_support,
        )
    with pytest.raises(TypeError, match="same backend"):
        _erp((16, 32)).with_field(
            "confidence",
            numpy_data,
            spec=RasterFieldSpec.continuous(),
            support=torch_support,
        )
    image = torch.zeros((2, 3, 16, 32), dtype=torch.float32)
    cam = torch.zeros((2, 4, 16, 32), dtype=torch.float32)
    erp = EquirectangularImage(image).with_field(
        "cam",
        cam,
        spec=RasterFieldSpec.continuous(component_ids=("a", "b", "c", "d")),
    )
    result = (
        erp.views("cube", size=8)
        .map(lambda value: value, input="cam")
        .reconstruct(modalities="cam")
    )
    assert isinstance(result.field("cam"), torch.Tensor)
    assert result.field("cam").shape == cam.shape
    assert result.field("cam").dtype == cam.dtype
    assert result.field("cam").device == cam.device
