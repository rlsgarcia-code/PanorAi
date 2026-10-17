from __future__ import annotations

from dataclasses import replace
import warnings

import numpy as np
import pytest

from panorai.applications import (
    FullSphereDenseStereo,
    PanoramicDenseStereo,
    StereoSGBMDepth,
)
from panorai.data import EquirectangularImage
from panorai.geometry import (
    GnomonicSpec,
    erp_pixels_to_rays,
    gnomonic_pixels_to_rays,
    rays_to_gnomonic_pixels,
)
from panorai.stereo import SphericalStereoOptions
from panorai.stereo.classical import StereoSGBMConfig, StereoValidationConfig
from panorai.stereo.classical._operations import _census
from panorai.stereo.panoramic import (
    FullSphereMultiViewStereo,
    FullSphereStereoOptions,
    OverlappingCubemapPlanner,
    PanoramicStereoPose,
    RectifiedStereoViewPlanner,
    RegisteredPanoramaSet,
    SphericalStereoViewPairer,
    StereoSourceView,
    StereoViewDescriptor,
    spec_from_axes,
)
from panorai.stereo.panoramic._full_sphere import (
    _ad_census_face_cost,
    _align_source_features,
    _erp_tangent_basis,
    _face_tap_stride,
    _local_cost_aggregation,
    _regularize_cost_volume,
)
from panorai.stereo.panoramic._fusion import compose_fusion_confidence


def _known_pose(baseline: float = 0.5) -> PanoramicStereoPose:
    return PanoramicStereoPose(
        np.eye(3),
        np.asarray((-baseline, 0.0, 0.0)),
        baseline,
        "m",
        "metric",
        "test",
    )


def test_fusion_confidence_removes_geometric_taper_from_score():
    valid = np.asarray(((True, True, False),), dtype=bool)
    matching, consistency, combined = compose_fusion_confidence(
        np.asarray(((0.20, 0.40, 0.0),), dtype=np.float64),
        np.asarray(((0.25, 0.50, 0.0),), dtype=np.float64),
        np.asarray(((2.0, 2.0, np.nan),), dtype=np.float32),
        np.asarray(((0.0, 0.0, np.nan),), dtype=np.float32),
        valid,
    )

    np.testing.assert_allclose(matching[valid], 0.8, rtol=0.0, atol=1e-7)
    np.testing.assert_array_equal(consistency[valid], 1.0)
    np.testing.assert_allclose(combined[valid], 0.8, rtol=0.0, atol=1e-7)
    assert matching[0, 2] == 0.0
    assert consistency[0, 2] == 0.0
    assert combined[0, 2] == 0.0


def _sphere_pair(shape: tuple[int, int], baseline: float, radius: float):
    height, width = shape
    yy, xx = np.indices(shape, dtype=np.float64)
    pixels = np.stack((xx, yy), axis=-1)
    rays = np.asarray(erp_pixels_to_rays(pixels, shape))

    def texture(points: np.ndarray) -> np.ndarray:
        unit = points / np.linalg.norm(points, axis=-1, keepdims=True)
        longitude = np.arctan2(unit[..., 0], unit[..., 2])
        latitude = np.arcsin(np.clip(unit[..., 1], -1.0, 1.0))
        channels = (
            0.5 + 0.24 * np.sin(13.0 * longitude + 3.0 * latitude),
            0.5 + 0.24 * np.sin(9.0 * longitude - 11.0 * latitude),
            0.5 + 0.24 * np.cos(17.0 * longitude + 7.0 * latitude),
        )
        return np.stack(channels, axis=-1).astype(np.float32)

    reference_points = radius * rays
    centre_b = np.asarray((baseline, 0.0, 0.0))
    dot = np.einsum("...i,i->...", rays, centre_b)
    root = -dot + np.sqrt(dot * dot + radius * radius - baseline * baseline)
    source_points = centre_b + root[..., None] * rays
    return texture(reference_points), texture(source_points)


def _textured_sphere_view(
    shape: tuple[int, int], camera_centre: np.ndarray, radius: float
) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float64)
    rays = np.asarray(erp_pixels_to_rays(np.stack((xx, yy), axis=-1), shape))
    centre = np.asarray(camera_centre, dtype=np.float64)
    projection = np.einsum("...i,i->...", rays, centre)
    distance = -projection + np.sqrt(
        projection * projection + radius * radius - float(centre @ centre)
    )
    points = centre + distance[..., None] * rays
    unit = points / np.linalg.norm(points, axis=-1, keepdims=True)
    longitude = np.arctan2(unit[..., 0], unit[..., 2])
    latitude = np.arcsin(np.clip(unit[..., 1], -1.0, 1.0))
    return np.stack(
        (
            0.5 + 0.24 * np.sin(13.0 * longitude + 3.0 * latitude),
            0.5 + 0.24 * np.sin(9.0 * longitude - 11.0 * latitude),
            0.5 + 0.24 * np.cos(17.0 * longitude + 7.0 * latitude),
        ),
        axis=-1,
    ).astype(np.float32)


def _registered_source(
    source_id: str,
    image: np.ndarray,
    camera_centre: tuple[float, float, float],
) -> StereoSourceView:
    centre = np.asarray(camera_centre, dtype=np.float64)
    baseline = float(np.linalg.norm(centre))
    return StereoSourceView(
        source_id=source_id,
        panorama=image,
        pose=PanoramicStereoPose(
            np.eye(3),
            -centre,
            baseline,
            "m",
            "metric",
            "analytic-render",
        ),
    )


def test_spec_from_axes_recovers_requested_camera_axes():
    forward = np.asarray((0.4, 0.3, 0.8660254))
    forward /= np.linalg.norm(forward)
    right = np.cross(np.asarray((0.0, 1.0, 0.0)), forward)
    right /= np.linalg.norm(right)
    spec = spec_from_axes(
        forward,
        right,
        hfov_deg=80.0,
        vfov_deg=70.0,
        output_shape_hw=(64, 80),
    )
    centre = np.asarray(
        ((spec.output_shape_hw[1] - 1) / 2, (spec.output_shape_hw[0] - 1) / 2)
    )
    ray = gnomonic_pixels_to_rays(centre, spec).rays_xyz
    horizontal = gnomonic_pixels_to_rays(centre + np.asarray((1.0, 0.0)), spec).rays_xyz
    assert np.allclose(ray, forward, atol=1e-10)
    assert np.dot(horizontal - ray, right) > 0.0


def test_rectified_plan_makes_epipolar_rows_horizontal():
    angle_y = np.deg2rad(17.0)
    angle_z = np.deg2rad(-9.0)
    rotation_y = np.asarray(
        (
            (np.cos(angle_y), 0.0, np.sin(angle_y)),
            (0.0, 1.0, 0.0),
            (-np.sin(angle_y), 0.0, np.cos(angle_y)),
        )
    )
    rotation_z = np.asarray(
        (
            (np.cos(angle_z), -np.sin(angle_z), 0.0),
            (np.sin(angle_z), np.cos(angle_z), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    rotation = rotation_z @ rotation_y
    baseline_axis_a = np.asarray((0.8, 0.2, -0.3))
    baseline_axis_a /= np.linalg.norm(baseline_axis_a)
    translation_b = -0.5 * (rotation @ baseline_axis_a)
    pose = PanoramicStereoPose(rotation, translation_b, 0.5, "m", "metric", "analytic")
    plan = RectifiedStereoViewPlanner(
        hfov_deg=80.0,
        vfov_deg=90.0,
        overlap_deg=20.0,
        output_shape_hw=(96, 128),
    ).plan(pose)
    pair = plan.pairs[1]
    pixels_a = np.asarray(((40.0, 31.0), (70.0, 55.0), (92.0, 72.0)))
    rays_a = gnomonic_pixels_to_rays(pixels_a, pair.reference_spec).rays_xyz
    points_a = rays_a * np.asarray((4.0, 6.0, 8.0))[:, None]
    points_b = (
        points_a @ pose.rotation_source_from_reference.T
        + pose.translation_source_from_reference
    )
    bearings_b = points_b / np.linalg.norm(points_b, axis=1, keepdims=True)
    projected_b = rays_to_gnomonic_pixels(bearings_b, pair.source_spec)
    assert np.all(projected_b.valid)
    assert np.allclose(projected_b.pixels_xy[:, 1], pixels_a[:, 1], atol=1e-9)
    assert np.all(pixels_a[:, 0] - projected_b.pixels_xy[:, 0] > 0.0)


def test_view_pairer_allows_many_to_many_overlap():
    pose = _known_pose()
    planner = RectifiedStereoViewPlanner(
        hfov_deg=100.0,
        vfov_deg=100.0,
        overlap_deg=40.0,
        output_shape_hw=(32, 32),
    )
    plan = planner.plan(pose)
    reference = StereoViewDescriptor("a", plan.pairs[0].reference_spec)
    sources = (
        StereoViewDescriptor("b0", plan.pairs[0].source_spec),
        StereoViewDescriptor("b1", plan.pairs[1].source_spec),
    )
    pairs = SphericalStereoViewPairer(
        min_score=0.01, max_pairs_per_reference=None
    ).match((reference,), sources, pose=pose)
    assert len(pairs) == 2
    assert pairs[0].score >= pairs[1].score


def test_overlapping_cubemap_planner_covers_every_erp_direction():
    shape = (65, 130)
    yy, xx = np.indices(shape, dtype=np.float64)
    rays = np.asarray(erp_pixels_to_rays(np.stack((xx, yy), axis=-1), shape))
    planner = OverlappingCubemapPlanner(face_size=32, face_fov_deg=100.0)
    coverage = np.zeros(shape, dtype=np.uint8)
    for _, spec in planner.plan():
        projection = rays_to_gnomonic_pixels(rays, spec)
        coverage += np.asarray(projection.valid, dtype=np.uint8)
    assert len(planner.plan()) == 6
    assert np.all(coverage >= 1)
    assert np.any(coverage >= 2)


def test_full_sphere_defaults_use_reviewed_cost_and_angular_support():
    options = FullSphereStereoOptions()

    assert options.matching_cost == "ad-census"
    assert options.local_aggregation == "box"
    assert options.regularizer == "sgm4"
    assert options.erp_feature_tap_deg == pytest.approx(0.703125)
    assert options.face_cost_tap_deg == pytest.approx(1.0667900072919114)

    legacy = FullSphereStereoOptions(
        matching_cost="intensity-gradient",
        erp_feature_tap_deg=None,
        face_cost_tap_deg=None,
    )
    assert legacy.matching_cost == "intensity-gradient"
    assert legacy.erp_feature_tap_deg is None
    assert legacy.face_cost_tap_deg is None


def test_full_sphere_legacy_config_without_candidate_fields_keeps_old_behavior():
    payload = FullSphereDenseStereo().to_config()
    serialized = payload["parameters"]["options"]
    serialized.pop("matching_cost")
    serialized.pop("erp_feature_tap_deg")
    serialized.pop("face_cost_tap_deg")

    restored = FullSphereDenseStereo.from_config(payload)

    assert restored.options.matching_cost == "intensity-gradient"
    assert restored.options.erp_feature_tap_deg is None
    assert restored.options.face_cost_tap_deg is None


@pytest.mark.parametrize(
    ("name", "value", "message"),
    (
        ("matching_cost", "unknown", "matching_cost"),
        ("local_aggregation", "bilateral", "local_aggregation"),
        ("regularizer", "mgm", "regularizer"),
        ("census_radius", 0, "census_radius"),
        ("erp_feature_tap_deg", 0.0, "erp_feature_tap_deg"),
        ("face_cost_tap_deg", np.inf, "face_cost_tap_deg"),
    ),
)
def test_full_sphere_cost_pipeline_options_reject_unknown_values(name, value, message):
    with pytest.raises(ValueError, match=message):
        FullSphereStereoOptions(**{name: value})


def test_ad_census_face_cost_is_zero_for_identical_supported_images():
    rng = np.random.default_rng(19)
    image = rng.uniform(0.0, 1.0, (17, 21)).astype(np.float32)
    options = FullSphereStereoOptions(matching_cost="ad-census", census_radius=2)
    cost, valid = _ad_census_face_cost(
        image,
        image.copy(),
        np.ones_like(image, dtype=bool),
        _census(image, options.census_radius),
        options,
    )

    assert np.all(valid[2:-2, 2:-2])
    np.testing.assert_allclose(cost[valid], 0.0)


def test_census_dilated_taps_preserve_bit_count_and_expand_invalid_border():
    yy, xx = np.indices((17, 19), dtype=np.float32)
    image = np.sin(xx * 0.7) + np.cos(yy * 0.4)

    dense, dense_valid = _census(image, 2)
    dilated, dilated_valid = _census(image, 2, tap_stride=2)

    assert dense.dtype == dilated.dtype == np.uint64
    assert np.any(dense[4:-4, 4:-4] != dilated[4:-4, 4:-4])
    assert np.all(dense_valid[2:-2, 2:-2])
    assert np.all(dilated_valid[4:-4, 4:-4])
    assert not np.any(dilated_valid[:4])
    assert not np.any(dilated_valid[:, :4])


def test_face_tap_stride_tracks_actual_centre_ray_spacing():
    low = OverlappingCubemapPlanner(face_size=128).plan()[0][1]
    high = OverlappingCubemapPlanner(face_size=256).plan()[0][1]
    centre = np.asarray(((127.0) / 2.0, (127.0) / 2.0))
    rays = gnomonic_pixels_to_rays(
        np.stack((centre, centre + np.asarray((1.0, 0.0)))), low
    ).rays_xyz
    native_step_deg = np.degrees(
        np.arccos(np.clip(np.dot(rays[0], rays[1]), -1.0, 1.0))
    )

    assert _face_tap_stride(low, None) == 1
    assert _face_tap_stride(low, float(native_step_deg)) == 1
    assert _face_tap_stride(high, float(native_step_deg)) == 2


def test_local_cost_aggregation_is_one_explicit_stage():
    volume = np.zeros((9, 11, 3), dtype=np.float32)
    volume[4, 5, 1] = 1.0
    valid = np.ones_like(volume, dtype=bool)

    unchanged, unchanged_valid = _local_cost_aggregation(volume, valid, 5, "none")
    aggregated, aggregated_valid = _local_cost_aggregation(volume, valid, 5, "box")

    np.testing.assert_array_equal(unchanged, volume)
    np.testing.assert_array_equal(unchanged_valid, valid)
    assert 0.0 < aggregated[4, 5, 1] < 1.0
    assert np.all(aggregated_valid)


def test_local_cost_aggregation_uses_declared_dilated_tap_grid():
    volume = np.zeros((9, 9, 1), dtype=np.float32)
    valid = np.ones_like(volume, dtype=bool)
    for dy in (-2, 0, 2):
        for dx in (-2, 0, 2):
            volume[4 + dy, 4 + dx, 0] = 1.0

    dense, _ = _local_cost_aggregation(volume, valid, 3, "box", tap_stride=1)
    dilated, dilated_valid = _local_cost_aggregation(
        volume, valid, 3, "box", tap_stride=2
    )

    assert dense[4, 4, 0] == pytest.approx(1.0 / 9.0)
    assert dilated[4, 4, 0] == pytest.approx(1.0)
    assert np.all(dilated_valid)


def test_regularizer_selection_preserves_dhw_contract():
    rng = np.random.default_rng(23)
    volume = rng.uniform(0.0, 1.0, (8, 10, 4)).astype(np.float32)
    valid = np.ones_like(volume, dtype=bool)
    guidance = rng.uniform(0.0, 1.0, (8, 10)).astype(np.float32)
    settings = SphericalStereoOptions(
        num_hypotheses=4,
        bidirectional_consistency=False,
    )

    raw = _regularize_cost_volume(volume, valid, guidance, settings, "none")
    sgm = _regularize_cost_volume(volume, valid, guidance, settings, "sgm4")
    tmgm = _regularize_cost_volume(volume, valid, guidance, settings, "tmgm")

    assert raw.shape == sgm.shape == tmgm.shape == (4, 8, 10)
    np.testing.assert_array_equal(raw, np.moveaxis(volume, -1, 0))
    assert not np.array_equal(sgm, raw)
    assert not np.array_equal(tmgm, raw)


def test_full_sphere_ad_census_route_reports_exact_cost_pipeline():
    shape = (17, 34)
    radius = 4.0
    baseline = 0.35
    reference = _textured_sphere_view(shape, np.zeros(3), radius)
    source = _registered_source(
        "source-x",
        _textured_sphere_view(shape, np.asarray((baseline, 0.0, 0.0)), radius),
        (baseline, 0.0, 0.0),
    )
    options = FullSphereStereoOptions(
        planner=OverlappingCubemapPlanner(face_size=16, face_fov_deg=100.0),
        stereo=SphericalStereoOptions(
            min_range=3.0,
            max_range=5.0,
            num_hypotheses=5,
            window_size=3,
            pole_margin_fraction=0.0,
            min_texture_std=0.0,
            min_confidence=0.0,
            max_matching_cost=1.0,
            bidirectional_consistency=False,
        ),
        min_parallax_deg=0.0,
        matching_cost="ad-census",
        local_aggregation="none",
        regularizer="none",
        census_radius=1,
        ad_weight=0.4,
        census_weight=0.6,
        ad_truncation=0.3,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = FullSphereDenseStereo(options)(
            RegisteredPanoramaSet(
                reference_id="A",
                reference=reference,
                sources=(source,),
            )
        )

    assert result.radial_range.shape == shape
    assert result.diagnostics["feature_cost"] == {
        "matching_cost": "ad-census",
        "census_radius": 1,
        "ad_weight": 0.4,
        "census_weight": 0.6,
        "ad_truncation": 0.3,
    }
    assert result.diagnostics["cost_aggregation"] == {
        "local": "none",
        "regularizer": "none",
        "window_size": 3,
    }


def test_source_tangent_gradients_are_transported_into_reference_basis():
    reference_rays = np.asarray(
        (
            ((0.3, 0.2, 0.9327379), (-0.4, 0.1, 0.9110434)),
            ((0.2, -0.5, 0.8426149), (-0.6, -0.2, 0.7745967)),
        ),
        dtype=np.float32,
    )
    reference_rays /= np.linalg.norm(reference_rays, axis=-1, keepdims=True)
    angle = np.deg2rad(37.0)
    rotation = np.asarray(
        (
            (np.cos(angle), 0.0, np.sin(angle)),
            (0.0, 1.0, 0.0),
            (-np.sin(angle), 0.0, np.cos(angle)),
        ),
        dtype=np.float32,
    )
    reference_east, reference_north = _erp_tangent_basis(reference_rays)
    expected_east = np.asarray(((0.2, -0.3), (0.4, -0.1)), dtype=np.float32)
    expected_north = np.asarray(((-0.4, 0.1), (0.2, 0.3)), dtype=np.float32)
    gradient_reference = (
        expected_east[..., None] * reference_east
        + expected_north[..., None] * reference_north
    )
    source_bearings = reference_rays @ rotation.T
    gradient_source = gradient_reference @ rotation.T
    source_east, source_north = _erp_tangent_basis(source_bearings)
    source_components = np.stack(
        (
            np.einsum("...i,...i->...", gradient_source, source_east),
            np.einsum("...i,...i->...", gradient_source, source_north),
        ),
        axis=-1,
    )
    intensity = np.full(reference_rays.shape[:2], 0.17, dtype=np.float32)
    warped = np.concatenate((intensity[..., None], source_components), axis=-1)

    aligned = _align_source_features(
        warped,
        source_bearings * 3.0,
        reference_rays,
        rotation,
    )
    assert np.allclose(aligned[..., 0], intensity, atol=1e-7)
    assert np.allclose(aligned[..., 1], expected_east, atol=1e-6)
    assert np.allclose(aligned[..., 2], expected_north, atol=1e-6)

    pole_east, pole_north = _erp_tangent_basis(
        np.asarray(((0.0, 1.0, 0.0), (0.0, -1.0, 0.0)))
    )
    assert np.allclose(pole_east, ((1.0, 0.0, 0.0),) * 2)
    assert np.allclose(pole_north, ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)))


def test_multibaseline_cube_stereo_covers_pairwise_epipoles_and_recovers_range():
    shape = (33, 66)
    radius = 4.0
    baseline = 0.35
    reference = _textured_sphere_view(shape, np.zeros(3), radius)
    centres = (
        (baseline, 0.0, 0.0),
        (0.0, baseline, 0.0),
        (0.0, 0.0, baseline),
    )
    sources = tuple(
        _registered_source(
            f"source-{axis}",
            _textured_sphere_view(shape, np.asarray(centre), radius),
            centre,
        )
        for axis, centre in zip("xyz", centres, strict=True)
    )
    options = FullSphereStereoOptions(
        planner=OverlappingCubemapPlanner(face_size=32, face_fov_deg=100.0),
        stereo=SphericalStereoOptions(
            min_range=3.0,
            max_range=5.0,
            num_hypotheses=9,
            window_size=3,
            pole_margin_fraction=0.0,
            min_texture_std=0.0,
            min_confidence=0.0,
            max_matching_cost=1.0,
            smoothness_penalty_small=0.0,
            smoothness_penalty_large=0.0,
            bidirectional_consistency=False,
        ),
        min_parallax_deg=2.0,
        min_source_evidence=1,
        max_sources_per_hypothesis=None,
        source_aggregation="median",
    )
    estimator = FullSphereMultiViewStereo(options)
    pairwise = estimator.estimate(
        RegisteredPanoramaSet(
            reference_id="A",
            reference=reference,
            sources=sources[:1],
        )
    )
    application = FullSphereDenseStereo(options)
    multibaseline = application(
        RegisteredPanoramaSet(
            reference_id="A",
            reference=reference,
            sources=sources,
        )
    )
    acceptance_order = multibaseline.diagnostics["face_acceptance_order"]
    assert acceptance_order == (
        "reference_support",
        "interior_hypothesis",
        "finite_range",
        "texture",
        "confidence",
        "matching_cost",
        "source_evidence",
        "range_consensus",
        "parallax",
        "accepted",
    )
    face_counts = multibaseline.diagnostics["face_acceptance_cumulative_counts"]
    assert set(face_counts) == {"front", "right", "back", "left", "up", "down"}
    for face_id, counts in face_counts.items():
        values = [counts[name] for name in acceptance_order]
        assert values == sorted(values, reverse=True)
        assert counts["accepted"] == round(
            multibaseline.diagnostics["face_valid_fractions"][face_id]
            * options.planner.face_size**2
        )
    totals = multibaseline.diagnostics["face_acceptance_cumulative_totals"]
    assert totals == {
        name: sum(counts[name] for counts in face_counts.values())
        for name in acceptance_order
    }
    consensus_application = FullSphereDenseStereo(
        replace(options, min_range_consensus_sources=2)
    )
    consensus = consensus_application(
        RegisteredPanoramaSet(
            reference_id="A",
            reference=reference,
            sources=sources,
        )
    )

    yy, xx = np.indices(shape, dtype=np.float64)
    rays = np.asarray(erp_pixels_to_rays(np.stack((xx, yy), axis=-1), shape))
    plus_x = np.unravel_index(np.argmax(rays[..., 0]), shape)
    assert not pairwise.observable[plus_x]
    assert multibaseline.observable[plus_x]
    assert np.all(multibaseline.sampling_support)
    assert np.all(multibaseline.observable)
    assert multibaseline.diagnostics["sampling_coverage_fraction"] == 1.0
    assert multibaseline.diagnostics["observable_fraction"] == 1.0
    assert multibaseline.diagnostics["reference_id"] == "A"
    assert float(multibaseline.validity.mean()) > 0.70
    assert float(consensus.validity.mean()) > 0.60
    assert np.all(consensus.source_consensus_count[consensus.validity] >= 2)
    median_range = float(np.median(multibaseline.radial_range[multibaseline.validity]))
    assert median_range == pytest.approx(radius, rel=0.12)
    assert np.all(np.isnan(multibaseline.radial_range[~multibaseline.validity]))
    np.testing.assert_allclose(
        multibaseline.confidence,
        multibaseline.matching_confidence * multibaseline.consistency_confidence,
        rtol=1e-6,
        atol=1e-7,
    )
    assert not multibaseline.matching_confidence.flags.writeable
    assert not multibaseline.consistency_confidence.flags.writeable
    assert FullSphereDenseStereo.from_config(application.to_config()) == application


def test_full_sphere_stereo_keeps_missing_source_evidence_invalid():
    shape = (17, 34)
    radius = 3.0
    baseline = 0.25
    reference = _textured_sphere_view(shape, np.zeros(3), radius)
    centre = (baseline, 0.0, 0.0)
    source_image = _textured_sphere_view(shape, np.asarray(centre), radius)
    source = _registered_source("masked", source_image, centre)
    source = StereoSourceView(
        source_id=source.source_id,
        panorama=source.panorama,
        pose=source.pose,
        validity=np.zeros(shape, dtype=bool),
    )
    options = FullSphereStereoOptions(
        planner=OverlappingCubemapPlanner(face_size=8, face_fov_deg=100.0),
        stereo=SphericalStereoOptions(
            min_range=2.0,
            max_range=4.0,
            num_hypotheses=3,
            window_size=3,
            pole_margin_fraction=0.0,
            min_texture_std=0.0,
            min_confidence=0.0,
            max_matching_cost=1.0,
            smoothness_penalty_small=0.0,
            smoothness_penalty_large=0.0,
            bidirectional_consistency=False,
        ),
        min_parallax_deg=0.0,
    )
    result = FullSphereDenseStereo(options)(
        RegisteredPanoramaSet(
            reference_id="A",
            reference=reference,
            sources=(source,),
        )
    )
    assert np.all(result.sampling_support)
    assert not np.any(result.observable)
    assert not np.any(result.validity)
    assert not np.any(result.source_consensus_count)
    assert np.all(np.isnan(result.radial_range))


def test_registered_panorama_set_owns_and_validates_the_view_relations():
    image = np.zeros((8, 16, 3), dtype=np.float32)
    first = _registered_source("B", image, (0.2, 0.0, 0.0))
    second = _registered_source("C", image, (0.0, 0.2, 0.0))

    registered = RegisteredPanoramaSet(
        reference_id="A",
        reference=image,
        sources=[first, second],
    )
    assert registered.sources == (first, second)

    with pytest.raises(ValueError, match="at least one"):
        RegisteredPanoramaSet(reference_id="A", reference=image, sources=())
    with pytest.raises(ValueError, match="reference_id must differ"):
        RegisteredPanoramaSet(reference_id="B", reference=image, sources=(first,))
    with pytest.raises(ValueError, match="source_id values must be unique"):
        RegisteredPanoramaSet(
            reference_id="A",
            reference=image,
            sources=(first, first),
        )

    scale_free = StereoSourceView(
        source_id="D",
        panorama=image,
        pose=PanoramicStereoPose(
            np.eye(3),
            np.asarray((0.0, -1.0, 0.0)),
            1.0,
            "baseline-units",
            "scale-free",
            "test",
        ),
    )
    with pytest.raises(ValueError, match="same units and scale status"):
        RegisteredPanoramaSet(
            reference_id="A",
            reference=image,
            sources=(first, scale_free),
        )

    with pytest.raises(TypeError, match="RegisteredPanoramaSet"):
        FullSphereMultiViewStereo().estimate(image)
    with pytest.raises(ValueError, match="cannot exceed"):
        FullSphereMultiViewStereo(
            FullSphereStereoOptions(min_range_consensus_sources=2)
        ).estimate(
            RegisteredPanoramaSet(
                reference_id="A",
                reference=image,
                sources=(first,),
            )
        )


def test_explicit_project_views_preserves_roll_and_iterates_lazily():
    panorama = EquirectangularImage(
        np.arange(32 * 64, dtype=np.float32).reshape(32, 64)
    )
    first = GnomonicSpec(
        center_lat_deg=20.0,
        center_lon_deg=-35.0,
        hfov_deg=75.0,
        vfov_deg=55.0,
        roll_deg=23.0,
        output_shape_hw=(24, 30),
    )
    second = GnomonicSpec(
        center_lat_deg=-10.0,
        center_lon_deg=70.0,
        hfov_deg=80.0,
        vfov_deg=60.0,
        roll_deg=-14.0,
        output_shape_hw=(20, 28),
    )
    views = panorama.project_views((first, second), face_ids=("left", "right"))
    assert [face.face_id for face in views] == ["left", "right"]
    assert views[0].spec == first and views[1].spec == second
    assert views[0].field("image").shape == first.output_shape_hw

    def delayed_specs():
        yield first
        raise RuntimeError("second spec requested")

    iterator = panorama.iter_views(delayed_specs())
    assert next(iterator).spec == first
    with pytest.raises(RuntimeError, match="second spec requested"):
        next(iterator)


def test_explicit_iter_views_preserves_torch_backend():
    torch = pytest.importorskip("torch")
    image = torch.linspace(0.0, 1.0, 16 * 32).reshape(1, 1, 16, 32)
    panorama = EquirectangularImage(image)
    spec = GnomonicSpec(
        center_lat_deg=15.0,
        center_lon_deg=25.0,
        hfov_deg=70.0,
        vfov_deg=60.0,
        roll_deg=11.0,
        output_shape_hw=(12, 14),
    )
    face = next(panorama.iter_views((spec,)))
    assert isinstance(face.field("image"), torch.Tensor)
    assert face.field("image").shape == (1, 1, 12, 14)
    assert face.validity("image").dtype == torch.bool


def test_rectified_application_streams_pairs_and_returns_radial_range():
    baseline = 0.5
    reference, source = _sphere_pair((64, 128), baseline, radius=5.0)
    # Real RGB inputs are generally uint8. This is also a regression for the
    # validity-normalized projector rejecting integer samples before the
    # application normalized them at its projection boundary.
    reference = np.rint(reference * 255.0).astype(np.uint8)
    source = np.rint(source * 255.0).astype(np.uint8)
    matcher = StereoSGBMDepth(
        StereoSGBMConfig(
            min_disparity=0,
            max_disparity=12,
            block_size=5,
            uniqueness_ratio=0,
            validation=StereoValidationConfig(
                uniqueness_margin=0.0,
                left_right_tolerance=2.0,
            ),
        )
    )
    app = PanoramicDenseStereo(
        matcher=matcher,
        pose_mode="provided",
        planner=RectifiedStereoViewPlanner(
            hfov_deg=80.0,
            vfov_deg=100.0,
            overlap_deg=20.0,
            output_shape_hw=(64, 64),
        ),
        min_joint_support=0.5,
    )
    partial_support = np.ones(reference.shape[:2], dtype=bool)
    partial_support[-8:] = False
    result = app.run(
        reference,
        source,
        rotation_source_from_reference=np.eye(3),
        translation_source_from_reference=np.asarray((-baseline, 0.0, 0.0)),
        units="m",
        reference_validity=partial_support,
        source_validity=partial_support,
    )
    assert result.radial_range.shape == (64, 128)
    assert result.pose.scale_status == "metric"
    assert result.pose.units == "m"
    assert result.diagnostics["route"] == "rectified-views"
    assert (
        result.diagnostics["stage_diagnostics"]["backproject_stereo_range"][
            "retained_cost_volumes"
        ]
        == 0
    )
    assert len(result.pair_diagnostics) > 0
    assert len(result.view_selection) == len(result.view_pairs)
    assert sum(item.accepted for item in result.view_selection) == len(
        result.pair_diagnostics
    )
    assert np.any(result.validity)
    assert not np.any(result.validity[-8:])
    np.testing.assert_allclose(
        result.confidence,
        result.matching_confidence * result.consistency_confidence,
        rtol=1e-6,
        atol=1e-7,
    )
    assert np.all(result.matching_confidence[~result.validity] == 0.0)
    assert np.all(result.consistency_confidence[~result.validity] == 0.0)
    median = float(np.median(result.radial_range[result.validity]))
    assert median == pytest.approx(5.0, rel=0.40)

    direct = PanoramicDenseStereo(
        route="sphere-native",
        pose_mode="provided",
        direct_options=SphericalStereoOptions(
            min_range=2.0,
            max_range=8.0,
            num_hypotheses=32,
            window_size=3,
            min_confidence=0.0,
            max_matching_cost=1.0,
            bidirectional_consistency=False,
        ),
    ).run(
        reference,
        source,
        rotation_source_from_reference=np.eye(3),
        translation_source_from_reference=np.asarray((-baseline, 0.0, 0.0)),
        units="m",
    )
    direct_median = float(np.median(direct.radial_range[direct.validity]))
    assert direct.radial_range.shape == result.radial_range.shape
    assert direct.pose.units == result.pose.units == "m"
    assert direct_median == pytest.approx(5.0, rel=0.15)
    assert abs(direct_median - median) < 2.0

    rejected = PanoramicDenseStereo(
        matcher=matcher,
        pose_mode="provided",
        planner=app.planner,
        min_joint_support=0.5,
        min_texture_std=1.0,
    ).run(
        reference,
        source,
        rotation_source_from_reference=np.eye(3),
        translation_source_from_reference=np.asarray((-baseline, 0.0, 0.0)),
        units="m",
    )
    assert not np.any(rejected.validity)
    assert rejected.pair_diagnostics == ()
    assert rejected.view_selection
    assert all(
        "insufficient-texture" in item.rejection_reasons
        for item in rejected.view_selection
    )


def test_estimated_pose_without_baseline_is_explicitly_scale_free(monkeypatch):
    class Matches:
        def __len__(self):
            return 8

        def to_bearing_correspondences(self):
            return self

    class Features:
        @classmethod
        def for_relative_pose(cls):
            return cls()

        def extract_and_match(self, *args, **kwargs):
            return Matches()

    class Estimate:
        rotation = np.eye(3)
        translation_direction = np.asarray((-1.0, 0.0, 0.0))
        num_inliers = 8
        quality_report = type(
            "Quality", (), {"accepted": True, "rejection_reasons": ()}
        )()

    class Estimator:
        def estimate(self, value):
            return Estimate()

    import panorai.stereo.panoramic._stages as stages

    monkeypatch.setattr(stages, "SphericalFeaturePipeline", Features)
    monkeypatch.setattr(stages, "SphericalRelativePoseEstimator", Estimator)
    image = np.zeros((16, 32, 3), dtype=np.float32)
    app = PanoramicDenseStereo(
        route="sphere-native",
        pose_mode="estimate",
        direct_options=SphericalStereoOptions(
            min_range=0.5,
            max_range=2.0,
            num_hypotheses=3,
            window_size=3,
            bidirectional_consistency=False,
        ),
    )
    result = app.run(image, image)
    assert result.pose.scale_status == "scale-free"
    assert result.pose.units == "baseline-units"

    Estimate.quality_report = type(
        "RejectedQuality",
        (),
        {"accepted": False, "rejection_reasons": ("weak-coverage",)},
    )()
    with pytest.raises(RuntimeError, match="quality gate rejected: weak-coverage"):
        app.run(image, image)


def test_real_feature_and_pose_stack_drives_common_stereo_result():
    reference, source = _sphere_pair((96, 192), baseline=0.5, radius=5.0)
    app = PanoramicDenseStereo(
        route="sphere-native",
        pose_mode="estimate",
        direct_options=SphericalStereoOptions(
            min_range=1.0,
            max_range=8.0,
            num_hypotheses=5,
            window_size=3,
            bidirectional_consistency=False,
        ),
    )
    result = app(reference, source, baseline=0.5, units="m")
    assert result.pose.provenance.startswith("estimated-spherical-features")
    assert np.linalg.norm(result.pose.rotation_source_from_reference - np.eye(3)) < 0.03
    assert result.pose.translation_source_from_reference[0] < -0.48
    pose_diagnostics = result.diagnostics["stage_diagnostics"]["resolve_pose"]
    assert pose_diagnostics["match_count"] >= 20
    assert pose_diagnostics["pose_inliers"] >= 15
    assert np.any(result.validity)


def test_direct_route_never_promotes_unsupported_reference_pixels():
    reference, source = _sphere_pair((32, 64), baseline=0.25, radius=3.0)
    support = np.ones((32, 64), dtype=bool)
    support[:, :16] = False
    app = PanoramicDenseStereo(
        route="sphere-native",
        pose_mode="provided",
        direct_options=SphericalStereoOptions(
            min_range=1.0,
            max_range=5.0,
            num_hypotheses=3,
            window_size=3,
            bidirectional_consistency=False,
        ),
    )
    result = app(
        reference,
        source,
        rotation_source_from_reference=np.eye(3),
        translation_source_from_reference=np.asarray((-0.25, 0.0, 0.0)),
        units="m",
        reference_validity=support,
    )
    assert not np.any(result.validity[:, :16])

    with pytest.raises(ValueError, match="partial source validity"):
        app(
            reference,
            source,
            rotation_source_from_reference=np.eye(3),
            translation_source_from_reference=np.asarray((-0.25, 0.0, 0.0)),
            units="m",
            source_validity=support,
        )


def test_application_configuration_round_trip(tmp_path):
    application = PanoramicDenseStereo(
        pose_mode="provided",
        planner=RectifiedStereoViewPlanner(output_shape_hw=(48, 64)),
    )
    path = application.save(tmp_path / "stereo.json")
    restored = PanoramicDenseStereo.load(path)
    assert isinstance(restored, PanoramicDenseStereo)
    assert restored.to_config() == application.to_config()

    full_sphere = FullSphereDenseStereo(
        FullSphereStereoOptions(
            planner=OverlappingCubemapPlanner(face_size=48, face_fov_deg=105.0),
            matching_cost="ad-census",
            local_aggregation="none",
            regularizer="tmgm",
            census_radius=3,
            ad_weight=0.4,
            census_weight=0.6,
            ad_truncation=0.15,
            erp_feature_tap_deg=0.703125,
            face_cost_tap_deg=1.06679,
        )
    )
    full_path = full_sphere.save(tmp_path / "full-sphere-stereo.json")
    restored_full = FullSphereDenseStereo.load(full_path)
    assert isinstance(restored_full, FullSphereDenseStereo)
    assert restored_full.to_config() == full_sphere.to_config()
