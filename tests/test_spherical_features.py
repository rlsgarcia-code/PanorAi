from __future__ import annotations

from dataclasses import fields
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from panorai.data.equirectangular_image import EquirectangularImage
from panorai.features import (
    FaceSetSpec,
    FeatureProvenance,
    FeatureExtractor,
    FeatureExtractorConfig,
    FeatureMatcher,
    FeatureMatcherConfig,
    MatchProvenance,
    OpenCVFeatureBackend,
    SphericalFeature,
    SphericalFeatureSet,
    SphericalFeaturePipeline,
    SphericalFeaturePipelineConfig,
    SphericalFeatureMatches,
    available_presets,
    deduplicate_spherical_keypoints,
    extract_opencv_features,
    gnomonic_feature_mask,
    match_opencv_features,
)
from panorai.geometry import (
    GnomonicSpec,
    equirectangular_to_gnomonic,
    gnomonic_face_geometry,
    gnomonic_intrinsics,
    gnomonic_pixel_map,
    gnomonic_pixels_to_rays,
    gnomonic_rotation,
    rays_to_gnomonic_pixels,
)


ROOT = Path(__file__).resolve().parents[1]


def _textured_panorama(height: int = 256, width: int = 512) -> np.ndarray:
    """Deterministic texture with edges, corners, and non-repeating detail."""

    y, x = np.indices((height, width))
    rng = np.random.default_rng(20261001)
    noise = rng.integers(0, 55, size=(height, width), dtype=np.uint8)
    checker = (((x // 13) + (y // 11)) % 2) * 115
    rings = (np.sin(np.hypot(x - width * 0.37, y - height * 0.61) / 3.7) + 1) * 35
    base = np.clip(noise.astype(np.int16) + checker + rings, 0, 255).astype(np.uint8)
    red = base
    green = np.roll(base, 7, axis=1)
    blue = np.bitwise_xor(base, ((x * 17 + y * 29) % 256).astype(np.uint8))
    return np.stack((red, green, blue), axis=-1)


def test_feature_import_does_not_eagerly_import_opencv_or_torch() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys; import panorai.features; "
            "print(json.dumps(sorted(set(sys.modules) & {'cv2','torch','pycolmap'})))",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(7, 9), (8, 10)])
def test_gnomonic_pixel_ray_roundtrip_preserves_numpy_contract(dtype, shape) -> None:
    spec = GnomonicSpec(
        center_lat_deg=31.0,
        center_lon_deg=-172.0,
        hfov_deg=103.0,
        vfov_deg=67.0,
        roll_deg=19.0,
        output_shape_hw=shape,
    )
    height, width = shape
    pixels = np.asarray(
        [
            [(width - 1) / 2, (height - 1) / 2],
            [0, 0],
            [width - 1, 0],
            [0, height - 1],
            [width - 1, height - 1],
            [1.25, 2.75],
        ],
        dtype=dtype,
    )
    projected = gnomonic_pixels_to_rays(pixels, spec)
    recovered = rays_to_gnomonic_pixels(projected.rays_xyz * dtype(4.25), spec)

    assert projected.rays_xyz.dtype == dtype
    assert recovered.pixels_xy.dtype == dtype
    assert recovered.ranges.dtype == dtype
    assert projected.valid.all() and recovered.valid.all()
    assert np.allclose(np.linalg.norm(projected.rays_xyz, axis=1), 1, atol=2e-6)
    assert np.allclose(recovered.pixels_xy, pixels, atol=3e-5)
    assert np.allclose(recovered.ranges, 4.25, atol=3e-5)


def test_gnomonic_pixels_match_independent_pinhole_oracle_and_invalid_rules() -> None:
    spec = GnomonicSpec(
        hfov_deg=90.0,
        vfov_deg=60.0,
        output_shape_hw=(5, 7),
    )
    pixels = np.asarray(((3.0, 2.0), (0.0, 0.0)), dtype=np.float64)
    result = gnomonic_pixels_to_rays(pixels, spec)

    x_local = ((0.0 - 3.0) / (7.0 / 2.0)) * math.tan(math.radians(45.0))
    y_local = ((0.0 - 2.0) / (5.0 / 2.0)) * math.tan(math.radians(30.0))
    expected_corner = np.asarray((x_local, -y_local, 1.0))
    expected_corner /= np.linalg.norm(expected_corner)
    assert np.allclose(result.rays_xyz[0], (0, 0, 1), atol=1e-15)
    assert np.allclose(result.rays_xyz[1], expected_corner, atol=1e-15)

    bad_pixels = np.asarray(((-0.5001, 0), (6.5001, 0), (0, np.nan)))
    bad = gnomonic_pixels_to_rays(bad_pixels, spec)
    assert not bad.valid.any()
    assert np.isnan(bad.rays_xyz).all()

    behind = rays_to_gnomonic_pixels(np.asarray(((0, 0, -1), (0, 0, 0))), spec)
    assert not behind.valid.any()
    assert np.isnan(behind.pixels_xy).all()


def test_intrinsics_rotation_and_projector_grid_are_consistent() -> None:
    spec = GnomonicSpec(
        center_lat_deg=-24.0,
        center_lon_deg=177.0,
        hfov_deg=111.0,
        vfov_deg=73.0,
        roll_deg=-13.0,
        output_shape_hw=(11, 14),
    )
    K = gnomonic_intrinsics(spec)
    R = gnomonic_rotation(spec)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-14)
    # Raster coordinates use y-down while the panorama frame uses +Y up.
    assert np.linalg.det(R) == pytest.approx(-1.0, abs=1e-14)

    y, x = np.indices(spec.output_shape_hw, dtype=np.float64)
    pixels = np.stack((x, y), axis=-1)
    local = np.stack(
        (
            (x - K[0, 2]) / K[0, 0],
            (y - K[1, 2]) / K[1, 1],
            np.ones_like(x),
        ),
        axis=-1,
    )
    expected = local @ R.T
    expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
    actual = gnomonic_pixels_to_rays(pixels, spec).rays_xyz
    assert np.allclose(actual, expected, atol=1e-14)

    erp = np.arange(24 * 48, dtype=np.float32).reshape(24, 48)
    projection = equirectangular_to_gnomonic(
        erp, spec, interpolation="nearest", return_source_pixels=True
    )
    explicit_map = gnomonic_pixel_map(spec, erp.shape)
    assert np.array_equal(projection.source_pixels_xy, explicit_map)


def test_gnomonic_face_exposes_complete_virtual_camera_geometry() -> None:
    panorama = EquirectangularImage(_textured_panorama(64, 128))
    face = panorama.views("cube", size=(13, 17), fov=(95, 75))[0]
    geometry = face.get_geometry(include_source_pixels=True)

    assert geometry.face_id == face.face_id
    assert geometry.spec == face.spec
    assert geometry.K.shape == (3, 3)
    assert geometry.R_panorama_from_face.shape == (3, 3)
    assert geometry.support_mask.shape == (13, 17)
    assert geometry.source_pixels_xy.shape == (13, 17, 2)
    assert np.array_equal(
        geometry.source_pixels_xy,
        gnomonic_pixel_map(face.spec, panorama.image.shape[:2]),
    )


def test_gnomonic_face_geometry_requires_exact_boolean_support() -> None:
    spec = GnomonicSpec(output_shape_hw=(2, 3))
    geometry = gnomonic_face_geometry(
        "face", spec, np.ones(spec.output_shape_hw, dtype=bool)
    )
    assert geometry.support_mask.dtype == np.bool_
    with pytest.raises(TypeError, match="boolean dtype"):
        gnomonic_face_geometry(
            "face", spec, np.ones(spec.output_shape_hw, dtype=np.float32)
        )
    with pytest.raises(ValueError, match="exact shape"):
        gnomonic_face_geometry("face", spec, np.ones((1, 2, 3), dtype=bool))


def test_gnomonic_face_geometry_preserves_torch_boolean_device() -> None:
    torch = pytest.importorskip("torch")
    spec = GnomonicSpec(output_shape_hw=(2, 3))
    support = torch.ones(spec.output_shape_hw, dtype=torch.bool)
    geometry = gnomonic_face_geometry("face", spec, support)
    assert geometry.support_mask is support
    assert geometry.K.device == support.device
    with pytest.raises(TypeError, match="boolean dtype"):
        gnomonic_face_geometry("face", spec, support.to(torch.float32))


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_torch_pixel_ray_roundtrip_preserves_dtype_device_and_gradients(
    dtype_name,
) -> None:
    torch = pytest.importorskip("torch")
    spec = GnomonicSpec(
        center_lat_deg=14,
        center_lon_deg=81,
        hfov_deg=92,
        vfov_deg=71,
        roll_deg=8,
        output_shape_hw=(8, 10),
    )
    dtype = getattr(torch, dtype_name)
    pixels = torch.tensor([[4.5, 3.5], [1.25, 6.0]], dtype=dtype, requires_grad=True)
    rays = gnomonic_pixels_to_rays(pixels, spec)
    recovered = rays_to_gnomonic_pixels(rays.rays_xyz * 2.0, spec)
    recovered.pixels_xy.sum().backward()

    assert rays.rays_xyz.dtype == dtype
    assert rays.rays_xyz.device == pixels.device
    assert recovered.pixels_xy.dtype == dtype
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    assert torch.allclose(recovered.pixels_xy, pixels, atol=tolerance, rtol=tolerance)
    assert pixels.grad is not None and torch.isfinite(pixels.grad).all()


def test_feature_mask_combines_support_validity_user_mask_and_edge_margin() -> None:
    class Projection:
        support_mask = np.ones((7, 9), dtype=bool)
        validity_mask = np.ones((7, 9), dtype=bool)

    Projection.support_mask[3, 4] = False
    Projection.validity_mask[2, 4] = False
    user = np.ones((7, 9), dtype=bool)
    user[4, 4] = False
    result = gnomonic_feature_mask(Projection, edge_margin_px=1, user_mask=user)

    assert not result[0].any() and not result[-1].any()
    assert not result[:, 0].any() and not result[:, -1].any()
    assert not result[2:5, 4].any()
    assert result[1, 1]


def test_angular_deduplication_is_deterministic_and_preserves_groups() -> None:
    angle = math.radians(0.05)
    bearings = np.asarray(
        (
            (0, 0, 1),
            (math.sin(angle), 0, math.cos(angle)),
            (0, 1, 0),
            (0, math.sin(angle), math.cos(angle)),
        ),
        dtype=np.float64,
    )
    responses = np.asarray((0.5, 0.9, 0.7, 0.4))
    result = deduplicate_spherical_keypoints(
        bearings, responses, angular_threshold_rad=math.radians(0.1)
    )
    repeated = deduplicate_spherical_keypoints(
        bearings, responses, angular_threshold_rad=math.radians(0.1)
    )

    assert np.array_equal(result.selected_indices, repeated.selected_indices)
    assert result.selected_indices.tolist() == [1, 2]
    assert result.duplicate_groups[0] == (1, 0, 3)
    assert result.group_index_for_input[0] == result.group_index_for_input[1]
    assert result.angular_distance_to_selected_rad[0] == pytest.approx(angle)


@pytest.mark.parametrize(
    ("preset", "extractor", "matcher", "metric"),
    [
        ("sift-flann", "sift", "flann", "l2"),
        ("sift-bf", "sift", "bf", "l2"),
        ("orb-hamming", "orb", "bf", "hamming"),
        ("akaze-hamming", "akaze", "bf", "hamming"),
    ],
)
def test_versioned_presets_are_serializable_and_execute_real_opencv(
    preset, extractor, matcher, metric
) -> None:
    cv2 = pytest.importorskip("cv2")
    pipeline = SphericalFeaturePipeline.from_preset(
        preset,
        face_sampler="cube",
        face_shape_hw=128,
        face_fov_deg=100.0,
        edge_margin_px=4,
        max_features=240,
    )
    serialized = json.loads(json.dumps(pipeline.describe()))
    assert serialized["interface"] == "panorai-spherical-features/v1"
    assert serialized["preset_version"] == 1
    assert serialized["minimum_opencv_version"] == "4.9.0"
    assert serialized["extractor"]["method"] == extractor
    assert serialized["matcher"]["method"] == matcher

    panorama = _textured_panorama()
    features = pipeline.extract(panorama, panorama_id="a")
    matches = pipeline.match(features, features)
    assert 0 < len(features) <= 240
    assert len(features) == features.descriptors.shape[0]
    assert features.descriptor_metric == metric
    assert np.allclose(np.linalg.norm(features.bearings, axis=1), 1, atol=1e-6)
    assert np.isfinite(features.source_erp_xy).all()
    assert len(matches) > 0
    assert matches.matcher_name == matcher
    assert matches.valid.all()
    assert matches.bearings_a.shape == (len(matches), 3)
    assert matches.keypoint_responses.shape == (len(matches), 2)
    assert not any(
        isinstance(item, (cv2.KeyPoint, cv2.DMatch)) for item in features.features
    )
    assert not any(
        isinstance(value, (cv2.KeyPoint, cv2.DMatch))
        for value in (getattr(matches, field.name) for field in fields(matches))
    )


def test_ratio_test_rejects_zero_distance_ties_and_uses_strict_inequality() -> None:
    backend = OpenCVFeatureBackend()
    config = FeatureMatcherConfig(method="bf", ratio_test=0.75)
    query = np.zeros((1, 32), dtype=np.uint8)
    tied_train = np.zeros((2, 32), dtype=np.uint8)
    tied, _ = backend.match(query, tied_train, "hamming", config)
    assert tied == []

    distinct_train = np.stack(
        (np.zeros(32, dtype=np.uint8), np.full(32, 255, dtype=np.uint8))
    )
    accepted, _ = backend.match(query, distinct_train, "hamming", config)
    assert len(accepted) == 1
    assert accepted[0]["distance"] == 0.0
    assert accepted[0]["ratio_score"] == 0.0

    class Match:
        def __init__(self, distance: float, index: int):
            self.distance = distance
            self.queryIdx = 0
            self.trainIdx = index

    class ImpossibleOrderMatcher:
        def knnMatch(self, descriptors_a, descriptors_b, k):
            assert k == 2
            return [[Match(1.0, 0), Match(0.0, 1)]]

    assert (
        backend._candidate_matches(ImpossibleOrderMatcher(), query, tied_train, config)
        == []
    )


def test_extraction_respects_validity_margin_alignment_and_is_reproducible() -> None:
    panorama = _textured_panorama()
    valid = np.ones(panorama.shape[:2], dtype=bool)
    valid[:, : panorama.shape[1] // 2] = False
    pipeline = SphericalFeaturePipeline.from_preset(
        "sift-bf",
        face_sampler="cube",
        face_shape_hw=128,
        face_fov_deg=100,
        edge_margin_px=12,
        max_features=300,
    )
    first = pipeline.extract(panorama, panorama_id="stable", validity_mask=valid)
    second = pipeline.extract(panorama, panorama_id="stable", validity_mask=valid)

    assert len(first) == first.descriptors.shape[0]
    assert np.array_equal(first.descriptors, second.descriptors)
    assert np.array_equal(first.pixels_xy, second.pixels_xy)
    assert np.array_equal(first.bearings, second.bearings)
    for feature in first.features:
        height, width = feature.projection_spec.output_shape_hw
        assert 11.5 <= feature.pixel_xy[0] <= width - 12.5
        assert 11.5 <= feature.pixel_xy[1] <= height - 12.5
        x, y = np.rint(feature.source_erp_xy).astype(int)
        assert valid[y, x % panorama.shape[1]]
        assert feature.descriptor_index < len(first)


def test_torch_chw_panorama_uses_opencv_backend_and_returns_public_numpy_results() -> (
    None
):
    torch = pytest.importorskip("torch")
    panorama = torch.from_numpy(_textured_panorama(128, 256)).permute(2, 0, 1)
    pipeline = SphericalFeaturePipeline.from_preset(
        "sift-bf",
        face_sampler="cube",
        face_shape_hw=96,
        face_fov_deg=100,
        edge_margin_px=4,
        max_features=80,
    )
    features = pipeline.extract(panorama)

    assert len(features) > 0
    assert isinstance(features.descriptors, np.ndarray)
    assert isinstance(features.bearings, np.ndarray)
    assert features.descriptors.dtype == np.float32


def test_high_level_pipeline_and_bearing_correspondence_contract() -> None:
    panorama = _textured_panorama()
    shifted = np.roll(panorama, 3, axis=1)
    pipeline = SphericalFeaturePipeline.from_preset(
        "sift-flann",
        face_sampler="icosahedron",
        face_shape_hw=96,
        face_fov_deg=(75, 70),
        face_overlap_deg=10,
        subdivisions=0,
        edge_margin_px=4,
        max_features=160,
    )
    matches = pipeline.extract_and_match(
        panorama, shifted, panorama_id_a="left", panorama_id_b="right"
    )
    correspondences = matches.to_bearing_correspondences()

    assert len(matches) > 0
    assert matches.panorama_id_a == "left" and matches.panorama_id_b == "right"
    assert np.array_equal(correspondences.bearings_a, matches.bearings_a)
    assert np.array_equal(correspondences.bearings_b, matches.bearings_b)
    assert np.array_equal(correspondences.valid, matches.valid)
    assert np.array_equal(correspondences.weights, matches.valid.astype(np.float32))


def test_virtual_camera_rig_matches_face_geometry() -> None:
    panorama = EquirectangularImage(_textured_panorama(64, 128))
    pipeline = SphericalFeaturePipeline.from_preset(
        "orb-hamming",
        face_sampler="cube",
        face_shape_hw=(20, 24),
        face_fov_deg=(90, 70),
    )
    rig = pipeline.build_virtual_camera_rig(panorama, panorama_id="rig-test")
    faces = panorama.views("cube", size=(20, 24), fov=(90, 70))

    assert rig.panorama_id == "rig-test"
    assert len(rig.cameras) == 6
    for camera, face in zip(rig.cameras, faces):
        assert camera.face_id == face.face_id
        assert (camera.height, camera.width) == (20, 24)
        assert np.allclose(camera.K, face.geometry.K)
        assert np.allclose(
            camera.R_panorama_from_face, face.geometry.R_panorama_from_face
        )
        assert np.array_equal(camera.translation_panorama_from_face, np.zeros(3))


def test_advanced_route_accepts_injected_opencv_objects() -> None:
    cv2 = pytest.importorskip("cv2")
    panorama = _textured_panorama()
    specs = [
        GnomonicSpec(
            center_lon_deg=0, hfov_deg=100, vfov_deg=100, output_shape_hw=(128, 128)
        ),
        GnomonicSpec(
            center_lon_deg=90, hfov_deg=100, vfov_deg=100, output_shape_hw=(128, 128)
        ),
    ]
    detector = cv2.SIFT_create(nfeatures=120)
    features = extract_opencv_features(panorama, specs, detector, edge_margin_px=4)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = match_opencv_features(features, features, matcher, knn=2)

    assert len(features) > 0
    assert len(matches) > 0
    assert matches.matcher_name == "injected"

    orb_features = extract_opencv_features(
        panorama,
        specs,
        cv2.ORB_create(nfeatures=120),
        edge_margin_px=4,
    )
    orb_matches = match_opencv_features(
        orb_features,
        orb_features,
        cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
        knn=2,
    )
    assert orb_features.extractor_name == "orb"
    assert orb_features.descriptor_type == "orb-binary"
    assert orb_features.descriptor_metric == "hamming"
    assert len(orb_matches) > 0


def test_matcher_rejects_incompatible_descriptor_semantics() -> None:
    panorama = _textured_panorama()
    base = SphericalFeaturePipeline.from_preset(
        "sift-bf", face_sampler="cube", face_shape_hw=96, max_features=60
    ).extract(panorama)
    incompatible = SphericalFeaturePipeline.from_preset(
        "orb-hamming", face_sampler="cube", face_shape_hw=96, max_features=60
    ).extract(panorama)

    with pytest.raises(ValueError, match="descriptor metric"):
        FeatureMatcher(FeatureMatcherConfig(method="bf")).match(base, incompatible)


def test_match_deduplication_is_bilateral_and_preserves_face_pair_groups() -> None:
    angle = math.radians(0.05)
    bearings = np.asarray(
        (
            (0, 0, 1),
            (math.sin(angle), 0, math.cos(angle)),
            (0, 1, 0),
        ),
        dtype=np.float64,
    )

    def feature_set(panorama_id: str, prefix: str) -> SphericalFeatureSet:
        features = []
        for index, bearing in enumerate(bearings):
            provenance = FeatureProvenance(
                interface="panorai-spherical-features/v1",
                source_panorama_checksum=panorama_id,
                projection_backend="panorai.geometry",
                projection_backend_version="test",
                face_id=f"{prefix}{index}",
            )
            features.append(
                SphericalFeature(
                    feature_id=f"{panorama_id}-{index}",
                    panorama_id=panorama_id,
                    face_id=f"{prefix}{index}",
                    pixel_xy=np.asarray((index, index), dtype=np.float64),
                    source_erp_xy=None,
                    bearing_xyz=bearing,
                    response=float(index + 1),
                    scale=1.0,
                    angle_deg=0.0,
                    octave=0,
                    descriptor_index=index,
                    valid=True,
                    projection_spec=GnomonicSpec(output_shape_hw=(8, 8)),
                    provenance=provenance,
                )
            )
        return SphericalFeatureSet(
            panorama_id=panorama_id,
            features=features,
            descriptors=np.eye(3, dtype=np.float32),
            descriptor_type="test-float32",
            descriptor_metric="l2",
            extractor_name="test",
            extractor_config={},
            backend_name="test",
            backend_version="1",
            face_set_spec=FaceSetSpec(shape_hw=(8, 8)),
            projection_backend="panorai.geometry",
            projection_backend_version="test",
            panorama_checksum=panorama_id,
        )

    class Backend:
        name = "test"
        version = "1"

        def require_version(self, minimum):
            assert minimum

        def match(self, descriptors_a, descriptors_b, descriptor_metric, config):
            return (
                [
                    {
                        "query_idx": 0,
                        "train_idx": 0,
                        "distance": 2.0,
                        "ratio_score": None,
                        "mutual": None,
                    },
                    {
                        "query_idx": 1,
                        "train_idx": 1,
                        "distance": 1.0,
                        "ratio_score": None,
                        "mutual": None,
                    },
                    {
                        "query_idx": 2,
                        "train_idx": 2,
                        "distance": 3.0,
                        "ratio_score": None,
                        "mutual": None,
                    },
                ],
                "test",
            )

    matches = FeatureMatcher(
        FeatureMatcherConfig(
            method="bf", ratio_test=None, angular_dedup_threshold_deg=0.1
        ),
        backend=Backend(),
    ).match(feature_set("a", "a"), feature_set("b", "b"))

    assert matches.feature_indices_a.tolist() == [1, 2]
    assert matches.provenance.face_pair_groups == (
        (("a1", "b1"), ("a0", "b0")),
        (("a2", "b2"),),
    )


def test_real_pycolmap_export_writes_rigs_features_and_matches(tmp_path) -> None:
    pycolmap = pytest.importorskip("pycolmap")
    panorama_a = _textured_panorama(128, 256)
    panorama_b = np.roll(panorama_a, 2, axis=1)
    pipeline = SphericalFeaturePipeline.from_preset(
        "sift-bf",
        face_sampler="cube",
        face_shape_hw=96,
        face_fov_deg=100,
        edge_margin_px=4,
        max_features=100,
    )
    features_a = pipeline.extract(panorama_a, panorama_id="a")
    features_b = pipeline.extract(panorama_b, panorama_id="b")
    matches = pipeline.match(features_a, features_b)
    rig_a = pipeline.build_virtual_camera_rig(panorama_a, panorama_id="a")
    rig_b = pipeline.build_virtual_camera_rig(panorama_b, panorama_id="b")
    database_path = tmp_path / "features.db"

    result = pipeline.export_pycolmap(
        rig=[rig_a, rig_b],
        features=[features_a, features_b],
        matches=matches,
        output_database=database_path,
    )

    assert result.interface == "panorai-pycolmap-export/v1"
    assert len(result.camera_ids) == 12
    assert len(result.image_ids) == 12
    assert set(result.panorama_rig_rotations) == {"a", "b"}
    assert all(
        np.linalg.det(
            camera.R_panorama_from_face.T @ rig.cameras[0].R_panorama_from_face
        )
        == pytest.approx(1.0, abs=1e-12)
        for rig in (rig_a, rig_b)
        for camera in rig.cameras
    )
    with pycolmap.Database.open(database_path) as database:
        assert database.num_cameras() == 12
        assert database.num_rigs() == 2
        assert database.num_frames() == 2
        assert database.num_images() == 12
        assert database.num_keypoints() == len(features_a) + len(features_b)
        assert database.num_descriptors() == len(features_a) + len(features_b)
        assert database.num_matches() == int(matches.valid.sum())
        stored_rigs = sorted(database.read_all_rigs(), key=lambda item: item.rig_id)
        assert len(stored_rigs) == 2
        for stored_rig, source_rig in zip(stored_rigs, (rig_a, rig_b)):
            reference = source_rig.cameras[0]
            reference_key = f"{source_rig.panorama_id}/{reference.face_id}"
            assert stored_rig.ref_sensor_id.id == result.camera_ids[reference_key]
            for camera in source_rig.cameras[1:]:
                camera_key = f"{source_rig.panorama_id}/{camera.face_id}"
                sensor = pycolmap.sensor_t(
                    pycolmap.SensorType.CAMERA, result.camera_ids[camera_key]
                )
                stored_transform = np.asarray(
                    stored_rig.sensor_from_rig(sensor).matrix()
                )
                expected_transform = np.concatenate(
                    (
                        camera.R_panorama_from_face.T @ reference.R_panorama_from_face,
                        np.zeros((3, 1)),
                    ),
                    axis=1,
                )
                assert np.allclose(stored_transform, expected_transform, atol=1e-12)
        for image_key, image_id in result.image_ids.items():
            expected_rows = len(result.feature_rows[image_key])
            stored_keypoints = database.read_keypoints(image_id)
            assert stored_keypoints.shape == (expected_rows, 2)
            stored_descriptors = database.read_descriptors(image_id).data
            assert stored_descriptors.shape == (expected_rows, 128)
            panorama_id, face_id = image_key.split("/", 1)
            source = features_a if panorama_id == "a" else features_b
            source_rig = rig_a if panorama_id == "a" else rig_b
            source_camera = next(
                camera for camera in source_rig.cameras if camera.face_id == face_id
            )
            rows = result.feature_rows[image_key]
            expected_keypoints = source.pixels_xy[list(rows)] + 0.5
            assert np.allclose(stored_keypoints, expected_keypoints, atol=1e-6)
            stored_K = database.read_camera(
                result.camera_ids[image_key]
            ).calibration_matrix()
            expected_K = source_camera.K.copy()
            expected_K[0, 2] += 0.5
            expected_K[1, 2] += 0.5
            assert np.allclose(stored_K, expected_K, atol=1e-12)
            if expected_rows:
                local = np.concatenate(
                    (
                        (stored_keypoints[:, :1] - stored_K[0, 2]) / stored_K[0, 0],
                        (stored_keypoints[:, 1:2] - stored_K[1, 2]) / stored_K[1, 1],
                        np.ones((expected_rows, 1)),
                    ),
                    axis=1,
                )
                reconstructed = local @ source_camera.R_panorama_from_face.T
                reconstructed /= np.linalg.norm(reconstructed, axis=1, keepdims=True)
                assert np.allclose(
                    reconstructed, source.bearings[list(rows)], atol=1e-6
                )
            assert np.array_equal(
                stored_descriptors,
                np.rint(source.descriptors[list(rows)]).astype(np.uint8),
            )
    with pytest.raises(FileExistsError, match="refusing to modify"):
        pipeline.export_pycolmap(
            rig=rig_a,
            features=features_a,
            output_database=database_path,
        )


def test_configuration_validation_is_explicit() -> None:
    assert available_presets() == (
        "sift-flann",
        "sift-bf",
        "orb-hamming",
        "akaze-hamming",
    )
    with pytest.raises(ValueError, match="unknown preset"):
        SphericalFeaturePipeline.from_preset("invented")
    with pytest.raises(ValueError, match="open interval"):
        FeatureMatcherConfig(ratio_test=1.0)
    with pytest.raises(TypeError, match="deduplicate_overlaps"):
        FeatureExtractorConfig(deduplicate_overlaps=1)
    with pytest.raises(TypeError, match="cross_check"):
        FeatureMatcherConfig(cross_check=1)
    with pytest.raises(TypeError, match="deduplicate_matches"):
        FeatureMatcherConfig(deduplicate_matches=1)
    for config_type in (FeatureExtractorConfig, FeatureMatcherConfig):
        with pytest.raises(ValueError, match="<= 180"):
            config_type(angular_dedup_threshold_deg=180.0001)
        assert (
            config_type(angular_dedup_threshold_deg=180).angular_dedup_threshold_deg
            == 180
        )
    assert SphericalFeaturePipelineConfig().minimum_opencv_version == "4.9.0"
    with pytest.raises(ValueError, match="at least 4.9.0"):
        SphericalFeaturePipelineConfig(minimum_opencv_version="4.8.0")
    with pytest.raises(ValueError, match="numeric X.Y"):
        SphericalFeaturePipelineConfig(minimum_opencv_version="latest")
    with pytest.raises(ValueError, match=r"fov_deg \+ overlap_deg"):
        SphericalFeaturePipeline.from_preset(
            "sift-flann", face_fov_deg=175, face_overlap_deg=10
        )
    with pytest.raises(ValueError, match="floating feature images"):
        FeatureExtractor(
            FeatureExtractorConfig(max_features=10, edge_margin_px=0)
        ).extract(np.full((32, 64), 999.0, dtype=np.float32))


def test_match_result_rejects_misaligned_optional_fields() -> None:
    common = dict(
        panorama_id_a="a",
        panorama_id_b="b",
        feature_indices_a=np.asarray([0]),
        feature_indices_b=np.asarray([0]),
        bearings_a=np.asarray([[0.0, 0.0, 1.0]]),
        bearings_b=np.asarray([[0.0, 0.0, 1.0]]),
        descriptor_distances=np.asarray([0.0]),
        mutual=None,
        valid=np.asarray([True]),
        matcher_name="test",
        matcher_config={},
        backend_name="test",
        backend_version="1",
        provenance=MatchProvenance(
            interface="panorai-spherical-features/v1",
            source_checksums=("a", "b"),
            face_pairs=(("face-a", "face-b"),),
            face_pair_groups=((("face-a", "face-b"),),),
            deduplicated=False,
        ),
        keypoint_responses=np.asarray([[1.0, 1.0]]),
        face_ids_a=np.asarray(["face-a"]),
        face_ids_b=np.asarray(["face-b"]),
    )
    with pytest.raises(ValueError, match="ratio_scores"):
        SphericalFeatureMatches(**common, ratio_scores=np.asarray([0.5, 0.6]))
    with pytest.raises(ValueError, match="feature_indices_a"):
        SphericalFeatureMatches(
            **{**common, "feature_indices_a": np.asarray([[0]])},
            ratio_scores=None,
        )
