from __future__ import annotations

import numpy as np
import pytest

from panorai.features import FeatureExtractorConfig, FeatureMatcherConfig
from panorai.reconstruction import SphericalGlobalMapperOptions
from panorai.slam import (
    EquidistantFisheyeCamera,
    SphericalSLAMOptions,
    SphericalVisualSLAM,
)
from scripts.run_hilti_slam import _load_tum, _reference_motion, _umeyama


def _camera(shape=(240, 320)) -> EquidistantFisheyeCamera:
    height, width = shape
    return EquidistantFisheyeCamera(
        image_shape_hw=shape,
        focal_xy=(95.0, 96.0),
        principal_xy=((width - 1) / 2, (height - 1) / 2),
        distortion=(0.02, -0.01, 0.001, -0.0001),
        max_theta_deg=100.0,
    )


def test_fisheye_pixels_to_rays_matches_opencv_oracle():
    cv2 = pytest.importorskip("cv2")
    camera = _camera()
    pixels = np.asarray(
        (
            camera.principal_xy,
            (40.25, 52.75),
            (278.5, 185.0),
            (160.0, 12.0),
        ),
        dtype=np.float64,
    )
    projection = camera.pixels_to_rays(pixels)
    K = np.asarray(
        (
            (camera.focal_xy[0], 0.0, camera.principal_xy[0]),
            (0.0, camera.focal_xy[1], camera.principal_xy[1]),
            (0.0, 0.0, 1.0),
        )
    )
    normalized = cv2.fisheye.undistortPoints(
        pixels.reshape(-1, 1, 2), K, np.asarray(camera.distortion)
    ).reshape(-1, 2)
    oracle = np.column_stack(
        (normalized[:, 0], -normalized[:, 1], np.ones(len(normalized)))
    )
    oracle /= np.linalg.norm(oracle, axis=1, keepdims=True)
    np.testing.assert_allclose(projection.rays_xyz, oracle, atol=2e-10, rtol=2e-10)
    assert projection.valid.tolist() == [True] * len(pixels)
    np.testing.assert_allclose(projection.rays_xyz[0], (0.0, 0.0, 1.0), atol=1e-14)


def test_fisheye_geometry_marks_outside_and_over_fov_invalid():
    camera = _camera()
    pixels = np.asarray(
        (
            (-1.0, 100.0),
            (1000.0, 100.0),
            camera.principal_xy,
            (0.0, 0.0),
        )
    )
    projection = camera.pixels_to_rays(pixels)
    assert projection.valid[0:2].tolist() == [False, False]
    assert projection.valid[2]
    assert np.isnan(projection.rays_xyz[~projection.valid]).all()
    mask = camera.valid_pixel_mask(edge_margin_px=8)
    assert mask.dtype == np.bool_
    assert mask.shape == camera.image_shape_hw
    assert not mask[:8].any()
    assert not mask[:, :8].any()


def test_kalibr_loader_accepts_opencv_yaml_marker(tmp_path):
    calibration = tmp_path / "calibration.yaml"
    calibration.write_text(
        "%YAML:1.0\n"
        "cam0:\n"
        "  camera_model: pinhole\n"
        "  distortion_model: equidistant\n"
        "  intrinsics: [100.0, 101.0, 160.0, 120.0]\n"
        "  distortion_coeffs: [0.1, 0.01, 0.001, 0.0001]\n"
        "  resolution: [320, 240]\n",
        encoding="utf-8",
    )
    camera = EquidistantFisheyeCamera.from_kalibr_yaml(calibration)
    assert camera.image_shape_hw == (240, 320)
    assert camera.focal_xy == (100.0, 101.0)
    assert camera.principal_xy == (160.0, 120.0)


def test_visual_slam_real_opencv_frontend_is_deterministic_and_non_mutating():
    pytest.importorskip("cv2")
    rng = np.random.default_rng(20261003)
    image = rng.integers(0, 256, size=(240, 320), dtype=np.uint8)
    original = image.copy()
    options = SphericalSLAMOptions(
        temporal_window=2,
        min_features_per_frame=20,
        min_matches_per_pair=10,
        edge_margin_px=8,
        extractor=FeatureExtractorConfig(
            method="sift", max_features=256, edge_margin_px=0
        ),
        matcher=FeatureMatcherConfig(
            method="bf", ratio_test=0.8, deduplicate_matches=False
        ),
        mapper=SphericalGlobalMapperOptions(min_panoramas=4),
    )
    slam = SphericalVisualSLAM(_camera(), options=options)
    for index in range(3):
        summary = slam.add_frame(
            image,
            timestamp_s=float(index),
            frame_id=f"frame-{index}",
        )
        assert summary is not None
        assert summary.feature_count >= 20
    result = slam.finish()
    assert not result.success
    assert result.failure_reasons == ("insufficient-keyframes",)
    assert result.diagnostics.input_frame_count == 3
    assert result.diagnostics.keyframe_count == 3
    assert result.diagnostics.candidate_pair_count == 3
    assert result.diagnostics.retained_pair_count == 3
    assert all(count >= 10 for _, count in result.diagnostics.pair_match_counts)
    assert result.describe()["interface"] == "panorai-spherical-slam/v1"
    np.testing.assert_array_equal(image, original)
    with pytest.raises(RuntimeError, match="only once"):
        slam.finish()


def test_visual_slam_rejects_bad_shape_and_nonincreasing_time():
    options = SphericalSLAMOptions(
        min_features_per_frame=1,
        mapper=SphericalGlobalMapperOptions(min_panoramas=3),
    )
    slam = SphericalVisualSLAM(_camera(), options=options)
    with pytest.raises(ValueError, match="spatial shape"):
        slam.add_frame(np.zeros((20, 20), dtype=np.uint8), timestamp_s=0.0)

    image = np.random.default_rng(4).integers(0, 256, size=(240, 320), dtype=np.uint8)
    assert slam.add_frame(image, timestamp_s=1.0, frame_id="a") is not None
    with pytest.raises(ValueError, match="increase strictly"):
        slam.add_frame(image, timestamp_s=1.0, frame_id="b")
    with pytest.raises(ValueError, match="duplicate frame_id"):
        slam.add_frame(image, timestamp_s=2.0, frame_id="a")


def test_hilti_evaluator_recovers_known_sim3_and_tum_format(tmp_path):
    rng = np.random.default_rng(90)
    predicted = rng.normal(size=(12, 3))
    angle = 0.37
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    target = (2.4 * (rotation @ predicted.T)).T + np.asarray((4.0, -2.0, 1.5))
    scale, recovered_rotation, translation = _umeyama(predicted, target)
    aligned = (scale * (recovered_rotation @ predicted.T)).T + translation
    np.testing.assert_allclose(aligned, target, atol=1e-12)

    trajectory = tmp_path / "groundtruth.txt"
    rows = np.column_stack(
        (
            np.arange(len(target), dtype=np.float64),
            target,
            np.zeros((len(target), 3)),
            np.ones(len(target)),
        )
    )
    np.savetxt(trajectory, rows, header="timestamp tx ty tz qx qy qz qw")
    timestamps, positions = _load_tum(trajectory)
    np.testing.assert_allclose(timestamps, np.arange(len(target)))
    np.testing.assert_allclose(positions, target)
    motion = _reference_motion(timestamps, rows)
    assert motion["step_count"] == len(target) - 1
    assert motion["interval_s"]["median"] == pytest.approx(1.0)
    assert motion["rotation_deg"]["max"] == pytest.approx(0.0)
