from __future__ import annotations

import math

import numpy as np
import pytest

from panorai.calibration import (
    PinholeIntrinsics,
    RotationCalibrationOptions,
    calibrate_rotation_pair,
    pinhole_pixels_to_rays,
    pinhole_rays_to_pixels,
)


def _rotation_y(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    return np.asarray(
        (
            (math.cos(angle), 0.0, math.sin(angle)),
            (0.0, 1.0, 0.0),
            (-math.sin(angle), 0.0, math.cos(angle)),
        )
    )


def _overlap_correspondences(
    intrinsics: PinholeIntrinsics, rotation_b_from_a: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.meshgrid(np.linspace(80.5, 559.5, 16), np.linspace(60.5, 419.5, 12))
    pixels_a = np.column_stack((x.ravel(), y.ravel()))
    rays_a = pinhole_pixels_to_rays(pixels_a, intrinsics)
    rays_b = rays_a @ rotation_b_from_a.T
    pixels_b, positive = pinhole_rays_to_pixels(rays_b, intrinsics)
    height, width = intrinsics.image_size_hw
    inside = (
        positive
        & (pixels_b[:, 0] >= -0.5)
        & (pixels_b[:, 0] <= width - 0.5)
        & (pixels_b[:, 1] >= -0.5)
        & (pixels_b[:, 1] <= height - 0.5)
    )
    return pixels_a[inside], pixels_b[inside]


def test_pinhole_pixel_ray_round_trip_uses_pixel_centres_and_y_up() -> None:
    intrinsics = PinholeIntrinsics.from_horizontal_fov((480, 640), 90.0)
    pixels = np.asarray(((319.5, 239.5), (100.25, 50.75), (600.0, 400.0)))

    rays = pinhole_pixels_to_rays(pixels, intrinsics)
    recovered, positive = pinhole_rays_to_pixels(rays, intrinsics)

    np.testing.assert_allclose(recovered, pixels, atol=1e-12)
    np.testing.assert_array_equal(positive, np.ones(3, dtype=bool))
    np.testing.assert_allclose(rays[0], (0.0, 0.0, 1.0), atol=1e-12)
    assert rays[1, 1] > 0.0


def test_rotation_pair_calibration_recovers_unknown_shared_fov_and_rotation() -> None:
    expected_intrinsics = PinholeIntrinsics.from_horizontal_fov((480, 640), 82.0)
    expected_rotation = _rotation_y(18.0)
    pixels_a, pixels_b = _overlap_correspondences(
        expected_intrinsics, expected_rotation
    )

    result = calibrate_rotation_pair(
        pixels_a,
        pixels_b,
        (480, 640),
        (480, 640),
        options=RotationCalibrationOptions(
            max_angular_error_deg=0.05,
            min_inliers=30,
            min_spatial_coverage=0.01,
        ),
    )

    assert result.estimated_fov
    assert result.observable
    assert result.degeneracy_reasons == ()
    assert result.intrinsics_a.horizontal_fov_deg == pytest.approx(82.0, abs=1e-3)
    assert result.intrinsics_b.horizontal_fov_deg == pytest.approx(82.0, abs=1e-3)
    np.testing.assert_allclose(result.rotation_b_from_a, expected_rotation, atol=1e-7)
    assert result.median_angular_error_deg < 1e-5


def test_calibration_reports_weak_spatial_evidence_without_hiding_pose() -> None:
    intrinsics = PinholeIntrinsics.from_horizontal_fov((240, 320), 75.0)
    pixels_a = np.asarray(
        ((158.0, 118.0), (160.0, 118.0), (158.0, 120.0), (160.0, 120.0), (159.0, 119.0))
    )
    rays = pinhole_pixels_to_rays(pixels_a, intrinsics) @ _rotation_y(5.0).T
    pixels_b, _ = pinhole_rays_to_pixels(rays, intrinsics)

    result = calibrate_rotation_pair(
        pixels_a,
        pixels_b,
        (240, 320),
        (240, 320),
        horizontal_fov_a_deg=75.0,
        horizontal_fov_b_deg=75.0,
        options=RotationCalibrationOptions(min_inliers=3, min_spatial_coverage=0.1),
    )

    assert not result.observable
    assert "insufficient-spatial-coverage" in result.degeneracy_reasons
    assert result.num_inliers == len(pixels_a)


def test_calibration_rejects_planar_translation_as_central_rotation() -> None:
    intrinsics = PinholeIntrinsics.from_horizontal_fov((480, 640), 82.0)
    x, y = np.meshgrid(np.linspace(-2.0, 2.0, 16), np.linspace(-1.5, 1.5, 12))
    world_points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 5.0)))
    pixels_a, _ = pinhole_rays_to_pixels(world_points, intrinsics)
    pixels_b, _ = pinhole_rays_to_pixels(
        world_points - np.asarray((0.75, 0.0, 0.0)), intrinsics
    )

    result = calibrate_rotation_pair(
        pixels_a,
        pixels_b,
        (480, 640),
        (480, 640),
        horizontal_fov_a_deg=82.0,
        horizontal_fov_b_deg=82.0,
        options=RotationCalibrationOptions(
            max_angular_error_deg=1.0,
            max_median_angular_error_deg=0.35,
            min_inliers=12,
            min_spatial_coverage=0.0,
        ),
    )

    assert not result.observable
    assert "central-rotation-residual-too-large" in result.degeneracy_reasons
    assert result.median_angular_error_deg > 0.35
