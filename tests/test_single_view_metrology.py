from __future__ import annotations

import math

import numpy as np
import pytest

from panorai.experimental.single_view_metrology import (
    SphericalLineObservation,
    calibrate_floor,
    erp_pixel_to_ray,
    estimate_direction_family,
    measure_floor_point,
    measure_vertical,
)


SHAPE = (180, 360)
HEIGHT = 1.6


def _pixel(ray: tuple[float, float, float]) -> tuple[float, float]:
    value = np.asarray(ray, dtype=np.float64)
    value /= np.linalg.norm(value)
    longitude = math.atan2(value[0], value[2])
    latitude = math.asin(value[1])
    return (
        ((longitude + math.pi) / (2 * math.pi)) * SHAPE[1] - 0.5,
        ((math.pi / 2 - latitude) / math.pi) * SHAPE[0] - 0.5,
    )


def _line(
    family: str, a: tuple[float, float, float], b: tuple[float, float, float]
) -> SphericalLineObservation:
    return SphericalLineObservation(family, _pixel(a), _pixel(b))


def _rotation(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    value = np.asarray(axis, dtype=np.float64)
    value /= np.linalg.norm(value)
    skew = np.asarray(
        [[0, -value[2], value[1]], [value[2], 0, -value[0]], [-value[1], value[0], 0]],
        dtype=np.float64,
    )
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def _calibration():
    lines = [
        _line("a", (-3, -HEIGHT, 2), (3, -HEIGHT, 2)),
        _line("a", (-3, -HEIGHT, 5), (3, -HEIGHT, 5)),
        _line("b", (2, -HEIGHT, 1), (2, -HEIGHT, 6)),
        _line("b", (-2, -HEIGHT, 1), (-2, -HEIGHT, 6)),
    ]
    return calibrate_floor(
        lines,
        SHAPE,
        family_a="a",
        family_b="b",
        camera_height_m=HEIGHT,
        floor_hint_xy=_pixel((0, -HEIGHT, 3)),
    )


def test_floor_calibration_and_metric_ray_plane_intersection() -> None:
    calibration = _calibration()
    measured = measure_floor_point(calibration, _pixel((2, -HEIGHT, 4)))

    assert measured.point_world_m[2] == pytest.approx(0, abs=1e-12)
    assert measured.horizontal_distance_m == pytest.approx(math.sqrt(20), rel=1e-12)
    assert measured.radial_distance_m == pytest.approx(
        math.sqrt(4 + HEIGHT**2 + 16), rel=1e-12
    )
    assert (
        max(family.median_residual_deg for family in calibration.direction_families)
        < 1e-10
    )
    assert calibration.family_separation_deg == pytest.approx(90.0, abs=1e-10)
    assert calibration.floor_normal_uncertainty_deg < 1e-10
    assert (
        float(
            calibration.floor_normal_panorama
            @ erp_pixel_to_ray(_pixel((0, -HEIGHT, 3)), SHAPE)
        )
        < 0
    )


def test_vertical_height_uses_floor_base_and_rejects_horizon() -> None:
    calibration = _calibration()
    base = _pixel((2, -HEIGHT, 4))
    top = _pixel((2, 1.0 - HEIGHT, 4))
    measured = measure_vertical(calibration, base, top)

    assert measured.height_m == pytest.approx(1.0, rel=1e-12)
    assert measured.horizontal_residual_m < 1e-12
    with pytest.raises(ValueError, match="horizon"):
        measure_floor_point(calibration, _pixel((0, 0, 1)))


def test_family_fit_is_robust_to_one_outlier_and_reports_membership() -> None:
    lines = [
        _line("a", (-3, -HEIGHT, 2), (3, -HEIGHT, 2)),
        _line("a", (-3, -HEIGHT, 5), (3, -HEIGHT, 5)),
        _line("a", (-4, -HEIGHT, -3), (4, -HEIGHT, -3)),
        _line("a", (2, -HEIGHT, 1), (2, -HEIGHT, 6)),
    ]

    estimate = estimate_direction_family("a", lines, SHAPE, max_residual_deg=0.5)

    assert estimate.inlier_indices == (0, 1, 2)
    assert estimate.outlier_indices == (3,)
    assert estimate.p90_residual_deg < 1e-9
    assert abs(
        float(estimate.direction_panorama @ np.asarray([1.0, 0.0, 0.0]))
    ) == pytest.approx(1.0, abs=1e-12)


def test_family_fit_handles_an_erp_seam_crossing() -> None:
    lines = [
        _line("a", (-2, -HEIGHT, -4), (2, -HEIGHT, -4)),
        _line("a", (-3, -HEIGHT, -7), (3, -HEIGHT, -7)),
    ]

    estimate = estimate_direction_family("a", lines, SHAPE)

    assert estimate.outlier_indices == ()
    assert abs(
        float(estimate.direction_panorama @ np.asarray([1.0, 0.0, 0.0]))
    ) == pytest.approx(1.0, abs=1e-12)


def test_non_axis_aligned_panorama_recovers_floor_normal_and_metric_height() -> None:
    panorama_from_base = _rotation((1.0, 2.0, 3.0), 0.57)

    def transformed_pixel(ray):
        return _pixel(tuple(panorama_from_base @ np.asarray(ray, dtype=np.float64)))

    def transformed_line(family, a, b):
        return SphericalLineObservation(
            family, transformed_pixel(a), transformed_pixel(b)
        )

    lines = [
        transformed_line("a", (-3, -HEIGHT, 2), (3, -HEIGHT, 2)),
        transformed_line("a", (-3, -HEIGHT, 5), (3, -HEIGHT, 5)),
        transformed_line("b", (2, -HEIGHT, 1), (2, -HEIGHT, 6)),
        transformed_line("b", (-2, -HEIGHT, 1), (-2, -HEIGHT, 6)),
    ]
    calibration = calibrate_floor(
        lines,
        SHAPE,
        family_a="a",
        family_b="b",
        camera_height_m=HEIGHT,
        floor_hint_xy=transformed_pixel((0, -HEIGHT, 3)),
    )

    expected_up = panorama_from_base @ np.asarray([0.0, 1.0, 0.0])
    floor = measure_floor_point(calibration, transformed_pixel((2, -HEIGHT, 4)))
    vertical = measure_vertical(
        calibration,
        transformed_pixel((2, -HEIGHT, 4)),
        transformed_pixel((2, 1.0 - HEIGHT, 4)),
    )

    assert calibration.floor_normal_panorama == pytest.approx(expected_up, abs=1e-12)
    assert floor.horizontal_distance_m == pytest.approx(math.sqrt(20), rel=1e-12)
    assert vertical.height_m == pytest.approx(1.0, rel=1e-12)


def test_pixel_center_poles_are_finite_and_longitude_independent() -> None:
    north_a = erp_pixel_to_ray((-0.5, -0.5), SHAPE)
    north_b = erp_pixel_to_ray((SHAPE[1] - 0.5, -0.5), SHAPE)
    south = erp_pixel_to_ray((SHAPE[1] / 2 - 0.5, SHAPE[0] - 0.5), SHAPE)

    assert north_a == pytest.approx([0.0, 1.0, 0.0], abs=1e-15)
    assert north_b == pytest.approx([0.0, 1.0, 0.0], abs=1e-15)
    assert south == pytest.approx([0.0, -1.0, 0.0], abs=1e-15)


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda: erp_pixel_to_ray((0, 0), (0, 360)), "dimensions"),
        (lambda: erp_pixel_to_ray((math.nan, 0), SHAPE), "finite"),
        (lambda: estimate_direction_family("a", [], SHAPE), "at least two"),
        (
            lambda: estimate_direction_family(
                "a",
                [
                    SphericalLineObservation("a", (1, 1), (1, 1)),
                    SphericalLineObservation("a", (2, 2), (3, 3)),
                ],
                SHAPE,
            ),
            "degenerate",
        ),
    ],
)
def test_invalid_or_unobservable_inputs_fail_explicitly(
    operation, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        operation()


def test_calibration_rejects_duplicate_or_nearly_parallel_families() -> None:
    lines = [
        _line("a", (-3, -HEIGHT, 2), (3, -HEIGHT, 2)),
        _line("a", (-3, -HEIGHT, 5), (3, -HEIGHT, 5)),
        _line("b", (-2, -HEIGHT, 1), (2, -HEIGHT, 1)),
        _line("b", (-2, -HEIGHT, 6), (2, -HEIGHT, 6)),
    ]
    kwargs = {
        "family_a": "a",
        "camera_height_m": HEIGHT,
        "floor_hint_xy": _pixel((0, -HEIGHT, 3)),
    }

    with pytest.raises(ValueError, match="distinct"):
        calibrate_floor(lines, SHAPE, family_b="a", **kwargs)
    with pytest.raises(ValueError, match="independently observable"):
        calibrate_floor(lines, SHAPE, family_b="b", **kwargs)


def test_vertical_can_reject_a_top_ray_inconsistent_with_the_floor_base() -> None:
    calibration = _calibration()
    with pytest.raises(ValueError, match="inconsistent"):
        measure_vertical(
            calibration,
            _pixel((2, -HEIGHT, 4)),
            _pixel((3, 1.0 - HEIGHT, 4)),
            max_horizontal_residual_m=0.05,
        )
