"""Experimental spherical single-view metrology.

The API estimates a level frame from parallel line families in one ERP, fixes
metric scale with a measured camera height, and intersects spherical rays with
the floor or a vertical through a floor point.  It is intentionally separate
from pairwise Essential geometry: no translation is inferred from one view.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _unit(value: FloatArray, *, name: str) -> FloatArray:
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm < 1e-10:
        raise ValueError(f"{name} is degenerate")
    return value / norm


def erp_pixel_to_ray(xy: tuple[float, float], shape_hw: tuple[int, int]) -> FloatArray:
    """Return a canonical ``(+X right, +Y up, +Z forward)`` unit ray."""

    height, width = shape_hw
    if height <= 0 or width <= 0:
        raise ValueError("ERP dimensions must be positive")
    x, y = xy
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("pixel coordinates must be finite")
    longitude = ((x + 0.5) / width) * 2.0 * math.pi - math.pi
    latitude = math.pi / 2.0 - ((y + 0.5) / height) * math.pi
    cos_latitude = math.cos(latitude)
    return np.asarray(
        [
            cos_latitude * math.sin(longitude),
            math.sin(latitude),
            cos_latitude * math.cos(longitude),
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class SphericalLineObservation:
    family: str
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]


@dataclass(frozen=True)
class DirectionFamilyEstimate:
    family: str
    direction_panorama: FloatArray
    inlier_indices: tuple[int, ...]
    outlier_indices: tuple[int, ...]
    residuals_deg: tuple[float, ...]
    median_residual_deg: float
    p90_residual_deg: float


@dataclass(frozen=True)
class FloorCalibration:
    shape_hw: tuple[int, int]
    rotation_panorama_to_world: FloatArray
    camera_height_m: float
    floor_normal_panorama: FloatArray
    direction_families: tuple[DirectionFamilyEstimate, DirectionFamilyEstimate]
    floor_hint_xy: tuple[float, float]
    family_separation_deg: float
    floor_normal_uncertainty_deg: float

    @property
    def camera_center_world(self) -> FloatArray:
        return np.asarray([0.0, 0.0, self.camera_height_m], dtype=np.float64)


@dataclass(frozen=True)
class FloorPointMeasurement:
    point_world_m: FloatArray
    horizontal_distance_m: float
    radial_distance_m: float


@dataclass(frozen=True)
class VerticalMeasurement:
    base_world_m: FloatArray
    top_world_m: FloatArray
    height_m: float
    horizontal_residual_m: float


def _line_normal(
    line: SphericalLineObservation, shape_hw: tuple[int, int]
) -> FloatArray:
    a = erp_pixel_to_ray(line.start_xy, shape_hw)
    b = erp_pixel_to_ray(line.end_xy, shape_hw)
    return _unit(np.cross(a, b), name=f"line family {line.family}")


def estimate_direction_family(
    family: str,
    observations: Iterable[SphericalLineObservation],
    shape_hw: tuple[int, int],
    *,
    max_residual_deg: float = 3.0,
) -> DirectionFamilyEstimate:
    if not math.isfinite(max_residual_deg) or not 0 < max_residual_deg < 90:
        raise ValueError("max_residual_deg must be finite and between 0 and 90")
    selected = [item for item in observations if item.family == family]
    if len(selected) < 2:
        raise ValueError(f"family {family!r} requires at least two lines")
    normals = np.stack([_line_normal(item, shape_hw) for item in selected])
    candidates: list[tuple[int, float, FloatArray, NDArray[np.int64]]] = []
    for first, second in combinations(range(len(selected)), 2):
        raw_direction = np.cross(normals[first], normals[second])
        if float(np.linalg.norm(raw_direction)) < 1e-10:
            continue
        candidate = _unit(raw_direction, name=f"direction family {family}")
        residuals = np.degrees(
            np.arcsin(np.clip(np.abs(normals @ candidate), 0.0, 1.0))
        )
        active = np.flatnonzero(residuals <= max_residual_deg)
        if len(active) >= 2:
            candidates.append(
                (len(active), -float(np.median(residuals[active])), candidate, active)
            )
    if not candidates:
        raise ValueError(f"family {family!r} has fewer than two robust inliers")
    _, _, _, active = max(candidates, key=lambda item: (item[0], item[1]))
    _, _, vh = np.linalg.svd(normals[active], full_matrices=True)
    direction = _unit(vh[-1], name=f"direction family {family}")
    residuals = np.degrees(np.arcsin(np.clip(np.abs(normals @ direction), 0.0, 1.0)))
    active = np.flatnonzero(residuals <= max_residual_deg)
    if len(active) < 2:
        raise ValueError(f"family {family!r} has fewer than two robust inliers")
    outliers = np.setdiff1d(np.arange(len(selected)), active)
    return DirectionFamilyEstimate(
        family=family,
        direction_panorama=direction,
        inlier_indices=tuple(int(index) for index in active),
        outlier_indices=tuple(int(index) for index in outliers),
        residuals_deg=tuple(float(value) for value in residuals),
        median_residual_deg=float(np.median(residuals[active])),
        p90_residual_deg=float(np.percentile(residuals[active], 90)),
    )


def calibrate_floor(
    observations: Iterable[SphericalLineObservation],
    shape_hw: tuple[int, int],
    *,
    family_a: str,
    family_b: str,
    camera_height_m: float,
    floor_hint_xy: tuple[float, float],
    max_residual_deg: float = 3.0,
    floor_hint_horizon_epsilon: float = 0.02,
) -> FloorCalibration:
    """Estimate a leveled frame from two independent floor direction families."""

    if not math.isfinite(camera_height_m) or camera_height_m <= 0:
        raise ValueError("camera_height_m must be positive and finite")
    if family_a == family_b:
        raise ValueError("floor direction families must be distinct")
    if (
        not math.isfinite(floor_hint_horizon_epsilon)
        or not 0 < floor_hint_horizon_epsilon < 1
    ):
        raise ValueError(
            "floor_hint_horizon_epsilon must be finite and between 0 and 1"
        )
    items = tuple(observations)
    first = estimate_direction_family(
        family_a, items, shape_hw, max_residual_deg=max_residual_deg
    )
    second = estimate_direction_family(
        family_b, items, shape_hw, max_residual_deg=max_residual_deg
    )
    x_axis = _unit(first.direction_panorama, name="first floor direction")
    second_axis = _unit(second.direction_panorama, name="second floor direction")
    acute_separation = math.degrees(
        math.acos(np.clip(abs(float(x_axis @ second_axis)), 0.0, 1.0))
    )
    if acute_separation < 15.0:
        raise ValueError("floor direction families are not independently observable")
    up_axis = _unit(np.cross(x_axis, second_axis), name="floor normal")
    floor_hint = erp_pixel_to_ray(floor_hint_xy, shape_hw)
    hint_height = float(floor_hint @ up_axis)
    if abs(hint_height) <= floor_hint_horizon_epsilon:
        raise ValueError("floor hint is too close to the fitted horizon")
    if hint_height > 0:
        up_axis = -up_axis
    y_axis = _unit(np.cross(up_axis, x_axis), name="second leveled axis")
    rotation = np.stack([x_axis, y_axis, up_axis])
    uncertainty = math.hypot(first.p90_residual_deg, second.p90_residual_deg)
    uncertainty /= max(math.sin(math.radians(acute_separation)), 1e-10)
    return FloorCalibration(
        shape_hw=shape_hw,
        rotation_panorama_to_world=rotation,
        camera_height_m=float(camera_height_m),
        floor_normal_panorama=up_axis,
        direction_families=(first, second),
        floor_hint_xy=floor_hint_xy,
        family_separation_deg=float(acute_separation),
        floor_normal_uncertainty_deg=float(uncertainty),
    )


def measure_floor_point(
    calibration: FloorCalibration,
    xy: tuple[float, float],
    *,
    horizon_epsilon: float = 1e-3,
) -> FloorPointMeasurement:
    if not math.isfinite(horizon_epsilon) or not 0 < horizon_epsilon < 1:
        raise ValueError("horizon_epsilon must be finite and between 0 and 1")
    ray_world = calibration.rotation_panorama_to_world @ erp_pixel_to_ray(
        xy, calibration.shape_hw
    )
    if ray_world[2] >= -horizon_epsilon:
        raise ValueError("ray does not intersect the floor safely below the horizon")
    scale = -calibration.camera_height_m / float(ray_world[2])
    point = calibration.camera_center_world + scale * ray_world
    return FloorPointMeasurement(
        point_world_m=point,
        horizontal_distance_m=float(np.linalg.norm(point[:2])),
        radial_distance_m=float(scale),
    )


def measure_vertical(
    calibration: FloorCalibration,
    base_xy: tuple[float, float],
    top_xy: tuple[float, float],
    *,
    max_horizontal_residual_m: float | None = None,
) -> VerticalMeasurement:
    if max_horizontal_residual_m is not None and (
        not math.isfinite(max_horizontal_residual_m) or max_horizontal_residual_m <= 0
    ):
        raise ValueError("max_horizontal_residual_m must be positive and finite")
    base = measure_floor_point(calibration, base_xy).point_world_m
    ray = calibration.rotation_panorama_to_world @ erp_pixel_to_ray(
        top_xy, calibration.shape_hw
    )
    horizontal_norm = float(ray[:2] @ ray[:2])
    if horizontal_norm < 1e-10:
        raise ValueError("top ray is not observable against the base vertical")
    scale = float(ray[:2] @ base[:2]) / horizontal_norm
    if scale <= 0:
        raise ValueError("top ray intersects behind the camera")
    ray_point = calibration.camera_center_world + scale * ray
    top = np.asarray([base[0], base[1], ray_point[2]], dtype=np.float64)
    height = float(top[2])
    if height < 0:
        raise ValueError("estimated top lies below the floor")
    horizontal_residual = float(np.linalg.norm(ray_point[:2] - base[:2]))
    if (
        max_horizontal_residual_m is not None
        and horizontal_residual > max_horizontal_residual_m
    ):
        raise ValueError("top ray is inconsistent with the base vertical")
    return VerticalMeasurement(
        base_world_m=base,
        top_world_m=top,
        height_m=height,
        horizontal_residual_m=horizontal_residual,
    )


__all__ = [
    "DirectionFamilyEstimate",
    "FloorCalibration",
    "FloorPointMeasurement",
    "SphericalLineObservation",
    "VerticalMeasurement",
    "calibrate_floor",
    "erp_pixel_to_ray",
    "estimate_direction_family",
    "measure_floor_point",
    "measure_vertical",
]
