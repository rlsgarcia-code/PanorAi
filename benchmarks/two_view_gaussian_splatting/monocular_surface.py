"""Landmark calibration and two-view filtering for monocular ERP surfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class PriorConsistencyOptions:
    """Conservative gates for accepting a monocular surface sample."""

    maximum_log_depth_disagreement: float = 0.15
    maximum_rgb_l1: float = 0.16
    min_range_m: float = 0.3
    max_range_m: float = 15.0

    def __post_init__(self) -> None:
        for name in (
            "maximum_log_depth_disagreement",
            "maximum_rgb_l1",
            "min_range_m",
            "max_range_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.maximum_rgb_l1 > 1.0:
            raise ValueError("maximum_rgb_l1 must not exceed one")
        if self.max_range_m <= self.min_range_m:
            raise ValueError("max_range_m must exceed min_range_m")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def load_and_calibrate_prior(
    path: str | Path,
    output_shape_hw: tuple[int, int],
    landmark_points_camera: Any,
    *,
    min_range_m: float = 0.3,
    max_range_m: float = 15.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a native prior and fit a robust monotonic log-depth calibration."""

    native = np.load(path, mmap_mode="r")
    if native.ndim != 2:
        raise ValueError("monocular prior must have shape HW")
    points = np.asarray(landmark_points_camera, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("landmark_points_camera must have shape (N, 3)")
    ranges = np.linalg.norm(points, axis=1)
    pixels = _landmark_pixels(points, native.shape)
    samples = np.asarray(native[pixels[:, 1], pixels[:, 0]], dtype=np.float64)
    valid_pairs = (
        np.isfinite(samples)
        & (samples > 0.0)
        & np.isfinite(ranges)
        & (ranges >= min_range_m)
        & (ranges <= max_range_m)
    )
    if int(valid_pairs.sum()) < 6:
        raise ValueError("at least six valid landmark/prior pairs are required")
    exponent, offset = _robust_log_affine(
        np.log(samples[valid_pairs]), np.log(ranges[valid_pairs])
    )

    native_valid = np.isfinite(native) & (native > 0.0)
    output_width = output_shape_hw[1]
    output_height = output_shape_hw[0]
    numerator = cv2.resize(
        np.where(native_valid, native, 0.0).astype(np.float32),
        (output_width, output_height),
        interpolation=cv2.INTER_AREA,
    )
    support = cv2.resize(
        native_valid.astype(np.float32),
        (output_width, output_height),
        interpolation=cv2.INTER_AREA,
    )
    reduced = numerator / np.maximum(support, 1e-6)
    calibrated = np.exp(offset) * np.maximum(reduced, 1e-6) ** exponent
    valid = (
        (support >= 0.99)
        & np.isfinite(calibrated)
        & (calibrated >= min_range_m)
        & (calibrated <= max_range_m)
    )
    radial = np.full(output_shape_hw, np.nan, dtype=np.float32)
    radial[valid] = calibrated[valid].astype(np.float32)
    fitted = np.exp(offset) * samples[valid_pairs] ** exponent
    residual = np.abs(np.log(fitted / ranges[valid_pairs]))
    return (
        radial,
        valid,
        {
            "source_path": str(path),
            "native_shape_hw": list(native.shape),
            "output_shape_hw": list(output_shape_hw),
            "landmark_count": int(valid_pairs.sum()),
            "log_depth_exponent": float(exponent),
            "log_depth_offset": float(offset),
            "median_landmark_abs_log": float(np.median(residual)),
            "p90_landmark_abs_log": float(np.quantile(residual, 0.9)),
            "valid_pixels": int(valid.sum()),
            "valid_fraction": float(valid.mean()),
        },
    )


def filter_prior_by_other_view(
    reference_radial_m: Any,
    reference_valid_hw: Any,
    other_radial_m: Any,
    other_valid_hw: Any,
    reference_rgb_hwc: Any,
    other_rgb_hwc: Any,
    reference_support_hw: Any,
    other_support_hw: Any,
    rotation_other_from_reference: Any,
    translation_other_from_reference_m: Any,
    *,
    options: PriorConsistencyOptions | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Keep prior samples corroborated geometrically and photometrically."""

    settings = options or PriorConsistencyOptions()
    reference = np.asarray(reference_radial_m, dtype=np.float32)
    other = np.asarray(other_radial_m, dtype=np.float32)
    reference_valid = np.asarray(reference_valid_hw, dtype=bool)
    other_valid = np.asarray(other_valid_hw, dtype=bool)
    reference_rgb = _normalized_rgb(reference_rgb_hwc)
    other_rgb = _normalized_rgb(other_rgb_hwc)
    reference_support = np.asarray(reference_support_hw, dtype=bool)
    other_support = np.asarray(other_support_hw, dtype=bool)
    shape = reference.shape
    for name, value in (
        ("other radial", other),
        ("reference validity", reference_valid),
        ("other validity", other_valid),
        ("reference support", reference_support),
        ("other support", other_support),
    ):
        if value.shape != shape:
            raise ValueError(f"{name} must match reference shape")
    if reference_rgb.shape != (*shape, 3) or other_rgb.shape != (*shape, 3):
        raise ValueError("RGB arrays must match radial shape")

    rays = _erp_rays_numpy(shape)
    points = rays * np.nan_to_num(reference, nan=1.0)[..., None]
    rotation = np.asarray(rotation_other_from_reference, dtype=np.float64)
    translation = np.asarray(
        translation_other_from_reference_m, dtype=np.float64
    ).reshape(3)
    transformed = points @ rotation.T + translation
    transformed_range = np.linalg.norm(transformed, axis=2)
    safe_range = np.maximum(transformed_range, 1e-8)
    bearings = transformed / safe_range[..., None]
    map_x, map_y = _bearings_to_maps(bearings, shape)
    vertical_in_bounds = (map_y >= -0.5) & (map_y <= shape[0] - 0.5)
    sampled_other = _remap_erp(
        other,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        border_value=float("nan"),
    )
    sampled_other_valid = _remap_erp(
        other_valid.astype(np.uint8),
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        border_value=0,
    ).astype(bool)
    sampled_other_support = _remap_erp(
        other_support.astype(np.uint8),
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        border_value=0,
    ).astype(bool)
    sampled_rgb = _remap_erp(
        other_rgb,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        border_value=0.0,
    )
    depth_disagreement = np.abs(
        np.log(np.maximum(sampled_other, 1e-8) / np.maximum(transformed_range, 1e-8))
    )
    rgb_l1 = np.mean(np.abs(sampled_rgb - reference_rgb), axis=2)
    base = (
        reference_valid
        & sampled_other_valid
        & vertical_in_bounds
        & reference_support
        & sampled_other_support
        & np.isfinite(depth_disagreement)
        & np.isfinite(rgb_l1)
        & (transformed_range >= settings.min_range_m)
        & (transformed_range <= settings.max_range_m)
    )
    accepted = (
        base
        & (depth_disagreement <= settings.maximum_log_depth_disagreement)
        & (rgb_l1 <= settings.maximum_rgb_l1)
    )
    confidence = np.zeros(shape, dtype=np.float32)
    confidence[accepted] = np.exp(
        -depth_disagreement[accepted] / settings.maximum_log_depth_disagreement
        - rgb_l1[accepted] / settings.maximum_rgb_l1
    ).astype(np.float32)
    radial = np.full(shape, np.nan, dtype=np.float32)
    radial[accepted] = reference[accepted]
    return (
        radial,
        accepted,
        confidence,
        {
            "options": settings.to_dict(),
            "projectable_pixels": int(base.sum()),
            "accepted_pixels": int(accepted.sum()),
            "accepted_fraction": float(accepted.mean()),
            "median_log_depth_disagreement": float(
                np.median(depth_disagreement[accepted])
                if np.any(accepted)
                else math.nan
            ),
            "median_rgb_l1": float(
                np.median(rgb_l1[accepted]) if np.any(accepted) else math.nan
            ),
        },
    )


def merge_prior_with_stereo(
    stereo_radial_m: Any,
    stereo_valid_hw: Any,
    stereo_confidence_hw: Any,
    prior_radial_m: Any,
    prior_valid_hw: Any,
    prior_confidence_hw: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill stereo holes with corroborated prior while preserving stereo exactly."""

    stereo = np.asarray(stereo_radial_m, dtype=np.float32)
    stereo_valid = np.asarray(stereo_valid_hw, dtype=bool)
    prior = np.asarray(prior_radial_m, dtype=np.float32)
    prior_valid = np.asarray(prior_valid_hw, dtype=bool)
    if prior.shape != stereo.shape or prior_valid.shape != stereo.shape:
        raise ValueError("prior and stereo arrays must share shape")
    merged_valid = stereo_valid | prior_valid
    merged = np.full(stereo.shape, np.nan, dtype=np.float32)
    merged[prior_valid] = prior[prior_valid]
    merged[stereo_valid] = stereo[stereo_valid]
    confidence = np.zeros(stereo.shape, dtype=np.float32)
    confidence[prior_valid] = np.asarray(prior_confidence_hw, dtype=np.float32)[
        prior_valid
    ]
    confidence[stereo_valid] = np.asarray(stereo_confidence_hw, dtype=np.float32)[
        stereo_valid
    ]
    return merged, merged_valid, confidence


def _robust_log_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack((x, np.ones_like(x)))
    parameters = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(12):
        residual = design @ parameters - y
        scale = max(1.4826 * float(np.median(np.abs(residual))), 1e-4)
        cutoff = 1.345 * scale
        weights = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), 1e-12))
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_y = y * np.sqrt(weights)
        updated = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]
        if np.max(np.abs(updated - parameters)) < 1e-8:
            parameters = updated
            break
        parameters = updated
    exponent = float(np.clip(parameters[0], 0.5, 2.0))
    offset = float(np.median(y - exponent * x))
    return exponent, offset


def _landmark_pixels(points: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    ranges = np.linalg.norm(points, axis=1)
    bearings = points / np.maximum(ranges[:, None], 1e-12)
    map_x, map_y = _bearings_to_maps(bearings, shape_hw)
    x = np.mod(np.rint(map_x).astype(np.int64), shape_hw[1])
    y = np.clip(np.rint(map_y).astype(np.int64), 0, shape_hw[0] - 1)
    return np.stack((x, y), axis=1)


def _erp_rays_numpy(shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    rows = np.arange(height, dtype=np.float32)
    columns = np.arange(width, dtype=np.float32)
    latitude = np.pi / 2.0 - (rows + 0.5) / height * np.pi
    longitude = (columns + 0.5) / width * (2.0 * np.pi) - np.pi
    lat, lon = np.meshgrid(latitude, longitude, indexing="ij")
    cos_lat = np.cos(lat)
    return np.stack((cos_lat * np.sin(lon), np.sin(lat), cos_lat * np.cos(lon)), axis=2)


def _bearings_to_maps(
    bearings: np.ndarray, shape_hw: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape_hw
    longitude = np.arctan2(bearings[..., 0], bearings[..., 2])
    latitude = np.arcsin(np.clip(bearings[..., 1], -1.0, 1.0))
    map_x = ((longitude + np.pi) / (2.0 * np.pi) * width - 0.5).astype(np.float32)
    map_y = ((np.pi / 2.0 - latitude) / np.pi * height - 0.5).astype(np.float32)
    return map_x, map_y


def _remap_erp(
    array: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    interpolation: int,
    *,
    border_value: float | int,
) -> np.ndarray:
    """Remap with periodic longitude and nonperiodic latitude."""

    horizontal = np.concatenate((array[:, -1:], array, array[:, :1]), axis=1)
    padded = np.concatenate((horizontal[:1], horizontal, horizontal[-1:]), axis=0)
    clipped_y = np.clip(map_y, 0.0, array.shape[0] - 1.0)
    nearest_y = np.rint(clipped_y)
    clipped_y = np.where(
        np.abs(clipped_y - nearest_y) < 1e-4, nearest_y, clipped_y
    ).astype(np.float32)
    return cv2.remap(
        padded,
        map_x + 1.0,
        clipped_y + 1.0,
        interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _normalized_rgb(value: Any) -> np.ndarray:
    rgb = np.asarray(value)
    result = rgb.astype(np.float32)
    if np.issubdtype(rgb.dtype, np.integer):
        result /= float(np.iinfo(rgb.dtype).max)
    return np.clip(result, 0.0, 1.0)


__all__ = [
    "PriorConsistencyOptions",
    "filter_prior_by_other_view",
    "load_and_calibrate_prior",
    "merge_prior_with_stereo",
]
