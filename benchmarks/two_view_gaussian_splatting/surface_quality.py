"""Conservative confidence and floater filtering for observed ERP surfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class SurfaceQualityOptions:
    """Quality gates that never create geometry outside observed support."""

    maximum_neighbor_log_jump: float = 0.28
    minimum_consistent_neighbors: int = 1
    depth_edge_log_scale: float = 0.12
    texture_std_scale: float = 0.035
    saturation_threshold: float = 0.98
    minimum_retained_opacity: float = 0.08

    def __post_init__(self) -> None:
        for name in (
            "maximum_neighbor_log_jump",
            "depth_edge_log_scale",
            "texture_std_scale",
            "saturation_threshold",
            "minimum_retained_opacity",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.minimum_consistent_neighbors < 0:
            raise ValueError("minimum_consistent_neighbors must be nonnegative")
        if self.minimum_consistent_neighbors > 4:
            raise ValueError("minimum_consistent_neighbors must not exceed four")
        if self.saturation_threshold > 1.0:
            raise ValueError("saturation_threshold must not exceed one")
        if self.minimum_retained_opacity > 1.0:
            raise ValueError("minimum_retained_opacity must not exceed one")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def assess_surface_quality(
    radial_m_hw: Any,
    valid_hw: Any,
    rgb_hwc: Any,
    *,
    base_confidence_hw: Any | None = None,
    options: SurfaceQualityOptions | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Remove isolated floaters and derive a bounded per-Gaussian opacity.

    Geometry is only removed, never added.  Opacity is reduced near depth
    discontinuities, in textureless regions, and in clipped highlights where
    photometric evidence is unreliable.
    """

    settings = options or SurfaceQualityOptions()
    radial = np.asarray(radial_m_hw, dtype=np.float32)
    valid = np.asarray(valid_hw, dtype=bool)
    rgb = np.asarray(rgb_hwc)
    if radial.ndim != 2 or valid.shape != radial.shape:
        raise ValueError("radial and validity must share shape (H, W)")
    if rgb.shape != (*radial.shape, 3):
        raise ValueError("RGB must have shape (H, W, 3) matching radial")
    finite = valid & np.isfinite(radial) & (radial > 0.0)
    safe_log = np.zeros_like(radial, dtype=np.float32)
    safe_log[finite] = np.log(radial[finite])

    neighbor_consistency: list[np.ndarray] = []
    neighbor_jump: list[np.ndarray] = []
    for shifted_log, shifted_valid in _four_neighbors(safe_log, finite):
        both = finite & shifted_valid
        jump = np.where(both, np.abs(safe_log - shifted_log), np.inf)
        neighbor_consistency.append(both & (jump <= settings.maximum_neighbor_log_jump))
        neighbor_jump.append(jump)
    consistent_count = np.sum(np.stack(neighbor_consistency), axis=0)
    cleaned = finite & (consistent_count >= settings.minimum_consistent_neighbors)

    finite_jumps = np.stack(neighbor_jump)
    minimum_jump = np.min(finite_jumps, axis=0)
    minimum_jump = np.where(np.isfinite(minimum_jump), minimum_jump, 0.0)
    edge_confidence = np.exp(-minimum_jump / settings.depth_edge_log_scale)
    edge_confidence = np.clip(edge_confidence, 0.15, 1.0)

    normalized_rgb = rgb.astype(np.float32)
    if np.issubdtype(rgb.dtype, np.integer):
        normalized_rgb /= float(np.iinfo(rgb.dtype).max)
    normalized_rgb = np.clip(normalized_rgb, 0.0, 1.0)
    luminance = (
        0.2126 * normalized_rgb[..., 0]
        + 0.7152 * normalized_rgb[..., 1]
        + 0.0722 * normalized_rgb[..., 2]
    )
    mean = _periodic_box_filter(luminance, 5)
    mean_square = _periodic_box_filter(luminance * luminance, 5)
    texture_std = np.sqrt(np.maximum(mean_square - mean * mean, 0.0))
    texture_confidence = np.clip(texture_std / settings.texture_std_scale, 0.25, 1.0)
    clipped = (np.max(normalized_rgb, axis=2) >= settings.saturation_threshold) | (
        np.min(normalized_rgb, axis=2) <= 1.0 - settings.saturation_threshold
    )
    saturation_confidence = np.where(clipped, 0.55, 1.0).astype(np.float32)

    if base_confidence_hw is None:
        base_confidence = np.ones_like(radial, dtype=np.float32)
    else:
        base_confidence = np.clip(
            np.nan_to_num(
                np.asarray(base_confidence_hw, dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            0.0,
            1.0,
        )
        if base_confidence.shape != radial.shape:
            raise ValueError("base confidence must match radial shape")
    opacity = (
        base_confidence
        * edge_confidence.astype(np.float32)
        * texture_confidence.astype(np.float32)
        * saturation_confidence
    )
    opacity = np.where(
        cleaned,
        np.clip(opacity, settings.minimum_retained_opacity, 1.0),
        0.0,
    ).astype(np.float32)
    removed = finite & ~cleaned
    return (
        cleaned,
        opacity,
        {
            "options": settings.to_dict(),
            "input_valid_pixels": int(finite.sum()),
            "retained_pixels": int(cleaned.sum()),
            "removed_floaters": int(removed.sum()),
            "retained_fraction_of_input": float(
                cleaned.sum() / max(int(finite.sum()), 1)
            ),
            "mean_opacity_retained": (
                float(np.mean(opacity[cleaned])) if np.any(cleaned) else 0.0
            ),
            "low_texture_pixels": int(
                np.count_nonzero(cleaned & (texture_confidence < 1.0))
            ),
            "clipped_pixels": int(np.count_nonzero(cleaned & clipped)),
        },
    )


def _four_neighbors(
    values: np.ndarray, valid: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    left_values = np.roll(values, 1, axis=1)
    left_valid = np.roll(valid, 1, axis=1)
    right_values = np.roll(values, -1, axis=1)
    right_valid = np.roll(valid, -1, axis=1)
    up_values = np.empty_like(values)
    up_values[0] = values[0]
    up_values[1:] = values[:-1]
    up_valid = np.zeros_like(valid)
    up_valid[1:] = valid[:-1]
    down_values = np.empty_like(values)
    down_values[-1] = values[-1]
    down_values[:-1] = values[1:]
    down_valid = np.zeros_like(valid)
    down_valid[:-1] = valid[1:]
    return [
        (left_values, left_valid),
        (right_values, right_valid),
        (up_values, up_valid),
        (down_values, down_valid),
    ]


def _periodic_box_filter(values: np.ndarray, kernel_size: int) -> np.ndarray:
    radius = kernel_size // 2
    horizontal = np.concatenate(
        (values[:, -radius:], values, values[:, :radius]), axis=1
    )
    padded = np.pad(horizontal, ((radius, radius), (0, 0)), mode="edge")
    filtered = cv2.boxFilter(
        padded,
        ddepth=-1,
        ksize=(kernel_size, kernel_size),
        normalize=True,
        borderType=cv2.BORDER_CONSTANT,
    )
    return filtered[radius:-radius, radius:-radius]


__all__ = ["SurfaceQualityOptions", "assess_surface_quality"]
