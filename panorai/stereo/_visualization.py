"""Dependency-light visualization helpers for spherical stereo results."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from ._dense import SphericalStereoResult


_COLORMAPS = {
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "plasma": cv2.COLORMAP_PLASMA,
    "turbo": cv2.COLORMAP_TURBO,
    "viridis": cv2.COLORMAP_VIRIDIS,
}


def colorize_spherical_range(
    range_map: np.ndarray,
    validity_mask: np.ndarray | None = None,
    *,
    value_range: tuple[float, float] | None = None,
    percentile_range: tuple[float, float] = (2.0, 98.0),
    colormap: str = "turbo",
    invalid_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Convert an HW radial-range map to an RGB ``uint8`` visualization.

    When ``value_range`` is omitted, the color limits are the requested
    percentiles of finite, positive, valid samples. Invalid pixels receive
    ``invalid_color``. The returned image is for display only; it must not be
    used as a numeric depth representation.
    """

    values, valid = _scalar_and_mask(range_map, validity_mask, positive=True)
    limits = _resolve_limits(values, valid, value_range, percentile_range)
    return _colorize(values, valid, limits, colormap, invalid_color)


def render_spherical_stereo_result(
    reference_erp: np.ndarray,
    result: SphericalStereoResult,
    *,
    target_erp: np.ndarray | None = None,
    reference_range: np.ndarray | None = None,
    value_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Render a labeled RGB panel for a spherical stereo result.

    The panel always contains the reference ERP, estimated radial range and
    accepted-pixel confidence. If supplied, the target ERP is included too.
    Supplying metric ``reference_range`` adds reference range and relative
    error panels using the same color limits as the estimate.
    """

    from ._dense import SphericalStereoResult

    if not isinstance(result, SphericalStereoResult):
        raise TypeError("result must be a SphericalStereoResult")
    shape_hw = result.range.shape
    reference_rgb = _rgb_u8(reference_erp, "reference_erp", shape_hw)
    range_values, range_valid = _scalar_and_mask(
        result.range, result.validity_mask, positive=True
    )

    reference_values: np.ndarray | None = None
    reference_valid: np.ndarray | None = None
    if reference_range is not None:
        reference_values, reference_valid = _scalar_and_mask(
            reference_range, None, positive=True
        )
        if reference_values.shape != shape_hw:
            raise ValueError("reference_range must match the result HW shape")

    limit_values = reference_values if reference_values is not None else range_values
    limit_valid = reference_valid if reference_valid is not None else range_valid
    limits = _resolve_limits(limit_values, limit_valid, value_range, (2.0, 98.0))
    estimated_rgb = _colorize(range_values, range_valid, limits, "turbo", (0, 0, 0))
    confidence_rgb = _colorize(
        result.confidence,
        result.validity_mask,
        (0.0, 1.0),
        "viridis",
        (0, 0, 0),
    )

    panels: list[tuple[str, np.ndarray]] = [("Reference ERP", reference_rgb)]
    if target_erp is not None:
        panels.append(("Target ERP", _rgb_u8(target_erp, "target_erp", shape_hw)))
    panels.extend(
        (
            ("Estimated radial range", estimated_rgb),
            ("Accepted-pixel confidence", confidence_rgb),
        )
    )
    if reference_values is not None and reference_valid is not None:
        panels.append(
            (
                "Reference radial range",
                _colorize(
                    reference_values,
                    reference_valid,
                    limits,
                    "turbo",
                    (0, 0, 0),
                ),
            )
        )
        error = np.full(shape_hw, np.nan, dtype=np.float32)
        error_valid = range_valid & reference_valid
        np.divide(
            np.abs(range_values - reference_values),
            reference_values,
            out=error,
            where=error_valid,
        )
        panels.append(
            (
                "Relative error (0 to 1)",
                _colorize(error, error_valid, (0.0, 1.0), "magma", (0, 0, 0)),
            )
        )
    return _tile_labeled(panels)


def _scalar_and_mask(
    values: np.ndarray,
    validity_mask: np.ndarray | None,
    *,
    positive: bool,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.number):
        raise ValueError("scalar map must be a numeric HW array")
    array = np.asarray(array, dtype=np.float32)
    valid = np.isfinite(array)
    if positive:
        valid &= array > 0.0
    if validity_mask is not None:
        mask = np.asarray(validity_mask)
        if mask.shape != array.shape:
            raise ValueError("validity_mask must match the scalar-map HW shape")
        if mask.dtype != np.bool_:
            raise TypeError("validity_mask must have boolean dtype")
        valid &= mask
    return array, valid


def _resolve_limits(
    values: np.ndarray,
    valid: np.ndarray,
    value_range: tuple[float, float] | None,
    percentile_range: tuple[float, float],
) -> tuple[float, float]:
    if value_range is not None:
        if len(value_range) != 2:
            raise ValueError("value_range must contain (minimum, maximum)")
        lower, upper = (float(item) for item in value_range)
        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            raise ValueError("value_range must be finite and strictly increasing")
        return lower, upper
    if len(percentile_range) != 2:
        raise ValueError("percentile_range must contain two values")
    low_percentile, high_percentile = (float(item) for item in percentile_range)
    if not (
        math.isfinite(low_percentile)
        and math.isfinite(high_percentile)
        and 0.0 <= low_percentile < high_percentile <= 100.0
    ):
        raise ValueError(
            "percentile_range must be finite, increasing, and inside [0, 100]"
        )
    if not np.any(valid):
        return 0.0, 1.0
    lower, upper = np.percentile(
        values[valid], (low_percentile, high_percentile)
    ).astype(float)
    if not lower < upper:
        scale = max(abs(lower), 1.0)
        lower -= 0.5e-6 * scale
        upper += 0.5e-6 * scale
    return lower, upper


def _colorize(
    values: np.ndarray,
    valid: np.ndarray,
    limits: tuple[float, float],
    colormap: str,
    invalid_color: tuple[int, int, int],
) -> np.ndarray:
    if colormap not in _COLORMAPS:
        supported = ", ".join(sorted(_COLORMAPS))
        raise ValueError(f"colormap must be one of: {supported}")
    if (
        len(invalid_color) != 3
        or any(isinstance(item, bool) for item in invalid_color)
        or any(not isinstance(item, (int, np.integer)) for item in invalid_color)
        or any(not 0 <= int(item) <= 255 for item in invalid_color)
    ):
        raise ValueError("invalid_color must be an RGB integer triplet in [0, 255]")
    lower, upper = limits
    normalized = np.zeros(values.shape, dtype=np.float32)
    np.subtract(values, lower, out=normalized, where=valid)
    normalized /= upper - lower
    np.clip(normalized, 0.0, 1.0, out=normalized)
    gray = np.rint(normalized * 255.0).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.applyColorMap(gray, _COLORMAPS[colormap]), cv2.COLOR_BGR2RGB)
    rgb[~valid] = np.asarray(invalid_color, dtype=np.uint8)
    return rgb


def _rgb_u8(value: np.ndarray, name: str, shape_hw: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape[:2] != shape_hw:
        raise ValueError(f"{name} must match the result HW shape")
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3 and array.shape[2] in {1, 3, 4}:
        array = np.repeat(array, 3, axis=2) if array.shape[2] == 1 else array[..., :3]
    else:
        raise ValueError(f"{name} must use HW, HW1, HWC-RGB, or HWC-RGBA layout")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.issubdtype(array.dtype, np.integer):
        maximum = float(np.iinfo(array.dtype).max)
        converted = np.rint(array.astype(np.float32) * (255.0 / maximum))
    else:
        converted = np.asarray(array, dtype=np.float32)
        if converted.size and (
            float(converted.min()) < 0.0 or float(converted.max()) > 1.0
        ):
            raise ValueError(f"floating {name} values must be in [0, 1]")
        converted = np.rint(converted * 255.0)
    return np.clip(converted, 0.0, 255.0).astype(np.uint8)


def _tile_labeled(panels: list[tuple[str, np.ndarray]]) -> np.ndarray:
    height, width = panels[0][1].shape[:2]
    columns = 2 if len(panels) <= 4 else 3
    rows = math.ceil(len(panels) / columns)
    caption_height = max(20, int(round(height * 0.12)))
    tile_height = height + caption_height
    canvas = np.zeros((rows * tile_height, columns * width, 3), dtype=np.uint8)
    font_scale = max(0.32, min(0.7, width / 640.0))
    for index, (label, image) in enumerate(panels):
        row, column = divmod(index, columns)
        top = row * tile_height
        left = column * width
        vertical = slice(top + caption_height, top + tile_height)
        horizontal = slice(left, left + width)
        canvas[vertical, horizontal] = image
        cv2.putText(
            canvas,
            label,
            (left + 5, top + caption_height - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas
