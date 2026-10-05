"""Spherical Canny edge detection."""

from __future__ import annotations

import math

import numpy as np

from ._filters import spherical_gaussian_blur, spherical_gradient
from ._sampling import row_chunks, sample_tangent, validate_image, validate_step


def _nonmaximum_suppression(
    magnitude: np.ndarray, orientation: np.ndarray, step: float
) -> np.ndarray:
    height, width = magnitude.shape
    kept = np.zeros_like(magnitude, dtype=np.float64)
    for rows in row_chunks(height, width):
        angle = orientation[rows]
        east = np.cos(angle) * step
        north = np.sin(angle) * step
        forward = sample_tangent(magnitude, rows, east, north)
        backward = sample_tangent(magnitude, rows, -east, -north)
        center = magnitude[rows]
        kept[rows] = np.where((center >= forward) & (center >= backward), center, 0.0)
    return kept


def _tangent_dilate(mask: np.ndarray, step: float) -> np.ndarray:
    height, width = mask.shape
    source = mask.astype(np.uint8, copy=False)
    result = mask.copy()
    for rows in row_chunks(height, width):
        expanded = result[rows].copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                expanded |= sample_tangent(
                    source,
                    rows,
                    dx * step,
                    -dy * step,
                    interpolation="nearest",
                ).astype(bool)
        result[rows] = expanded
    return result


def spherical_canny(
    image: np.ndarray,
    threshold_low: float,
    threshold_high: float,
    *,
    gaussian_ksize: int = 5,
    gaussian_sigma: float = 1.0,
    gradient_operator: str = "sobel",
    angular_step_deg: float | None = None,
    l2_gradient: bool = True,
    backend: str = "auto",
) -> np.ndarray:
    """Detect thin, hysteresis-connected edges over geodesic neighbourhoods.

    The stages mirror Canny—Gaussian denoising, tangent Sobel/Scharr
    derivatives, directional non-maximum suppression and double-threshold
    hysteresis—but all spatial samples use spherical exponential-map offsets.
    The result is an ``HW uint8`` image containing only 0 and 255.
    """

    source = validate_image(image, grayscale=True)
    for name, value in (
        ("threshold_low", threshold_low),
        ("threshold_high", threshold_high),
    ):
        if (
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"{name} must be a finite non-negative real number")
    low, high = float(threshold_low), float(threshold_high)
    if low > high:
        raise ValueError("threshold_low must be less than or equal to threshold_high")
    step = validate_step(angular_step_deg, source.shape[0])
    smooth = spherical_gaussian_blur(
        source,
        gaussian_ksize,
        gaussian_sigma,
        angular_step_deg=angular_step_deg,
        backend=backend,
    )
    gradient = spherical_gradient(
        smooth,
        operator=gradient_operator,
        angular_step_deg=angular_step_deg,
        backend=backend,
    )
    magnitude = (
        gradient.magnitude
        if l2_gradient
        else np.abs(gradient.east) + np.abs(gradient.north)
    )
    thinned = _nonmaximum_suppression(magnitude, gradient.orientation, step)
    positive = thinned > 0.0
    strong = positive & (thinned >= high)
    weak = positive & (thinned >= low)
    connected = strong.copy()
    while True:
        grown = weak & _tangent_dilate(connected, step)
        updated = connected | grown
        if np.array_equal(updated, connected):
            break
        connected = updated
    return np.where(connected, 255, 0).astype(np.uint8)
