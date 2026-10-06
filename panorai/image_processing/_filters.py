"""Geometry-aware filtering and smoothing on equirectangular panoramas."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from . import _native
from ._sampling import (
    output_dtype,
    row_chunks,
    sample_tangent,
    validate_image,
    validate_step,
)


def _validate_kernel(kernel: np.ndarray) -> np.ndarray:
    values = np.asarray(kernel)
    if values.ndim != 2:
        raise ValueError("kernel must be a two-dimensional array")
    if values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("kernel dimensions must be positive")
    if values.shape[0] % 2 == 0 or values.shape[1] % 2 == 0:
        raise ValueError("kernel dimensions must be odd")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise TypeError("kernel must use a real numeric dtype")
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("kernel values must be finite")
    return values


def _filter_numpy(
    image: np.ndarray, kernel: np.ndarray, angular_step_rad: float
) -> np.ndarray:
    height, width = image.shape[:2]
    trailing = image.shape[2:]
    result = np.empty((height, width, *trailing), dtype=np.float64)
    anchor_y, anchor_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    offsets = [
        (coefficient, x - anchor_x, y - anchor_y)
        for y, row in enumerate(kernel)
        for x, coefficient in enumerate(row)
        if coefficient != 0.0
    ]
    for rows in row_chunks(height, width):
        filtered = np.zeros(
            (rows.stop - rows.start, width, *trailing), dtype=np.float64
        )
        for coefficient, dx, dy in offsets:
            filtered += coefficient * sample_tangent(
                image,
                rows,
                dx * angular_step_rad,
                -dy * angular_step_rad,
            )
        result[rows] = filtered
    return result.astype(output_dtype(image), copy=False)


def spherical_filter2d(
    image: np.ndarray,
    kernel: np.ndarray,
    *,
    angular_step_deg: float | None = None,
    backend: str = "auto",
) -> np.ndarray:
    """Correlate an ``HW``/``HWC`` ERP with a tangent-plane kernel.

    The kernel is placed in the local east/down tangent frame at each ERP
    pixel and its taps are carried onto the sphere with the exponential map.
    This matches OpenCV's correlation convention (``filter2D`` does not flip
    the kernel) while removing the latitude-dependent scale of planar ERP
    filtering.
    """

    source = validate_image(image)
    weights = _validate_kernel(kernel)
    step = validate_step(angular_step_deg, source.shape[0])
    radius = math.hypot(weights.shape[1] // 2, weights.shape[0] // 2) * step
    if radius >= math.pi:
        raise ValueError("kernel radius must be smaller than 180 degrees")
    if backend not in {"auto", "numpy", "native"}:
        raise ValueError("backend must be 'auto', 'numpy', or 'native'")
    supported = _native.supports_native_filter(source)
    if backend == "native" and not supported:
        raise RuntimeError(
            "native backend requires the compiled extension and float32/float64 HW/HWC input"
        )
    if backend != "numpy" and supported:
        return _native.native_spherical_filter2d(source, weights, step)
    return _filter_numpy(source, weights, step)


def spherical_box_blur(
    image: np.ndarray,
    ksize: int = 3,
    *,
    normalize: bool = True,
    angular_step_deg: float | None = None,
    backend: str = "auto",
) -> np.ndarray:
    """Apply a local tangent-plane averaging (or unnormalized box) filter."""

    size = _validate_ksize(ksize)
    kernel = np.ones((size, size), dtype=np.float64)
    if normalize:
        kernel /= kernel.size
    return spherical_filter2d(
        image, kernel, angular_step_deg=angular_step_deg, backend=backend
    )


def _validate_ksize(ksize: int) -> int:
    if isinstance(ksize, bool) or not isinstance(ksize, (int, np.integer)):
        raise TypeError("ksize must be a positive odd integer")
    size = int(ksize)
    if size < 1 or size % 2 == 0:
        raise ValueError("ksize must be a positive odd integer")
    return size


def _gaussian_kernel(ksize: int, sigma: float) -> np.ndarray:
    size = _validate_ksize(ksize)
    if isinstance(sigma, bool):
        raise TypeError("sigma must be a finite positive real number")
    sigma_value = float(sigma)
    if not math.isfinite(sigma_value) or sigma_value <= 0.0:
        raise ValueError("sigma must be a finite positive real number")
    axis = np.arange(size, dtype=np.float64) - size // 2
    one_dimensional = np.exp(-0.5 * (axis / sigma_value) ** 2)
    one_dimensional /= one_dimensional.sum()
    return np.outer(one_dimensional, one_dimensional)


def spherical_gaussian_blur(
    image: np.ndarray,
    ksize: int = 5,
    sigma: float = 1.0,
    *,
    angular_step_deg: float | None = None,
    backend: str = "auto",
) -> np.ndarray:
    """Smooth an ERP with a Gaussian defined in every local tangent plane."""

    return spherical_filter2d(
        image,
        _gaussian_kernel(ksize, sigma),
        angular_step_deg=angular_step_deg,
        backend=backend,
    )


def spherical_median_blur(
    image: np.ndarray,
    ksize: int = 3,
    *,
    angular_step_deg: float | None = None,
) -> np.ndarray:
    """Replace each sample with its geodesic tangent-neighbourhood median."""

    source = validate_image(image)
    size = _validate_ksize(ksize)
    step = validate_step(angular_step_deg, source.shape[0])
    height, width = source.shape[:2]
    trailing = source.shape[2:]
    result = np.empty((height, width, *trailing), dtype=output_dtype(source))
    radius = size // 2
    for rows in row_chunks(height, width):
        samples = [
            sample_tangent(source, rows, dx * step, -dy * step)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
        ]
        result[rows] = np.median(np.stack(samples, axis=0), axis=0)
    return result


def spherical_bilateral_filter(
    image: np.ndarray,
    ksize: int = 5,
    sigma_color: float = 25.0,
    sigma_space: float = 1.5,
    *,
    angular_step_deg: float | None = None,
) -> np.ndarray:
    """Edge-preserving bilateral smoothing in a geodesic neighbourhood."""

    source = validate_image(image)
    size = _validate_ksize(ksize)
    for name, value in (("sigma_color", sigma_color), ("sigma_space", sigma_space)):
        if (
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(f"{name} must be a finite positive real number")
    color_scale = 2.0 * float(sigma_color) ** 2
    space_scale = 2.0 * float(sigma_space) ** 2
    step = validate_step(angular_step_deg, source.shape[0])
    height, width = source.shape[:2]
    trailing = source.shape[2:]
    result = np.empty((height, width, *trailing), dtype=output_dtype(source))
    radius = size // 2
    for rows in row_chunks(height, width):
        center = source[rows].astype(np.float64)
        numerator = np.zeros_like(center, dtype=np.float64)
        denominator = np.zeros(center.shape[:2], dtype=np.float64)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                sample = sample_tangent(source, rows, dx * step, -dy * step)
                difference = sample - center
                if source.ndim == 3:
                    difference = np.sum(difference * difference, axis=-1)
                else:
                    difference = difference * difference
                weight = np.exp(
                    -difference / color_scale - (dx * dx + dy * dy) / space_scale
                )
                numerator += sample * (
                    weight[..., None] if source.ndim == 3 else weight
                )
                denominator += weight
        result[rows] = numerator / (
            denominator[..., None] if source.ndim == 3 else denominator
        )
    return result


@dataclass(frozen=True)
class SphericalGradient:
    """East/north first derivatives and their magnitude/orientation."""

    east: np.ndarray
    north: np.ndarray
    magnitude: np.ndarray
    orientation: np.ndarray


def _gradient_kernels(operator: str) -> tuple[np.ndarray, np.ndarray]:
    if operator == "sobel":
        east = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        north = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
        return east / 8.0, north / 8.0
    if operator == "scharr":
        east = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float64)
        north = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=np.float64)
        return east / 32.0, north / 32.0
    raise ValueError("operator must be 'sobel' or 'scharr'")


def spherical_sobel(
    image: np.ndarray,
    *,
    direction: str,
    operator: str = "sobel",
    angular_step_deg: float | None = None,
    normalize_by_angle: bool = False,
    backend: str = "auto",
) -> np.ndarray:
    """Return the local east or north Sobel/Scharr derivative."""

    source = validate_image(image)
    east, north = _gradient_kernels(operator)
    if direction not in {"east", "north"}:
        raise ValueError("direction must be 'east' or 'north'")
    result = spherical_filter2d(
        source,
        east if direction == "east" else north,
        angular_step_deg=angular_step_deg,
        backend=backend,
    )
    if normalize_by_angle:
        result = result / validate_step(angular_step_deg, source.shape[0])
    return result


def spherical_gradient(
    image: np.ndarray,
    *,
    operator: str = "sobel",
    angular_step_deg: float | None = None,
    normalize_by_angle: bool = False,
    backend: str = "auto",
) -> SphericalGradient:
    """Compute signed east/north derivatives in every local tangent frame."""

    east = spherical_sobel(
        image,
        direction="east",
        operator=operator,
        angular_step_deg=angular_step_deg,
        normalize_by_angle=normalize_by_angle,
        backend=backend,
    )
    north = spherical_sobel(
        image,
        direction="north",
        operator=operator,
        angular_step_deg=angular_step_deg,
        normalize_by_angle=normalize_by_angle,
        backend=backend,
    )
    return SphericalGradient(
        east=east,
        north=north,
        magnitude=np.hypot(east, north),
        orientation=np.arctan2(north, east),
    )


def spherical_laplacian(
    image: np.ndarray,
    *,
    angular_step_deg: float | None = None,
    normalize_by_angle: bool = False,
    backend: str = "auto",
) -> np.ndarray:
    """Apply the four-neighbour tangent-plane Laplacian."""

    source = validate_image(image)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    result = spherical_filter2d(
        source, kernel, angular_step_deg=angular_step_deg, backend=backend
    )
    if normalize_by_angle:
        step = validate_step(angular_step_deg, source.shape[0])
        result = result / (step * step)
    return result
