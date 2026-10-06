"""Gaussian and Laplacian pyramids for equirectangular panoramas."""

from __future__ import annotations

import numpy as np

from ._filters import spherical_gaussian_blur
from ._sampling import validate_image
from ._transforms import spherical_resize


def _validate_levels(levels: int) -> int:
    if isinstance(levels, bool) or not isinstance(levels, (int, np.integer)):
        raise TypeError("levels must be a positive integer")
    value = int(levels)
    if value < 1:
        raise ValueError("levels must be a positive integer")
    return value


def spherical_gaussian_pyramid(
    image: np.ndarray,
    levels: int,
    *,
    sigma: float = 1.0,
    ksize: int = 5,
    backend: str = "auto",
) -> tuple[np.ndarray, ...]:
    """Build ``levels`` ERP octaves after spherical Gaussian prefiltering."""

    source = validate_image(image)
    count = _validate_levels(levels)
    pyramid = [source.copy()]
    for _ in range(1, count):
        previous = pyramid[-1]
        if min(previous.shape[:2]) <= 2:
            raise ValueError("requested levels would reduce an ERP below 2x2")
        blurred = spherical_gaussian_blur(
            previous, ksize=ksize, sigma=sigma, backend=backend
        )
        next_shape = (
            max(2, (previous.shape[0] + 1) // 2),
            max(2, (previous.shape[1] + 1) // 2),
        )
        pyramid.append(spherical_resize(blurred, next_shape))
    return tuple(pyramid)


def spherical_laplacian_pyramid(
    image: np.ndarray,
    levels: int,
    *,
    sigma: float = 1.0,
    ksize: int = 5,
    backend: str = "auto",
) -> tuple[np.ndarray, ...]:
    """Build a reconstructible spherical Laplacian pyramid."""

    gaussian = spherical_gaussian_pyramid(
        image, levels, sigma=sigma, ksize=ksize, backend=backend
    )
    laplacian = []
    for fine, coarse in zip(gaussian[:-1], gaussian[1:], strict=True):
        expanded = spherical_resize(coarse, fine.shape[:2])
        laplacian.append(fine.astype(np.float64) - expanded.astype(np.float64))
    laplacian.append(gaussian[-1].astype(np.float64, copy=True))
    return tuple(laplacian)


def reconstruct_laplacian_pyramid(
    pyramid: tuple[np.ndarray, ...] | list[np.ndarray],
) -> np.ndarray:
    """Reconstruct the finest ERP from ``spherical_laplacian_pyramid``."""

    if not isinstance(pyramid, (tuple, list)) or not pyramid:
        raise ValueError("pyramid must be a non-empty tuple or list of arrays")
    for level in pyramid:
        validate_image(level)
    result = np.asarray(pyramid[-1], dtype=np.float64).copy()
    for level in reversed(pyramid[:-1]):
        if level.ndim != result.ndim or level.shape[2:] != result.shape[2:]:
            raise ValueError("all pyramid levels must use a consistent layout")
        result = spherical_resize(result, level.shape[:2]) + level
    return result
