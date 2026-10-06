"""Solid-angle-aware histogram operations for ERP imagery."""

from __future__ import annotations

import numpy as np


def spherical_equalize_histogram(
    image: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    area_weighted: bool = True,
) -> np.ndarray:
    """Equalize an ``HW uint8`` histogram over spherical solid angle.

    By default, each ERP row is weighted by its exact relative solid angle.
    Set ``area_weighted=False`` for the ordinary pixel-count algorithm used by
    ``cv2.equalizeHist``.  ``mask`` selects samples used to build the mapping;
    the resulting lookup table is applied to the complete image.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim != 2:
        raise ValueError("image must use grayscale HW layout")
    if image.dtype != np.uint8:
        raise TypeError("image must use uint8 dtype")
    height, width = image.shape
    if height < 1 or width < 1:
        raise ValueError("image dimensions must be positive")
    if mask is None:
        selected = np.ones(image.shape, dtype=bool)
    else:
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
            raise TypeError("mask must be a boolean numpy.ndarray")
        if mask.shape != image.shape:
            raise ValueError(f"mask must have shape {image.shape}; got {mask.shape}")
        selected = mask
    if not selected.any():
        raise ValueError("mask must select at least one pixel")

    if area_weighted:
        latitude = np.pi / 2.0 - (np.arange(height) + 0.5) * np.pi / height
        weights = np.broadcast_to(np.cos(latitude)[:, None], image.shape)[selected]
    else:
        weights = np.ones(int(selected.sum()), dtype=np.float64)
    histogram = np.bincount(image[selected], weights=weights, minlength=256)
    nonzero = np.flatnonzero(histogram)
    if nonzero.size <= 1:
        return image.copy()
    first = nonzero[0]
    cumulative = np.cumsum(histogram)
    denominator = cumulative[-1] - histogram[first]
    lookup = np.zeros(256, dtype=np.uint8)
    values = (cumulative[first + 1 :] - histogram[first]) * (255.0 / denominator)
    lookup[first + 1 :] = np.rint(values).clip(0, 255).astype(np.uint8)
    return lookup[image]
