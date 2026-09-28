"""Shared input and mask handling for NumPy blenders.

Masks carry validity. Pixel values never do: zero and black are valid values
whenever their corresponding mask entry is true.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def prepare_blend_inputs(
    images: Sequence[np.ndarray], masks: Sequence[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Validate equal-shaped images and normalize masks to ``(H, W)`` bool."""

    if images is None or masks is None or len(images) == 0 or len(images) != len(masks):
        raise ValueError("Images and masks must have the same non-zero length.")

    arrays = [np.asarray(image) for image in images]
    expected_shape = arrays[0].shape
    if len(expected_shape) not in (2, 3):
        raise ValueError("Images must have shape (H, W) or (H, W, C).")
    if any(array.shape != expected_shape for array in arrays):
        raise ValueError("All images must have the same shape.")

    spatial_shape = expected_shape[:2]
    normalized_masks: list[np.ndarray] = []
    for index, (array, mask) in enumerate(zip(arrays, masks)):
        normalized = np.asarray(mask)
        if normalized.shape == spatial_shape:
            pass
        elif normalized.shape == spatial_shape + (1,):
            normalized = normalized[..., 0]
        elif array.ndim == 3 and normalized.shape == array.shape:
            normalized = np.all(normalized.astype(bool), axis=-1)
        else:
            raise ValueError(
                f"Mask {index} must have shape {spatial_shape}, "
                f"{spatial_shape + (1,)}, or {array.shape}; got {normalized.shape}."
            )

        normalized = normalized.astype(bool, copy=False)
        finite = np.isfinite(array)
        if array.ndim == 3:
            finite = np.all(finite, axis=-1)
        if np.any(normalized & ~finite):
            raise ValueError(f"Image {index} has non-finite values marked as valid.")
        normalized_masks.append(normalized)

    return arrays, normalized_masks


def expand_mask(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Broadcast a spatial mask over channels when needed."""

    return mask[..., None] if image.ndim == 3 else mask


def masked_values(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return ``image`` with unsupported samples replaced by zero safely."""

    return np.where(expand_mask(mask, image), image, 0)


def finish_blend(
    result: np.ndarray, masks: Sequence[np.ndarray], return_mask: bool
):
    """Return the array and, when requested, the union validity mask."""

    support_mask = np.logical_or.reduce(np.stack(masks, axis=0), axis=0)
    return (result, support_mask) if return_mask else result
