"""Private dispatch for the optional geometry extension."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

try:
    from panorai._native import _geometry as _native_geometry
except ImportError as exc:  # pragma: no cover - environment-dependent
    _native_geometry = None
    _native_import_error: Exception | None = exc
else:
    _native_import_error = None


def native_geometry_available() -> bool:
    """Return whether the optional geometry extension is importable."""

    return _native_geometry is not None


def supports_native_cubemap(faces: Sequence[Any]) -> bool:
    """Return whether six NumPy faces fit the fused kernel's private ABI."""

    if _native_geometry is None or len(faces) != 6:
        return False
    first = faces[0]
    if not isinstance(first, np.ndarray) or first.ndim not in {2, 3}:
        return False
    if first.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        return False
    return all(
        isinstance(face, np.ndarray)
        and face.ndim == first.ndim
        and face.shape == first.shape
        and face.dtype == first.dtype
        for face in faces
    )


def native_cubemap_to_equirectangular(
    faces: Sequence[np.ndarray],
    plans: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_shape_hw: tuple[int, int],
) -> np.ndarray:
    """Run fused selective bilinear sampling through the private extension."""

    if _native_geometry is None:  # pragma: no cover - guarded by dispatcher
        detail = (
            "unknown" if _native_import_error is None else str(_native_import_error)
        )
        raise RuntimeError(f"native geometry kernels are unavailable: {detail}")
    contiguous_faces = tuple(np.ascontiguousarray(face) for face in faces)
    contiguous_plans = tuple(
        (
            np.ascontiguousarray(indices, dtype=np.int64),
            np.ascontiguousarray(map_x, dtype=np.float64),
            np.ascontiguousarray(map_y, dtype=np.float64),
        )
        for indices, map_x, map_y in plans
    )
    payload = _native_geometry.cubemap_to_equirectangular(
        contiguous_faces,
        contiguous_plans,
        *output_shape_hw,
    )
    trailing_shape = contiguous_faces[0].shape[2:]
    return np.frombuffer(payload, dtype=contiguous_faces[0].dtype).reshape(
        (*output_shape_hw, *trailing_shape)
    )


def supports_native_gnomonic_forward(image: Any) -> bool:
    """Return whether an ERP array fits the native arbitrary-N sampler ABI."""

    return bool(
        _native_geometry is not None
        and isinstance(image, np.ndarray)
        and image.ndim in {2, 3}
        and image.dtype in {np.dtype(np.float32), np.dtype(np.float64)}
    )


def native_equirectangular_to_gnomonic_batch(
    image: np.ndarray,
    plans: Sequence[np.ndarray],
    *,
    interpolation: str,
) -> tuple[np.ndarray, ...]:
    """Sample one NumPy ERP into an ordered arbitrary-size gnomonic batch."""

    if _native_geometry is None:  # pragma: no cover - guarded by dispatcher
        detail = (
            "unknown" if _native_import_error is None else str(_native_import_error)
        )
        raise RuntimeError(f"native geometry kernels are unavailable: {detail}")
    if interpolation not in {"nearest", "bilinear"}:
        raise ValueError("interpolation must be 'nearest' or 'bilinear'")
    contiguous_image = np.ascontiguousarray(image)
    contiguous_plans = tuple(
        np.ascontiguousarray(pixel_map, dtype=np.float64) for pixel_map in plans
    )
    payloads = _native_geometry.equirectangular_to_gnomonic_batch(
        contiguous_image,
        contiguous_plans,
        1 if interpolation == "bilinear" else 0,
    )
    trailing_shape = contiguous_image.shape[2:]
    return tuple(
        np.frombuffer(payload, dtype=contiguous_image.dtype).reshape(
            (*plan.shape[:2], *trailing_shape)
        )
        for payload, plan in zip(payloads, contiguous_plans, strict=True)
    )


def supports_native_gnomonic_gaussian(
    faces: Sequence[Any], validity_masks: Sequence[Any]
) -> bool:
    """Return whether inputs fit the fused Gaussian reconstruction ABI."""

    if _native_geometry is None or not faces or len(faces) != len(validity_masks):
        return False
    first = faces[0]
    if (
        not isinstance(first, np.ndarray)
        or first.ndim not in {2, 3}
        or first.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}
    ):
        return False
    channels = first.shape[2:] if first.ndim == 3 else ()
    return all(
        isinstance(face, np.ndarray)
        and face.ndim == first.ndim
        and face.shape[2:] == channels
        and face.dtype == first.dtype
        and isinstance(mask, np.ndarray)
        and mask.dtype == np.bool_
        and tuple(mask.shape) == tuple(face.shape[:2])
        for face, mask in zip(faces, validity_masks, strict=True)
    )


def native_gnomonic_gaussian_to_equirectangular(
    faces: Sequence[np.ndarray],
    validity_masks: Sequence[np.ndarray],
    plans: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    output_shape_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse selective sampling, strict validity, and Gaussian accumulation."""

    if _native_geometry is None:  # pragma: no cover - guarded by dispatcher
        detail = (
            "unknown" if _native_import_error is None else str(_native_import_error)
        )
        raise RuntimeError(f"native geometry kernels are unavailable: {detail}")
    contiguous_faces = tuple(np.ascontiguousarray(face) for face in faces)
    contiguous_masks = tuple(
        np.ascontiguousarray(mask, dtype=np.bool_) for mask in validity_masks
    )
    contiguous_plans = tuple(
        (
            np.ascontiguousarray(indices, dtype=np.int64),
            np.ascontiguousarray(map_x, dtype=np.float64),
            np.ascontiguousarray(map_y, dtype=np.float64),
            np.ascontiguousarray(center_score, dtype=np.float64),
        )
        for indices, map_x, map_y, center_score in plans
    )
    data_payload, valid_payload = _native_geometry.gnomonic_gaussian_to_equirectangular(
        contiguous_faces,
        contiguous_masks,
        contiguous_plans,
        *output_shape_hw,
    )
    trailing_shape = contiguous_faces[0].shape[2:]
    data = np.frombuffer(data_payload, dtype=contiguous_faces[0].dtype).reshape(
        (*output_shape_hw, *trailing_shape)
    )
    valid = (
        np.frombuffer(valid_payload, dtype=np.uint8)
        .reshape(output_shape_hw)
        .astype(bool, copy=False)
    )
    return data, valid
