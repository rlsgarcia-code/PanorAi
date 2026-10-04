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
