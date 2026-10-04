"""Optional first-party native kernels for spherical reconstruction."""

from __future__ import annotations

from typing import Any

import numpy as np


try:
    from panorai._native import _essential as _native_extension
except ImportError as exc:  # pragma: no cover - environment-dependent
    _native_extension = None
    _native_import_error: Exception | None = exc
else:
    _native_import_error = None


def native_bundle_kernels_available() -> bool:
    """Return whether the compiled spherical-BA kernel is importable."""

    return _native_extension is not None and hasattr(
        _native_extension, "spherical_ba_residual_jacobians"
    )


def resolve_bundle_backend(backend: str) -> str:
    if not isinstance(backend, str):
        raise TypeError("bundle compute backend must be a string")
    normalized = backend.strip().lower()
    if normalized not in {"auto", "numpy", "native"}:
        raise ValueError("bundle compute backend must be 'auto', 'numpy', or 'native'")
    if normalized == "auto":
        return "native" if native_bundle_kernels_available() else "numpy"
    if normalized == "native" and not native_bundle_kernels_available():
        detail = (
            "unknown" if _native_import_error is None else str(_native_import_error)
        )
        raise RuntimeError(f"native spherical-BA kernels are unavailable: {detail}")
    return normalized


def spherical_ba_residual_jacobians(
    measured_bearings: Any,
    rotations: Any,
    centers: Any,
    points: Any,
    camera_indices: Any,
    point_indices: Any,
    rotation_deltas: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate native residuals and local ``(2, 3)`` Jacobian blocks."""

    resolve_bundle_backend("native")
    measured = np.ascontiguousarray(measured_bearings, dtype=np.float64)
    rotation_values = np.ascontiguousarray(rotations, dtype=np.float64)
    center_values = np.ascontiguousarray(centers, dtype=np.float64)
    point_values = np.ascontiguousarray(points, dtype=np.float64)
    camera_values = np.ascontiguousarray(camera_indices, dtype=np.int64)
    point_index_values = np.ascontiguousarray(point_indices, dtype=np.int64)
    delta_values = np.ascontiguousarray(rotation_deltas, dtype=np.float64)
    count = measured.shape[0] if measured.ndim == 2 else 0
    if measured.shape != (count, 3) or not np.all(np.isfinite(measured)):
        raise ValueError("measured_bearings must be finite with shape (N, 3)")
    if count and not np.allclose(
        np.linalg.norm(measured, axis=1), 1.0, atol=1e-10, rtol=1e-10
    ):
        raise ValueError("measured_bearings must contain unit vectors")
    if (
        rotation_values.ndim != 3
        or rotation_values.shape[1:] != (3, 3)
        or not np.all(np.isfinite(rotation_values))
    ):
        raise ValueError("rotations must be finite with shape (C, 3, 3)")
    camera_count = rotation_values.shape[0]
    if center_values.shape != (camera_count, 3) or not np.all(
        np.isfinite(center_values)
    ):
        raise ValueError("centers must be finite with shape (C, 3)")
    if (
        point_values.ndim != 2
        or point_values.shape[1:] != (3,)
        or not np.all(np.isfinite(point_values))
    ):
        raise ValueError("points must be finite with shape (P, 3)")
    if camera_values.shape != (count,) or point_index_values.shape != (count,):
        raise ValueError("observation indices must have shape (N,)")
    if delta_values.shape != (camera_count, 3) or not np.all(np.isfinite(delta_values)):
        raise ValueError("rotation_deltas must be finite with shape (C, 3)")
    payload = _native_extension.spherical_ba_residual_jacobians(
        measured,
        rotation_values,
        center_values,
        point_values,
        camera_values,
        point_index_values,
        delta_values,
    )
    residuals = np.frombuffer(payload[0], dtype=np.float64).copy().reshape(count, 2)
    blocks = tuple(
        np.frombuffer(item, dtype=np.float64).copy().reshape(count, 2, 3)
        for item in payload[1:]
    )
    return residuals, blocks[0], blocks[1], blocks[2]
