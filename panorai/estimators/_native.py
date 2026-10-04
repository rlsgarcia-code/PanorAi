"""Dispatch to optional first-party compiled estimator kernels."""

from __future__ import annotations

from typing import Any

import numpy as np


try:
    from panorai._native import _essential as _native_essential
except ImportError as exc:  # pragma: no cover - environment-dependent
    _native_essential = None
    _native_import_error: Exception | None = exc
else:
    _native_import_error = None


def native_kernels_available() -> bool:
    """Return whether the optional first-party C++ kernels are importable."""

    return _native_essential is not None


def resolve_compute_backend(backend: str) -> str:
    if not isinstance(backend, str):
        raise TypeError("compute backend must be a string")
    normalized = backend.strip().lower()
    if normalized not in {"auto", "numpy", "native"}:
        raise ValueError("compute backend must be 'auto', 'numpy', or 'native'")
    if normalized == "auto":
        return "native" if native_kernels_available() else "numpy"
    if normalized == "native" and not native_kernels_available():
        detail = (
            "unknown" if _native_import_error is None else str(_native_import_error)
        )
        raise RuntimeError(f"native estimator kernels are unavailable: {detail}")
    return normalized


def native_five_point_coefficients(nullspace: Any) -> np.ndarray:
    resolve_compute_backend("native")
    matrix = np.ascontiguousarray(nullspace, dtype=np.float64)
    if matrix.shape != (9, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("nullspace must be a finite array with shape (9, 4)")
    payload = _native_essential.five_point_coefficients(matrix)
    return np.frombuffer(payload, dtype=np.float64).reshape(4, 10, 20)


def native_sampson_residuals(
    bearings_a: Any,
    bearings_b: Any,
    essential_matrix: Any,
    *,
    squared: bool,
) -> np.ndarray:
    resolve_compute_backend("native")
    first = np.ascontiguousarray(bearings_a, dtype=np.float64)
    second = np.ascontiguousarray(bearings_b, dtype=np.float64)
    essential = np.ascontiguousarray(essential_matrix, dtype=np.float64)
    payload = _native_essential.sampson_residuals(
        first, second, essential, bool(squared)
    )
    return np.frombuffer(payload, dtype=np.float64).copy()
