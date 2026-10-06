"""Private dispatch for the optional spherical-convolution kernel."""

from __future__ import annotations

import numpy as np

try:
    from panorai._native import _geometry as _native_geometry  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - environment dependent
    _native_geometry = None
    _native_import_error: Exception | None = exc
else:
    _native_import_error = None


def native_filter_available() -> bool:
    """Return whether the optional first-party filter kernel is importable."""

    return bool(
        _native_geometry is not None and hasattr(_native_geometry, "spherical_filter2d")
    )


def supports_native_filter(image: object) -> bool:
    return bool(
        native_filter_available()
        and isinstance(image, np.ndarray)
        and image.ndim in {2, 3}
        and image.dtype in {np.dtype(np.float32), np.dtype(np.float64)}
    )


def native_spherical_filter2d(
    image: np.ndarray, kernel: np.ndarray, angular_step_rad: float
) -> np.ndarray:
    if not native_filter_available():  # pragma: no cover - guarded by dispatcher
        detail = (
            "unknown" if _native_import_error is None else str(_native_import_error)
        )
        raise RuntimeError(f"native spherical filter is unavailable: {detail}")
    contiguous = np.ascontiguousarray(image)
    kernel64 = np.ascontiguousarray(kernel, dtype=np.float64)
    payload = _native_geometry.spherical_filter2d(
        contiguous, kernel64, angular_step_rad
    )
    return np.frombuffer(payload, dtype=contiguous.dtype).reshape(contiguous.shape)
