"""Private dispatch for optional first-party native spherical kernels."""

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


def native_conv_available() -> bool:
    """Return whether the fused channel-mixing convolution is importable."""

    return bool(
        _native_geometry is not None and hasattr(_native_geometry, "spherical_conv2d")
    )


def native_extrema_available() -> bool:
    """Return whether the fused spherical DoG-extrema kernel is importable."""

    return bool(
        _native_geometry is not None
        and hasattr(_native_geometry, "spherical_extrema3d")
    )


def native_rotation_available() -> bool:
    """Return whether reusable native ERP rotation kernels are importable."""

    return bool(
        _native_geometry is not None
        and hasattr(_native_geometry, "spherical_rotation_map")
        and hasattr(_native_geometry, "sample_equirectangular_map")
    )


def native_spherical_rotation_map(
    source_shape_hw: tuple[int, int],
    output_shape_hw: tuple[int, int],
    rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one float64 inverse map in the first-party C++ extension."""

    if not native_rotation_available():  # pragma: no cover - guarded by caller
        detail = "unknown" if _native_import_error is None else str(_native_import_error)
        raise RuntimeError(f"native spherical rotation is unavailable: {detail}")
    matrix = np.ascontiguousarray(rotation, dtype=np.float64)
    payload_x, payload_y = _native_geometry.spherical_rotation_map(
        matrix,
        source_shape_hw[0],
        source_shape_hw[1],
        output_shape_hw[0],
        output_shape_hw[1],
    )
    map_x = np.frombuffer(payload_x, dtype=np.float64).reshape(output_shape_hw)
    map_y = np.frombuffer(payload_y, dtype=np.float64).reshape(output_shape_hw)
    map_x.flags.writeable = False
    map_y.flags.writeable = False
    return map_x, map_y


def supports_native_rotation_sample(image: object) -> bool:
    """Return whether a field can use the native rotation sampler."""

    return bool(
        native_rotation_available()
        and isinstance(image, np.ndarray)
        and image.ndim in {2, 3}
        and image.dtype in {np.dtype(np.float32), np.dtype(np.float64)}
    )


def native_sample_equirectangular_map(
    image: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    *,
    interpolation: str,
) -> np.ndarray:
    """Sample one compatible field through a reusable native rotation map."""

    if interpolation not in {"nearest", "bilinear"}:
        raise ValueError("interpolation must be 'nearest' or 'bilinear'")
    if not supports_native_rotation_sample(image):
        raise TypeError("native rotation sampling requires float32 or float64 HW/HWC")
    contiguous = np.ascontiguousarray(image)
    payload = _native_geometry.sample_equirectangular_map(
        contiguous,
        np.ascontiguousarray(map_x, dtype=np.float64),
        np.ascontiguousarray(map_y, dtype=np.float64),
        interpolation == "bilinear",
    )
    return np.frombuffer(payload, dtype=contiguous.dtype).reshape(
        (*map_x.shape, *contiguous.shape[2:])
    )


def supports_native_filter(image: object) -> bool:
    return bool(
        native_filter_available()
        and isinstance(image, np.ndarray)
        and image.ndim in {2, 3}
        and image.dtype in {np.dtype(np.float32), np.dtype(np.float64)}
    )


def supports_native_conv(
    values: object, weight: object, bias: object | None = None
) -> bool:
    """Return whether values can use the exact first-party convolution kernel."""

    arrays = (values, weight)
    return bool(
        native_conv_available()
        and all(
            isinstance(array, np.ndarray)
            and array.dtype in {np.dtype(np.float32), np.dtype(np.float64)}
            for array in arrays
        )
        and values.ndim == 4
        and weight.ndim == 4
        and (
            bias is None
            or (
                isinstance(bias, np.ndarray)
                and bias.ndim == 1
                and bias.dtype in {np.dtype(np.float32), np.dtype(np.float64)}
            )
        )
    )


def supports_native_extrema(
    previous: object, current: object, following: object
) -> bool:
    arrays = (previous, current, following)
    return bool(
        native_extrema_available()
        and all(
            isinstance(array, np.ndarray)
            and array.ndim == 2
            and array.dtype in {np.dtype(np.float32), np.dtype(np.float64)}
            for array in arrays
        )
        and previous.shape == current.shape == following.shape
        and previous.dtype == current.dtype == following.dtype
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


def native_spherical_conv2d(
    values: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    *,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
    angular_step_rad: tuple[float, float],
    output_shape_hw: tuple[int, int],
) -> np.ndarray:
    """Run exact fused NCHW spherical correlation without a global im2col."""

    if not native_conv_available():  # pragma: no cover - guarded by dispatcher
        detail = (
            "unknown" if _native_import_error is None else str(_native_import_error)
        )
        raise RuntimeError(f"native spherical convolution is unavailable: {detail}")
    dtype = np.result_type(values.dtype, weight.dtype)
    if dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise TypeError("native spherical convolution requires float32 or float64")
    contiguous = np.ascontiguousarray(values, dtype=dtype)
    weights = np.ascontiguousarray(weight, dtype=dtype)
    biases = None if bias is None else np.ascontiguousarray(bias, dtype=dtype)
    payload = _native_geometry.spherical_conv2d(
        contiguous,
        weights,
        biases,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        groups,
        angular_step_rad[0],
        angular_step_rad[1],
    )
    return np.frombuffer(payload, dtype=dtype).reshape(
        values.shape[0], weight.shape[0], *output_shape_hw
    )


def native_spherical_extrema3d(
    previous: np.ndarray,
    current: np.ndarray,
    following: np.ndarray,
    angular_step_rad: float,
    contrast_threshold: float,
) -> np.ndarray:
    """Find strict tangent-neighbourhood extrema without materializing neighbours."""

    if not native_extrema_available():  # pragma: no cover - guarded by dispatcher
        detail = (
            "unknown" if _native_import_error is None else str(_native_import_error)
        )
        raise RuntimeError(f"native spherical extrema kernel is unavailable: {detail}")
    previous_contiguous = np.ascontiguousarray(previous)
    current_contiguous = np.ascontiguousarray(current)
    following_contiguous = np.ascontiguousarray(following)
    payload = _native_geometry.spherical_extrema3d(
        previous_contiguous,
        current_contiguous,
        following_contiguous,
        angular_step_rad,
        contrast_threshold,
    )
    return np.frombuffer(payload, dtype=np.bool_).reshape(current_contiguous.shape)
