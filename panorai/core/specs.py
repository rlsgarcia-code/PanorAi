"""Immutable contracts shared by spherical operation backends."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Literal


BackendName = Literal["auto", "numpy", "native"]
Interpolation = Literal["bilinear", "nearest"]
Layout = Literal["NCHW", "CHW"]


def _positive_pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    if isinstance(value, bool):
        raise TypeError(f"{name} must contain integers")
    if isinstance(value, Integral):
        pair = (int(value), int(value))
    else:
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError(f"{name} must be an integer or a two-integer tuple")
        if any(isinstance(item, bool) or not isinstance(item, Integral) for item in value):
            raise TypeError(f"{name} must contain integers")
        pair = (int(value[0]), int(value[1]))
    if min(pair) < 1:
        raise ValueError(f"{name} values must be positive")
    return pair


def _nonnegative_pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    if isinstance(value, bool):
        raise TypeError(f"{name} must contain integers")
    if isinstance(value, Integral):
        pair = (int(value), int(value))
    else:
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError(f"{name} must be an integer or a two-integer tuple")
        if any(isinstance(item, bool) or not isinstance(item, Integral) for item in value):
            raise TypeError(f"{name} must contain integers")
        pair = (int(value[0]), int(value[1]))
    if min(pair) < 0:
        raise ValueError(f"{name} values must be non-negative")
    return pair


def _optional_step(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("angular_step_deg must be a real number or None")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 180.0:
        raise ValueError("angular_step_deg must be finite and in (0, 180)")
    return result


@dataclass(frozen=True, slots=True)
class SphericalGridSpec:
    """An ERP tensor grid following the geometry-v1 pixel-centre contract."""

    shape_hw: tuple[int, int]
    layout: Layout = "NCHW"

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape_hw", _positive_pair(self.shape_hw, "shape_hw"))
        if self.layout not in {"NCHW", "CHW"}:
            raise ValueError("layout must be 'NCHW' or 'CHW'")


@dataclass(frozen=True, slots=True)
class SphericalNeighborhoodSpec:
    """A tangent-plane neighbourhood sampled around ERP lattice anchors."""

    kernel_size: tuple[int, int] | int
    dilation: tuple[int, int] | int = 1
    angular_step_deg: float | None = None
    interpolation: Interpolation = "bilinear"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kernel_size", _positive_pair(self.kernel_size, "kernel_size")
        )
        object.__setattr__(self, "dilation", _positive_pair(self.dilation, "dilation"))
        object.__setattr__(self, "angular_step_deg", _optional_step(self.angular_step_deg))
        if self.interpolation not in {"bilinear", "nearest"}:
            raise ValueError("interpolation must be 'bilinear' or 'nearest'")


@dataclass(frozen=True, slots=True)
class SphericalConv2dSpec:
    """Backend-independent channel-mixing spherical correlation contract."""

    in_channels: int
    out_channels: int
    kernel_size: tuple[int, int] | int
    stride: tuple[int, int] | int = 1
    padding: tuple[int, int] | int = 0
    dilation: tuple[int, int] | int = 1
    groups: int = 1
    angular_step_deg: float | None = None

    def __post_init__(self) -> None:
        for name in ("in_channels", "out_channels", "groups"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, int(value))
        if self.in_channels % self.groups or self.out_channels % self.groups:
            raise ValueError("groups must divide both channel counts")
        object.__setattr__(
            self, "kernel_size", _positive_pair(self.kernel_size, "kernel_size")
        )
        object.__setattr__(self, "stride", _positive_pair(self.stride, "stride"))
        object.__setattr__(self, "padding", _nonnegative_pair(self.padding, "padding"))
        object.__setattr__(self, "dilation", _positive_pair(self.dilation, "dilation"))
        object.__setattr__(self, "angular_step_deg", _optional_step(self.angular_step_deg))

    def output_shape(self, input_shape_hw: tuple[int, int]) -> tuple[int, int]:
        height, width = _positive_pair(input_shape_hw, "input_shape_hw")
        result = tuple(
            (size + 2 * pad - dil * (kernel - 1) - 1) // stride + 1
            for size, pad, dil, kernel, stride in zip(
                (height, width),
                self.padding,
                self.dilation,
                self.kernel_size,
                self.stride,
                strict=True,
            )
        )
        if min(result) < 1:
            raise ValueError("input is too small for the requested output lattice")
        return result


@dataclass(frozen=True, slots=True)
class SphericalPool2dSpec:
    """Backend-independent spherical pooling contract."""

    kernel_size: tuple[int, int] | int
    stride: tuple[int, int] | int | None = None
    padding: tuple[int, int] | int = 0
    dilation: tuple[int, int] | int = 1
    ceil_mode: bool = False
    angular_step_deg: float | None = None

    def __post_init__(self) -> None:
        kernel = _positive_pair(self.kernel_size, "kernel_size")
        object.__setattr__(self, "kernel_size", kernel)
        object.__setattr__(
            self,
            "stride",
            kernel if self.stride is None else _positive_pair(self.stride, "stride"),
        )
        object.__setattr__(self, "padding", _nonnegative_pair(self.padding, "padding"))
        object.__setattr__(self, "dilation", _positive_pair(self.dilation, "dilation"))
        if not isinstance(self.ceil_mode, bool):
            raise TypeError("ceil_mode must be boolean")
        object.__setattr__(self, "angular_step_deg", _optional_step(self.angular_step_deg))


@dataclass(frozen=True, slots=True)
class SphericalResizeSpec:
    """Target ERP lattice for a geometry-safe resize."""

    output_shape_hw: tuple[int, int]
    interpolation: Interpolation = "bilinear"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_shape_hw",
            _positive_pair(self.output_shape_hw, "output_shape_hw"),
        )
        if self.interpolation not in {"bilinear", "nearest"}:
            raise ValueError("interpolation must be 'bilinear' or 'nearest'")


@dataclass(frozen=True, slots=True)
class SphericalExecutionSpec:
    """Execution controls that do not alter spherical mathematics."""

    backend: BackendName = "auto"
    max_sampled_elements: int | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "numpy", "native"}:
            raise ValueError("backend must be 'auto', 'numpy', or 'native'")
        if self.max_sampled_elements is not None:
            value = self.max_sampled_elements
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError("max_sampled_elements must be an integer or None")
            if int(value) < 1:
                raise ValueError("max_sampled_elements must be positive")
            object.__setattr__(self, "max_sampled_elements", int(value))


__all__ = [
    "BackendName",
    "Interpolation",
    "Layout",
    "SphericalConv2dSpec",
    "SphericalExecutionSpec",
    "SphericalGridSpec",
    "SphericalNeighborhoodSpec",
    "SphericalPool2dSpec",
    "SphericalResizeSpec",
]
