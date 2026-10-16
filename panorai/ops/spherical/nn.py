"""Optional Torch layers and validation for direct ERP execution."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import math
from numbers import Real
from typing import Any

try:
    from torch import nn
except ImportError as exc:  # pragma: no cover - optional dependency boundary
    raise ImportError(
        "Torch spherical operations are optional; install with "
        "`pip install panorai[torch]`."
    ) from exc

from panorai.core.torch import (
    SphericalConv2d,
    SphericalConvTranspose2d,
    SphericalMaxPool2d,
    SphericalPortReport,
    port_module_with_report,
    spherical_area_average,
)
from panorai.core.torch import SphericalAvgPool2d as _CoreSphericalAvgPool2d
from panorai.core.torch import SphericalResize as _CoreSphericalResize


_SPHERICAL_TORCH_LAYERS_INTERFACE = "panorai-spherical-ops-torch/v1"


class PlanarSpatialOperationError(ValueError):
    """Raised when a conventional spatial layer is placed in ``OnSphere``."""


def _valid_weight_policy(
    invalid_policy: str, min_valid_weight: float | None
) -> float | None:
    if invalid_policy not in {"propagate", "renormalize"}:
        raise ValueError("invalid_policy must be 'propagate' or 'renormalize'")
    if invalid_policy == "propagate":
        if min_valid_weight is not None:
            raise ValueError("min_valid_weight requires invalid_policy='renormalize'")
        return None
    if (
        min_valid_weight is None
        or isinstance(min_valid_weight, bool)
        or not isinstance(min_valid_weight, Real)
    ):
        raise TypeError(
            "renormalize requires min_valid_weight as a finite real in (0, 1]"
        )
    threshold = float(min_valid_weight)
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("renormalize requires min_valid_weight in the interval (0, 1]")
    return threshold


def _output_shape(
    input_shape: tuple[int, int],
    *,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    ceil_mode: bool = False,
) -> tuple[int, int]:
    result = []
    for size, kernel, step, pad, dil in zip(
        input_shape,
        kernel_size,
        stride,
        padding,
        dilation,
        strict=True,
    ):
        numerator = size + 2 * pad - dil * (kernel - 1) - 1
        value = math.floor((numerator + (step - 1 if ceil_mode else 0)) / step + 1)
        if value < 1:
            raise ValueError("input grid is too small for the spatial operation")
        result.append(value)
    return tuple(result)


def _neighborhood_grid(layer: Any, input_grid: Any, *, operation: str) -> Any:
    output_shape = _output_shape(
        input_grid.shape_hw,
        kernel_size=tuple(layer.kernel_size),
        stride=tuple(layer.stride),
        padding=tuple(layer.padding),
        dilation=tuple(layer.dilation),
        ceil_mode=bool(getattr(layer, "ceil_mode", False)),
    )
    offset = tuple(
        ((kernel - 1) * dilation / 2.0 - padding) * input_stride
        for kernel, dilation, padding, input_stride in zip(
            layer.kernel_size,
            layer.dilation,
            layer.padding,
            input_grid.pixel_stride_yx,
            strict=True,
        )
    )
    first_center = tuple(
        center + delta
        for center, delta in zip(input_grid.first_center_px_yx, offset, strict=True)
    )
    output_stride = tuple(
        current * step
        for current, step in zip(input_grid.pixel_stride_yx, layer.stride, strict=True)
    )
    unchanged = (
        output_shape == input_grid.shape_hw
        and first_center == input_grid.first_center_px_yx
        and output_stride == input_grid.pixel_stride_yx
    )
    return input_grid.with_transform(
        operation=operation,
        output_shape_hw=output_shape,
        domain=input_grid.domain if unchanged else "derived",
        alignment=input_grid.alignment if unchanged else "derived-spherical-lattice",
        first_center_px_yx=first_center,
        pixel_stride_yx=output_stride,
        parameters={
            "kernel_size": tuple(layer.kernel_size),
            "stride": tuple(layer.stride),
            "padding": tuple(layer.padding),
            "dilation": tuple(layer.dilation),
            "ceil_mode": bool(getattr(layer, "ceil_mode", False)),
            "angular_step_deg": getattr(layer, "angular_step_deg", None),
        },
    )


def _mask_nchw(mask: Any) -> tuple[Any, bool]:
    import torch

    if mask.dtype != torch.bool:
        raise TypeError("support and validity masks must use torch.bool")
    if mask.ndim == 2:
        return mask[None, None], True
    if mask.ndim == 3:
        return mask[:, None], False
    raise ValueError("support and validity masks must use HW or NHW layout")


def _restore_mask(mask: Any, unbatched: bool) -> Any:
    result = mask[:, 0]
    return result[0] if unbatched else result


def _strict_neighborhood_mask(mask: Any, layer: Any) -> Any:
    import torch
    from panorai.core.torch._modules import _sample_tangent_neighbourhood

    values, unbatched = _mask_nchw(mask)
    sampled, _ = _sample_tangent_neighbourhood(
        (~values).to(dtype=torch.float32),
        kernel_size=tuple(layer.kernel_size),
        stride=tuple(layer.stride),
        padding=tuple(layer.padding),
        dilation=tuple(layer.dilation),
        angular_step_deg=getattr(layer, "angular_step_deg", None),
        ceil_mode=bool(getattr(layer, "ceil_mode", False)),
        max_sampled_elements=getattr(layer, "max_sampled_elements", None),
    )
    touched_invalid = sampled.amax(dim=2) > 0.0
    return _restore_mask(~touched_invalid, unbatched)


def _any_neighborhood_mask(mask: Any, layer: Any) -> Any:
    import torch
    from panorai.core.torch._modules import _sample_tangent_neighbourhood

    values, unbatched = _mask_nchw(mask)
    sampled, _ = _sample_tangent_neighbourhood(
        values.to(dtype=torch.float32),
        kernel_size=tuple(layer.kernel_size),
        stride=tuple(layer.stride),
        padding=tuple(layer.padding),
        dilation=tuple(layer.dilation),
        angular_step_deg=getattr(layer, "angular_step_deg", None),
        ceil_mode=bool(getattr(layer, "ceil_mode", False)),
        max_sampled_elements=getattr(layer, "max_sampled_elements", None),
    )
    any_valid = sampled.amax(dim=2) > 0.0
    return _restore_mask(any_valid, unbatched)


def _strict_transposed_mask(mask: Any, layer: Any) -> Any:
    import torch
    from panorai.core.torch import SphericalConvTranspose2d

    values, unbatched = _mask_nchw(mask)
    source = nn.ConvTranspose2d(
        1,
        1,
        tuple(layer.kernel_size),
        stride=tuple(layer.stride),
        padding=tuple(layer.padding),
        output_padding=tuple(layer.output_padding),
        dilation=tuple(layer.dilation),
        bias=False,
        device=values.device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        source.weight.fill_(1.0)
    transpose = SphericalConvTranspose2d(
        source,
        angular_step_deg=getattr(layer, "angular_step_deg", None),
        max_sampled_elements=getattr(layer, "max_sampled_elements", None),
    )
    with torch.no_grad():
        full = transpose(torch.ones_like(values, dtype=torch.float32))
        accepted = transpose(values.to(dtype=torch.float32))
    tolerance = torch.finfo(accepted.dtype).eps * full.clamp_min(1.0) * 8.0
    valid = (full > 0.0) & (accepted >= full - tolerance)
    return _restore_mask(valid, unbatched)


def _transposed_grid(layer: Any, input_grid: Any) -> Any:
    output_shape = tuple(
        (size - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
        for size, stride, padding, dilation, kernel, output_padding in zip(
            input_grid.shape_hw,
            layer.stride,
            layer.padding,
            layer.dilation,
            layer.kernel_size,
            layer.output_padding,
            strict=True,
        )
    )
    output_stride = tuple(
        current / step
        for current, step in zip(input_grid.pixel_stride_yx, layer.stride, strict=True)
    )
    offset = tuple(
        ((kernel - 1) * dilation / 2.0 - padding) * stride
        for kernel, dilation, padding, stride in zip(
            layer.kernel_size,
            layer.dilation,
            layer.padding,
            output_stride,
            strict=True,
        )
    )
    first_center = tuple(
        center - delta
        for center, delta in zip(input_grid.first_center_px_yx, offset, strict=True)
    )
    canonical = (
        output_shape == input_grid.erp_shape_hw
        and first_center == (0.5, 0.5)
        and output_stride == (1.0, 1.0)
    )
    return input_grid.with_transform(
        operation="conv_transpose2d",
        output_shape_hw=output_shape,
        domain="erp" if canonical else "derived",
        alignment=("erp-pixel-centres" if canonical else "derived-spherical-lattice"),
        first_center_px_yx=first_center,
        pixel_stride_yx=output_stride,
        parameters={
            "kernel_size": tuple(layer.kernel_size),
            "stride": tuple(layer.stride),
            "padding": tuple(layer.padding),
            "dilation": tuple(layer.dilation),
            "output_padding": tuple(layer.output_padding),
            "angular_step_deg": getattr(layer, "angular_step_deg", None),
        },
    )


@dataclass(frozen=True, slots=True)
class ValidatedLayer:
    path: str
    layer_type: str
    role: str
    kernel_size: tuple[int, int] | None = None
    stride: tuple[int, int] | None = None
    input_grid: dict[str, Any] | None = None
    output_grid: dict[str, Any] | None = None
    resolution_changed: bool = False


@dataclass(frozen=True, slots=True)
class SphericalValidationReport:
    interface: str
    stability: str
    layers: tuple[ValidatedLayer, ...]
    input_grid: dict[str, Any] | None = None
    output_grid: dict[str, Any] | None = None
    backend: str = "torch"
    dtype: str | None = None
    device: str | None = None
    unverifiable_functional_calls: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "stability": self.stability,
            "layers": [asdict(layer) for layer in self.layers],
            "input_grid": self.input_grid,
            "output_grid": self.output_grid,
            "backend": self.backend,
            "dtype": self.dtype,
            "device": self.device,
            "unverifiable_functional_calls": list(self.unverifiable_functional_calls),
        }


class Conv2d(SphericalConv2d):
    """Construct a spherical Conv2d directly or adapt a planar layer."""

    interface = _SPHERICAL_TORCH_LAYERS_INTERFACE
    stability = "stable"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Any,
        stride: Any = 1,
        padding: Any = 0,
        dilation: Any = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Any = None,
        dtype: Any = None,
        *,
        max_sampled_elements: int | None = None,
    ) -> None:
        if padding_mode != "zeros":
            raise ValueError(
                "spherical Conv2d does not accept planar padding modes; "
                "sampling is defined by the spherical operator"
            )
        source = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        super().__init__(source, max_sampled_elements=max_sampled_elements)

    @classmethod
    def from_planar(
        cls, source: nn.Conv2d, *, max_sampled_elements: int | None = None
    ) -> "Conv2d":
        result = cls.__new__(cls)
        SphericalConv2d.__init__(
            result, source, max_sampled_elements=max_sampled_elements
        )
        return result

    def transform_grid(self, input_grid: Any) -> Any:
        return _neighborhood_grid(self, input_grid, operation="conv2d")

    def transform_mask(self, mask: Any) -> Any:
        return _strict_neighborhood_mask(mask, self)


class ConvTranspose2d(SphericalConvTranspose2d):
    """Direct spherical transposed convolution constructor."""

    interface = _SPHERICAL_TORCH_LAYERS_INTERFACE
    stability = "stable"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Any,
        stride: Any = 1,
        padding: Any = 0,
        output_padding: Any = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Any = 1,
        padding_mode: str = "zeros",
        device: Any = None,
        dtype: Any = None,
        *,
        max_sampled_elements: int | None = None,
    ) -> None:
        if padding_mode != "zeros":
            raise ValueError("spherical ConvTranspose2d requires padding_mode='zeros'")
        source = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        super().__init__(source, max_sampled_elements=max_sampled_elements)

    @classmethod
    def from_planar(
        cls,
        source: nn.ConvTranspose2d,
        *,
        max_sampled_elements: int | None = None,
    ) -> "ConvTranspose2d":
        result = cls.__new__(cls)
        SphericalConvTranspose2d.__init__(
            result, source, max_sampled_elements=max_sampled_elements
        )
        return result

    def transform_grid(self, input_grid: Any) -> Any:
        return _transposed_grid(self, input_grid)

    def transform_mask(self, mask: Any) -> Any:
        return _strict_transposed_mask(mask, self)


class MaxPool2d(SphericalMaxPool2d):
    """Direct spherical max-pooling constructor."""

    interface = _SPHERICAL_TORCH_LAYERS_INTERFACE
    stability = "stable"

    def __init__(
        self,
        kernel_size: Any,
        stride: Any = None,
        padding: Any = 0,
        dilation: Any = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
        *,
        invalid_policy: str = "propagate",
        max_sampled_elements: int | None = None,
    ) -> None:
        if invalid_policy not in {"propagate", "ignore-invalid"}:
            raise ValueError(
                "MaxPool2d invalid_policy must be 'propagate' or 'ignore-invalid'"
            )
        source = nn.MaxPool2d(
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )
        super().__init__(source, max_sampled_elements=max_sampled_elements)
        self.invalid_policy = invalid_policy

    @classmethod
    def from_planar(
        cls,
        source: nn.MaxPool2d,
        *,
        invalid_policy: str = "propagate",
        max_sampled_elements: int | None = None,
    ) -> "MaxPool2d":
        if invalid_policy not in {"propagate", "ignore-invalid"}:
            raise ValueError(
                "MaxPool2d invalid_policy must be 'propagate' or 'ignore-invalid'"
            )
        result = cls.__new__(cls)
        SphericalMaxPool2d.__init__(
            result, source, max_sampled_elements=max_sampled_elements
        )
        result.invalid_policy = invalid_policy
        return result

    def transform_grid(self, input_grid: Any) -> Any:
        return _neighborhood_grid(self, input_grid, operation="max_pool2d")

    def transform_mask(self, mask: Any) -> Any:
        return _strict_neighborhood_mask(mask, self)

    def forward_with_masks(
        self, values: Any, support: Any, validity: Any
    ) -> tuple[Any, Any, Any]:
        if self.invalid_policy == "propagate":
            return (
                self(values),
                _strict_neighborhood_mask(support, self),
                _strict_neighborhood_mask(validity, self),
            )
        import torch

        usable = support & validity
        usable_nchw, _ = _mask_nchw(usable)
        expanded = usable_nchw.expand(-1, values.shape[1], -1, -1)
        masked = torch.where(
            expanded,
            values,
            torch.full(
                (),
                torch.finfo(values.dtype).min,
                dtype=values.dtype,
                device=values.device,
            ),
        )
        result = self(masked)
        output_support = _any_neighborhood_mask(support, self)
        output_validity = _any_neighborhood_mask(usable, self)
        valid_nchw, _ = _mask_nchw(output_validity)
        result = torch.where(
            valid_nchw.expand(-1, result.shape[1], -1, -1),
            result,
            torch.full((), torch.nan, dtype=result.dtype, device=result.device),
        )
        return result, output_support, output_validity


class AvgPool2d(_CoreSphericalAvgPool2d):
    """Spherical average pooling with strict or weighted-validity semantics."""

    interface = _SPHERICAL_TORCH_LAYERS_INTERFACE
    stability = "stable"

    def __init__(
        self,
        kernel_size: Any,
        stride: Any = None,
        padding: Any = 0,
        dilation: Any = 1,
        ceil_mode: bool = False,
        *,
        invalid_policy: str = "propagate",
        min_valid_weight: float | None = None,
        max_sampled_elements: int | None = None,
    ) -> None:
        threshold = _valid_weight_policy(invalid_policy, min_valid_weight)
        super().__init__(
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            max_sampled_elements=max_sampled_elements,
        )
        self.invalid_policy = invalid_policy
        self.min_valid_weight = threshold

    def transform_grid(self, input_grid: Any) -> Any:
        return _neighborhood_grid(self, input_grid, operation="avg_pool2d")

    def transform_mask(self, mask: Any) -> Any:
        return _strict_neighborhood_mask(mask, self)

    def forward_with_masks(
        self, values: Any, support: Any, validity: Any
    ) -> tuple[Any, Any, Any]:
        if self.invalid_policy == "propagate":
            return (
                self(values),
                _strict_neighborhood_mask(support, self),
                _strict_neighborhood_mask(validity, self),
            )
        import torch

        usable = support & validity
        usable_nchw, _ = _mask_nchw(usable)
        expanded = usable_nchw.expand(-1, values.shape[1], -1, -1)
        numerator = self(torch.where(expanded, values, torch.zeros_like(values)))
        valid_weight = self(usable_nchw.to(dtype=values.dtype))
        support_weight = self(_mask_nchw(support)[0].to(dtype=values.dtype))
        output_support = _restore_mask(support_weight > 0.0, support.ndim == 2)
        output_validity_nchw = (support_weight > 0.0) & (
            valid_weight >= self.min_valid_weight
        )
        result = torch.divide(
            numerator,
            valid_weight.expand(-1, numerator.shape[1], -1, -1).clamp_min(
                torch.finfo(values.dtype).tiny
            ),
        )
        result = torch.where(
            output_validity_nchw.expand(-1, result.shape[1], -1, -1),
            result,
            torch.full((), torch.nan, dtype=result.dtype, device=result.device),
        )
        return (
            result,
            output_support,
            _restore_mask(output_validity_nchw, validity.ndim == 2),
        )


def _resize_mask(layer: Any, mask: Any) -> Any:
    import torch

    values, unbatched = _mask_nchw(mask)
    resized_invalid = layer((~values).to(dtype=torch.float32))
    return _restore_mask(resized_invalid <= 0.0, unbatched)


def _resize_grid(layer: Any, input_grid: Any, *, operation: str) -> Any:
    if not input_grid.is_canonical_erp:
        raise ValueError(
            f"{operation} requires canonical ERP input; derived lattices need "
            "coordinate-aware regridding"
        )
    output_shape = layer.output_shape_hw
    if output_shape is None:
        output_shape = (
            max(2, round(input_grid.shape_hw[0] * layer.scale_factor)),
            max(2, round(input_grid.shape_hw[1] * layer.scale_factor)),
        )
    return input_grid.with_transform(
        operation=operation,
        output_shape_hw=output_shape,
        domain="erp",
        alignment="erp-pixel-centres",
        erp_shape_hw=output_shape,
        first_center_px_yx=(0.5, 0.5),
        pixel_stride_yx=(1.0, 1.0),
        longitude_periodic=True,
        parameters={
            "output_shape_hw": layer.output_shape_hw,
            "scale_factor": layer.scale_factor,
            "interpolation": layer.interpolation,
        },
    )


class Resize(_CoreSphericalResize):
    """Explicit Torch resampling onto a canonical ERP pixel-centre lattice."""

    interface = _SPHERICAL_TORCH_LAYERS_INTERFACE
    stability = "stable"

    def __init__(
        self,
        output_shape_hw: tuple[int, int] | None = None,
        *,
        scale_factor: float | None = None,
        interpolation: str = "bilinear",
    ) -> None:
        super().__init__(
            output_shape_hw,
            scale_factor=scale_factor,
            interpolation=interpolation,
        )

    def transform_grid(self, input_grid: Any) -> Any:
        return _resize_grid(self, input_grid, operation="resize")

    def transform_mask(self, mask: Any) -> Any:
        return _resize_mask(self, mask)


class Interpolate(Resize):
    """Torch-style name for explicit spherical ERP interpolation."""

    def __init__(
        self,
        size: tuple[int, int] | None = None,
        *,
        scale_factor: float | None = None,
        mode: str = "bilinear",
    ) -> None:
        super().__init__(
            size,
            scale_factor=scale_factor,
            interpolation=mode,
        )

    def transform_grid(self, input_grid: Any) -> Any:
        return _resize_grid(self, input_grid, operation="interpolate")


class AdaptiveAvgPool2d(nn.Module):
    """Solid-angle spherical global average pooling (output size one)."""

    interface = _SPHERICAL_TORCH_LAYERS_INTERFACE
    stability = "stable"

    def __init__(self, output_size: Any) -> None:
        super().__init__()
        resolved = (
            (output_size, output_size)
            if isinstance(output_size, int)
            else tuple(output_size)
        )
        if resolved != (1, 1):
            raise ValueError(
                "spherical AdaptiveAvgPool2d currently supports output_size=1 only"
            )
        self.output_size = resolved

    def forward(self, value: Any) -> Any:
        return spherical_area_average(value)[..., None, None]


class GlobalAveragePool(nn.Module):
    """Solid-angle global mean with optional validity renormalization."""

    interface = _SPHERICAL_TORCH_LAYERS_INTERFACE
    stability = "stable"

    def __init__(
        self,
        *,
        keepdim: bool = False,
        invalid_policy: str = "propagate",
        min_valid_weight: float | None = None,
    ) -> None:
        super().__init__()
        threshold = _valid_weight_policy(invalid_policy, min_valid_weight)
        self.keepdim = bool(keepdim)
        self.invalid_policy = invalid_policy
        self.min_valid_weight = threshold

    def forward(self, value: Any) -> Any:
        result = spherical_area_average(value)
        return result[..., None, None] if self.keepdim else result

    def forward_with_masks(
        self, values: Any, support: Any, validity: Any
    ) -> tuple[Any, Any, Any]:
        import torch

        support_nchw, support_unbatched = _mask_nchw(support)
        validity_nchw, validity_unbatched = _mask_nchw(validity)
        if self.invalid_policy == "propagate":
            result = self(values)
            output_support = support_nchw.all(dim=(-2, -1), keepdim=True)
            output_validity = validity_nchw.all(dim=(-2, -1), keepdim=True)
        else:
            usable = support_nchw & validity_nchw
            numerator = spherical_area_average(
                torch.where(
                    usable.expand(-1, values.shape[1], -1, -1),
                    values,
                    torch.zeros_like(values),
                )
            )
            valid_weight = spherical_area_average(usable.to(dtype=values.dtype))
            support_weight = spherical_area_average(support_nchw.to(dtype=values.dtype))
            result = numerator / valid_weight.clamp_min(torch.finfo(values.dtype).tiny)
            output_support = (support_weight > 0.0)[..., None, None]
            output_validity = (
                (support_weight > 0.0) & (valid_weight >= self.min_valid_weight)
            )[..., None, None]
            result = torch.where(
                output_validity[..., 0, 0].expand_as(result),
                result,
                torch.full((), torch.nan, dtype=result.dtype, device=result.device),
            )
            if self.keepdim:
                result = result[..., None, None]
        return (
            result,
            _restore_mask(output_support, support_unbatched),
            _restore_mask(output_validity, validity_unbatched),
        )


_SPHERICAL = (
    Conv2d,
    ConvTranspose2d,
    MaxPool2d,
    AvgPool2d,
    Resize,
    Interpolate,
    SphericalConv2d,
    SphericalConvTranspose2d,
    SphericalMaxPool2d,
    _CoreSphericalAvgPool2d,
    _CoreSphericalResize,
    AdaptiveAvgPool2d,
    GlobalAveragePool,
)

_POINTWISE = (
    nn.Identity,
    nn.ReLU,
    nn.ReLU6,
    nn.LeakyReLU,
    nn.PReLU,
    nn.ELU,
    nn.CELU,
    nn.GELU,
    nn.SELU,
    nn.SiLU,
    nn.Mish,
    nn.Sigmoid,
    nn.Tanh,
    nn.Hardtanh,
    nn.Softplus,
    nn.Softsign,
    nn.Hardsigmoid,
    nn.Hardswish,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
)

_PLANAR_SPATIAL = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
    nn.AdaptiveMaxPool1d,
    nn.AdaptiveMaxPool2d,
    nn.AdaptiveMaxPool3d,
    nn.ReflectionPad1d,
    nn.ReflectionPad2d,
    nn.ReflectionPad3d,
    nn.ReplicationPad1d,
    nn.ReplicationPad2d,
    nn.ReplicationPad3d,
    nn.ZeroPad1d,
    nn.ZeroPad2d,
    nn.ZeroPad3d,
    nn.Upsample,
)

_REGISTERED_MODULE_SEMANTICS: dict[type[nn.Module], str] = {}


def _register_module_semantics(module_type: type[nn.Module], semantics: str) -> None:
    """Record an explicit trusted semantic declaration from Pipeline registration."""

    if semantics != "pointwise":
        raise ValueError("custom Torch module semantics must be 'pointwise'")
    existing = _REGISTERED_MODULE_SEMANTICS.get(module_type)
    if existing is not None and existing != semantics:
        raise ValueError(
            f"Torch module type {module_type.__qualname__} already declares "
            f"{existing!r} semantics"
        )
    _REGISTERED_MODULE_SEMANTICS[module_type] = semantics


def _declared_module_semantics(module: nn.Module) -> str | None:
    return _REGISTERED_MODULE_SEMANTICS.get(type(module))


def _has_forward_hooks(module: nn.Module) -> bool:
    local_names = ("_forward_pre_hooks", "_forward_hooks")
    if any(bool(getattr(module, name, None)) for name in local_names):
        return True
    torch_module = nn.modules.module
    global_names = ("_global_forward_pre_hooks", "_global_forward_hooks")
    return any(bool(getattr(torch_module, name, None)) for name in global_names)


def _pair(value: Any) -> tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _pointwise_conv(layer: nn.Conv2d) -> bool:
    return (
        _pair(layer.kernel_size) == (1, 1)
        and _pair(layer.stride) == (1, 1)
        and _pair(layer.padding) == (0, 0)
        and _pair(layer.dilation) == (1, 1)
        and layer.padding_mode == "zeros"
    )


def transform_grid(module: nn.Module, input_grid: Any) -> Any:
    """Apply declared layer lattice transforms in execution order."""

    if isinstance(module, nn.Sequential):
        result = input_grid
        for layer in module:
            result = transform_grid(layer, result)
        return result
    declared = getattr(module, "transform_grid", None)
    if callable(declared):
        return declared(input_grid)
    if isinstance(module, SphericalConvTranspose2d):
        return _transposed_grid(module, input_grid)
    if isinstance(
        module, (SphericalConv2d, SphericalMaxPool2d, _CoreSphericalAvgPool2d)
    ):
        return _neighborhood_grid(
            module,
            input_grid,
            operation=(
                "conv2d"
                if isinstance(module, SphericalConv2d)
                else (
                    "max_pool2d"
                    if isinstance(module, SphericalMaxPool2d)
                    else "avg_pool2d"
                )
            ),
        )
    if isinstance(module, _CoreSphericalResize):
        return _resize_grid(module, input_grid, operation="resize")
    if type(module) is nn.Conv2d and _pointwise_conv(module):
        return input_grid
    if _declared_module_semantics(module) == "pointwise":
        return input_grid
    if type(module) in _POINTWISE:
        return input_grid
    if isinstance(module, AdaptiveAvgPool2d):
        raise ValueError(
            "AdaptiveAvgPool2d removes the spatial lattice; declare a "
            "classification or embedding output"
        )
    raise PlanarSpatialOperationError(
        f"{type(module).__module__}.{type(module).__name__} has no grid transform"
    )


def transform_mask(module: nn.Module, mask: Any) -> Any:
    """Propagate one support or validity mask through a validated module."""

    if isinstance(module, nn.Sequential):
        result = mask
        for layer in module:
            result = transform_mask(layer, result)
        return result
    declared = getattr(module, "transform_mask", None)
    if callable(declared):
        return declared(mask)
    if isinstance(module, SphericalConvTranspose2d):
        return _strict_transposed_mask(mask, module)
    if isinstance(
        module, (SphericalConv2d, SphericalMaxPool2d, _CoreSphericalAvgPool2d)
    ):
        return _strict_neighborhood_mask(mask, module)
    if isinstance(module, _CoreSphericalResize):
        return _resize_mask(module, mask)
    if type(module) is nn.Conv2d and _pointwise_conv(module):
        return mask.clone()
    if _declared_module_semantics(module) == "pointwise":
        return mask.clone()
    if type(module) in _POINTWISE:
        return mask.clone()
    if isinstance(module, AdaptiveAvgPool2d):
        values, unbatched = _mask_nchw(mask)
        result = values.all(dim=(-2, -1), keepdim=True)
        return _restore_mask(result, unbatched)
    raise PlanarSpatialOperationError(
        f"{type(module).__module__}.{type(module).__name__} has no mask transform"
    )


def execute_with_masks(
    module: nn.Module,
    values: Any,
    support: Any,
    validity: Any,
) -> tuple[Any, Any, Any]:
    """Execute a validated module while carrying independent boolean masks."""

    if isinstance(module, nn.Sequential):
        result = values
        result_support = support
        result_validity = validity
        for layer in module:
            result, result_support, result_validity = execute_with_masks(
                layer, result, result_support, result_validity
            )
        return result, result_support, result_validity
    declared = getattr(module, "forward_with_masks", None)
    if callable(declared):
        outcome = declared(values, support, validity)
        if not isinstance(outcome, tuple) or len(outcome) != 3:
            raise TypeError(
                f"{type(module).__module__}.{type(module).__name__}."
                "forward_with_masks must return (values, support, validity)"
            )
        result, result_support, result_validity = outcome
    else:
        result = module(values)
        result_support = transform_mask(module, support)
        result_validity = transform_mask(module, validity)
    support_shape = tuple(result_support.shape[-2:])
    validity_shape = tuple(result_validity.shape[-2:])
    result_shape = tuple(result.shape[-2:]) if result.ndim >= 3 else None
    expected_mask_shape = result_shape if result_shape is not None else (1, 1)
    if support_shape != expected_mask_shape or validity_shape != expected_mask_shape:
        output_description = (
            f"spatial shape {result_shape}"
            if result_shape is not None
            else "an output without a spatial lattice"
        )
        raise PlanarSpatialOperationError(
            f"{type(module).__module__}.{type(module).__name__} produced "
            f"{output_description}, but support and validity produced "
            f"{support_shape} and {validity_shape}"
        )
    return result, result_support, result_validity


def _grid_flow(
    module: nn.Module, input_shape: tuple[int, int]
) -> tuple[Any, dict[str, tuple[Any, Any]]]:
    from panorai.pipeline._fields import RasterGrid

    initial = RasterGrid(
        domain="erp",
        shape_hw=input_shape,
        erp_shape_hw=input_shape,
        alignment="erp-pixel-centres",
    )
    transitions: dict[str, tuple[Any, Any]] = {}

    def visit(layer: nn.Module, path: str, grid: Any | None) -> Any | None:
        if isinstance(layer, nn.Sequential):
            result = grid
            for name, child in layer.named_children():
                child_path = f"{path}.{name}" if path else name
                result = visit(child, child_path, result)
            return result
        before = grid
        if isinstance(layer, (AdaptiveAvgPool2d, GlobalAveragePool)):
            after = None
        elif grid is None:
            if (
                type(layer) in _POINTWISE
                or (type(layer) is nn.Conv2d and _pointwise_conv(layer))
                or _declared_module_semantics(layer) == "pointwise"
            ):
                after = None
            else:
                raise PlanarSpatialOperationError(
                    f"layer {path or '<root>'!r} applies a spatial operation after "
                    "the spatial lattice was removed"
                )
        else:
            after = transform_grid(layer, grid)
        transitions[path or "<root>"] = (before, after)
        return after

    return visit(module, "", initial), transitions


def validate(
    module: nn.Module,
    *,
    input_shape: tuple[int, int] | None = None,
) -> SphericalValidationReport:
    """Validate visible module leaves for direct spherical execution.

    The validator is deliberately fail-closed for unknown leaf modules and
    custom parents. Functional spatial calls hidden in a custom ``forward``
    cannot be inspected; compose declared layers with ``nn.Sequential`` in
    this first slice.
    """

    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    records: list[ValidatedLayer] = []
    for path, layer in module.named_modules():
        children = tuple(layer.children())
        display_path = path or "<root>"
        if _has_forward_hooks(layer):
            raise PlanarSpatialOperationError(
                f"layer {display_path!r} has forward hooks that can change output "
                "semantics; remove hooks before spherical validation"
            )
        if "forward" in layer.__dict__:
            raise PlanarSpatialOperationError(
                f"layer {display_path!r} has an instance-level forward override "
                "that cannot be proven sphere-safe"
            )
        if type(layer) in _SPHERICAL:
            records.append(
                ValidatedLayer(
                    display_path,
                    type(layer).__name__,
                    "spherical-spatial",
                    tuple(layer.kernel_size) if hasattr(layer, "kernel_size") else None,
                    tuple(layer.stride) if hasattr(layer, "stride") else None,
                )
            )
            continue
        if type(layer) is nn.Conv2d and _pointwise_conv(layer):
            records.append(
                ValidatedLayer(
                    display_path,
                    type(layer).__name__,
                    "pointwise",
                    (1, 1),
                    (1, 1),
                )
            )
            continue
        if _declared_module_semantics(layer) == "pointwise":
            records.append(
                ValidatedLayer(display_path, type(layer).__name__, "pointwise")
            )
            continue
        if isinstance(layer, _PLANAR_SPATIAL):
            kernel = getattr(layer, "kernel_size", None)
            raise PlanarSpatialOperationError(
                f"layer {display_path!r} is {type(layer).__module__}."
                f"{type(layer).__name__}(kernel_size={kernel!r}). Use a "
                "panorai.ops.spherical.nn layer, from_planar(), or "
                "SampleViews/PerView."
            )
        if children:
            if type(layer) is nn.Sequential:
                continue
            raise PlanarSpatialOperationError(
                f"container {display_path!r} ({type(layer).__module__}."
                f"{type(layer).__name__}) has custom forward semantics that cannot "
                "be proven sphere-safe; compose validated layers with nn.Sequential"
            )
        if type(layer) in _POINTWISE:
            records.append(
                ValidatedLayer(display_path, type(layer).__name__, "pointwise")
            )
            continue
        raise PlanarSpatialOperationError(
            f"leaf layer {display_path!r} ({type(layer).__module__}."
            f"{type(layer).__name__}) has no declared spherical or pointwise "
            "semantics"
        )
    initial_description = None
    output_description = None
    if input_shape is not None:
        output_grid, transitions = _grid_flow(module, input_shape)
        first_before = next(iter(transitions.values()))[0] if transitions else None
        initial_description = None if first_before is None else first_before.describe()
        output_description = None if output_grid is None else output_grid.describe()
        records = [
            replace(
                record,
                input_grid=(
                    None
                    if transitions[record.path][0] is None
                    else transitions[record.path][0].describe()
                ),
                output_grid=(
                    None
                    if transitions[record.path][1] is None
                    else transitions[record.path][1].describe()
                ),
                resolution_changed=(
                    transitions[record.path][0] is not None
                    and transitions[record.path][1] is not None
                    and transitions[record.path][0].shape_hw
                    != transitions[record.path][1].shape_hw
                ),
            )
            for record in records
        ]
    parameter = next(module.parameters(), None)
    return SphericalValidationReport(
        interface=_SPHERICAL_TORCH_LAYERS_INTERFACE,
        stability="stable",
        layers=tuple(records),
        input_grid=initial_description,
        output_grid=output_description,
        dtype=(
            None if parameter is None else str(parameter.dtype).replace("torch.", "")
        ),
        device=None if parameter is None else str(parameter.device),
    )


def from_planar(
    module: nn.Module,
    *,
    inplace: bool = False,
    max_sampled_elements: int | None = None,
) -> tuple[nn.Module, SphericalPortReport]:
    """Port supported planar spatial layers and return the existing audit report."""

    target = module if inplace else copy.deepcopy(module)
    report = port_module_with_report(target, max_sampled_elements=max_sampled_elements)
    if report.remaining_planar_spatial_layers:
        raise PlanarSpatialOperationError(
            "spherical port left planar spatial layers: "
            + ", ".join(report.remaining_planar_spatial_layers)
        )
    validate(target)
    return target, report


__all__ = [
    "AdaptiveAvgPool2d",
    "AvgPool2d",
    "Conv2d",
    "ConvTranspose2d",
    "GlobalAveragePool",
    "Interpolate",
    "MaxPool2d",
    "PlanarSpatialOperationError",
    "Resize",
    "SphericalConv2d",
    "SphericalConvTranspose2d",
    "SphericalMaxPool2d",
    "SphericalValidationReport",
    "ValidatedLayer",
    "from_planar",
    "execute_with_masks",
    "spherical_area_average",
    "transform_grid",
    "transform_mask",
    "validate",
]
