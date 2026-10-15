"""Optional Torch layers and validation for direct ERP execution."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
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


class PlanarSpatialOperationError(ValueError):
    """Raised when a conventional spatial layer is placed in ``OnSphere``."""


@dataclass(frozen=True, slots=True)
class ValidatedLayer:
    path: str
    layer_type: str
    role: str
    kernel_size: tuple[int, int] | None = None
    stride: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class SphericalValidationReport:
    interface: str
    stability: str
    layers: tuple[ValidatedLayer, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "stability": self.stability,
            "layers": [asdict(layer) for layer in self.layers],
        }


class Conv2d(SphericalConv2d):
    """Construct a spherical Conv2d directly or adapt a planar layer."""

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


class ConvTranspose2d(SphericalConvTranspose2d):
    """Direct spherical transposed convolution constructor."""

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


class MaxPool2d(SphericalMaxPool2d):
    """Direct spherical max-pooling constructor."""

    def __init__(
        self,
        kernel_size: Any,
        stride: Any = None,
        padding: Any = 0,
        dilation: Any = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
        *,
        max_sampled_elements: int | None = None,
    ) -> None:
        source = nn.MaxPool2d(
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )
        super().__init__(source, max_sampled_elements=max_sampled_elements)

    @classmethod
    def from_planar(
        cls, source: nn.MaxPool2d, *, max_sampled_elements: int | None = None
    ) -> "MaxPool2d":
        result = cls.__new__(cls)
        SphericalMaxPool2d.__init__(
            result, source, max_sampled_elements=max_sampled_elements
        )
        return result


class AdaptiveAvgPool2d(nn.Module):
    """Solid-angle spherical global average pooling (output size one)."""

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


_SPHERICAL = (
    SphericalConv2d,
    SphericalConvTranspose2d,
    SphericalMaxPool2d,
    AdaptiveAvgPool2d,
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


def validate(module: nn.Module) -> SphericalValidationReport:
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
        if isinstance(layer, _SPHERICAL):
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
        if isinstance(layer, nn.Conv2d) and _pointwise_conv(layer):
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
        if isinstance(layer, _PLANAR_SPATIAL):
            kernel = getattr(layer, "kernel_size", None)
            raise PlanarSpatialOperationError(
                f"layer {display_path!r} is {type(layer).__module__}."
                f"{type(layer).__name__}(kernel_size={kernel!r}). Use a "
                "panorai.ops.spherical.nn layer, from_planar(), or "
                "SampleViews/PerView."
            )
        if children:
            if isinstance(layer, nn.Sequential):
                continue
            raise PlanarSpatialOperationError(
                f"container {display_path!r} ({type(layer).__module__}."
                f"{type(layer).__name__}) has custom forward semantics that cannot "
                "be proven sphere-safe; compose validated layers with nn.Sequential"
            )
        if isinstance(layer, _POINTWISE):
            records.append(
                ValidatedLayer(display_path, type(layer).__name__, "pointwise")
            )
            continue
        raise PlanarSpatialOperationError(
            f"leaf layer {display_path!r} ({type(layer).__module__}."
            f"{type(layer).__name__}) has no declared spherical or pointwise "
            "semantics"
        )
    return SphericalValidationReport(
        interface="panorai-spherical-torch-validation/v1-experimental",
        stability="experimental",
        layers=tuple(records),
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
    "Conv2d",
    "ConvTranspose2d",
    "MaxPool2d",
    "PlanarSpatialOperationError",
    "SphericalConv2d",
    "SphericalConvTranspose2d",
    "SphericalMaxPool2d",
    "SphericalValidationReport",
    "ValidatedLayer",
    "from_planar",
    "spherical_area_average",
    "validate",
]
