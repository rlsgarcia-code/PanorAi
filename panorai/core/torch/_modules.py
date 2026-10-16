"""Differentiable fundamental spherical operators for ERP tensors.

This optional Stable module is explicit opt-in so importing
``panorai.core`` remains NumPy-only. It follows geometry-v1:
pixel-centre ERP sampling, top-left image origin, longitude increasing right,
latitude decreasing down, and the panorama frame ``+X`` right, ``+Y`` up,
``+Z`` forward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from panorai.core.specs import SphericalConv2dSpec


SPHERICAL_TORCH_CONVOLUTION_INTERFACE = "panorai-spherical-torch-convolution/v1"


def _pair(value: int | Iterable[int], name: str) -> tuple[int, int]:
    if isinstance(value, Integral):
        result = (int(value), int(value))
    else:
        result = tuple(int(item) for item in value)
        if len(result) != 2:
            raise ValueError(f"{name} must have exactly two elements")
    if min(result) < 1:
        raise ValueError(f"{name} values must be positive")
    return result


def _padding_pair(value: int | Iterable[int]) -> tuple[int, int]:
    if isinstance(value, Integral):
        result = (int(value), int(value))
    else:
        result = tuple(int(item) for item in value)
        if len(result) != 2:
            raise ValueError("padding must have exactly two elements")
    if min(result) < 0:
        raise ValueError("padding values must be non-negative")
    return result


def _output_size(
    size: int,
    *,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool = False,
) -> int:
    numerator = size + 2 * padding - dilation * (kernel - 1) - 1
    if ceil_mode:
        result = math.floor((numerator + stride - 1) / stride + 1)
    else:
        result = math.floor(numerator / stride + 1)
    if result < 1:
        raise ValueError(
            "the input is too small for the source operator's output lattice"
        )
    return result


def _tangent_coordinates(
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    angular_step_deg: float | None = None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    row_start: int = 0,
    row_stop: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Return ``(K, rows, W_out)`` ERP pixel coordinates for tangent samples."""

    input_height, input_width = input_shape
    output_height, output_width = output_shape
    kernel_height, kernel_width = kernel_size
    if row_stop is None:
        row_stop = output_height
    if not 0 <= row_start < row_stop <= output_height:
        raise ValueError("tangent-grid row interval is outside the output lattice")
    row_count = row_stop - row_start
    y = torch.arange(row_start, row_stop, device=device, dtype=dtype)[:, None]
    x = torch.arange(output_width, device=device, dtype=dtype)[None, :]
    anchor_y = y * stride[0] - padding[0] + (kernel_height - 1.0) * dilation[0] / 2.0
    anchor_x = x * stride[1] - padding[1] + (kernel_width - 1.0) * dilation[1] / 2.0
    longitude = ((anchor_x + 0.5) / input_width) * (2.0 * math.pi) - math.pi
    latitude = (math.pi / 2.0) - ((anchor_y + 0.5) / input_height) * math.pi
    longitude = longitude.expand(row_count, output_width)
    latitude = latitude.expand(row_count, output_width)

    sin_lon = torch.sin(longitude)
    cos_lon = torch.cos(longitude)
    sin_lat = torch.sin(latitude)
    cos_lat = torch.cos(latitude)
    rays = torch.stack((sin_lon * cos_lat, sin_lat, cos_lon * cos_lat), dim=-1)
    east = torch.stack((cos_lon, torch.zeros_like(cos_lon), -sin_lon), dim=-1)
    north = torch.stack((-sin_lon * sin_lat, cos_lat, -cos_lon * sin_lat), dim=-1)

    if angular_step_deg is None:
        step_east = 2.0 * math.pi / input_width
        step_north = math.pi / input_height
    else:
        step_east = step_north = math.radians(float(angular_step_deg))
    kernel_y = torch.arange(kernel_height, device=device, dtype=dtype)
    kernel_x = torch.arange(kernel_width, device=device, dtype=dtype)
    north_offset = -(kernel_y - (kernel_height - 1.0) / 2.0) * dilation[0] * step_north
    east_offset = (kernel_x - (kernel_width - 1.0) / 2.0) * dilation[1] * step_east
    north_offset, east_offset = torch.meshgrid(north_offset, east_offset, indexing="ij")
    north_offset = north_offset.reshape(-1, 1, 1)
    east_offset = east_offset.reshape(-1, 1, 1)

    radius = torch.hypot(east_offset, north_offset)
    sinc = torch.where(
        radius == 0.0, torch.ones_like(radius), torch.sinc(radius / math.pi)
    )
    tangent = (
        east_offset[..., None] * east[None] + north_offset[..., None] * north[None]
    )
    sample_rays = torch.cos(radius)[..., None] * rays[None] + sinc[..., None] * tangent
    sample_rays = F.normalize(sample_rays, dim=-1)

    sample_lon = torch.atan2(sample_rays[..., 0], sample_rays[..., 2])
    sample_lat = torch.asin(sample_rays[..., 1].clamp(-1.0, 1.0))
    sample_x = (sample_lon + math.pi) / (2.0 * math.pi) * input_width - 0.5
    sample_y = (math.pi / 2.0 - sample_lat) / math.pi * input_height - 0.5
    return sample_x, sample_y


def _tangent_grid(
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    angular_step_deg: float | None = None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    row_start: int = 0,
    row_stop: int | None = None,
) -> Tensor:
    """Return ``(K*rows, W_out, 2)`` tangent samples for ``grid_sample``."""

    input_height, input_width = input_shape
    output_width = output_shape[1]
    if row_stop is None:
        row_stop = output_shape[0]
    sample_x, sample_y = _tangent_coordinates(
        input_shape,
        output_shape,
        kernel_size,
        stride,
        padding,
        dilation,
        angular_step_deg,
        device=device,
        dtype=dtype,
        row_start=row_start,
        row_stop=row_stop,
    )

    # The copied boundary columns provide exact ERP longitude wrap. Latitude
    # stays on the sphere and uses border clamping for the limiting pole row.
    extended_x = sample_x + 1.0
    normalized_x = 2.0 * extended_x / (input_width + 1.0) - 1.0
    if input_height == 1:
        normalized_y = torch.zeros_like(sample_y)
    else:
        normalized_y = 2.0 * sample_y / (input_height - 1.0) - 1.0
    return torch.stack((normalized_x, normalized_y), dim=-1).reshape(
        -1, output_width, 2
    )


def _scatter_tangent_adjoint_rows(
    sample_values: Tensor,
    *,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    angular_step_deg: float | None,
    row_start: int,
    row_stop: int,
) -> Tensor:
    """Scatter sample gradients through the exact bilinear tangent sampler."""

    batch, channels, sample_count, row_count, output_width = sample_values.shape
    if sample_count != kernel_size[0] * kernel_size[1]:
        raise ValueError("sample_values kernel axis does not match kernel_size")
    if row_count != row_stop - row_start or output_width != output_shape[1]:
        raise ValueError("sample_values does not match the declared output rows")
    input_height, input_width = input_shape
    sample_x, sample_y = _tangent_coordinates(
        input_shape,
        output_shape,
        kernel_size,
        stride,
        padding,
        dilation,
        angular_step_deg,
        device=sample_values.device,
        dtype=sample_values.dtype,
        row_start=row_start,
        row_stop=row_stop,
    )
    sample_x = torch.remainder(sample_x, input_width)
    sample_y = sample_y.clamp(0.0, input_height - 1.0)
    x0_raw = torch.floor(sample_x).to(torch.long)
    x0 = torch.remainder(x0_raw, input_width)
    y0 = torch.floor(sample_y).to(torch.long)
    x1 = torch.remainder(x0 + 1, input_width)
    y1 = (y0 + 1).clamp(max=input_height - 1)
    wx = sample_x - x0_raw.to(sample_x.dtype)
    wy = sample_y - y0.to(sample_y.dtype)

    result = sample_values.new_zeros(batch, channels, input_height * input_width)
    for xx, yy, weight in (
        (x0, y0, (1.0 - wx) * (1.0 - wy)),
        (x1, y0, wx * (1.0 - wy)),
        (x0, y1, (1.0 - wx) * wy),
        (x1, y1, wx * wy),
    ):
        indices = (yy * input_width + xx).reshape(1, 1, -1)
        indices = indices.expand(batch, channels, -1)
        source = (sample_values * weight[None, None]).reshape(batch, channels, -1)
        result.scatter_add_(2, indices, source)
    return result.reshape(batch, channels, input_height, input_width)


def _sample_tangent_neighbourhood(
    values: Tensor,
    *,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    angular_step_deg: float | None = None,
    ceil_mode: bool = False,
    max_sampled_elements: int | None = None,
) -> tuple[Tensor, tuple[int, int]]:
    if values.ndim != 4:
        raise ValueError("values must use NCHW layout")
    if not values.is_floating_point():
        raise TypeError("values must use a floating dtype")
    batch, channels, height, width = values.shape
    output_shape = (
        _output_size(
            height,
            kernel=kernel_size[0],
            stride=stride[0],
            padding=padding[0],
            dilation=dilation[0],
            ceil_mode=ceil_mode,
        ),
        _output_size(
            width,
            kernel=kernel_size[1],
            stride=stride[1],
            padding=padding[1],
            dilation=dilation[1],
            ceil_mode=ceil_mode,
        ),
    )
    if max_sampled_elements is not None and max_sampled_elements < 1:
        raise ValueError("max_sampled_elements must be positive")
    sample_count = kernel_size[0] * kernel_size[1]
    if max_sampled_elements is None:
        rows_per_chunk = output_shape[0]
    else:
        elements_per_row = batch * channels * sample_count * output_shape[1]
        rows_per_chunk = max(1, max_sampled_elements // elements_per_row)
    parts: list[Tensor] = []
    for row_start in range(0, output_shape[0], rows_per_chunk):
        row_stop = min(output_shape[0], row_start + rows_per_chunk)
        parts.append(
            _sample_tangent_rows(
                values,
                output_shape=output_shape,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                angular_step_deg=angular_step_deg,
                row_start=row_start,
                row_stop=row_stop,
            )
        )
    return torch.cat(parts, dim=-2), output_shape


def _sample_tangent_rows(
    values: Tensor,
    *,
    output_shape: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    angular_step_deg: float | None = None,
    row_start: int,
    row_stop: int,
) -> Tensor:
    """Sample one output-row interval without materializing the full lattice."""

    batch, channels, height, width = values.shape
    grid = _tangent_grid(
        (height, width),
        output_shape,
        kernel_size,
        stride,
        padding,
        dilation,
        angular_step_deg,
        device=values.device,
        dtype=values.dtype,
        row_start=row_start,
        row_stop=row_stop,
    )
    wrapped = torch.cat((values[..., -1:], values, values[..., :1]), dim=-1)
    sampled = F.grid_sample(
        wrapped,
        grid[None].expand(batch, -1, -1, -1),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.reshape(
        batch,
        channels,
        kernel_size[0] * kernel_size[1],
        row_stop - row_start,
        output_shape[1],
    )


def _chunk_rows(
    values: Tensor,
    *,
    kernel_size: tuple[int, int],
    output_width: int,
    max_sampled_elements: int,
) -> int:
    elements_per_row = (
        values.shape[0]
        * values.shape[1]
        * kernel_size[0]
        * kernel_size[1]
        * output_width
    )
    return max(1, max_sampled_elements // elements_per_row)


class SphericalConv2d(nn.Module):
    """Port a ``Conv2d`` and its exact parameters to ERP tangent sampling.

    ``weight`` and ``bias`` are the same ``Parameter`` objects owned by the
    source layer. The operator therefore preserves pretrained weights and
    gradient flow without a state-dict translation or hidden copy.
    """

    interface = SPHERICAL_TORCH_CONVOLUTION_INTERFACE
    stability = "stable"

    def __init__(
        self,
        source: nn.Conv2d | None = None,
        *,
        in_channels: int | None = None,
        out_channels: int | None = None,
        kernel_size: int | tuple[int, int] | None = None,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        angular_step_deg: float | None = None,
        max_sampled_elements: int | None = None,
    ) -> None:
        super().__init__()
        if source is None:
            if in_channels is None or out_channels is None or kernel_size is None:
                raise TypeError(
                    "provide source=nn.Conv2d or in_channels, out_channels, and kernel_size"
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
        elif not isinstance(source, nn.Conv2d):
            raise TypeError("source must be torch.nn.Conv2d")
        elif any(
            value is not None for value in (in_channels, out_channels, kernel_size)
        ):
            raise TypeError("channel and kernel arguments cannot accompany source")
        if source.padding_mode != "zeros":
            raise ValueError(
                "non-zero planar padding modes have no implicit spherical meaning"
            )
        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self.kernel_size = _pair(source.kernel_size, "kernel_size")
        self.stride = _pair(source.stride, "stride")
        self.padding = _padding_pair(source.padding)
        self.dilation = _pair(source.dilation, "dilation")
        self.groups = source.groups
        self.weight = source.weight
        self.bias = source.bias
        if max_sampled_elements is not None and max_sampled_elements < 1:
            raise ValueError("max_sampled_elements must be positive")
        self.max_sampled_elements = max_sampled_elements
        self.spec = SphericalConv2dSpec(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            angular_step_deg=angular_step_deg,
        )
        self.angular_step_deg = self.spec.angular_step_deg

    @classmethod
    def from_torch(
        cls,
        source: nn.Conv2d,
        *,
        share_parameters: bool = True,
        angular_step_deg: float | None = None,
        max_sampled_elements: int | None = None,
    ) -> "SphericalConv2d":
        if share_parameters:
            return cls(
                source,
                angular_step_deg=angular_step_deg,
                max_sampled_elements=max_sampled_elements,
            )
        copied = nn.Conv2d(
            source.in_channels,
            source.out_channels,
            source.kernel_size,
            stride=source.stride,
            padding=source.padding,
            dilation=source.dilation,
            groups=source.groups,
            bias=source.bias is not None,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        with torch.no_grad():
            copied.weight.copy_(source.weight)
            if source.bias is not None and copied.bias is not None:
                copied.bias.copy_(source.bias)
        return cls(
            copied,
            angular_step_deg=angular_step_deg,
            max_sampled_elements=max_sampled_elements,
        )

    @classmethod
    def from_kernel(
        cls,
        kernel: Tensor,
        *,
        trainable: bool = False,
        angular_step_deg: float | None = None,
        max_sampled_elements: int | None = None,
    ) -> "SphericalConv2d":
        values = torch.as_tensor(kernel)
        if values.ndim == 2:
            values = values[None, None]
        if values.ndim != 4 or not values.is_floating_point():
            raise TypeError("kernel must be floating HW or OIHW")
        source = nn.Conv2d(
            values.shape[1],
            values.shape[0],
            values.shape[-2:],
            padding=(values.shape[-2] // 2, values.shape[-1] // 2),
            bias=False,
            device=values.device,
            dtype=values.dtype,
        )
        source.weight = nn.Parameter(values.clone(), requires_grad=trainable)
        return cls(
            source,
            angular_step_deg=angular_step_deg,
            max_sampled_elements=max_sampled_elements,
        )

    def forward(self, values: Tensor) -> Tensor:
        if values.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels, received {values.shape[1]}"
            )
        if self.kernel_size == (1, 1) and self.stride == (1, 1):
            return F.conv2d(values, self.weight, self.bias, groups=self.groups)
        output_shape = (
            _output_size(
                values.shape[-2],
                kernel=self.kernel_size[0],
                stride=self.stride[0],
                padding=self.padding[0],
                dilation=self.dilation[0],
            ),
            _output_size(
                values.shape[-1],
                kernel=self.kernel_size[1],
                stride=self.stride[1],
                padding=self.padding[1],
                dilation=self.dilation[1],
            ),
        )
        batch = values.shape[0]
        sample_count = self.kernel_size[0] * self.kernel_size[1]
        input_per_group = self.in_channels // self.groups
        output_per_group = self.out_channels // self.groups
        grouped_weights = self.weight.reshape(
            self.groups, output_per_group, input_per_group, sample_count
        )

        def convolve(sampled: Tensor) -> Tensor:
            row_count = sampled.shape[-2]
            grouped_samples = sampled.reshape(
                batch,
                self.groups,
                input_per_group,
                sample_count,
                row_count,
                output_shape[1],
            )
            result = torch.einsum(
                "ngckhw,gock->ngohw", grouped_samples, grouped_weights
            ).reshape(batch, self.out_channels, row_count, output_shape[1])
            if self.bias is not None:
                result = result + self.bias[None, :, None, None]
            return result

        if self.max_sampled_elements is None:
            sampled, _ = _sample_tangent_neighbourhood(
                values,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                angular_step_deg=self.angular_step_deg,
            )
            return convolve(sampled)
        rows_per_chunk = _chunk_rows(
            values,
            kernel_size=self.kernel_size,
            output_width=output_shape[1],
            max_sampled_elements=self.max_sampled_elements,
        )
        parts: list[Tensor] = []
        for row_start in range(0, output_shape[0], rows_per_chunk):
            row_stop = min(output_shape[0], row_start + rows_per_chunk)
            sampled = _sample_tangent_rows(
                values,
                output_shape=output_shape,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                angular_step_deg=self.angular_step_deg,
                row_start=row_start,
                row_stop=row_stop,
            )
            parts.append(convolve(sampled))
        return torch.cat(parts, dim=-2)


class SphericalConvTranspose2d(nn.Module):
    """Adjoint of spherical ``Conv2d`` for an ERP tangent neighbourhood.

    The learned ``weight`` and ``bias`` remain the exact source ``Parameter``
    objects. Sample contributions are scattered through the same bilinear
    tangent map used by :class:`SphericalConv2d`; this is the actual linear
    transpose rather than a convolution over a zero-inserted ERP lattice.
    """

    interface = SPHERICAL_TORCH_CONVOLUTION_INTERFACE
    stability = "stable"

    def __init__(
        self,
        source: nn.ConvTranspose2d | None = None,
        *,
        in_channels: int | None = None,
        out_channels: int | None = None,
        kernel_size: int | tuple[int, int] | None = None,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        output_padding: int | tuple[int, int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int, int] = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        angular_step_deg: float | None = None,
        max_sampled_elements: int | None = None,
    ) -> None:
        super().__init__()
        if source is None:
            if in_channels is None or out_channels is None or kernel_size is None:
                raise TypeError(
                    "provide source=nn.ConvTranspose2d or channels and kernel_size"
                )
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
                device=device,
                dtype=dtype,
            )
        elif not isinstance(source, nn.ConvTranspose2d):
            raise TypeError("source must be torch.nn.ConvTranspose2d")
        elif any(
            value is not None for value in (in_channels, out_channels, kernel_size)
        ):
            raise TypeError("channel and kernel arguments cannot accompany source")
        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self.kernel_size = _pair(source.kernel_size, "kernel_size")
        self.stride = _pair(source.stride, "stride")
        self.padding = _padding_pair(source.padding)
        self.output_padding = _padding_pair(source.output_padding)
        self.dilation = _pair(source.dilation, "dilation")
        self.groups = source.groups
        self.weight = source.weight
        self.bias = source.bias
        if angular_step_deg is not None and (
            not math.isfinite(float(angular_step_deg))
            or not 0.0 < float(angular_step_deg) < 180.0
        ):
            raise ValueError("angular_step_deg must be finite and in (0, 180)")
        self.angular_step_deg = (
            None if angular_step_deg is None else float(angular_step_deg)
        )
        if max_sampled_elements is not None and max_sampled_elements < 1:
            raise ValueError("max_sampled_elements must be positive")
        self.max_sampled_elements = max_sampled_elements

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 4 or values.shape[1] != self.in_channels:
            raise ValueError(
                f"expected NCHW with {self.in_channels} channels; received "
                f"{tuple(values.shape)}"
            )
        output_shape = tuple(
            (size - 1) * stride
            - 2 * padding
            + dilation * (kernel - 1)
            + output_padding
            + 1
            for size, stride, padding, dilation, kernel, output_padding in zip(
                values.shape[-2:],
                self.stride,
                self.padding,
                self.dilation,
                self.kernel_size,
                self.output_padding,
                strict=True,
            )
        )
        if min(output_shape) < 2:
            raise ValueError("transposed spherical output must be at least 2x2")
        batch = values.shape[0]
        sample_count = self.kernel_size[0] * self.kernel_size[1]
        input_per_group = self.in_channels // self.groups
        output_per_group = self.out_channels // self.groups
        grouped_weights = self.weight.reshape(
            self.groups, input_per_group, output_per_group, sample_count
        )

        def transpose_rows(row_start: int, row_stop: int) -> Tensor:
            row_count = row_stop - row_start
            grouped_values = values[..., row_start:row_stop, :].reshape(
                batch,
                self.groups,
                input_per_group,
                row_count,
                values.shape[-1],
            )
            sample_values = torch.einsum(
                "ngihw,giok->ngokhw", grouped_values, grouped_weights
            ).reshape(
                batch,
                self.out_channels,
                sample_count,
                row_count,
                values.shape[-1],
            )
            return _scatter_tangent_adjoint_rows(
                sample_values,
                input_shape=output_shape,
                output_shape=tuple(values.shape[-2:]),
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                angular_step_deg=self.angular_step_deg,
                row_start=row_start,
                row_stop=row_stop,
            )

        if self.max_sampled_elements is None:
            rows_per_chunk = values.shape[-2]
        else:
            elements_per_row = (
                batch * self.out_channels * sample_count * values.shape[-1]
            )
            rows_per_chunk = max(1, self.max_sampled_elements // elements_per_row)
        result = values.new_zeros(batch, self.out_channels, *output_shape)
        for row_start in range(0, values.shape[-2], rows_per_chunk):
            row_stop = min(values.shape[-2], row_start + rows_per_chunk)
            result = result + transpose_rows(row_start, row_stop)
        if self.bias is not None:
            result = result + self.bias[None, :, None, None]
        return result


class SphericalResizeConv2d(nn.Module):
    """Artifact-resistant spherical rewrite of a learned transposed convolution.

    This is deliberately not called a transposed convolution: it performs
    nearest ERP resize followed by spherical convolution. The source kernel is
    flipped, channel-transposed, and divided by the stride area so that a
    repeated low-frequency field retains the zero-insertion path's mean gain.
    It is intended as an explicit inference-porting policy for pretrained
    decoders whose stacked transposed convolutions expose stride-period bands.
    """

    interface = SPHERICAL_TORCH_CONVOLUTION_INTERFACE
    stability = "stable"

    def __init__(
        self,
        source: nn.ConvTranspose2d,
        *,
        max_sampled_elements: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(source, nn.ConvTranspose2d):
            raise TypeError("source must be torch.nn.ConvTranspose2d")
        kernel_size = _pair(source.kernel_size, "kernel_size")
        dilation = _pair(source.dilation, "dilation")
        effective_kernel = tuple(
            dil * (kernel - 1) + 1
            for kernel, dil in zip(kernel_size, dilation, strict=True)
        )
        if any(size % 2 == 0 for size in effective_kernel):
            raise ValueError(
                "resize-convolution port requires odd effective kernel dimensions"
            )
        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(source.stride, "stride")
        self.padding = _padding_pair(source.padding)
        self.output_padding = _padding_pair(source.output_padding)
        self.dilation = dilation
        self.groups = source.groups
        input_per_group = self.in_channels // self.groups
        output_per_group = self.out_channels // self.groups
        converted = torch.flip(source.weight.detach(), dims=(-2, -1)).reshape(
            self.groups,
            input_per_group,
            output_per_group,
            *self.kernel_size,
        ).permute(0, 2, 1, 3, 4).reshape(
            self.out_channels, input_per_group, *self.kernel_size
        ) / float(self.stride[0] * self.stride[1])
        planar = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=1,
            padding=tuple((size - 1) // 2 for size in effective_kernel),
            dilation=self.dilation,
            groups=self.groups,
            bias=source.bias is not None,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        planar.weight = nn.Parameter(
            converted.clone(), requires_grad=source.weight.requires_grad
        )
        if source.bias is not None:
            planar.bias = nn.Parameter(
                source.bias.detach().clone(), requires_grad=source.bias.requires_grad
            )
        self.convolution = SphericalConv2d(
            planar, max_sampled_elements=max_sampled_elements
        )

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 4 or values.shape[1] != self.in_channels:
            raise ValueError(
                f"expected NCHW with {self.in_channels} channels; received "
                f"{tuple(values.shape)}"
            )
        output_shape = tuple(
            (size - 1) * stride
            - 2 * padding
            + dilation * (kernel - 1)
            + output_padding
            + 1
            for size, stride, padding, dilation, kernel, output_padding in zip(
                values.shape[-2:],
                self.stride,
                self.padding,
                self.dilation,
                self.kernel_size,
                self.output_padding,
                strict=True,
            )
        )
        resized = F.interpolate(values, size=output_shape, mode="nearest")
        return self.convolution(resized)


class SphericalMaxPool2d(nn.Module):
    """Port a ``MaxPool2d`` neighbourhood to ERP tangent sampling."""

    interface = SPHERICAL_TORCH_CONVOLUTION_INTERFACE
    stability = "stable"

    def __init__(
        self,
        source: nn.MaxPool2d | None = None,
        *,
        kernel_size: int | tuple[int, int] | None = None,
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        ceil_mode: bool = False,
        max_sampled_elements: int | None = None,
    ) -> None:
        super().__init__()
        if source is None:
            if kernel_size is None:
                raise TypeError("provide source=nn.MaxPool2d or kernel_size")
            source = nn.MaxPool2d(
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
            )
        elif not isinstance(source, nn.MaxPool2d):
            raise TypeError("source must be torch.nn.MaxPool2d")
        elif kernel_size is not None:
            raise TypeError("kernel_size cannot accompany source")
        if source.return_indices:
            raise ValueError("return_indices=True is not supported")
        self.kernel_size = _pair(source.kernel_size, "kernel_size")
        stride = source.stride if source.stride is not None else source.kernel_size
        self.stride = _pair(stride, "stride")
        self.padding = _padding_pair(source.padding)
        self.dilation = _pair(source.dilation, "dilation")
        self.ceil_mode = source.ceil_mode
        if max_sampled_elements is not None and max_sampled_elements < 1:
            raise ValueError("max_sampled_elements must be positive")
        self.max_sampled_elements = max_sampled_elements

    def forward(self, values: Tensor) -> Tensor:
        output_shape = (
            _output_size(
                values.shape[-2],
                kernel=self.kernel_size[0],
                stride=self.stride[0],
                padding=self.padding[0],
                dilation=self.dilation[0],
                ceil_mode=self.ceil_mode,
            ),
            _output_size(
                values.shape[-1],
                kernel=self.kernel_size[1],
                stride=self.stride[1],
                padding=self.padding[1],
                dilation=self.dilation[1],
                ceil_mode=self.ceil_mode,
            ),
        )
        if self.max_sampled_elements is None:
            sampled, _ = _sample_tangent_neighbourhood(
                values,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                ceil_mode=self.ceil_mode,
            )
            return sampled.amax(dim=2)
        rows_per_chunk = _chunk_rows(
            values,
            kernel_size=self.kernel_size,
            output_width=output_shape[1],
            max_sampled_elements=self.max_sampled_elements,
        )
        parts: list[Tensor] = []
        for row_start in range(0, output_shape[0], rows_per_chunk):
            row_stop = min(output_shape[0], row_start + rows_per_chunk)
            sampled = _sample_tangent_rows(
                values,
                output_shape=output_shape,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                row_start=row_start,
                row_stop=row_stop,
            )
            parts.append(sampled.amax(dim=2))
        return torch.cat(parts, dim=-2)


@dataclass(frozen=True, slots=True)
class PortedLayer:
    """One conventional spatial layer replaced by its spherical counterpart."""

    path: str
    source_type: str
    target_type: str
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    parameter_identity_preserved: bool | None
    weight_shape: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class SphericalPortReport:
    """Auditable layer-by-layer record of an in-place spherical port."""

    interface: str
    stability: str
    layers: tuple[PortedLayer, ...]
    remaining_planar_spatial_layers: tuple[str, ...]
    conv_transpose_policy: str = "adjoint"
    collapsed_reflection_pads: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "interface": self.interface,
            "stability": self.stability,
            "ported_layer_count": len(self.layers),
            "conv_transpose_policy": self.conv_transpose_policy,
            "layers": [asdict(layer) for layer in self.layers],
            "collapsed_reflection_pads": list(self.collapsed_reflection_pads),
            "remaining_planar_spatial_layers": list(
                self.remaining_planar_spatial_layers
            ),
        }


def _port_module(
    module: nn.Module,
    *,
    prefix: str,
    records: list[PortedLayer],
    collapsed_reflection_pads: list[str],
    max_sampled_elements: int | None,
    conv_transpose_policy: str,
) -> None:
    children = tuple(module.named_children())
    for index, (name, child) in enumerate(children):
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.ReflectionPad2d):
            if index + 1 >= len(children) or not isinstance(
                children[index + 1][1], nn.Conv2d
            ):
                raise ValueError(
                    f"ReflectionPad2d at {path!r} is not followed by Conv2d"
                )
            padding = child.padding
            if isinstance(padding, Integral):
                symmetric = (int(padding), int(padding))
            else:
                left, right, top, bottom = tuple(int(item) for item in padding)
                if left != right or top != bottom:
                    raise ValueError(
                        f"ReflectionPad2d at {path!r} has asymmetric padding"
                    )
                symmetric = (top, left)
            children[index + 1][1].padding = symmetric
            setattr(module, name, nn.Identity())
            collapsed_reflection_pads.append(path)
            continue
        if isinstance(child, nn.Conv2d):
            replacement: nn.Module = SphericalConv2d(
                child, max_sampled_elements=max_sampled_elements
            )
            records.append(
                PortedLayer(
                    path=path,
                    source_type="Conv2d",
                    target_type="SphericalConv2d",
                    kernel_size=replacement.kernel_size,
                    stride=replacement.stride,
                    parameter_identity_preserved=(
                        replacement.weight is child.weight
                        and replacement.bias is child.bias
                    ),
                    weight_shape=tuple(replacement.weight.shape),
                )
            )
        elif isinstance(child, nn.ConvTranspose2d):
            if conv_transpose_policy == "adjoint":
                replacement = SphericalConvTranspose2d(
                    child, max_sampled_elements=max_sampled_elements
                )
            else:
                replacement = SphericalResizeConv2d(
                    child, max_sampled_elements=max_sampled_elements
                )
            records.append(
                PortedLayer(
                    path=path,
                    source_type="ConvTranspose2d",
                    target_type=type(replacement).__name__,
                    kernel_size=replacement.kernel_size,
                    stride=replacement.stride,
                    parameter_identity_preserved=(
                        isinstance(replacement, SphericalConvTranspose2d)
                        and replacement.weight is child.weight
                        and replacement.bias is child.bias
                    ),
                    weight_shape=tuple(child.weight.shape),
                )
            )
        elif isinstance(child, nn.MaxPool2d):
            replacement = SphericalMaxPool2d(
                child, max_sampled_elements=max_sampled_elements
            )
            records.append(
                PortedLayer(
                    path=path,
                    source_type="MaxPool2d",
                    target_type="SphericalMaxPool2d",
                    kernel_size=replacement.kernel_size,
                    stride=replacement.stride,
                    parameter_identity_preserved=None,
                    weight_shape=None,
                )
            )
        else:
            _port_module(
                child,
                prefix=path,
                records=records,
                collapsed_reflection_pads=collapsed_reflection_pads,
                max_sampled_elements=max_sampled_elements,
                conv_transpose_policy=conv_transpose_policy,
            )
            continue
        setattr(module, name, replacement)


def port_module_with_report(
    module: nn.Module,
    *,
    max_sampled_elements: int | None = None,
    conv_transpose_policy: str = "adjoint",
) -> SphericalPortReport:
    """Port spatial layers in place and return a complete replacement trace."""

    if max_sampled_elements is not None and max_sampled_elements < 1:
        raise ValueError("max_sampled_elements must be positive")
    if conv_transpose_policy not in {"adjoint", "resize_conv"}:
        raise ValueError("conv_transpose_policy must be 'adjoint' or 'resize_conv'")
    records: list[PortedLayer] = []
    collapsed_reflection_pads: list[str] = []
    _port_module(
        module,
        prefix="",
        records=records,
        collapsed_reflection_pads=collapsed_reflection_pads,
        max_sampled_elements=max_sampled_elements,
        conv_transpose_policy=conv_transpose_policy,
    )
    remaining = tuple(
        name
        for name, child in module.named_modules()
        if isinstance(
            child, (nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.ReflectionPad2d)
        )
    )
    return SphericalPortReport(
        interface=SPHERICAL_TORCH_CONVOLUTION_INTERFACE,
        stability="stable",
        layers=tuple(records),
        conv_transpose_policy=conv_transpose_policy,
        collapsed_reflection_pads=tuple(collapsed_reflection_pads),
        remaining_planar_spatial_layers=remaining,
    )


def port_module(
    module: nn.Module,
    *,
    max_sampled_elements: int | None = None,
    conv_transpose_policy: str = "adjoint",
) -> nn.Module:
    """Recursively port ``Conv2d``/``MaxPool2d`` layers in place.

    Other modules, including normalization and nonlinearities, retain their
    original objects and state. This function intentionally does not alter
    classifier topology; callers first decide how linear heads become dense.
    """

    report = port_module_with_report(
        module,
        max_sampled_elements=max_sampled_elements,
        conv_transpose_policy=conv_transpose_policy,
    )
    if report.remaining_planar_spatial_layers:
        raise RuntimeError(
            "spherical port left planar spatial layers: "
            + ", ".join(report.remaining_planar_spatial_layers)
        )
    return module


def spherical_area_average(values: Tensor) -> Tensor:
    """Average ``NCHW`` ERP values with exact pixel-cell solid angles."""

    if values.ndim != 4:
        raise ValueError("values must use NCHW layout")
    height = values.shape[-2]
    edges = (
        math.pi / 2.0
        - torch.arange(height + 1, device=values.device, dtype=values.dtype)
        / height
        * math.pi
    )
    weights = torch.sin(edges[:-1]) - torch.sin(edges[1:])
    weights = weights / weights.sum()
    return (values * weights[None, None, :, None]).sum(dim=(-2, -1)) / values.shape[-1]


__all__ = [
    "SPHERICAL_TORCH_CONVOLUTION_INTERFACE",
    "PortedLayer",
    "SphericalConv2d",
    "SphericalConvTranspose2d",
    "SphericalResizeConv2d",
    "SphericalMaxPool2d",
    "SphericalPortReport",
    "port_module",
    "port_module_with_report",
    "spherical_area_average",
]
