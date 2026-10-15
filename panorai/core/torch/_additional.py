"""Torch spherical primitives composed from the shared tangent sampler."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ._modules import _padding_pair, _pair, _sample_tangent_neighbourhood


class SphericalFilter2d(nn.Module):
    """Apply a fixed tangent-plane kernel independently to every channel."""

    def __init__(
        self,
        kernel: Tensor,
        *,
        trainable: bool = False,
        max_sampled_elements: int | None = None,
    ) -> None:
        super().__init__()
        values = torch.as_tensor(kernel)
        if values.ndim != 2 or not values.is_floating_point():
            raise TypeError("kernel must be a floating two-dimensional tensor")
        if min(values.shape) < 1 or any(size % 2 == 0 for size in values.shape):
            raise ValueError("kernel dimensions must be positive and odd")
        if not torch.isfinite(values).all():
            raise ValueError("kernel values must be finite")
        if trainable:
            self.kernel = nn.Parameter(values.clone())
        else:
            self.register_buffer("kernel", values.clone())
        self.max_sampled_elements = max_sampled_elements

    @classmethod
    def from_kernel(cls, kernel: Tensor, **kwargs) -> "SphericalFilter2d":
        return cls(kernel, **kwargs)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 4 or not values.is_floating_point():
            raise TypeError("values must be a floating NCHW tensor")
        kernel_size = (int(self.kernel.shape[0]), int(self.kernel.shape[1]))
        sampled, _ = _sample_tangent_neighbourhood(
            values,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            dilation=(1, 1),
            max_sampled_elements=self.max_sampled_elements,
        )
        weights = self.kernel.reshape(1, 1, -1, 1, 1).to(
            device=values.device, dtype=values.dtype
        )
        return (sampled * weights).sum(dim=2)


class SphericalGaussianBlur2d(SphericalFilter2d):
    def __init__(
        self,
        sigma: float,
        *,
        kernel_size: int | None = None,
        truncate: float = 3.0,
        trainable: bool = False,
        max_sampled_elements: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        sigma_value = float(sigma)
        if not math.isfinite(sigma_value) or sigma_value <= 0.0:
            raise ValueError("sigma must be finite and positive")
        if kernel_size is None:
            radius = max(1, math.ceil(float(truncate) * sigma_value))
            kernel_size = radius * 2 + 1
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        axis = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
        one_dimensional = torch.exp(-0.5 * (axis / sigma_value).square())
        one_dimensional = one_dimensional / one_dimensional.sum()
        super().__init__(
            torch.outer(one_dimensional, one_dimensional),
            trainable=trainable,
            max_sampled_elements=max_sampled_elements,
        )
        self.sigma = sigma_value


class SphericalAvgPool2d(nn.Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        ceil_mode: bool = False,
        max_sampled_elements: int | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size, "kernel_size")
        self.stride = _pair(kernel_size if stride is None else stride, "stride")
        self.padding = _padding_pair(padding)
        self.dilation = _pair(dilation, "dilation")
        self.ceil_mode = bool(ceil_mode)
        self.max_sampled_elements = max_sampled_elements

    def forward(self, values: Tensor) -> Tensor:
        sampled, _ = _sample_tangent_neighbourhood(
            values,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            max_sampled_elements=self.max_sampled_elements,
        )
        return sampled.mean(dim=2)


class SphericalNeighborhoodMax(nn.Module):
    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size, "kernel_size")

    def forward(self, values: Tensor) -> Tensor:
        sampled, _ = _sample_tangent_neighbourhood(
            values,
            kernel_size=self.kernel_size,
            stride=(1, 1),
            padding=(self.kernel_size[0] // 2, self.kernel_size[1] // 2),
            dilation=(1, 1),
        )
        return sampled.amax(dim=2)


class SphericalNeighborhoodMin(SphericalNeighborhoodMax):
    def forward(self, values: Tensor) -> Tensor:
        return -super().forward(-values)


class SphericalResize(nn.Module):
    """Resample an ERP tensor onto another pixel-centre ERP lattice."""

    def __init__(
        self,
        output_shape_hw: tuple[int, int] | None = None,
        *,
        scale_factor: float | None = None,
        interpolation: str = "bilinear",
    ) -> None:
        super().__init__()
        if (output_shape_hw is None) == (scale_factor is None):
            raise ValueError("provide exactly one of output_shape_hw or scale_factor")
        if output_shape_hw is not None:
            self.output_shape_hw = _pair(output_shape_hw, "output_shape_hw")
        else:
            self.output_shape_hw = None
        if scale_factor is not None and (
            not math.isfinite(float(scale_factor)) or float(scale_factor) <= 0.0
        ):
            raise ValueError("scale_factor must be finite and positive")
        self.scale_factor = None if scale_factor is None else float(scale_factor)
        if interpolation not in {"bilinear", "nearest"}:
            raise ValueError("interpolation must be 'bilinear' or 'nearest'")
        self.interpolation = interpolation

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 4 or not values.is_floating_point():
            raise TypeError("values must be a floating NCHW tensor")
        if self.output_shape_hw is None:
            output_shape = (
                max(2, round(values.shape[-2] * self.scale_factor)),
                max(2, round(values.shape[-1] * self.scale_factor)),
            )
        else:
            output_shape = self.output_shape_hw
        output_height, output_width = output_shape
        y = torch.arange(output_height, device=values.device, dtype=values.dtype)[
            :, None
        ]
        x = torch.arange(output_width, device=values.device, dtype=values.dtype)[
            None, :
        ]
        source_x = (x + 0.5) / output_width * values.shape[-1] - 0.5
        source_y = (y + 0.5) / output_height * values.shape[-2] - 0.5
        source_x = source_x.expand(output_height, output_width)
        source_y = source_y.expand(output_height, output_width)
        wrapped = torch.cat((values[..., -1:], values, values[..., :1]), dim=-1)
        normalized_x = 2.0 * (source_x + 1.0) / (values.shape[-1] + 1.0) - 1.0
        normalized_y = 2.0 * source_y / (values.shape[-2] - 1.0) - 1.0
        grid = torch.stack((normalized_x, normalized_y), dim=-1)
        return F.grid_sample(
            wrapped,
            grid[None].expand(values.shape[0], -1, -1, -1),
            mode=self.interpolation,
            padding_mode="border",
            align_corners=True,
        )


class SphericalAreaSum(nn.Module):
    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 4 or not values.is_floating_point():
            raise TypeError("values must be a floating NCHW tensor")
        height, width = values.shape[-2:]
        edges = (
            math.pi / 2.0
            - torch.arange(height + 1, device=values.device, dtype=values.dtype)
            / height
            * math.pi
        )
        row_area = (torch.sin(edges[:-1]) - torch.sin(edges[1:])) * (
            2.0 * math.pi / width
        )
        return (values * row_area[None, None, :, None]).sum(dim=(-2, -1))


class SphericalAreaMean(nn.Module):
    def forward(self, values: Tensor) -> Tensor:
        return SphericalAreaSum()(values) / (4.0 * math.pi)


class SphericalGradient(nn.Module):
    def __init__(self, *, operator: str = "sobel") -> None:
        super().__init__()
        if operator == "sobel":
            east = (
                torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
                / 8.0
            )
            north = (
                torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
                / 8.0
            )
        elif operator == "scharr":
            east = (
                torch.tensor(
                    [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32
                )
                / 32.0
            )
            north = (
                torch.tensor(
                    [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32
                )
                / 32.0
            )
        else:
            raise ValueError("operator must be 'sobel' or 'scharr'")
        self.east = SphericalFilter2d(east)
        self.north = SphericalFilter2d(north)

    def forward(self, values: Tensor) -> Tensor:
        return torch.cat((self.east(values), self.north(values)), dim=1)


class SphericalLaplacian(SphericalFilter2d):
    def __init__(self) -> None:
        super().__init__(
            torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        )


class SphericalScaleExtrema(nn.Module):
    """Strict extrema over space and adjacent scales for N,C,S,H,W."""

    def __init__(
        self,
        *,
        spatial_radius: int = 1,
        scale_radius: int = 1,
        contrast_threshold: float = 0.0,
    ):
        super().__init__()
        if spatial_radius < 1:
            raise ValueError("spatial_radius must be positive")
        if scale_radius < 1:
            raise ValueError("scale_radius must be positive")
        self.spatial_radius = int(spatial_radius)
        self.scale_radius = int(scale_radius)
        self.contrast_threshold = float(contrast_threshold)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 5 or not values.is_floating_point():
            raise TypeError("values must be floating N,C,S,H,W")
        result = torch.zeros_like(values, dtype=torch.bool)
        size = self.spatial_radius * 2 + 1
        center_index = size * size // 2
        for scale in range(self.scale_radius, values.shape[2] - self.scale_radius):
            current = values[:, :, scale]
            sampled, _ = _sample_tangent_neighbourhood(
                current,
                kernel_size=(size, size),
                stride=(1, 1),
                padding=(self.spatial_radius, self.spatial_radius),
                dilation=(1, 1),
            )
            neighbours = torch.cat(
                (sampled[:, :, :center_index], sampled[:, :, center_index + 1 :]),
                dim=2,
            )
            other_scales = torch.cat(
                (
                    values[:, :, scale - self.scale_radius : scale],
                    values[:, :, scale + 1 : scale + self.scale_radius + 1],
                ),
                dim=2,
            )
            result[:, :, scale] = (
                (
                    (current[:, :, None] > neighbours).all(dim=2)
                    & (current[:, :, None] > other_scales).all(dim=2)
                )
                | (
                    (current[:, :, None] < neighbours).all(dim=2)
                    & (current[:, :, None] < other_scales).all(dim=2)
                )
            ) & (current.abs() >= self.contrast_threshold)
        return result


Sequential = nn.Sequential


__all__ = [
    "Sequential",
    "SphericalAreaMean",
    "SphericalAreaSum",
    "SphericalAvgPool2d",
    "SphericalFilter2d",
    "SphericalGaussianBlur2d",
    "SphericalGradient",
    "SphericalLaplacian",
    "SphericalNeighborhoodMax",
    "SphericalNeighborhoodMin",
    "SphericalResize",
    "SphericalScaleExtrema",
]
