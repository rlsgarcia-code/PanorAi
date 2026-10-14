"""Composable NumPy modules for spherical ERP tensors."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from panorai.core._sampling import sample_map
from panorai.core.specs import (
    SphericalConv2dSpec,
    SphericalExecutionSpec,
    SphericalPool2dSpec,
    SphericalResizeSpec,
)

from ._functional import (
    _gaussian_kernel,
    spherical_filter2d,
    spherical_gradient,
    spherical_laplacian,
)
from ._transforms import spherical_resize
from . import _native


def _as_nchw(values: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(values)
    if array.ndim == 3:
        array = array[None]
        squeezed = True
    elif array.ndim == 4:
        squeezed = False
    else:
        raise ValueError("values must use CHW or NCHW layout")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("spherical modules require floating CHW/NCHW values")
    if min(array.shape[-2:]) < 2:
        raise ValueError("ERP height and width must both be at least 2")
    return array, squeezed


def _restore_rank(values: np.ndarray, squeezed: bool) -> np.ndarray:
    return values[0] if squeezed else values


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
    result = math.floor(
        (numerator + (stride - 1 if ceil_mode else 0)) / stride + 1
    )
    if result < 1:
        raise ValueError("input is too small for the requested output lattice")
    return result


def _sample_neighborhood(
    values: np.ndarray,
    *,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    angular_step_deg: float | None,
    ceil_mode: bool = False,
    max_sampled_elements: int | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Return ``N,C,K,Hout,Wout`` exponential-map tangent samples."""

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
    kernel_count = kernel_size[0] * kernel_size[1]
    if max_sampled_elements is None:
        rows_per_chunk = output_shape[0]
    else:
        per_row = batch * channels * kernel_count * output_shape[1]
        rows_per_chunk = max(1, max_sampled_elements // max(per_row, 1))

    if angular_step_deg is None:
        step_north = math.pi / height
        step_east = 2.0 * math.pi / width
    else:
        step_north = step_east = math.radians(angular_step_deg)
    ky = np.arange(kernel_size[0], dtype=np.float64)
    kx = np.arange(kernel_size[1], dtype=np.float64)
    north_offset = -(
        ky - (kernel_size[0] - 1.0) / 2.0
    ) * dilation[0] * step_north
    east_offset = (
        kx - (kernel_size[1] - 1.0) / 2.0
    ) * dilation[1] * step_east
    north_offset, east_offset = np.meshgrid(
        north_offset, east_offset, indexing="ij"
    )
    north_offset = north_offset.reshape(-1, 1, 1)
    east_offset = east_offset.reshape(-1, 1, 1)
    rho = np.hypot(east_offset, north_offset)
    sinc = np.ones_like(rho)
    np.divide(np.sin(rho), rho, out=sinc, where=rho != 0.0)

    packed = values.transpose(2, 3, 0, 1).reshape(height, width, -1)
    parts: list[np.ndarray] = []
    kernel_center_y = (kernel_size[0] - 1.0) * dilation[0] / 2.0
    kernel_center_x = (kernel_size[1] - 1.0) * dilation[1] / 2.0
    for row_start in range(0, output_shape[0], rows_per_chunk):
        row_stop = min(output_shape[0], row_start + rows_per_chunk)
        output_y = np.arange(row_start, row_stop, dtype=np.float64)[:, None]
        output_x = np.arange(output_shape[1], dtype=np.float64)[None, :]
        anchor_y = output_y * stride[0] - padding[0] + kernel_center_y
        anchor_x = output_x * stride[1] - padding[1] + kernel_center_x
        longitude = ((anchor_x + 0.5) / width) * (2.0 * math.pi) - math.pi
        latitude = math.pi / 2.0 - ((anchor_y + 0.5) / height) * math.pi
        shape = (row_stop - row_start, output_shape[1])
        longitude = np.broadcast_to(longitude, shape)
        latitude = np.broadcast_to(latitude, shape)
        sin_lon, cos_lon = np.sin(longitude), np.cos(longitude)
        sin_lat, cos_lat = np.sin(latitude), np.cos(latitude)
        rays = np.stack((sin_lon * cos_lat, sin_lat, cos_lon * cos_lat), axis=-1)
        east = np.stack((cos_lon, np.zeros_like(cos_lon), -sin_lon), axis=-1)
        north = np.stack((-sin_lon * sin_lat, cos_lat, -cos_lon * sin_lat), axis=-1)
        tangent = (
            east_offset[..., None] * east[None]
            + north_offset[..., None] * north[None]
        )
        sample_rays = (
            np.cos(rho)[..., None] * rays[None] + sinc[..., None] * tangent
        )
        sample_rays /= np.linalg.norm(sample_rays, axis=-1, keepdims=True)
        sample_lon = np.arctan2(sample_rays[..., 0], sample_rays[..., 2])
        sample_lat = np.arcsin(np.clip(sample_rays[..., 1], -1.0, 1.0))
        map_x = np.mod(
            (sample_lon + math.pi) / (2.0 * math.pi) * width - 0.5, width
        )
        map_y = np.clip(
            (math.pi / 2.0 - sample_lat) / math.pi * height - 0.5,
            0,
            height - 1,
        )
        sampled = sample_map(packed, map_x, map_y, interpolation="bilinear")
        sampled = sampled.reshape(
            kernel_count,
            row_stop - row_start,
            output_shape[1],
            batch,
            channels,
        ).transpose(3, 4, 0, 1, 2)
        parts.append(sampled)
    return np.concatenate(parts, axis=-2), output_shape


class Module:
    """Small callable protocol for inference-oriented NumPy composition."""

    def forward(self, values: np.ndarray) -> Any:
        raise NotImplementedError

    def __call__(self, values: np.ndarray) -> Any:
        return self.forward(values)


class Sequential(Module):
    """Apply NumPy spherical modules in order."""

    def __init__(self, *modules: Module) -> None:
        if not modules:
            raise ValueError("Sequential requires at least one module")
        if any(not callable(module) for module in modules):
            raise TypeError("every Sequential item must be callable")
        self.modules = tuple(modules)

    def forward(self, values: np.ndarray) -> np.ndarray:
        result = values
        for module in self.modules:
            result = module(result)
        return result


class SphericalFilter2d(Module):
    """Apply one fixed tangent-plane kernel independently to every channel."""

    def __init__(
        self,
        kernel: np.ndarray,
        *,
        angular_step_deg: float | None = None,
        execution: SphericalExecutionSpec | None = None,
    ) -> None:
        array = np.asarray(kernel, dtype=np.float64)
        if array.ndim != 2 or min(array.shape) < 1:
            raise ValueError("kernel must be a non-empty two-dimensional array")
        if any(size % 2 == 0 for size in array.shape):
            raise ValueError("kernel dimensions must be odd")
        if not np.isfinite(array).all():
            raise ValueError("kernel values must be finite")
        self.kernel = array.copy()
        self.kernel.setflags(write=False)
        self.angular_step_deg = angular_step_deg
        self.execution = execution or SphericalExecutionSpec()

    @classmethod
    def from_kernel(cls, kernel: np.ndarray, **kwargs: Any) -> "SphericalFilter2d":
        return cls(kernel, **kwargs)

    def forward(self, values: np.ndarray) -> np.ndarray:
        array, squeezed = _as_nchw(values)
        result = np.empty_like(array)
        for batch in range(array.shape[0]):
            hwc = np.moveaxis(array[batch], 0, -1)
            filtered = spherical_filter2d(
                hwc,
                self.kernel,
                angular_step_deg=self.angular_step_deg,
                backend=self.execution.backend,
            )
            result[batch] = np.moveaxis(filtered, -1, 0)
        return _restore_rank(result, squeezed)


class SphericalGaussianBlur2d(SphericalFilter2d):
    """Fixed Gaussian correlation built on :class:`SphericalFilter2d`."""

    def __init__(
        self,
        sigma: float,
        *,
        kernel_size: int | None = None,
        truncate: float = 3.0,
        angular_step_deg: float | None = None,
        execution: SphericalExecutionSpec | None = None,
    ) -> None:
        sigma_value = float(sigma)
        if not math.isfinite(sigma_value) or sigma_value <= 0.0:
            raise ValueError("sigma must be finite and positive")
        if kernel_size is None:
            radius = max(1, math.ceil(float(truncate) * sigma_value))
            kernel_size = radius * 2 + 1
        super().__init__(
            _gaussian_kernel(kernel_size, sigma_value),
            angular_step_deg=angular_step_deg,
            execution=execution,
        )
        self.sigma = sigma_value


class SphericalConv2d(Module):
    """Channel-mixing spherical correlation for NumPy CHW/NCHW tensors."""

    def __init__(
        self,
        weight: np.ndarray,
        bias: np.ndarray | None = None,
        *,
        spec: SphericalConv2dSpec | None = None,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        angular_step_deg: float | None = None,
        execution: SphericalExecutionSpec | None = None,
    ) -> None:
        weights = np.asarray(weight)
        if weights.ndim != 4 or not np.issubdtype(weights.dtype, np.floating):
            raise TypeError("weight must be a floating OIHW array")
        inferred = SphericalConv2dSpec(
            in_channels=weights.shape[1] * groups,
            out_channels=weights.shape[0],
            kernel_size=weights.shape[-2:],
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            angular_step_deg=angular_step_deg,
        )
        self.spec = inferred if spec is None else spec
        expected = (
            self.spec.out_channels,
            self.spec.in_channels // self.spec.groups,
            *self.spec.kernel_size,
        )
        if weights.shape != expected:
            raise ValueError(f"weight shape must be {expected}, received {weights.shape}")
        if bias is None:
            biases = None
        else:
            biases = np.asarray(bias)
            if biases.shape != (self.spec.out_channels,):
                raise ValueError("bias must have shape (out_channels,)")
            if not np.issubdtype(biases.dtype, np.floating):
                raise TypeError("bias must use a floating dtype")
        self.weight = weights.copy()
        self.weight.setflags(write=False)
        self.bias = None if biases is None else biases.copy()
        if self.bias is not None:
            self.bias.setflags(write=False)
        self.execution = execution or SphericalExecutionSpec()

    @classmethod
    def from_kernel(
        cls, kernel: np.ndarray, *, execution: SphericalExecutionSpec | None = None
    ) -> "SphericalConv2d":
        values = np.asarray(kernel)
        if values.ndim != 2:
            raise ValueError("kernel must be two-dimensional")
        return cls(values[None, None], execution=execution)

    @classmethod
    def from_spec(
        cls,
        spec: SphericalConv2dSpec,
        weight: np.ndarray,
        bias: np.ndarray | None = None,
        *,
        execution: SphericalExecutionSpec | None = None,
    ) -> "SphericalConv2d":
        return cls(weight, bias, spec=spec, execution=execution)

    def forward(self, values: np.ndarray) -> np.ndarray:
        array, squeezed = _as_nchw(values)
        if array.shape[1] != self.spec.in_channels:
            raise ValueError(
                f"expected {self.spec.in_channels} channels, received {array.shape[1]}"
            )
        use_native = self.execution.backend == "native" or (
            self.execution.backend == "auto"
            and _native.supports_native_conv(array, self.weight, self.bias)
        )
        if use_native:
            if self.spec.angular_step_deg is None:
                angular_step = (
                    math.pi / array.shape[-2],
                    2.0 * math.pi / array.shape[-1],
                )
            else:
                step = math.radians(self.spec.angular_step_deg)
                angular_step = (step, step)
            result = _native.native_spherical_conv2d(
                array,
                self.weight,
                self.bias,
                stride=self.spec.stride,
                padding=self.spec.padding,
                dilation=self.spec.dilation,
                groups=self.spec.groups,
                angular_step_rad=angular_step,
                output_shape_hw=self.spec.output_shape(array.shape[-2:]),
            )
            return _restore_rank(result, squeezed)
        sampled, _ = _sample_neighborhood(
            array,
            kernel_size=self.spec.kernel_size,
            stride=self.spec.stride,
            padding=self.spec.padding,
            dilation=self.spec.dilation,
            angular_step_deg=self.spec.angular_step_deg,
            max_sampled_elements=self.execution.max_sampled_elements,
        )
        batch = array.shape[0]
        input_per_group = self.spec.in_channels // self.spec.groups
        output_per_group = self.spec.out_channels // self.spec.groups
        kernel_count = self.spec.kernel_size[0] * self.spec.kernel_size[1]
        grouped_samples = sampled.reshape(
            batch,
            self.spec.groups,
            input_per_group,
            kernel_count,
            sampled.shape[-2],
            sampled.shape[-1],
        )
        grouped_weights = self.weight.reshape(
            self.spec.groups, output_per_group, input_per_group, kernel_count
        )
        result = np.einsum(
            "ngckhw,gock->ngohw", grouped_samples, grouped_weights, optimize=True
        ).reshape(batch, self.spec.out_channels, sampled.shape[-2], sampled.shape[-1])
        if self.bias is not None:
            result = result + self.bias[None, :, None, None]
        result = result.astype(np.result_type(array.dtype, self.weight.dtype), copy=False)
        return _restore_rank(result, squeezed)


class SphericalConvTranspose2d(Module):
    """NumPy counterpart of a spherical ``ConvTranspose2d`` inference layer."""

    def __init__(
        self,
        weight: np.ndarray,
        bias: np.ndarray | None = None,
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        output_padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        execution: SphericalExecutionSpec | None = None,
    ) -> None:
        weights = np.asarray(weight)
        if weights.ndim != 4 or not np.issubdtype(weights.dtype, np.floating):
            raise TypeError("weight must be a floating IOHW array")
        from panorai.core.specs import _nonnegative_pair, _positive_pair

        self.in_channels = int(weights.shape[0])
        self.groups = int(groups)
        if self.in_channels % self.groups:
            raise ValueError("groups must divide input channels")
        self.out_channels = int(weights.shape[1] * self.groups)
        self.kernel_size = (int(weights.shape[2]), int(weights.shape[3]))
        self.stride = _positive_pair(stride, "stride")
        self.padding = _nonnegative_pair(padding, "padding")
        self.output_padding = _nonnegative_pair(output_padding, "output_padding")
        self.dilation = _positive_pair(dilation, "dilation")
        if any(
            output >= step
            for output, step in zip(
                self.output_padding, self.stride, strict=True
            )
        ):
            raise ValueError("output_padding must be smaller than stride")
        self.weight = weights.copy()
        self.weight.setflags(write=False)
        self.bias = None if bias is None else np.asarray(bias).copy()
        if self.bias is not None:
            if self.bias.shape != (self.out_channels,):
                raise ValueError("bias must have shape (out_channels,)")
            self.bias.setflags(write=False)
        self.execution = execution or SphericalExecutionSpec()

    def forward(self, values: np.ndarray) -> np.ndarray:
        array, squeezed = _as_nchw(values)
        if array.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels, received {array.shape[1]}"
            )
        expanded_shape = (
            (array.shape[-2] - 1) * self.stride[0] + 1 + self.output_padding[0],
            (array.shape[-1] - 1) * self.stride[1] + 1 + self.output_padding[1],
        )
        expanded = np.zeros(
            (array.shape[0], array.shape[1], *expanded_shape), dtype=array.dtype
        )
        expanded[..., :: self.stride[0], :: self.stride[1]] = array
        effective_padding = (
            self.dilation[0] * (self.kernel_size[0] - 1) - self.padding[0],
            self.dilation[1] * (self.kernel_size[1] - 1) - self.padding[1],
        )
        if min(effective_padding) < 0:
            raise ValueError("negative effective spherical padding is unsupported")
        input_per_group = self.in_channels // self.groups
        output_per_group = self.out_channels // self.groups
        converted = np.flip(self.weight, axis=(-2, -1)).reshape(
            self.groups,
            input_per_group,
            output_per_group,
            *self.kernel_size,
        ).transpose(0, 2, 1, 3, 4).reshape(
            self.out_channels, input_per_group, *self.kernel_size
        )
        result = SphericalConv2d(
            converted,
            self.bias,
            stride=1,
            padding=effective_padding,
            dilation=self.dilation,
            groups=self.groups,
            execution=self.execution,
        )(expanded)
        return _restore_rank(result, squeezed)


class _Pool2d(Module):
    reduction = "max"

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        ceil_mode: bool = False,
        angular_step_deg: float | None = None,
        execution: SphericalExecutionSpec | None = None,
    ) -> None:
        self.spec = SphericalPool2dSpec(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            angular_step_deg=angular_step_deg,
        )
        self.execution = execution or SphericalExecutionSpec()

    def forward(self, values: np.ndarray) -> np.ndarray:
        array, squeezed = _as_nchw(values)
        sampled, _ = _sample_neighborhood(
            array,
            kernel_size=self.spec.kernel_size,
            stride=self.spec.stride,
            padding=self.spec.padding,
            dilation=self.spec.dilation,
            angular_step_deg=self.spec.angular_step_deg,
            ceil_mode=self.spec.ceil_mode,
            max_sampled_elements=self.execution.max_sampled_elements,
        )
        result = sampled.max(axis=2) if self.reduction == "max" else sampled.mean(axis=2)
        return _restore_rank(result.astype(array.dtype, copy=False), squeezed)


class SphericalMaxPool2d(_Pool2d):
    reduction = "max"


class SphericalAvgPool2d(_Pool2d):
    reduction = "mean"


class SphericalNeighborhoodMax(SphericalMaxPool2d):
    def __init__(self, kernel_size: int = 3, **kwargs: Any) -> None:
        if "stride" in kwargs or "padding" in kwargs:
            raise TypeError("neighborhood operations fix stride=1 and centred padding")
        size = int(kernel_size)
        super().__init__(size, stride=1, padding=size // 2, **kwargs)


class SphericalNeighborhoodMin(SphericalMaxPool2d):
    def forward(self, values: np.ndarray) -> np.ndarray:
        return -super().forward(-np.asarray(values))


class SphericalResize(Module):
    def __init__(
        self,
        output_shape_hw: tuple[int, int] | None = None,
        *,
        scale_factor: float | None = None,
        interpolation: str = "bilinear",
    ) -> None:
        if (output_shape_hw is None) == (scale_factor is None):
            raise ValueError("provide exactly one of output_shape_hw or scale_factor")
        self.spec = (
            None
            if output_shape_hw is None
            else SphericalResizeSpec(output_shape_hw, interpolation)
        )
        if scale_factor is not None and (
            not math.isfinite(float(scale_factor)) or float(scale_factor) <= 0.0
        ):
            raise ValueError("scale_factor must be finite and positive")
        self.scale_factor = None if scale_factor is None else float(scale_factor)
        if interpolation not in {"bilinear", "nearest"}:
            raise ValueError("interpolation must be 'bilinear' or 'nearest'")
        self.interpolation = interpolation

    def forward(self, values: np.ndarray) -> np.ndarray:
        array, squeezed = _as_nchw(values)
        output_shape = (
            (
                max(2, round(array.shape[-2] * self.scale_factor)),
                max(2, round(array.shape[-1] * self.scale_factor)),
            )
            if self.spec is None
            else self.spec.output_shape_hw
        )
        packed = array.transpose(2, 3, 0, 1).reshape(
            array.shape[-2], array.shape[-1], -1
        )
        resized = spherical_resize(
            packed,
            output_shape,
            interpolation=self.interpolation,
        )
        result = resized.reshape(
            *output_shape, array.shape[0], array.shape[1]
        ).transpose(2, 3, 0, 1)
        return _restore_rank(result.astype(array.dtype, copy=False), squeezed)


def _solid_angle_row_weights(height: int, dtype: np.dtype[Any]) -> np.ndarray:
    edges = (
        math.pi / 2.0
        - np.arange(height + 1, dtype=np.float64) / height * math.pi
    )
    return (np.sin(edges[:-1]) - np.sin(edges[1:])).astype(dtype, copy=False)


def spherical_solid_angle_weights(
    shape_hw: tuple[int, int], *, normalize: bool = False
) -> np.ndarray:
    """Return exact ERP pixel-cell solid angles for every raster location."""

    height, width = (int(shape_hw[0]), int(shape_hw[1]))
    if height < 1 or width < 1:
        raise ValueError("shape_hw must contain positive dimensions")
    row_weights = _solid_angle_row_weights(height, np.dtype(np.float64))
    result = np.broadcast_to(
        (row_weights * (2.0 * math.pi / width))[:, None], (height, width)
    ).copy()
    if normalize:
        result /= 4.0 * math.pi
    result.setflags(write=False)
    return result


class SphericalAreaSum(Module):
    """Integrate an ERP using exact pixel-cell solid angles."""

    def forward(self, values: np.ndarray) -> np.ndarray:
        array, squeezed = _as_nchw(values)
        row_weights = _solid_angle_row_weights(array.shape[-2], array.dtype)
        pixel_area = row_weights * (2.0 * math.pi / array.shape[-1])
        result = (array * pixel_area[None, None, :, None]).sum(axis=(-2, -1))
        return result[0] if squeezed else result


class SphericalAreaMean(Module):
    """Solid-angle mean over a complete ERP."""

    def forward(self, values: np.ndarray) -> np.ndarray:
        return SphericalAreaSum()(values) / (4.0 * math.pi)


class SphericalGradient(Module):
    """Return east and north derivatives as two channels per input channel."""

    def __init__(self, *, operator: str = "sobel", backend: str = "auto") -> None:
        self.operator = operator
        self.backend = backend

    def forward(self, values: np.ndarray) -> np.ndarray:
        array, squeezed = _as_nchw(values)
        outputs = []
        for batch in array:
            channels = []
            for channel in batch:
                gradient = spherical_gradient(
                    channel, operator=self.operator, backend=self.backend
                )
                channels.extend((gradient.east, gradient.north))
            outputs.append(np.stack(channels))
        result = np.stack(outputs).astype(array.dtype, copy=False)
        return _restore_rank(result, squeezed)


class SphericalLaplacian(Module):
    def __init__(self, *, backend: str = "auto") -> None:
        self.backend = backend

    def forward(self, values: np.ndarray) -> np.ndarray:
        array, squeezed = _as_nchw(values)
        result = np.stack(
            [
                np.stack(
                    [spherical_laplacian(channel, backend=self.backend) for channel in batch]
                )
                for batch in array
            ]
        ).astype(array.dtype, copy=False)
        return _restore_rank(result, squeezed)


class SphericalScaleExtrema(Module):
    """Strict spatial-and-scale extrema for ``N,C,S,H,W`` responses."""

    def __init__(
        self,
        *,
        spatial_radius: int = 1,
        scale_radius: int = 1,
        contrast_threshold: float = 0.0,
    ):
        if spatial_radius < 1:
            raise ValueError("spatial_radius must be positive")
        if scale_radius < 1:
            raise ValueError("scale_radius must be positive")
        self.spatial_radius = int(spatial_radius)
        self.scale_radius = int(scale_radius)
        self.contrast_threshold = float(contrast_threshold)

    def forward(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim != 5 or not np.issubdtype(array.dtype, np.floating):
            raise TypeError("values must be a floating N,C,S,H,W array")
        result = np.zeros(array.shape, dtype=np.bool_)
        size = self.spatial_radius * 2 + 1
        center_index = size * size // 2
        for scale in range(self.scale_radius, array.shape[2] - self.scale_radius):
            current = array[:, :, scale]
            sampled, _ = _sample_neighborhood(
                current,
                kernel_size=(size, size),
                stride=(1, 1),
                padding=(self.spatial_radius, self.spatial_radius),
                dilation=(1, 1),
                angular_step_deg=None,
            )
            neighbours = np.concatenate(
                (sampled[:, :, :center_index], sampled[:, :, center_index + 1 :]),
                axis=2,
            )
            spatial_max = np.all(current[:, :, None] > neighbours, axis=2)
            spatial_min = np.all(current[:, :, None] < neighbours, axis=2)
            other_scales = np.concatenate(
                (
                    array[:, :, scale - self.scale_radius : scale],
                    array[:, :, scale + 1 : scale + self.scale_radius + 1],
                ),
                axis=2,
            )
            greater_scale = np.all(current[:, :, None] > other_scales, axis=2)
            lower_scale = np.all(current[:, :, None] < other_scales, axis=2)
            result[:, :, scale] = (
                (spatial_max & greater_scale) | (spatial_min & lower_scale)
            ) & (np.abs(current) >= self.contrast_threshold)
        return result


__all__ = [
    "Module",
    "Sequential",
    "SphericalAreaMean",
    "SphericalAreaSum",
    "SphericalAvgPool2d",
    "SphericalConv2d",
    "SphericalConvTranspose2d",
    "SphericalFilter2d",
    "SphericalGaussianBlur2d",
    "SphericalGradient",
    "SphericalLaplacian",
    "SphericalMaxPool2d",
    "SphericalNeighborhoodMax",
    "SphericalNeighborhoodMin",
    "SphericalResize",
    "SphericalScaleExtrema",
    "spherical_solid_angle_weights",
]
