"""Differentiable geometry-safe transformations of Torch ERP tensors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import torch
from torch import Tensor
from torch.nn import functional as F


def _validate_output_shape(
    output_shape_hw: tuple[int, int] | None,
    source_shape_hw: tuple[int, int],
) -> tuple[int, int]:
    if output_shape_hw is None:
        return source_shape_hw
    if (
        not isinstance(output_shape_hw, tuple)
        or len(output_shape_hw) != 2
        or any(isinstance(value, bool) for value in output_shape_hw)
        or any(not isinstance(value, Integral) for value in output_shape_hw)
    ):
        raise TypeError("output_shape_hw must be a (height, width) integer tuple")
    height, width = (int(value) for value in output_shape_hw)
    if height < 2 or width < 2:
        raise ValueError("output height and width must both be at least 2")
    return height, width


def _validate_values(values: Tensor) -> Tensor:
    if not isinstance(values, Tensor):
        raise TypeError("values must be a torch.Tensor")
    if values.ndim not in {2, 3, 4}:
        raise ValueError("values must use HW, CHW, or NCHW layout")
    if values.shape[-2] < 2 or values.shape[-1] < 2:
        raise ValueError("input height and width must both be at least 2")
    if values.is_complex():
        raise TypeError("values must use a real dtype")
    return values


def _validate_rotation(
    rotation: Tensor,
    *,
    like: Tensor,
    dtype: torch.dtype,
    batch_size: int,
) -> Tensor:
    matrix = torch.as_tensor(rotation, dtype=dtype, device=like.device)
    if matrix.shape == (3, 3):
        matrix = matrix[None]
    elif matrix.ndim != 3 or matrix.shape[-2:] != (3, 3):
        raise ValueError("rotation must have shape (3, 3) or (N, 3, 3)")
    if matrix.shape[0] not in {1, batch_size}:
        raise ValueError("rotation batch must contain one or N matrices")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("rotation must contain only finite values")
    identity = torch.eye(3, dtype=dtype, device=like.device)[None]
    tolerance = 2e-5 if dtype == torch.float32 else 1e-7
    if not torch.allclose(
        matrix @ matrix.transpose(-1, -2), identity, atol=tolerance, rtol=0.0
    ):
        raise ValueError("rotation must be orthonormal")
    determinants = torch.linalg.det(matrix)
    if not torch.allclose(
        determinants,
        torch.ones_like(determinants),
        atol=tolerance,
        rtol=0.0,
    ):
        raise ValueError("rotation must be a proper rotation with determinant +1")
    return matrix


def _as_nchw(values: Tensor) -> tuple[Tensor, int]:
    if values.ndim == 2:
        return values[None, None], 2
    if values.ndim == 3:
        return values[None], 3
    return values, 4


def _restore_layout(values: Tensor, ndim: int) -> Tensor:
    if ndim == 2:
        return values[0, 0]
    if ndim == 3:
        return values[0]
    return values


def _source_pixel_map(
    output_shape_hw: tuple[int, int],
    source_shape_hw: tuple[int, int],
    rotations: Tensor,
) -> tuple[Tensor, Tensor]:
    output_height, output_width = output_shape_hw
    source_height, source_width = source_shape_hw
    dtype, device = rotations.dtype, rotations.device
    y = torch.arange(output_height, dtype=dtype, device=device)[:, None]
    x = torch.arange(output_width, dtype=dtype, device=device)[None, :]
    longitude = ((x + 0.5) / output_width) * (2.0 * math.pi) - math.pi
    latitude = (math.pi / 2.0) - ((y + 0.5) / output_height) * math.pi
    cos_latitude = torch.cos(latitude)
    output_rays = torch.stack(
        (
            torch.sin(longitude) * cos_latitude,
            torch.sin(latitude).expand(output_height, output_width),
            torch.cos(longitude) * cos_latitude,
        ),
        dim=-1,
    )
    source_rays = torch.einsum("hwc,bcd->bhwd", output_rays, rotations)
    source_longitude = torch.atan2(source_rays[..., 0], source_rays[..., 2])
    source_latitude = torch.asin(source_rays[..., 1].clamp(-1.0, 1.0))
    source_x = torch.remainder(
        (source_longitude + math.pi) / (2.0 * math.pi) * source_width - 0.5,
        source_width,
    )
    source_y = (
        (math.pi / 2.0 - source_latitude) / math.pi * source_height - 0.5
    ).clamp(0.0, source_height - 1.0)
    return source_x, source_y


@dataclass(frozen=True, slots=True)
class SphericalRotationPlan:
    """Reusable Torch sampling plan for one or one-per-sample ERP rotations."""

    source_shape_hw: tuple[int, int]
    output_shape_hw: tuple[int, int]
    grid: Tensor
    nearest_flat_index: Tensor

    @classmethod
    def from_rotation(
        cls,
        source_shape_hw: tuple[int, int],
        rotation: Tensor,
        *,
        like: Tensor,
        batch_size: int | None = None,
        output_shape_hw: tuple[int, int] | None = None,
    ) -> "SphericalRotationPlan":
        source_shape = _validate_output_shape(source_shape_hw, source_shape_hw)
        output_shape = _validate_output_shape(output_shape_hw, source_shape)
        if batch_size is None:
            batch_size = int(like.shape[0]) if like.ndim == 4 else 1
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        work_dtype = torch.float64 if like.dtype == torch.float64 else torch.float32
        matrices = _validate_rotation(
            rotation,
            like=like,
            dtype=work_dtype,
            batch_size=batch_size,
        )
        source_x, source_y = _source_pixel_map(
            output_shape, source_shape, matrices
        )
        normalized_x = 2.0 * (source_x + 1.0) / (source_shape[1] + 1.0) - 1.0
        normalized_y = 2.0 * source_y / (source_shape[0] - 1.0) - 1.0
        grid = torch.stack((normalized_x, normalized_y), dim=-1)
        x_index = torch.floor(source_x + 0.5).to(torch.int64) % source_shape[1]
        y_index = (
            torch.floor(source_y + 0.5)
            .to(torch.int64)
            .clamp(0, source_shape[0] - 1)
        )
        nearest_flat_index = y_index * source_shape[1] + x_index
        return cls(source_shape, output_shape, grid, nearest_flat_index)

    def apply(
        self,
        values: Tensor,
        *,
        interpolation: str = "bilinear",
    ) -> Tensor:
        """Sample an aligned Torch ERP field through this plan."""

        source = _validate_values(values)
        if interpolation not in {"nearest", "bilinear"}:
            raise ValueError("interpolation must be 'nearest' or 'bilinear'")
        if interpolation == "bilinear" and not source.is_floating_point():
            raise TypeError("bilinear interpolation requires floating-point values")
        if tuple(source.shape[-2:]) != self.source_shape_hw:
            raise ValueError(
                "values shape does not match rotation plan source_shape_hw"
            )
        if source.device != self.grid.device:
            raise ValueError("values and rotation plan must use the same device")

        nchw, original_ndim = _as_nchw(source)
        batch_size, channels = int(nchw.shape[0]), int(nchw.shape[1])
        plan_batch = int(self.grid.shape[0])
        if plan_batch not in {1, batch_size}:
            raise ValueError("rotation plan batch must contain one or N grids")

        if interpolation == "nearest":
            indices = self.nearest_flat_index
            if plan_batch == 1 and batch_size != 1:
                indices = indices.expand(batch_size, -1, -1)
            gathered = torch.gather(
                nchw.reshape(batch_size, channels, -1),
                2,
                indices.reshape(batch_size, 1, -1).expand(-1, channels, -1),
            )
            output = gathered.reshape(
                batch_size, channels, *self.output_shape_hw
            )
            return _restore_layout(output, original_ndim)

        original_dtype = nchw.dtype
        working = nchw
        if (
            original_dtype in {torch.float16, torch.bfloat16}
            and source.device.type == "cpu"
        ):
            working = nchw.to(torch.float32)
        wrapped = torch.cat((working[..., -1:], working, working[..., :1]), dim=-1)
        grid = self.grid
        if plan_batch == 1 and batch_size != 1:
            grid = grid.expand(batch_size, -1, -1, -1)
        output = F.grid_sample(
            wrapped,
            grid.to(working.dtype),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        return _restore_layout(output, original_ndim)


def spherical_rotate(
    values: Tensor,
    rotation: Tensor,
    *,
    output_shape_hw: tuple[int, int] | None = None,
    interpolation: str = "bilinear",
) -> Tensor:
    """Actively rotate an ERP tensor with inverse ray mapping.

    ``rotation`` maps source rays to output rays in PanorAi's canonical frame.
    Input and output use the same ``HW``, ``CHW``, or ``NCHW`` layout. Nearest
    sampling preserves integer and boolean values exactly; bilinear sampling
    is differentiable with respect to floating input values.
    """

    source = _validate_values(values)
    if interpolation not in {"nearest", "bilinear"}:
        raise ValueError("interpolation must be 'nearest' or 'bilinear'")
    if interpolation == "bilinear" and not source.is_floating_point():
        raise TypeError("bilinear interpolation requires floating-point values")

    source_shape = (int(source.shape[-2]), int(source.shape[-1]))
    output_shape = _validate_output_shape(output_shape_hw, source_shape)
    plan = SphericalRotationPlan.from_rotation(
        source_shape,
        rotation,
        like=source,
        output_shape_hw=output_shape,
    )
    return plan.apply(source, interpolation=interpolation)


__all__ = ["SphericalRotationPlan", "spherical_rotate"]
