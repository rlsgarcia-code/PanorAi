"""Differentiable geometry-safe transformations of Torch ERP tensors."""

from __future__ import annotations

import math
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


def _validate_rotation(rotation: Tensor, *, like: Tensor, dtype: torch.dtype) -> Tensor:
    matrix = torch.as_tensor(rotation, dtype=dtype, device=like.device)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("rotation must contain only finite values")
    identity = torch.eye(3, dtype=dtype, device=like.device)
    tolerance = 2e-5 if dtype == torch.float32 else 1e-7
    if not torch.allclose(matrix @ matrix.T, identity, atol=tolerance, rtol=0.0):
        raise ValueError("rotation must be orthonormal")
    determinant = float(torch.linalg.det(matrix).detach().cpu())
    if not math.isclose(determinant, 1.0, abs_tol=tolerance):
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
    rotation: Tensor,
) -> tuple[Tensor, Tensor]:
    output_height, output_width = output_shape_hw
    source_height, source_width = source_shape_hw
    dtype, device = rotation.dtype, rotation.device
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
    source_rays = output_rays @ rotation
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
    work_dtype = torch.float64 if source.dtype == torch.float64 else torch.float32
    matrix = _validate_rotation(rotation, like=source, dtype=work_dtype)
    source_x, source_y = _source_pixel_map(output_shape, source_shape, matrix)
    nchw, original_ndim = _as_nchw(source)

    if interpolation == "nearest":
        x_index = torch.floor(source_x + 0.5).to(torch.int64) % source_shape[1]
        y_index = (
            torch.floor(source_y + 0.5).to(torch.int64).clamp(0, source_shape[0] - 1)
        )
        return _restore_layout(nchw[..., y_index, x_index], original_ndim)

    original_dtype = nchw.dtype
    working = nchw
    if (
        original_dtype in {torch.float16, torch.bfloat16}
        and source.device.type == "cpu"
    ):
        working = nchw.to(torch.float32)
    wrapped = torch.cat((working[..., -1:], working, working[..., :1]), dim=-1)
    normalized_x = 2.0 * (source_x + 1.0) / (source_shape[1] + 1.0) - 1.0
    normalized_y = 2.0 * source_y / (source_shape[0] - 1.0) - 1.0
    grid = torch.stack((normalized_x, normalized_y), dim=-1).to(working.dtype)
    output = F.grid_sample(
        wrapped,
        grid[None].expand(working.shape[0], -1, -1, -1),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    if output.dtype != original_dtype:
        output = output.to(original_dtype)
    return _restore_layout(output, original_ndim)


__all__ = ["spherical_rotate"]
