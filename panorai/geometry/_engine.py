"""Backend-neutral spherical geometry with NumPy and optional Torch sampling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real
from types import MappingProxyType
from typing import Any

import numpy as np

from ._contracts import (
    ERPPointProjection,
    GnomonicFaceGeometry,
    GnomonicPointProjection,
    GnomonicRayProjection,
    GnomonicSpec,
    ProjectionResult,
    _shape_hw,
)
from ._typing import (
    ArrayT,
    Interpolation,
    InvalidPolicy,
    ShapeHW,
    _validate_interpolation,
    _validate_invalid_policy,
)

CUBE_FACE_ORDER = ("front", "right", "back", "left", "up", "down")

# PERF-007 reference evidence measured the selective Torch route at 5.01x the
# cached full-grid route on CPU. CUDA and MPS remain on the full-grid route
# until a device-specific measurement clears the required 20 percent margin.
_TORCH_SELECTIVE_CUBEMAP_DEVICE_TYPES = frozenset(("cpu",))

# Values are (forward, right, up), in PanorAi's +X right, +Y up, +Z forward
# Cartesian frame. Tuples keep the public constant immutable and serializable.
CUBE_FACE_BASES = MappingProxyType(
    {
        "front": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        "right": ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
        "back": ((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        "left": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        "up": ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
        "down": ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    }
)


@dataclass(frozen=True, slots=True)
class _GnomonicForwardPlan:
    pixels_xy: Any


@dataclass(frozen=True, slots=True)
class _GnomonicBackPlan:
    map_x: Any
    map_y: Any
    support: Any


@dataclass(frozen=True, slots=True)
class _GnomonicSelectiveFacePlan:
    """Compact ERP destinations and face coordinates for one gnomonic view."""

    flat_indices: Any
    map_x: Any
    map_y: Any
    center_score: Any


@dataclass(frozen=True, slots=True)
class _GnomonicBatchBackPlan:
    """Reusable selective plan for an ordered, arbitrary-size view set."""

    output_shape_hw: ShapeHW
    faces: tuple[_GnomonicSelectiveFacePlan, ...]


@dataclass(frozen=True, slots=True)
class _GnomonicSparseProjection:
    """Sampled values whose ERP locations are carried by ``flat_indices``."""

    flat_indices: Any
    data: Any
    center_score: Any
    validity_mask: Any | None = None
    valid_weight: Any | None = None


@dataclass(frozen=True, slots=True)
class _CubemapForwardPlan:
    pixels_xy: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _CubemapFullBackPlan:
    selected: Any
    maps_xy: tuple[tuple[Any, Any], ...]


@dataclass(frozen=True, slots=True)
class _CubemapSelectiveFacePlan:
    flat_indices: Any
    map_x: Any
    map_y: Any


@dataclass(frozen=True, slots=True)
class _CubemapSelectiveBackPlan:
    output_shape_hw: ShapeHW
    faces: tuple[_CubemapSelectiveFacePlan, ...]


def _torch_module():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in clean-wheel CI
        raise ImportError(
            "Torch support is optional; install PanorAi with `pip install panorai[torch]`."
        ) from exc
    return torch


def _is_torch(value: Any) -> bool:
    module = type(value).__module__
    return module == "torch" or module.startswith("torch.")


def _require_array(value: Any, name: str, *, image: bool = False) -> None:
    if not (_is_torch(value) or isinstance(value, np.ndarray)):
        raise TypeError(f"{name} must be a numpy.ndarray or torch.Tensor")
    if image:
        if _is_torch(value) and value.ndim not in {2, 3, 4}:
            raise ValueError("Torch images must use HW, CHW, or NCHW layout")
        if not _is_torch(value) and value.ndim not in {2, 3}:
            raise ValueError("NumPy images must use HW or HWC layout")


def _validate_shape(image_shape_hw: object, name: str = "image_shape_hw") -> ShapeHW:
    return _shape_hw(image_shape_hw, name)


def erp_pixels_to_rays(pixels_xy: ArrayT, image_shape_hw: ShapeHW) -> ArrayT:
    """Convert ERP pixel-centre coordinates to unit Cartesian rays.

    ``pixels_xy`` must have a final dimension of length two. Coordinates use a
    top-left image origin and may be fractional. Horizontal coordinates wrap.
    """

    _require_array(pixels_xy, "pixels_xy")
    height, width = _validate_shape(image_shape_hw)
    if pixels_xy.shape[-1] != 2:
        raise ValueError("pixels_xy must have a final dimension of length 2")
    if _is_torch(pixels_xy):
        torch = _torch_module()
        if not (pixels_xy.dtype.is_floating_point or pixels_xy.dtype.is_complex):
            pixels_xy = pixels_xy.to(torch.get_default_dtype())
        x, y = pixels_xy.unbind(dim=-1)
        lon = ((x + 0.5) / width) * (2.0 * torch.pi) - torch.pi
        lat = (torch.pi / 2.0) - ((y + 0.5) / height) * torch.pi
        cos_lat = torch.cos(lat)
        return torch.stack(
            (torch.sin(lon) * cos_lat, torch.sin(lat), torch.cos(lon) * cos_lat),
            dim=-1,
        )

    pixels = np.asarray(pixels_xy)
    if not np.issubdtype(pixels.dtype, np.floating):
        pixels = pixels.astype(np.float64)
    x, y = np.moveaxis(pixels, -1, 0)
    lon = ((x + 0.5) / width) * (2.0 * np.pi) - np.pi
    lat = (np.pi / 2.0) - ((y + 0.5) / height) * np.pi
    cos_lat = np.cos(lat)
    return np.stack(
        (np.sin(lon) * cos_lat, np.sin(lat), np.cos(lon) * cos_lat), axis=-1
    )


def rays_to_erp_pixels(
    rays: ArrayT, image_shape_hw: ShapeHW
) -> ERPPointProjection[ArrayT]:
    """Project Cartesian rays or points onto ERP pixel-centre coordinates."""

    _require_array(rays, "rays")
    height, width = _validate_shape(image_shape_hw)
    if rays.shape[-1] != 3:
        raise ValueError("rays must have a final dimension of length 3")
    if _is_torch(rays):
        torch = _torch_module()
        if not (rays.dtype.is_floating_point or rays.dtype.is_complex):
            rays = rays.to(torch.get_default_dtype())
        ranges = torch.linalg.vector_norm(rays, dim=-1)
        finite = torch.isfinite(rays).all(dim=-1) & torch.isfinite(ranges)
        valid = finite & (ranges > 0)
        safe = torch.where(valid, ranges, torch.ones_like(ranges))
        unit = rays / safe.unsqueeze(-1)
        lon = torch.atan2(unit[..., 0], unit[..., 2])
        lat = torch.asin(torch.clamp(unit[..., 1], -1.0, 1.0))
        x = torch.remainder((lon + torch.pi) / (2.0 * torch.pi) * width - 0.5, width)
        y = (torch.pi / 2.0 - lat) / torch.pi * height - 0.5
        nan = torch.full_like(x, float("nan"))
        pixels = torch.stack(
            (torch.where(valid, x, nan), torch.where(valid, y, nan)), -1
        )
        return ERPPointProjection(pixels, ranges, valid)

    values = np.asarray(rays)
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float64)
    ranges = np.linalg.norm(values, axis=-1)
    valid = np.isfinite(values).all(axis=-1) & np.isfinite(ranges) & (ranges > 0)
    safe = np.where(valid, ranges, 1.0)
    unit = values / safe[..., None]
    lon = np.arctan2(unit[..., 0], unit[..., 2])
    lat = np.arcsin(np.clip(unit[..., 1], -1.0, 1.0))
    x = np.mod((lon + np.pi) / (2.0 * np.pi) * width - 0.5, width)
    y = (np.pi / 2.0 - lat) / np.pi * height - 0.5
    pixels = np.stack((np.where(valid, x, np.nan), np.where(valid, y, np.nan)), axis=-1)
    return ERPPointProjection(pixels, ranges, valid)


def _floating_geometry_input(value: ArrayT, name: str, final_size: int) -> ArrayT:
    _require_array(value, name)
    if value.ndim == 0 or value.shape[-1] != final_size:
        raise ValueError(f"{name} must have a final dimension of length {final_size}")
    if _is_torch(value):
        torch = _torch_module()
        if value.dtype.is_complex:
            raise TypeError(f"{name} must use a real-valued dtype")
        return (
            value
            if value.dtype.is_floating_point
            else value.to(torch.get_default_dtype())
        )
    values = np.asarray(value)
    if np.issubdtype(values.dtype, np.complexfloating):
        raise TypeError(f"{name} must use a real-valued dtype")
    return (
        values
        if np.issubdtype(values.dtype, np.floating)
        else values.astype(np.float64)
    )


def _geometry_constant(value: np.ndarray, like: ArrayT | None = None) -> ArrayT:
    if like is not None and _is_torch(like):
        torch = _torch_module()
        dtype = (
            like.dtype if like.dtype.is_floating_point else torch.get_default_dtype()
        )
        return torch.as_tensor(value, dtype=dtype, device=like.device)
    if (
        like is not None
        and isinstance(like, np.ndarray)
        and np.issubdtype(like.dtype, np.floating)
    ):
        return value.astype(like.dtype, copy=False)
    return value.copy()


def gnomonic_intrinsics(spec: GnomonicSpec, *, like: ArrayT | None = None) -> ArrayT:
    """Return the pinhole intrinsic matrix equivalent to ``spec``.

    Pixel coordinates follow PanorAi's pixel-centre convention.  The local
    camera raster uses ``+x`` right and ``+y`` down, as expected by OpenCV and
    COLMAP.
    """

    height, width = spec.output_shape_hw
    x_limit = math.tan(math.radians(spec.hfov_deg) / 2.0)
    y_limit = math.tan(math.radians(spec.vfov_deg) / 2.0)
    matrix = np.asarray(
        (
            (width / (2.0 * x_limit), 0.0, (width - 1.0) / 2.0),
            (0.0, height / (2.0 * y_limit), (height - 1.0) / 2.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    return _geometry_constant(matrix, like)


def gnomonic_rotation(spec: GnomonicSpec, *, like: ArrayT | None = None) -> ArrayT:
    """Return ``R_panorama_from_face`` for the rolled virtual camera.

    Columns are the face camera's right, down, and forward axes expressed in
    the canonical panorama frame ``(+X right, +Y up, +Z forward)``. Because
    raster y is down while panorama Y is up, this orthogonal direction
    transform has determinant -1; relative face transforms are rotations.
    """

    forward, right, up = _gnomonic_basis(spec)
    roll = math.radians(spec.roll_deg)
    down = -up
    camera_right = math.cos(roll) * right + math.sin(roll) * down
    camera_down = -math.sin(roll) * right + math.cos(roll) * down
    matrix = np.stack((camera_right, camera_down, forward), axis=1)
    return _geometry_constant(matrix, like)


def gnomonic_pixels_to_rays(
    pixels_xy: ArrayT, spec: GnomonicSpec
) -> GnomonicRayProjection[ArrayT]:
    """Convert gnomonic pixel-centre coordinates to panorama-frame unit rays.

    The final dimension of ``pixels_xy`` is ``(x, y)``.  ``valid`` is true for
    finite coordinates inside the closed raster footprint
    ``[-0.5, W-0.5] x [-0.5, H-0.5]``.  Invalid rays are returned as NaN.
    Floating NumPy dtype and Torch dtype/device are preserved.
    """

    pixels = _floating_geometry_input(pixels_xy, "pixels_xy", 2)
    height, width = spec.output_shape_hw
    K = gnomonic_intrinsics(spec, like=pixels)
    R = gnomonic_rotation(spec, like=pixels)
    if _is_torch(pixels):
        torch = _torch_module()
        x, y = pixels.unbind(dim=-1)
        finite = torch.isfinite(pixels).all(dim=-1)
        valid = (
            finite
            & (x >= -0.5)
            & (x <= width - 0.5)
            & (y >= -0.5)
            & (y <= height - 0.5)
        )
        local = torch.stack(
            ((x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1], torch.ones_like(x)),
            dim=-1,
        )
        rays = local @ R.transpose(0, 1)
        rays = rays / torch.linalg.vector_norm(rays, dim=-1, keepdim=True)
        rays = torch.where(
            valid.unsqueeze(-1), rays, torch.full_like(rays, float("nan"))
        )
        return GnomonicRayProjection(rays, valid)

    values = np.asarray(pixels)
    x, y = np.moveaxis(values, -1, 0)
    valid = (
        np.isfinite(values).all(axis=-1)
        & (x >= -0.5)
        & (x <= width - 0.5)
        & (y >= -0.5)
        & (y <= height - 0.5)
    )
    local = np.stack(
        ((x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1], np.ones_like(x)),
        axis=-1,
    )
    rays = local @ R.T
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    rays = np.where(valid[..., None], rays, np.nan).astype(values.dtype, copy=False)
    return GnomonicRayProjection(rays, valid)


def rays_to_gnomonic_pixels(
    rays_xyz: ArrayT, spec: GnomonicSpec
) -> GnomonicPointProjection[ArrayT]:
    """Project panorama-frame rays or points onto a gnomonic raster."""

    rays = _floating_geometry_input(rays_xyz, "rays_xyz", 3)
    height, width = spec.output_shape_hw
    K = gnomonic_intrinsics(spec, like=rays)
    R = gnomonic_rotation(spec, like=rays)
    if _is_torch(rays):
        torch = _torch_module()
        ranges = torch.linalg.vector_norm(rays, dim=-1)
        finite = torch.isfinite(rays).all(dim=-1) & torch.isfinite(ranges)
        nonzero = finite & (ranges > 0)
        safe_range = torch.where(nonzero, ranges, torch.ones_like(ranges))
        unit = rays / safe_range.unsqueeze(-1)
        local = unit @ R
        in_front = local[..., 2] > 0
        safe_z = torch.where(in_front, local[..., 2], torch.ones_like(local[..., 2]))
        x = K[0, 0] * (local[..., 0] / safe_z) + K[0, 2]
        y = K[1, 1] * (local[..., 1] / safe_z) + K[1, 2]
        valid = (
            nonzero
            & in_front
            & (x >= -0.5)
            & (x <= width - 0.5)
            & (y >= -0.5)
            & (y <= height - 0.5)
        )
        nan = torch.full_like(x, float("nan"))
        pixels = torch.stack(
            (torch.where(valid, x, nan), torch.where(valid, y, nan)), dim=-1
        )
        return GnomonicPointProjection(pixels, ranges, valid)

    values = np.asarray(rays)
    ranges = np.linalg.norm(values, axis=-1)
    finite = np.isfinite(values).all(axis=-1) & np.isfinite(ranges)
    nonzero = finite & (ranges > 0)
    safe_range = np.where(nonzero, ranges, 1.0)
    unit = values / safe_range[..., None]
    local = unit @ R
    in_front = local[..., 2] > 0
    safe_z = np.where(in_front, local[..., 2], 1.0)
    x = K[0, 0] * (local[..., 0] / safe_z) + K[0, 2]
    y = K[1, 1] * (local[..., 1] / safe_z) + K[1, 2]
    valid = (
        nonzero
        & in_front
        & (x >= -0.5)
        & (x <= width - 0.5)
        & (y >= -0.5)
        & (y <= height - 0.5)
    )
    pixels = np.stack((np.where(valid, x, np.nan), np.where(valid, y, np.nan)), axis=-1)
    return GnomonicPointProjection(
        pixels.astype(values.dtype, copy=False), ranges, valid
    )


def gnomonic_pixel_map(
    spec: GnomonicSpec,
    erp_shape_hw: ShapeHW,
    *,
    like: ArrayT | None = None,
) -> ArrayT:
    """Return the face-to-ERP source pixel map used by canonical sampling."""

    height, width = spec.output_shape_hw
    if like is not None and _is_torch(like):
        torch = _torch_module()
        dtype = (
            like.dtype if like.dtype.is_floating_point else torch.get_default_dtype()
        )
        y, x = torch.meshgrid(
            torch.arange(height, dtype=dtype, device=like.device),
            torch.arange(width, dtype=dtype, device=like.device),
            indexing="ij",
        )
        pixels = torch.stack((x, y), dim=-1)
    else:
        dtype = (
            like.dtype
            if isinstance(like, np.ndarray) and np.issubdtype(like.dtype, np.floating)
            else np.float64
        )
        y, x = np.indices((height, width), dtype=dtype)
        pixels = np.stack((x, y), axis=-1)
    rays = gnomonic_pixels_to_rays(pixels, spec).rays_xyz
    return rays_to_erp_pixels(rays, erp_shape_hw).pixels_xy


def gnomonic_face_geometry(
    face_id: str,
    spec: GnomonicSpec,
    support_mask: ArrayT,
    *,
    erp_shape_hw: ShapeHW | None = None,
    include_source_pixels: bool = False,
) -> GnomonicFaceGeometry[ArrayT]:
    """Construct explicit virtual-camera metadata for a materialized face."""

    _require_array(support_mask, "support_mask")
    if _is_torch(support_mask):
        if support_mask.dtype != _torch_module().bool:
            raise TypeError("support_mask must have boolean dtype")
    elif support_mask.dtype != np.bool_:
        raise TypeError("support_mask must have boolean dtype")
    if tuple(support_mask.shape) != tuple(spec.output_shape_hw):
        raise ValueError("support_mask must have exact shape spec.output_shape_hw")
    if include_source_pixels and erp_shape_hw is None:
        raise ValueError("erp_shape_hw is required when include_source_pixels=True")
    source_pixels = (
        gnomonic_pixel_map(spec, erp_shape_hw, like=support_mask)
        if include_source_pixels
        else None
    )
    return GnomonicFaceGeometry(
        face_id=str(face_id),
        spec=spec,
        K=gnomonic_intrinsics(spec, like=support_mask),
        R_panorama_from_face=gnomonic_rotation(spec, like=support_mask),
        support_mask=support_mask,
        source_pixels_xy=source_pixels,
    )


def _numpy_grid(
    shape_hw: tuple[int, int], x_limit: float, y_limit: float
) -> tuple[np.ndarray, np.ndarray]:
    height, width = _validate_shape(shape_hw)
    x = (((np.arange(width, dtype=np.float64) + 0.5) / width) * 2.0 - 1.0) * x_limit
    y = (((np.arange(height, dtype=np.float64) + 0.5) / height) * 2.0 - 1.0) * y_limit
    return np.meshgrid(x, y)


def _gnomonic_basis(spec: GnomonicSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.deg2rad(spec.center_lat_deg)
    lon = np.deg2rad(spec.center_lon_deg)
    forward = np.array(
        (np.sin(lon) * np.cos(lat), np.sin(lat), np.cos(lon) * np.cos(lat))
    )
    right = np.array((np.cos(lon), 0.0, -np.sin(lon)))
    up = np.array((-np.sin(lon) * np.sin(lat), np.cos(lat), -np.cos(lon) * np.sin(lat)))
    return forward, right, up


def _gnomonic_rays_numpy(spec: GnomonicSpec) -> np.ndarray:
    height, width = spec.output_shape_hw
    y, x = np.indices((height, width), dtype=np.float64)
    pixels = np.stack((x, y), axis=-1)
    return gnomonic_pixels_to_rays(pixels, spec).rays_xyz


def _erp_rays_numpy(shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = _validate_shape(shape_hw)
    y, x = np.indices((height, width), dtype=np.float64)
    return erp_pixels_to_rays(np.stack((x, y), axis=-1), shape_hw)


def _geometry_like(image: Any) -> Any | None:
    if not _is_torch(image):
        return None
    torch = _torch_module()
    # Build geometry in float64 wherever the device supports it, matching the
    # canonical NumPy oracle before the sampler casts its grid to input dtype.
    dtype = torch.float32 if image.device.type == "mps" else torch.float64
    return torch.empty((), dtype=dtype, device=image.device)


def _sampling_grid_like(image: Any) -> Any | None:
    if not _is_torch(image):
        return None
    torch = _torch_module()
    dtype = (
        torch.float64
        if image.dtype == torch.float64 and image.device.type != "mps"
        else torch.float32
    )
    return torch.empty((), dtype=dtype, device=image.device)


def _erp_rays(shape_hw: ShapeHW, *, like: Any | None = None) -> Any:
    if like is None or not _is_torch(like):
        return _erp_rays_numpy(shape_hw)
    torch = _torch_module()
    height, width = _validate_shape(shape_hw)
    y, x = torch.meshgrid(
        torch.arange(height, dtype=like.dtype, device=like.device),
        torch.arange(width, dtype=like.dtype, device=like.device),
        indexing="ij",
    )
    return erp_pixels_to_rays(torch.stack((x, y), dim=-1), shape_hw)


def _freeze_numpy(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


def _gnomonic_forward_plan(
    spec: GnomonicSpec, source_shape_hw: ShapeHW, image: Any
) -> _GnomonicForwardPlan:
    pixels = gnomonic_pixel_map(spec, source_shape_hw, like=_sampling_grid_like(image))
    if isinstance(pixels, np.ndarray):
        _freeze_numpy(pixels)
    return _GnomonicForwardPlan(pixels)


def _gnomonic_back_plan(
    spec: GnomonicSpec,
    output_shape_hw: ShapeHW,
    face_shape_hw: ShapeHW,
    image: Any,
) -> _GnomonicBackPlan:
    like = _geometry_like(image)
    rays = _erp_rays(output_shape_hw, like=like)
    forward, right, up = _gnomonic_basis(spec)
    if _is_torch(rays):
        torch = _torch_module()
        forward = torch.as_tensor(forward, dtype=rays.dtype, device=rays.device)
        right = torch.as_tensor(right, dtype=rays.dtype, device=rays.device)
        up = torch.as_tensor(up, dtype=rays.dtype, device=rays.device)
        denominator = rays @ forward
        safe_denominator = torch.where(
            denominator != 0, denominator, torch.ones_like(denominator)
        )
        plane_x = (rays @ right) / safe_denominator
        plane_y = -(rays @ up) / safe_denominator
        roll = math.radians(spec.roll_deg)
        x = math.cos(roll) * plane_x + math.sin(roll) * plane_y
        y = -math.sin(roll) * plane_x + math.cos(roll) * plane_y
        x_limit = math.tan(math.radians(spec.hfov_deg) / 2.0)
        y_limit = math.tan(math.radians(spec.vfov_deg) / 2.0)
        tolerance = 1e-12 if rays.dtype == torch.float64 else 0.0
        support = (
            (denominator > 0.0)
            & (torch.abs(x) <= x_limit + tolerance)
            & (torch.abs(y) <= y_limit + tolerance)
        )
    else:
        denominator = rays @ forward
        safe_denominator = np.where(denominator != 0, denominator, 1.0)
        plane_x = (rays @ right) / safe_denominator
        plane_y = -(rays @ up) / safe_denominator
        roll = np.deg2rad(spec.roll_deg)
        x = np.cos(roll) * plane_x + np.sin(roll) * plane_y
        y = -np.sin(roll) * plane_x + np.cos(roll) * plane_y
        x_limit = np.tan(np.deg2rad(spec.hfov_deg) / 2.0)
        y_limit = np.tan(np.deg2rad(spec.vfov_deg) / 2.0)
        support = (
            (denominator > 0.0)
            & (np.abs(x) <= x_limit + 1e-12)
            & (np.abs(y) <= y_limit + 1e-12)
        )
    face_height, face_width = face_shape_hw
    map_x = ((x / x_limit + 1.0) * 0.5) * face_width - 0.5
    map_y = ((y / y_limit + 1.0) * 0.5) * face_height - 0.5
    if isinstance(map_x, np.ndarray):
        _freeze_numpy(map_x)
        _freeze_numpy(map_y)
        _freeze_numpy(support)
    return _GnomonicBackPlan(map_x, map_y, support)


def _gnomonic_selective_face_plan(
    rays: Any,
    spec: GnomonicSpec,
    face_shape_hw: ShapeHW,
) -> _GnomonicSelectiveFacePlan:
    """Build one compact plan from a shared ERP ray lattice."""

    forward, right, up = _gnomonic_basis(spec)
    if _is_torch(rays):
        torch = _torch_module()
        forward = torch.as_tensor(forward, dtype=rays.dtype, device=rays.device)
        right = torch.as_tensor(right, dtype=rays.dtype, device=rays.device)
        up = torch.as_tensor(up, dtype=rays.dtype, device=rays.device)
        denominator = rays @ forward
        safe_denominator = torch.where(
            denominator != 0, denominator, torch.ones_like(denominator)
        )
        plane_x = (rays @ right) / safe_denominator
        plane_y = -(rays @ up) / safe_denominator
        roll = math.radians(spec.roll_deg)
        x = math.cos(roll) * plane_x + math.sin(roll) * plane_y
        y = -math.sin(roll) * plane_x + math.cos(roll) * plane_y
        x_limit = math.tan(math.radians(spec.hfov_deg) / 2.0)
        y_limit = math.tan(math.radians(spec.vfov_deg) / 2.0)
        tolerance = 1e-12 if rays.dtype == torch.float64 else 0.0
        support = (
            (denominator > 0.0)
            & (torch.abs(x) <= x_limit + tolerance)
            & (torch.abs(y) <= y_limit + tolerance)
        )
        indices = torch.nonzero(support, as_tuple=False).squeeze(1)
        face_height, face_width = face_shape_hw
        map_x = (((x[indices] / x_limit + 1.0) * 0.5) * face_width - 0.5).unsqueeze(1)
        map_y = (((y[indices] / y_limit + 1.0) * 0.5) * face_height - 0.5).unsqueeze(1)
        return _GnomonicSelectiveFacePlan(
            indices,
            map_x,
            map_y,
            denominator[indices],
        )

    denominator = rays @ forward
    safe_denominator = np.where(denominator != 0, denominator, 1.0)
    plane_x = (rays @ right) / safe_denominator
    plane_y = -(rays @ up) / safe_denominator
    roll = np.deg2rad(spec.roll_deg)
    x = np.cos(roll) * plane_x + np.sin(roll) * plane_y
    y = -np.sin(roll) * plane_x + np.cos(roll) * plane_y
    x_limit = np.tan(np.deg2rad(spec.hfov_deg) / 2.0)
    y_limit = np.tan(np.deg2rad(spec.vfov_deg) / 2.0)
    support = (
        (denominator > 0.0)
        & (np.abs(x) <= x_limit + 1e-12)
        & (np.abs(y) <= y_limit + 1e-12)
    )
    indices = np.flatnonzero(support).astype(np.int32, copy=False)
    face_height, face_width = face_shape_hw
    map_x = ((x[indices] / x_limit + 1.0) * 0.5) * face_width - 0.5
    map_y = ((y[indices] / y_limit + 1.0) * 0.5) * face_height - 0.5
    return _GnomonicSelectiveFacePlan(
        _freeze_numpy(indices),
        _freeze_numpy(map_x),
        _freeze_numpy(map_y),
        _freeze_numpy(denominator[indices]),
    )


def _gnomonic_batch_back_plan(
    specs: tuple[GnomonicSpec, ...] | list[GnomonicSpec],
    output_shape_hw: ShapeHW,
    face_shapes_hw: tuple[ShapeHW, ...] | list[ShapeHW],
    image: Any,
) -> _GnomonicBatchBackPlan:
    """Build a selective plan for N views while creating ERP rays only once."""

    output_shape_hw = _validate_shape(output_shape_hw, "output_shape_hw")
    if not specs:
        raise ValueError("specs must contain at least one gnomonic view")
    if len(specs) != len(face_shapes_hw):
        raise ValueError("specs and face_shapes_hw must have the same length")
    rays = _erp_rays(output_shape_hw, like=_geometry_like(image)).reshape(-1, 3)
    plans = tuple(
        _gnomonic_selective_face_plan(rays, spec, _validate_shape(face_shape))
        for spec, face_shape in zip(specs, face_shapes_hw, strict=True)
    )
    return _GnomonicBatchBackPlan(output_shape_hw, plans)


def _squeeze_sparse_sample(value: Any) -> Any:
    """Remove the artificial one-column dimension used by Torch grid_sample."""

    if _is_torch(value):
        return value.squeeze(-1)
    return value


def _gnomonic_batch_to_sparse(
    images: tuple[ArrayT, ...] | list[ArrayT],
    plan: _GnomonicBatchBackPlan,
    *,
    interpolation: Interpolation,
    invalid_policy: InvalidPolicy = "propagate",
    validity_masks: tuple[ArrayT, ...] | list[ArrayT] | None = None,
    min_valid_weight: float | None = None,
) -> tuple[_GnomonicSparseProjection, ...]:
    """Sample N gnomonic views only at their supported ERP destinations."""

    if len(images) != len(plan.faces):
        raise ValueError("images and batch plan must have the same number of views")
    if validity_masks is not None and len(validity_masks) != len(images):
        raise ValueError("validity_masks and images must have the same length")
    interpolation = _validate_interpolation(interpolation)
    results = []
    first_is_torch = _is_torch(images[0])
    for index, (image, face_plan) in enumerate(zip(images, plan.faces, strict=True)):
        _require_array(image, f"images[{index}]", image=True)
        if _is_torch(image) != first_is_torch:
            raise TypeError("all gnomonic views must use the same backend")
        sampled, sampled_validity, sampled_weight = _sample_by_policy(
            image,
            face_plan.map_x,
            face_plan.map_y,
            interpolation,
            wrap_x=False,
            invalid_policy=invalid_policy,
            validity_mask=None if validity_masks is None else validity_masks[index],
            min_valid_weight=min_valid_weight,
        )
        results.append(
            _GnomonicSparseProjection(
                face_plan.flat_indices,
                _squeeze_sparse_sample(sampled),
                face_plan.center_score,
                None
                if sampled_validity is None
                else _squeeze_sparse_sample(sampled_validity),
                None
                if sampled_weight is None
                else _squeeze_sparse_sample(sampled_weight),
            )
        )
    return tuple(results)


def _cubemap_forward_plan(
    face_shape_hw: ShapeHW, source_shape_hw: ShapeHW, image: Any
) -> _CubemapForwardPlan:
    like = _sampling_grid_like(image)
    pixels = []
    for face in CUBE_FACE_ORDER:
        face_pixels = rays_to_erp_pixels(
            _cube_face_rays(face, face_shape_hw, like=like), source_shape_hw
        ).pixels_xy
        if isinstance(face_pixels, np.ndarray):
            _freeze_numpy(face_pixels)
        pixels.append(face_pixels)
    return _CubemapForwardPlan(tuple(pixels))


def _cube_selection(rays: Any) -> Any:
    if _is_torch(rays):
        torch = _torch_module()
        forwards = torch.as_tensor(
            [CUBE_FACE_BASES[face][0] for face in CUBE_FACE_ORDER],
            dtype=rays.dtype,
            device=rays.device,
        )
        face_scores = rays @ forwards.transpose(0, 1)
        tie_atol = 8.0 * torch.finfo(face_scores.dtype).eps
        maximum_score = torch.max(face_scores, dim=-1, keepdim=True).values
        return torch.argmax(
            (face_scores >= maximum_score - tie_atol).to(torch.int8), dim=-1
        )
    forwards = np.stack([CUBE_FACE_BASES[face][0] for face in CUBE_FACE_ORDER])
    face_scores = rays @ forwards.T
    tie_atol = 8.0 * np.finfo(face_scores.dtype).eps
    maximum_score = np.max(face_scores, axis=-1, keepdims=True)
    return np.argmax(face_scores >= maximum_score - tie_atol, axis=-1)


def _face_pixel_map(rays: Any, face: str, face_shape_hw: ShapeHW) -> tuple[Any, Any]:
    forward, right, up = (np.asarray(vector) for vector in CUBE_FACE_BASES[face])
    if _is_torch(rays):
        torch = _torch_module()
        forward = torch.as_tensor(forward, dtype=rays.dtype, device=rays.device)
        right = torch.as_tensor(right, dtype=rays.dtype, device=rays.device)
        up = torch.as_tensor(up, dtype=rays.dtype, device=rays.device)
        denominator = rays @ forward
        safe = torch.where(denominator != 0, denominator, torch.ones_like(denominator))
    else:
        denominator = rays @ forward
        safe = np.where(denominator != 0, denominator, 1.0)
    x = (rays @ right) / safe
    y = -(rays @ up) / safe
    face_height, face_width = face_shape_hw
    return (
        ((x + 1.0) * 0.5) * face_width - 0.5,
        ((y + 1.0) * 0.5) * face_height - 0.5,
    )


def _cubemap_full_back_plan(
    output_shape_hw: ShapeHW, face_shape_hw: ShapeHW, image: Any
) -> _CubemapFullBackPlan:
    rays = _erp_rays(output_shape_hw, like=_geometry_like(image))
    selected = _cube_selection(rays)
    maps = tuple(_face_pixel_map(rays, face, face_shape_hw) for face in CUBE_FACE_ORDER)
    if isinstance(selected, np.ndarray):
        _freeze_numpy(selected)
        for map_x, map_y in maps:
            _freeze_numpy(map_x)
            _freeze_numpy(map_y)
    return _CubemapFullBackPlan(selected, maps)


def _cubemap_selective_back_plan(
    output_shape_hw: ShapeHW, face_shape_hw: ShapeHW, image: Any | None = None
) -> _CubemapSelectiveBackPlan:
    rays = _erp_rays(output_shape_hw, like=_geometry_like(image))
    selected = _cube_selection(rays)
    plans = []
    if _is_torch(rays):
        torch = _torch_module()
        flat_rays = rays.reshape(-1, 3)
        flat_selected = selected.reshape(-1)
        for index, face in enumerate(CUBE_FACE_ORDER):
            flat_indices = torch.nonzero(
                flat_selected == index, as_tuple=False
            ).squeeze(1)
            map_x, map_y = _face_pixel_map(flat_rays[flat_indices], face, face_shape_hw)
            plans.append(
                _CubemapSelectiveFacePlan(
                    flat_indices, map_x.unsqueeze(1), map_y.unsqueeze(1)
                )
            )
        return _CubemapSelectiveBackPlan(output_shape_hw, tuple(plans))
    flat_rays = rays.reshape(-1, 3)
    flat_selected = selected.ravel()
    for index, face in enumerate(CUBE_FACE_ORDER):
        flat_indices = np.flatnonzero(flat_selected == index)
        map_x, map_y = _face_pixel_map(flat_rays[flat_indices], face, face_shape_hw)
        plans.append(
            _CubemapSelectiveFacePlan(
                _freeze_numpy(flat_indices),
                _freeze_numpy(map_x),
                _freeze_numpy(map_y),
            )
        )
    return _CubemapSelectiveBackPlan(output_shape_hw, tuple(plans))


def _use_selective_cubemap_plan(image: Any) -> bool:
    return not _is_torch(image) or (
        image.device.type in _TORCH_SELECTIVE_CUBEMAP_DEVICE_TYPES
    )


def _as_torch_nchw(image: Any):
    if image.ndim == 2:
        return image[None, None], "HW"
    if image.ndim == 3:
        return image[None], "CHW"
    if image.ndim == 4:
        return image, "NCHW"
    raise ValueError("Torch images must use HW, CHW, or NCHW layout")


def _from_torch_nchw(image: Any, layout: str):
    if layout == "HW":
        return image[0, 0]
    if layout == "CHW":
        return image[0]
    return image


def _numpy_nearest(
    image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, *, wrap_x: bool
) -> np.ndarray:
    height, width = image.shape[:2]
    finite = np.isfinite(map_x) & np.isfinite(map_y)
    safe_x = np.where(finite, map_x, 0.0)
    safe_y = np.where(finite, map_y, 0.0)
    safe_x = np.mod(safe_x, width) if wrap_x else np.clip(safe_x, -0.5, width - 0.5)
    safe_y = np.clip(safe_y, -0.5, height - 0.5)
    x = np.floor(safe_x + 0.5).astype(np.int64)
    y = np.floor(safe_y + 0.5).astype(np.int64)
    if wrap_x:
        x %= width
    else:
        x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)
    sampled = image[y, x]
    if finite.all():
        return sampled
    expanded = finite if sampled.ndim == 2 else finite[..., None]
    invalid = np.nan if np.issubdtype(sampled.dtype, np.floating) else 0
    return np.where(expanded, sampled, invalid).astype(sampled.dtype, copy=False)


def _numpy_sample(
    image: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    interpolation: Interpolation,
    *,
    wrap_x: bool,
) -> np.ndarray:
    if image.ndim not in {2, 3}:
        raise ValueError("NumPy images must use HW or HWC layout")
    if interpolation == "nearest":
        return _numpy_nearest(image, map_x, map_y, wrap_x=wrap_x)
    if not np.issubdtype(image.dtype, np.floating):
        raise TypeError("bilinear interpolation requires floating-point NumPy data")
    height, width = image.shape[:2]
    x0_raw = np.floor(map_x).astype(np.int64)
    y0_raw = np.floor(map_y).astype(np.int64)
    x1_raw = x0_raw + 1
    y1_raw = y0_raw + 1
    wx = map_x - x0_raw
    wy = map_y - y0_raw
    if wrap_x:
        x0 = np.mod(x0_raw, width)
        x1 = np.mod(x1_raw, width)
    else:
        x0 = np.clip(x0_raw, 0, width - 1)
        x1 = np.clip(x1_raw, 0, width - 1)
    y0 = np.clip(y0_raw, 0, height - 1)
    y1 = np.clip(y1_raw, 0, height - 1)
    if image.ndim == 3:
        wx = wx[..., None]
        wy = wy[..., None]
    top = image[y0, x0] * (1.0 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1.0 - wx) + image[y1, x1] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(image.dtype, copy=False)


def _torch_nearest(image: Any, map_x: Any, map_y: Any, *, wrap_x: bool):
    torch = _torch_module()
    nchw, layout = _as_torch_nchw(image)
    height, width = nchw.shape[-2:]
    x = torch.floor(map_x + 0.5).to(torch.long)
    y = torch.floor(map_y + 0.5).to(torch.long).clamp(0, height - 1)
    x = torch.remainder(x, width) if wrap_x else x.clamp(0, width - 1)
    flat_indices = (
        (y * width + x).reshape(1, 1, -1).expand(nchw.shape[0], nchw.shape[1], -1)
    )
    sampled = torch.gather(
        nchw.reshape(nchw.shape[0], nchw.shape[1], -1), 2, flat_indices
    )
    sampled = sampled.reshape(nchw.shape[0], nchw.shape[1], *map_x.shape)
    return _from_torch_nchw(sampled, layout)


def _torch_sample(
    image: Any, map_x: Any, map_y: Any, interpolation: Interpolation, *, wrap_x: bool
):
    torch = _torch_module()
    if interpolation == "nearest":
        return _torch_nearest(image, map_x, map_y, wrap_x=wrap_x)
    if not image.dtype.is_floating_point:
        raise TypeError("bilinear interpolation requires a floating-point Torch tensor")
    nchw, layout = _as_torch_nchw(image)
    original_dtype = nchw.dtype
    working = nchw
    if wrap_x:
        working = torch.cat((working[..., -1:], working, working[..., :1]), dim=-1)
        map_x = map_x + 1.0
    input_height, input_width = working.shape[-2:]
    grid_x = (2.0 * (map_x + 0.5) / input_width) - 1.0
    grid_y = (2.0 * (map_y + 0.5) / input_height) - 1.0
    # Clamping to the outer pixel centres is equivalent to border padding and
    # also works on MPS, where grid_sample's native border mode is unavailable.
    grid_x = grid_x.clamp(-1.0 + 1.0 / input_width, 1.0 - 1.0 / input_width)
    grid_y = grid_y.clamp(-1.0 + 1.0 / input_height, 1.0 - 1.0 / input_height)
    grid = torch.stack((grid_x, grid_y), dim=-1)
    grid = grid.unsqueeze(0).expand(working.shape[0], -1, -1, -1)
    if working.device.type == "cpu" and working.dtype in {
        torch.float16,
        torch.bfloat16,
    }:
        working = working.float()
        grid = grid.float()
    else:
        grid = grid.to(dtype=working.dtype)
    sampled = torch.nn.functional.grid_sample(
        working,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).to(original_dtype)
    return _from_torch_nchw(sampled, layout)


def _sample(
    image: Any, map_x: Any, map_y: Any, interpolation: Interpolation, *, wrap_x: bool
):
    if _is_torch(image):
        torch = _torch_module()
        grid_dtype = (
            torch.float64
            if image.dtype == torch.float64 and image.device.type != "mps"
            else torch.float32
        )
        x = torch.as_tensor(map_x, device=image.device, dtype=grid_dtype)
        y = torch.as_tensor(map_y, device=image.device, dtype=grid_dtype)
        return _torch_sample(image, x, y, interpolation, wrap_x=wrap_x)
    return _numpy_sample(
        np.asarray(image),
        np.asarray(map_x),
        np.asarray(map_y),
        interpolation,
        wrap_x=wrap_x,
    )


def _validate_sampling_options(
    image: Any,
    interpolation: Interpolation,
    invalid_policy: str,
    validity_mask: Any | None,
    min_valid_weight: float | None,
) -> tuple[InvalidPolicy, Any | None, float | None]:
    policy = _validate_invalid_policy(invalid_policy)
    if policy == "propagate":
        if validity_mask is not None or min_valid_weight is not None:
            raise ValueError(
                "validity_mask and min_valid_weight are only valid with "
                "invalid_policy='renormalize'"
            )
        return policy, None, None
    if interpolation != "bilinear":
        raise ValueError("invalid_policy='renormalize' requires bilinear interpolation")
    floating = (
        image.dtype.is_floating_point
        if _is_torch(image)
        else np.issubdtype(image.dtype, np.floating)
    )
    if not floating:
        raise TypeError(
            "validity-normalized interpolation requires floating-point data"
        )
    if validity_mask is None:
        raise ValueError("validity_mask is required for invalid_policy='renormalize'")
    if min_valid_weight is None:
        raise ValueError(
            "min_valid_weight is required for invalid_policy='renormalize'"
        )
    if isinstance(min_valid_weight, bool) or not isinstance(min_valid_weight, Real):
        raise TypeError("min_valid_weight must be a finite real number")
    threshold = float(min_valid_weight)
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("min_valid_weight must be finite and in the interval (0, 1]")
    return policy, _validate_validity_mask(image, validity_mask), threshold


def _validate_validity_mask(image: Any, validity_mask: Any) -> Any:
    image_is_torch = _is_torch(image)
    if image_is_torch != _is_torch(validity_mask):
        raise TypeError("image and validity_mask must use the same backend")
    if image_is_torch:
        torch = _torch_module()
        if validity_mask.dtype != torch.bool:
            raise TypeError("validity_mask must have boolean dtype")
        if validity_mask.device != image.device:
            raise ValueError("validity_mask must use the same Torch device as image")
        expected_shape = (
            tuple(image.shape[-2:])
            if image.ndim in {2, 3}
            else (image.shape[0], image.shape[-2], image.shape[-1])
        )
    else:
        if not isinstance(validity_mask, np.ndarray):
            raise TypeError("image and validity_mask must use the same backend")
        if validity_mask.dtype != np.bool_:
            raise TypeError("validity_mask must have boolean dtype")
        expected_shape = tuple(image.shape[:2])
    if tuple(validity_mask.shape) != tuple(expected_shape):
        raise ValueError(
            f"validity_mask must have shape {tuple(expected_shape)}; "
            f"got {tuple(validity_mask.shape)}"
        )
    return validity_mask


def _effective_validity(image: Any, validity_mask: Any) -> Any:
    if _is_torch(image):
        torch = _torch_module()
        finite = torch.isfinite(image)
        if image.ndim == 3:
            finite = finite.all(dim=0)
        elif image.ndim == 4:
            finite = finite.all(dim=1)
        return validity_mask & finite
    finite = np.isfinite(image)
    if image.ndim == 3:
        finite = finite.all(axis=-1)
    return validity_mask & finite


def _expand_spatial(mask: Any, data: Any) -> Any:
    if _is_torch(data):
        if data.ndim == 3:
            return mask.unsqueeze(0)
        if data.ndim == 4:
            return mask.unsqueeze(1)
        return mask
    return mask[..., None] if data.ndim == mask.ndim + 1 else mask


def _sample_normalized(
    image: Any,
    validity_mask: Any,
    map_x: Any,
    map_y: Any,
    *,
    wrap_x: bool,
    min_valid_weight: float,
) -> tuple[Any, Any, Any]:
    effective = _effective_validity(image, validity_mask)
    if _is_torch(image):
        torch = _torch_module()
        sanitized = torch.where(
            _expand_spatial(effective, image), image, torch.zeros_like(image)
        )
        weights_source = effective.to(dtype=image.dtype)
        if image.ndim == 4:
            weights_source = weights_source.unsqueeze(1)
    else:
        sanitized = np.where(_expand_spatial(effective, image), image, 0)
        weights_source = effective.astype(image.dtype, copy=False)

    numerator = _sample(sanitized, map_x, map_y, "bilinear", wrap_x=wrap_x)
    valid_weight = _sample(weights_source, map_x, map_y, "bilinear", wrap_x=wrap_x)
    if _is_torch(image) and image.ndim == 4:
        valid_weight = valid_weight[:, 0]

    output_validity = valid_weight >= min_valid_weight
    expanded_weight = _expand_spatial(valid_weight, numerator)
    expanded_validity = _expand_spatial(output_validity, numerator)
    if _is_torch(image):
        torch = _torch_module()
        safe_weight = torch.where(
            expanded_validity, expanded_weight, torch.ones_like(expanded_weight)
        )
        normalized = numerator / safe_weight
        normalized = torch.where(
            expanded_validity,
            normalized,
            torch.full_like(normalized, float("nan")),
        )
    else:
        safe_weight = np.where(expanded_validity, expanded_weight, 1.0)
        normalized = numerator / safe_weight
        normalized = np.where(expanded_validity, normalized, np.nan).astype(
            image.dtype, copy=False
        )
    return normalized, output_validity, valid_weight


def _sample_by_policy(
    image: Any,
    map_x: Any,
    map_y: Any,
    interpolation: Interpolation,
    *,
    wrap_x: bool,
    invalid_policy: str,
    validity_mask: Any | None,
    min_valid_weight: float | None,
) -> tuple[Any, Any | None, Any | None]:
    policy, validity_mask, threshold = _validate_sampling_options(
        image,
        interpolation,
        invalid_policy,
        validity_mask,
        min_valid_weight,
    )
    if policy == "propagate":
        return _sample(image, map_x, map_y, interpolation, wrap_x=wrap_x), None, None
    assert threshold is not None
    return _sample_normalized(
        image,
        validity_mask,
        map_x,
        map_y,
        wrap_x=wrap_x,
        min_valid_weight=threshold,
    )


def _mask_data(data: Any, support: np.ndarray, fill_value: Any):
    if _is_torch(data):
        torch = _torch_module()
        mask = torch.as_tensor(support, device=data.device, dtype=torch.bool)
        expanded = mask
        if data.ndim == 3:
            expanded = mask.unsqueeze(0)
        elif data.ndim == 4:
            expanded = mask.unsqueeze(0).unsqueeze(0)
        fill = torch.as_tensor(fill_value, device=data.device, dtype=data.dtype)
        return torch.where(expanded, data, fill), mask
    mask = np.asarray(support, dtype=bool)
    expanded = mask if data.ndim == 2 else mask[..., None]
    return np.where(expanded, data, np.asarray(fill_value, dtype=data.dtype)), mask


def _mask_validity_outputs(
    validity: Any | None,
    valid_weight: Any | None,
    support: np.ndarray,
) -> tuple[Any | None, Any | None]:
    if validity is None:
        return None, None
    if _is_torch(validity):
        torch = _torch_module()
        mask = torch.as_tensor(support, device=validity.device, dtype=torch.bool)
        if validity.ndim == 3:
            mask = mask.unsqueeze(0)
        return validity & mask, torch.where(mask, valid_weight, 0.0)
    mask = np.asarray(support, dtype=bool)
    return validity & mask, np.where(mask, valid_weight, 0.0).astype(
        valid_weight.dtype, copy=False
    )


def equirectangular_to_gnomonic(
    image: ArrayT,
    spec: GnomonicSpec,
    *,
    interpolation: Interpolation = "bilinear",
    invalid_policy: InvalidPolicy = "propagate",
    validity_mask: ArrayT | None = None,
    min_valid_weight: float | None = None,
    return_source_pixels: bool = False,
) -> ProjectionResult[ArrayT]:
    """Sample an ERP image into a canonical rectangular gnomonic view."""

    return _equirectangular_to_gnomonic_with_plan(
        image,
        spec,
        interpolation=interpolation,
        invalid_policy=invalid_policy,
        validity_mask=validity_mask,
        min_valid_weight=min_valid_weight,
        return_source_pixels=return_source_pixels,
    )


def _equirectangular_to_gnomonic_with_plan(
    image: ArrayT,
    spec: GnomonicSpec,
    *,
    interpolation: Interpolation,
    invalid_policy: InvalidPolicy,
    validity_mask: ArrayT | None,
    min_valid_weight: float | None,
    return_source_pixels: bool,
    plan: _GnomonicForwardPlan | None = None,
) -> ProjectionResult[ArrayT]:

    _require_array(image, "image", image=True)
    interpolation = _validate_interpolation(interpolation)
    source_shape = image.shape[-2:] if _is_torch(image) else image.shape[:2]
    if plan is None:
        plan = _gnomonic_forward_plan(spec, source_shape, image)
    pixels = plan.pixels_xy
    data, output_validity, valid_weight = _sample_by_policy(
        image,
        pixels[..., 0],
        pixels[..., 1],
        interpolation,
        wrap_x=True,
        invalid_policy=invalid_policy,
        validity_mask=validity_mask,
        min_valid_weight=min_valid_weight,
    )
    if _is_torch(data):
        torch = _torch_module()
        mask = torch.ones(spec.output_shape_hw, dtype=torch.bool, device=data.device)
    else:
        mask = np.ones(spec.output_shape_hw, dtype=bool)
    source_pixels = None
    if return_source_pixels:
        source_pixels = pixels.clone() if _is_torch(pixels) else pixels.copy()
    return ProjectionResult(
        data,
        mask,
        output_validity,
        valid_weight,
        source_pixels,
    )


def gnomonic_to_equirectangular(
    image: ArrayT,
    spec: GnomonicSpec,
    output_shape_hw: tuple[int, int],
    *,
    interpolation: Interpolation = "bilinear",
    fill_value: Any | None = None,
    invalid_policy: InvalidPolicy = "propagate",
    validity_mask: ArrayT | None = None,
    min_valid_weight: float | None = None,
) -> ProjectionResult[ArrayT]:
    """Back-project a gnomonic view and return its exact rectangular support."""

    return _gnomonic_to_equirectangular_with_plan(
        image,
        spec,
        output_shape_hw,
        interpolation=interpolation,
        fill_value=fill_value,
        invalid_policy=invalid_policy,
        validity_mask=validity_mask,
        min_valid_weight=min_valid_weight,
    )


def _gnomonic_to_equirectangular_with_plan(
    image: ArrayT,
    spec: GnomonicSpec,
    output_shape_hw: tuple[int, int],
    *,
    interpolation: Interpolation,
    fill_value: Any | None,
    invalid_policy: InvalidPolicy,
    validity_mask: ArrayT | None,
    min_valid_weight: float | None,
    plan: _GnomonicBackPlan | None = None,
) -> ProjectionResult[ArrayT]:

    _require_array(image, "image", image=True)
    interpolation = _validate_interpolation(interpolation)
    output_shape_hw = _validate_shape(output_shape_hw, "output_shape_hw")
    face_shape = image.shape[-2:] if _is_torch(image) else image.shape[:2]
    if plan is None:
        plan = _gnomonic_back_plan(spec, output_shape_hw, face_shape, image)
    data, output_validity, valid_weight = _sample_by_policy(
        image,
        plan.map_x,
        plan.map_y,
        interpolation,
        wrap_x=False,
        invalid_policy=invalid_policy,
        validity_mask=validity_mask,
        min_valid_weight=min_valid_weight,
    )
    if fill_value is None:
        dtype = image.dtype
        if (_is_torch(image) and dtype.is_floating_point) or (
            not _is_torch(image) and np.issubdtype(dtype, np.floating)
        ):
            fill_value = float("nan")
        else:
            fill_value = 0
    data, mask = _mask_data(data, plan.support, fill_value)
    output_validity, valid_weight = _mask_validity_outputs(
        output_validity, valid_weight, plan.support
    )
    return ProjectionResult(data, mask, output_validity, valid_weight)


def _cube_face_rays(
    face: str, face_shape_hw: tuple[int, int], *, like: Any | None = None
) -> Any:
    if face not in CUBE_FACE_BASES:
        raise KeyError(f"Unknown cubemap face: {face!r}")
    if like is not None and _is_torch(like):
        torch = _torch_module()
        height, width = _validate_shape(face_shape_hw)
        x_values = (
            (torch.arange(width, dtype=like.dtype, device=like.device) + 0.5) / width
        ) * 2.0 - 1.0
        y_values = (
            (torch.arange(height, dtype=like.dtype, device=like.device) + 0.5) / height
        ) * 2.0 - 1.0
        y, x = torch.meshgrid(y_values, x_values, indexing="ij")
        forward, right, up = (
            torch.as_tensor(vector, dtype=like.dtype, device=like.device)
            for vector in CUBE_FACE_BASES[face]
        )
        rays = forward + x[..., None] * right - y[..., None] * up
        return rays / torch.linalg.vector_norm(rays, dim=-1, keepdim=True)
    x, y = _numpy_grid(face_shape_hw, 1.0, 1.0)
    forward, right, up = (np.asarray(vector) for vector in CUBE_FACE_BASES[face])
    rays = forward + x[..., None] * right - y[..., None] * up
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def equirectangular_to_cubemap(
    image: ArrayT,
    face_shape_hw: int | tuple[int, int],
    *,
    interpolation: Interpolation = "bilinear",
    invalid_policy: InvalidPolicy = "propagate",
    validity_mask: ArrayT | None = None,
    min_valid_weight: float | None = None,
) -> dict[str, ProjectionResult[ArrayT]]:
    """Convert ERP data into the six canonical cubemap faces."""

    return _equirectangular_to_cubemap_with_plan(
        image,
        face_shape_hw,
        interpolation=interpolation,
        invalid_policy=invalid_policy,
        validity_mask=validity_mask,
        min_valid_weight=min_valid_weight,
    )


def _equirectangular_to_cubemap_with_plan(
    image: ArrayT,
    face_shape_hw: int | tuple[int, int],
    *,
    interpolation: Interpolation,
    invalid_policy: InvalidPolicy,
    validity_mask: ArrayT | None,
    min_valid_weight: float | None,
    plan: _CubemapForwardPlan | None = None,
) -> dict[str, ProjectionResult[ArrayT]]:

    _require_array(image, "image", image=True)
    interpolation = _validate_interpolation(interpolation)
    shape = (
        (face_shape_hw, face_shape_hw)
        if isinstance(face_shape_hw, int)
        else face_shape_hw
    )
    shape = _validate_shape(shape)
    source_shape = image.shape[-2:] if _is_torch(image) else image.shape[:2]
    if plan is None:
        plan = _cubemap_forward_plan(shape, source_shape, image)
    result: dict[str, ProjectionResult[ArrayT]] = {}
    for face, pixels in zip(CUBE_FACE_ORDER, plan.pixels_xy, strict=True):
        data, output_validity, valid_weight = _sample_by_policy(
            image,
            pixels[..., 0],
            pixels[..., 1],
            interpolation,
            wrap_x=True,
            invalid_policy=invalid_policy,
            validity_mask=validity_mask,
            min_valid_weight=min_valid_weight,
        )
        if _is_torch(data):
            torch = _torch_module()
            mask = torch.ones(shape, dtype=torch.bool, device=data.device)
        else:
            mask = np.ones(shape, dtype=bool)
        result[face] = ProjectionResult(data, mask, output_validity, valid_weight)
    return result


def cubemap_to_equirectangular(
    faces: Mapping[str, ArrayT],
    output_shape_hw: tuple[int, int],
    *,
    interpolation: Interpolation = "bilinear",
    invalid_policy: InvalidPolicy = "propagate",
    validity_masks: Mapping[str, ArrayT] | None = None,
    min_valid_weight: float | None = None,
) -> ProjectionResult[ArrayT]:
    """Convert six canonical cubemap faces into an ERP image."""

    return _cubemap_to_equirectangular_with_plan(
        faces,
        output_shape_hw,
        interpolation=interpolation,
        invalid_policy=invalid_policy,
        validity_masks=validity_masks,
        min_valid_weight=min_valid_weight,
    )


def _validate_cubemap_inputs(
    faces: Mapping[str, ArrayT],
    output_shape_hw: tuple[int, int],
    interpolation: Interpolation,
    validity_masks: Mapping[str, ArrayT] | None,
) -> tuple[Interpolation, ShapeHW, ArrayT, bool, ShapeHW]:

    interpolation = _validate_interpolation(interpolation)
    output_shape_hw = _validate_shape(output_shape_hw, "output_shape_hw")
    missing = [face for face in CUBE_FACE_ORDER if face not in faces]
    if missing:
        raise ValueError(f"Missing cubemap faces: {', '.join(missing)}")
    if validity_masks is not None:
        missing_masks = [face for face in CUBE_FACE_ORDER if face not in validity_masks]
        if missing_masks:
            raise ValueError(
                f"Missing cubemap validity masks: {', '.join(missing_masks)}"
            )
    first = faces[CUBE_FACE_ORDER[0]]
    _require_array(first, f"faces[{CUBE_FACE_ORDER[0]}]", image=True)
    torch_backend = _is_torch(first)
    expected_shape = first.shape[-2:] if torch_backend else first.shape[:2]
    for face in CUBE_FACE_ORDER:
        value = faces[face]
        _require_array(value, f"faces[{face}]", image=True)
        if _is_torch(value) != torch_backend:
            raise TypeError("All cubemap faces must use the same backend")
        shape = value.shape[-2:] if torch_backend else value.shape[:2]
        if tuple(shape) != tuple(expected_shape):
            raise ValueError("All cubemap faces must have the same spatial shape")
    return interpolation, output_shape_hw, first, torch_backend, expected_shape


def _cubemap_to_equirectangular_selective(
    faces: Mapping[str, ArrayT],
    plan: _CubemapSelectiveBackPlan,
    *,
    interpolation: Interpolation,
    invalid_policy: InvalidPolicy,
    validity_masks: Mapping[str, ArrayT] | None,
    min_valid_weight: float | None,
) -> ProjectionResult[ArrayT]:
    if _is_torch(faces[CUBE_FACE_ORDER[0]]):
        return _cubemap_to_equirectangular_selective_torch(
            faces,
            plan,
            interpolation=interpolation,
            invalid_policy=invalid_policy,
            validity_masks=validity_masks,
            min_valid_weight=min_valid_weight,
        )
    # Keep importing ``panorai.geometry`` lightweight; the compiled extension
    # is resolved only when NumPy cubemap back-projection is actually used.
    from . import _native as _native_geometry

    ordered_faces = tuple(np.asarray(faces[face]) for face in CUBE_FACE_ORDER)
    if (
        interpolation == "bilinear"
        and invalid_policy == "propagate"
        and validity_masks is None
        and min_valid_weight is None
        and _native_geometry.supports_native_cubemap(ordered_faces)
    ):
        native_plans = tuple(
            (face_plan.flat_indices, face_plan.map_x, face_plan.map_y)
            for face_plan in plan.faces
        )
        output = _native_geometry.native_cubemap_to_equirectangular(
            ordered_faces, native_plans, plan.output_shape_hw
        )
        support = np.ones(plan.output_shape_hw, dtype=bool)
        return ProjectionResult(output, support, None, None)
    first = np.asarray(faces[CUBE_FACE_ORDER[0]])
    pixel_count = plan.output_shape_hw[0] * plan.output_shape_hw[1]
    trailing_shape = first.shape[2:]
    output = np.empty((pixel_count, *trailing_shape), dtype=first.dtype)
    output_validity = None
    output_valid_weight = None
    for face, face_plan in zip(CUBE_FACE_ORDER, plan.faces, strict=True):
        sampled, sampled_validity, sampled_valid_weight = _sample_by_policy(
            faces[face],
            face_plan.map_x,
            face_plan.map_y,
            interpolation,
            wrap_x=False,
            invalid_policy=invalid_policy,
            validity_mask=None if validity_masks is None else validity_masks[face],
            min_valid_weight=min_valid_weight,
        )
        output[face_plan.flat_indices] = sampled
        if sampled_validity is not None:
            if output_validity is None:
                output_validity = np.empty(pixel_count, dtype=bool)
                output_valid_weight = np.empty(
                    pixel_count, dtype=sampled_valid_weight.dtype
                )
            output_validity[face_plan.flat_indices] = sampled_validity
            output_valid_weight[face_plan.flat_indices] = sampled_valid_weight
    output = output.reshape((*plan.output_shape_hw, *trailing_shape))
    if output_validity is not None:
        output_validity = output_validity.reshape(plan.output_shape_hw)
        output_valid_weight = output_valid_weight.reshape(plan.output_shape_hw)
    support = np.ones(plan.output_shape_hw, dtype=bool)
    return ProjectionResult(output, support, output_validity, output_valid_weight)


def _cubemap_to_equirectangular_selective_torch(
    faces: Mapping[str, ArrayT],
    plan: _CubemapSelectiveBackPlan,
    *,
    interpolation: Interpolation,
    invalid_policy: InvalidPolicy,
    validity_masks: Mapping[str, ArrayT] | None,
    min_valid_weight: float | None,
) -> ProjectionResult[ArrayT]:
    torch = _torch_module()
    first = faces[CUBE_FACE_ORDER[0]]
    first_nchw, layout = _as_torch_nchw(first)
    pixel_count = plan.output_shape_hw[0] * plan.output_shape_hw[1]
    output = torch.empty(
        (first_nchw.shape[0], first_nchw.shape[1], pixel_count),
        dtype=first.dtype,
        device=first.device,
    )
    output_validity = None
    output_valid_weight = None
    for face, face_plan in zip(CUBE_FACE_ORDER, plan.faces, strict=True):
        sampled, sampled_validity, sampled_valid_weight = _sample_by_policy(
            faces[face],
            face_plan.map_x,
            face_plan.map_y,
            interpolation,
            wrap_x=False,
            invalid_policy=invalid_policy,
            validity_mask=None if validity_masks is None else validity_masks[face],
            min_valid_weight=min_valid_weight,
        )
        sampled_nchw, _ = _as_torch_nchw(sampled)
        output[..., face_plan.flat_indices] = sampled_nchw[..., 0]
        if sampled_validity is not None:
            metric = (
                sampled_validity[..., 0]
                if layout == "NCHW"
                else sampled_validity.reshape(1, -1)
            )
            weight = (
                sampled_valid_weight[..., 0]
                if layout == "NCHW"
                else sampled_valid_weight.reshape(1, -1)
            )
            if output_validity is None:
                output_validity = torch.empty(
                    (first_nchw.shape[0], pixel_count),
                    dtype=torch.bool,
                    device=first.device,
                )
                output_valid_weight = torch.empty(
                    (first_nchw.shape[0], pixel_count),
                    dtype=weight.dtype,
                    device=first.device,
                )
            output_validity[..., face_plan.flat_indices] = metric
            output_valid_weight[..., face_plan.flat_indices] = weight
    output = _from_torch_nchw(
        output.reshape(
            first_nchw.shape[0],
            first_nchw.shape[1],
            *plan.output_shape_hw,
        ),
        layout,
    )
    if output_validity is not None:
        output_validity = output_validity.reshape(
            first_nchw.shape[0], *plan.output_shape_hw
        )
        output_valid_weight = output_valid_weight.reshape(
            first_nchw.shape[0], *plan.output_shape_hw
        )
        if layout != "NCHW":
            output_validity = output_validity[0]
            output_valid_weight = output_valid_weight[0]
    support = torch.ones(plan.output_shape_hw, dtype=torch.bool, device=first.device)
    return ProjectionResult(output, support, output_validity, output_valid_weight)


def _cubemap_to_equirectangular_full(
    faces: Mapping[str, ArrayT],
    output_shape_hw: ShapeHW,
    plan: _CubemapFullBackPlan,
    *,
    interpolation: Interpolation,
    invalid_policy: InvalidPolicy,
    validity_masks: Mapping[str, ArrayT] | None,
    min_valid_weight: float | None,
) -> ProjectionResult[ArrayT]:
    first = faces[CUBE_FACE_ORDER[0]]
    torch_backend = _is_torch(first)

    output = None
    output_validity = None
    output_valid_weight = None
    for index, (face, maps) in enumerate(
        zip(CUBE_FACE_ORDER, plan.maps_xy, strict=True)
    ):
        map_x, map_y = maps
        sampled, sampled_validity, sampled_valid_weight = _sample_by_policy(
            faces[face],
            map_x,
            map_y,
            interpolation,
            wrap_x=False,
            invalid_policy=invalid_policy,
            validity_mask=None if validity_masks is None else validity_masks[face],
            min_valid_weight=min_valid_weight,
        )
        face_mask = plan.selected == index
        if output is None:
            if torch_backend:
                torch = _torch_module()
                output = torch.zeros_like(sampled)
            else:
                output = np.zeros_like(sampled)
            if sampled_validity is not None:
                if torch_backend:
                    output_validity = torch.zeros_like(sampled_validity)
                    output_valid_weight = torch.zeros_like(sampled_valid_weight)
                else:
                    output_validity = np.zeros_like(sampled_validity)
                    output_valid_weight = np.zeros_like(sampled_valid_weight)
        if torch_backend:
            torch = _torch_module()
            mask = torch.as_tensor(face_mask, device=sampled.device)
            if sampled.ndim == 3:
                mask = mask.unsqueeze(0)
            elif sampled.ndim == 4:
                mask = mask.unsqueeze(0).unsqueeze(0)
            output = torch.where(mask, sampled, output)
            if sampled_validity is not None:
                metric_mask = torch.as_tensor(face_mask, device=sampled.device)
                if sampled_validity.ndim == 3:
                    metric_mask = metric_mask.unsqueeze(0)
                output_validity = torch.where(
                    metric_mask, sampled_validity, output_validity
                )
                output_valid_weight = torch.where(
                    metric_mask, sampled_valid_weight, output_valid_weight
                )
        else:
            mask = face_mask if sampled.ndim == 2 else face_mask[..., None]
            output = np.where(mask, sampled, output)
            if sampled_validity is not None:
                output_validity = np.where(face_mask, sampled_validity, output_validity)
                output_valid_weight = np.where(
                    face_mask, sampled_valid_weight, output_valid_weight
                )
    if torch_backend:
        torch = _torch_module()
        support = torch.ones(output_shape_hw, dtype=torch.bool, device=first.device)
    else:
        support = np.ones(output_shape_hw, dtype=bool)
    return ProjectionResult(output, support, output_validity, output_valid_weight)


def _cubemap_to_equirectangular_with_plan(
    faces: Mapping[str, ArrayT],
    output_shape_hw: tuple[int, int],
    *,
    interpolation: Interpolation,
    invalid_policy: InvalidPolicy,
    validity_masks: Mapping[str, ArrayT] | None,
    min_valid_weight: float | None,
    plan: _CubemapSelectiveBackPlan | _CubemapFullBackPlan | None = None,
) -> ProjectionResult[ArrayT]:
    (
        interpolation,
        output_shape_hw,
        first,
        torch_backend,
        expected_shape,
    ) = _validate_cubemap_inputs(faces, output_shape_hw, interpolation, validity_masks)
    if plan is None:
        plan = (
            _cubemap_selective_back_plan(output_shape_hw, expected_shape, first)
            if _use_selective_cubemap_plan(first)
            else _cubemap_full_back_plan(output_shape_hw, expected_shape, first)
        )
    if isinstance(plan, _CubemapSelectiveBackPlan):
        return _cubemap_to_equirectangular_selective(
            faces,
            plan,
            interpolation=interpolation,
            invalid_policy=invalid_policy,
            validity_masks=validity_masks,
            min_valid_weight=min_valid_weight,
        )
    return _cubemap_to_equirectangular_full(
        faces,
        output_shape_hw,
        plan,
        interpolation=interpolation,
        invalid_policy=invalid_policy,
        validity_masks=validity_masks,
        min_valid_weight=min_valid_weight,
    )
