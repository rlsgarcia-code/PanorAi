"""Canonical camera/world transforms for radial-range observations.

The pose convention is ``x_camera = R_world_to_camera @ (X_world - C_world)``.
Array inputs use row-vector storage in their final dimension; the equations
below are the exact row-vector equivalents of that column-vector contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _is_torch(value: Any) -> bool:
    module = type(value).__module__
    return module == "torch" or module.startswith("torch.")


def _backend(reference: Any):
    if _is_torch(reference):
        import torch

        return torch
    return np


def _array(value: Any, *, like: Any):
    if _is_torch(like):
        import torch

        return torch.as_tensor(value, dtype=like.dtype, device=like.device)
    return np.asarray(value, dtype=np.asarray(like).dtype)


def _validate_last_dimension(value: Any, name: str, size: int) -> None:
    if getattr(value, "ndim", 0) < 1 or value.shape[-1] != size:
        raise ValueError(f"{name} must have final dimension {size}")


def camera_rays_to_world_points(
    bearings_camera: Any,
    radial_range: Any,
    rotation_world_to_camera: Any,
    center_world: Any,
) -> Any:
    """Back-project camera bearings and radial range into the world frame.

    NumPy arrays and Torch tensors are supported. Torch inputs preserve dtype,
    device, and autograd. ``radial_range`` must be broadcast-compatible with
    ``bearings_camera[..., 0]``. Validity is deliberately not inferred from
    numeric values; callers must apply their explicit validity mask first.
    """

    _validate_last_dimension(bearings_camera, "bearings_camera", 3)
    rotation = _array(rotation_world_to_camera, like=bearings_camera)
    center = _array(center_world, like=bearings_camera)
    if rotation.shape != (3, 3):
        raise ValueError("rotation_world_to_camera must have shape (3, 3)")
    if center.shape != (3,):
        raise ValueError("center_world must have shape (3,)")
    distance = _array(radial_range, like=bearings_camera)
    try:
        camera_points = bearings_camera * distance[..., None]
    except (IndexError, RuntimeError, ValueError) as exc:
        raise ValueError(
            "radial_range must be broadcast-compatible with the bearing shape"
        ) from exc
    return camera_points @ rotation + center


def world_points_to_camera_rays(
    points_world: Any,
    rotation_world_to_camera: Any,
    center_world: Any,
) -> tuple[Any, Any]:
    """Project world points to unit camera bearings and radial ranges.

    Points at the camera centre have no defined bearing and are rejected.
    The returned radial range uses the same units as ``points_world`` and
    ``center_world``.
    """

    _validate_last_dimension(points_world, "points_world", 3)
    rotation = _array(rotation_world_to_camera, like=points_world)
    center = _array(center_world, like=points_world)
    if rotation.shape != (3, 3):
        raise ValueError("rotation_world_to_camera must have shape (3, 3)")
    if center.shape != (3,):
        raise ValueError("center_world must have shape (3,)")
    camera_points = (points_world - center) @ rotation.T
    backend = _backend(points_world)
    radial_range = (
        backend.linalg.vector_norm(camera_points, dim=-1)
        if _is_torch(points_world)
        else backend.linalg.norm(camera_points, axis=-1)
    )
    if _is_torch(radial_range):
        if bool((radial_range <= 0).any().item()):
            raise ValueError("points_world cannot contain the camera centre")
    elif np.any(radial_range <= 0):
        raise ValueError("points_world cannot contain the camera centre")
    return camera_points / radial_range[..., None], radial_range


def relative_camera_pose(
    rotation_world_to_reference: Any,
    center_reference_world: Any,
    rotation_world_to_source: Any,
    center_source_world: Any,
) -> tuple[Any, Any]:
    """Return the source-from-reference rigid transform.

    The result satisfies ``x_source = R @ x_reference + t`` and is the pose
    convention consumed by PanorAi's registered spherical stereo engines.
    Inputs must all use one backend; the reference rotation selects the output
    dtype and device.
    """

    reference_rotation = rotation_world_to_reference
    _validate_last_dimension(reference_rotation, "rotation_world_to_reference", 3)
    if reference_rotation.shape != (3, 3):
        raise ValueError("rotation_world_to_reference must have shape (3, 3)")
    source_rotation = _array(rotation_world_to_source, like=reference_rotation)
    reference_center = _array(center_reference_world, like=reference_rotation)
    source_center = _array(center_source_world, like=reference_rotation)
    if source_rotation.shape != (3, 3):
        raise ValueError("rotation_world_to_source must have shape (3, 3)")
    if reference_center.shape != (3,) or source_center.shape != (3,):
        raise ValueError("camera centres must have shape (3,)")
    rotation_source_from_reference = source_rotation @ reference_rotation.T
    translation_source_from_reference = source_rotation @ (
        reference_center - source_center
    )
    return rotation_source_from_reference, translation_source_from_reference


__all__ = [
    "camera_rays_to_world_points",
    "relative_camera_pose",
    "world_points_to_camera_rays",
]
