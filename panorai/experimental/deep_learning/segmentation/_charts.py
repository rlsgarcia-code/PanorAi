"""Canonical gnomonic chart operations for streamed segmentation."""

from __future__ import annotations

import math

import numpy as np

from panorai.geometry import (
    GnomonicSpec,
    equirectangular_to_gnomonic,
    gnomonic_pixels_to_rays,
    gnomonic_to_equirectangular,
    rays_to_gnomonic_pixels,
)


def extract_rgb_face(rgb: np.ndarray, spec: GnomonicSpec) -> np.ndarray:
    values = np.asarray(rgb)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("rgb must have shape (H,W,3)")
    result = equirectangular_to_gnomonic(
        values.astype(np.float32), spec, interpolation="bilinear"
    )
    return np.clip(result.data, 0, 255).astype(np.uint8)


def extract_mask_face(mask: np.ndarray, spec: GnomonicSpec) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("mask must have shape (H,W)")
    return equirectangular_to_gnomonic(
        values.astype(np.uint8), spec, interpolation="nearest"
    ).data.astype(bool)


def backproject_face_masks(
    masks: np.ndarray,
    spec: GnomonicSpec,
    output_shape_hw: tuple[int, int],
    *,
    support: np.ndarray | None = None,
) -> np.ndarray:
    """Back-project exactly three masks using nearest sampling."""

    values = np.asarray(masks, dtype=bool)
    if values.ndim != 3 or values.shape[0] != 3:
        raise ValueError("masks must have shape (3,H,W)")
    output = []
    for mask in values:
        projected = gnomonic_to_equirectangular(
            mask.astype(np.uint8),
            spec,
            output_shape_hw,
            interpolation="nearest",
            fill_value=0,
        )
        result = projected.data.astype(bool) & projected.support_mask
        if support is not None:
            valid = np.asarray(support, dtype=bool)
            if valid.shape != output_shape_hw:
                raise ValueError("support must match output_shape_hw")
            result &= valid
        output.append(result)
    return np.stack(output)


def backproject_face_mask_sets(
    mask_sets: np.ndarray,
    spec: GnomonicSpec,
    output_shape_hw: tuple[int, int],
    *,
    support: np.ndarray | None = None,
) -> np.ndarray:
    """Back-project N three-mask sets with one geometry plan.

    Channels are sampled together so one SAM prompt batch pays for one
    gnomonic-to-ERP plan rather than three plans per accepted prompt.
    """

    values = np.asarray(mask_sets, dtype=bool)
    if values.ndim != 4 or values.shape[1] != 3:
        raise ValueError("mask_sets must have shape (N,3,H,W)")
    if values.shape[0] == 0:
        return np.empty((0, 3, *output_shape_hw), dtype=bool)
    count, _, face_height, face_width = values.shape
    channels = values.transpose(2, 3, 0, 1).reshape(face_height, face_width, count * 3)
    projected = gnomonic_to_equirectangular(
        channels.astype(np.uint8),
        spec,
        output_shape_hw,
        interpolation="nearest",
        fill_value=0,
    )
    output = projected.data.reshape(*output_shape_hw, count, 3).transpose(2, 3, 0, 1)
    output = output.astype(bool) & projected.support_mask[None, None]
    if support is not None:
        valid = np.asarray(support, dtype=bool)
        if valid.shape != output_shape_hw:
            raise ValueError("support must match output_shape_hw")
        output &= valid[None, None]
    return output


def ray_to_longitude_latitude(ray: np.ndarray) -> tuple[float, float]:
    direction = np.asarray(ray, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    return (
        math.degrees(math.atan2(direction[0], direction[2])),
        math.degrees(math.asin(direction[1])),
    )


def face_point_direction(
    spec: GnomonicSpec, point_xy: tuple[float, float]
) -> tuple[float, float]:
    ray = gnomonic_pixels_to_rays(np.asarray(point_xy, dtype=np.float64), spec).rays_xyz
    return ray_to_longitude_latitude(ray)


def advance_chart(
    spec: GnomonicSpec,
    frontier_xy: tuple[float, float],
    *,
    step_fraction: float,
) -> tuple[GnomonicSpec, tuple[float, float]]:
    """Move beyond a frontier while retaining that frontier in the next view."""

    frontier_ray = gnomonic_pixels_to_rays(
        np.asarray(frontier_xy, dtype=np.float64), spec
    ).rays_xyz
    longitude = math.radians(spec.center_lon_deg)
    latitude = math.radians(spec.center_lat_deg)
    center_ray = np.asarray(
        [
            math.cos(latitude) * math.sin(longitude),
            math.sin(latitude),
            math.cos(latitude) * math.cos(longitude),
        ]
    )
    cosine = float(np.clip(np.dot(center_ray, frontier_ray), -1.0, 1.0))
    angle = math.acos(cosine)
    if angle <= 1e-8:
        raise ValueError("frontier and chart center must differ")
    tangent = (frontier_ray - cosine * center_ray) / math.sin(angle)
    step = math.radians(step_fraction * min(spec.hfov_deg, spec.vfov_deg))
    next_ray = math.cos(step) * center_ray + math.sin(step) * tangent
    next_lon, next_lat = ray_to_longitude_latitude(next_ray)
    next_spec = GnomonicSpec(
        center_lon_deg=next_lon,
        center_lat_deg=next_lat,
        hfov_deg=spec.hfov_deg,
        vfov_deg=spec.vfov_deg,
        output_shape_hw=spec.output_shape_hw,
    )
    projected = rays_to_gnomonic_pixels(frontier_ray, next_spec)
    if not bool(projected.valid):
        raise RuntimeError("frontier is outside the advanced chart")
    return next_spec, tuple(float(value) for value in projected.pixels_xy)
