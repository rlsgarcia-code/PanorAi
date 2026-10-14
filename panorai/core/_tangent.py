"""Application-independent tangent-vector operations on the unit sphere."""

from __future__ import annotations

import numpy as np


def tangent_offsets_to_rays(
    bearings_xyz: np.ndarray,
    east_offset_rad: float | np.ndarray,
    north_offset_rad: float | np.ndarray,
) -> np.ndarray:
    """Move unit bearings by local east/north exponential-map offsets.

    ``bearings_xyz`` may contain any leading dimensions ending in three.
    Offsets broadcast against those leading dimensions and any added sampling
    dimensions. The result follows PanorAi's ``+X,+Y,+Z`` panorama frame.
    """

    bearings = np.asarray(bearings_xyz, dtype=np.float64)
    if bearings.ndim < 1 or bearings.shape[-1] != 3:
        raise ValueError("bearings_xyz must end with three Cartesian components")
    if not np.isfinite(bearings).all():
        raise ValueError("bearings_xyz must contain finite values")
    norms = np.linalg.norm(bearings, axis=-1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("bearings_xyz must be non-zero")
    centers = bearings / norms
    longitude = np.arctan2(centers[..., 0], centers[..., 2])
    east = np.stack(
        (np.cos(longitude), np.zeros_like(longitude), -np.sin(longitude)), axis=-1
    )
    north = np.cross(centers, east)
    north /= np.linalg.norm(north, axis=-1, keepdims=True)
    east_offset, north_offset = np.broadcast_arrays(
        np.asarray(east_offset_rad, dtype=np.float64),
        np.asarray(north_offset_rad, dtype=np.float64),
    )
    bearing_shape = centers.shape[:-1]
    offset_shape = east_offset.shape
    if not offset_shape:
        target_shape = bearing_shape
    elif len(offset_shape) >= len(bearing_shape) and all(
        bearing in {1, offset}
        for bearing, offset in zip(bearing_shape, offset_shape, strict=False)
    ):
        # A point batch may carry additional per-point sample dimensions, for
        # example bearings (N, 3) with offsets (N, Kh, Kw).  NumPy's trailing
        # dimension broadcasting cannot express that prefix relationship.
        target_shape = offset_shape
    else:
        try:
            target_shape = np.broadcast_shapes(bearing_shape, offset_shape)
        except ValueError as error:
            raise ValueError(
                "offsets must broadcast with the bearing dimensions or extend "
                "them with per-bearing sampling dimensions"
            ) from error
    sample_dimensions = len(target_shape) - len(bearing_shape)
    centers = np.broadcast_to(
        centers.reshape(*bearing_shape, *(1,) * sample_dimensions, 3),
        (*target_shape, 3),
    )
    east = np.broadcast_to(
        east.reshape(*bearing_shape, *(1,) * sample_dimensions, 3),
        (*target_shape, 3),
    )
    north = np.broadcast_to(
        north.reshape(*bearing_shape, *(1,) * sample_dimensions, 3),
        (*target_shape, 3),
    )
    east_offset = np.broadcast_to(east_offset, target_shape)
    north_offset = np.broadcast_to(north_offset, target_shape)
    radius = np.hypot(east_offset, north_offset)
    sinc = np.ones_like(radius, dtype=np.float64)
    np.divide(np.sin(radius), radius, out=sinc, where=radius != 0.0)
    tangent = east_offset[..., None] * east + north_offset[..., None] * north
    rays = np.cos(radius)[..., None] * centers + sinc[..., None] * tangent
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


__all__ = ["tangent_offsets_to_rays"]
