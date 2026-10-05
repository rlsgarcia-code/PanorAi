"""Private spherical sampling primitives for equirectangular images."""

from __future__ import annotations

from collections.abc import Iterator
import math

import numpy as np


def validate_image(image: np.ndarray, *, grayscale: bool = False) -> np.ndarray:
    """Validate a NumPy ``HW`` or ``HWC`` ERP without copying it."""

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    expected = {2} if grayscale else {2, 3}
    if image.ndim not in expected:
        layout = "HW" if grayscale else "HW or HWC"
        raise ValueError(f"image must use {layout} layout")
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError("image height and width must both be at least 2")
    if image.ndim == 3 and image.shape[2] < 1:
        raise ValueError("image must contain at least one channel")
    if not (
        np.issubdtype(image.dtype, np.integer)
        or np.issubdtype(image.dtype, np.floating)
    ):
        raise TypeError("image must use a real integer or floating dtype")
    return image


def validate_step(angular_step_deg: float | None, height: int) -> float:
    """Return a finite tangent-sampling step in radians."""

    if angular_step_deg is None:
        return math.pi / height
    if isinstance(angular_step_deg, bool):
        raise TypeError("angular_step_deg must be a finite positive real number")
    try:
        value = float(angular_step_deg)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "angular_step_deg must be a finite positive real number"
        ) from exc
    if not math.isfinite(value) or not 0.0 < value < 180.0:
        raise ValueError("angular_step_deg must be finite and in the interval (0, 180)")
    return math.radians(value)


def row_chunks(
    height: int, width: int, target_pixels: int = 262_144
) -> Iterator[slice]:
    rows = max(1, target_pixels // width)
    for start in range(0, height, rows):
        yield slice(start, min(height, start + rows))


def output_dtype(image: np.ndarray) -> np.dtype:
    if image.dtype == np.float32:
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def _rays_and_basis(
    shape_hw: tuple[int, int], rows: slice
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = shape_hw
    y = np.arange(rows.start, rows.stop, dtype=np.float64)[:, None]
    x = np.arange(width, dtype=np.float64)[None, :]
    lon = ((x + 0.5) / width) * (2.0 * np.pi) - np.pi
    lat = (np.pi / 2.0) - ((y + 0.5) / height) * np.pi
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)

    rays = np.stack(
        (
            sin_lon * cos_lat,
            np.broadcast_to(sin_lat, (rows.stop - rows.start, width)),
            cos_lon * cos_lat,
        ),
        axis=-1,
    )
    east = np.stack(
        (
            np.broadcast_to(cos_lon, rays.shape[:2]),
            np.zeros(rays.shape[:2], dtype=np.float64),
            np.broadcast_to(-sin_lon, rays.shape[:2]),
        ),
        axis=-1,
    )
    north = np.stack(
        (
            -sin_lon * sin_lat,
            np.broadcast_to(cos_lat, rays.shape[:2]),
            -cos_lon * sin_lat,
        ),
        axis=-1,
    )
    return rays, east, north


def tangent_map(
    shape_hw: tuple[int, int],
    rows: slice,
    east_offset_rad: float | np.ndarray,
    north_offset_rad: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map tangent-plane angular offsets to source ERP coordinates."""

    height, width = shape_hw
    rays, east, north = _rays_and_basis(shape_hw, rows)
    east_offset = np.asarray(east_offset_rad, dtype=np.float64)
    north_offset = np.asarray(north_offset_rad, dtype=np.float64)
    rho = np.hypot(east_offset, north_offset)
    scale = np.ones_like(rho, dtype=np.float64)
    np.divide(np.sin(rho), rho, out=scale, where=rho != 0.0)
    tangent = east_offset[..., None] * east + north_offset[..., None] * north
    sample_rays = np.cos(rho)[..., None] * rays + scale[..., None] * tangent
    sample_rays /= np.linalg.norm(sample_rays, axis=-1, keepdims=True)

    lon = np.arctan2(sample_rays[..., 0], sample_rays[..., 2])
    lat = np.arcsin(np.clip(sample_rays[..., 1], -1.0, 1.0))
    map_x = np.mod((lon + np.pi) / (2.0 * np.pi) * width - 0.5, width)
    map_y = np.clip((np.pi / 2.0 - lat) / np.pi * height - 0.5, 0, height - 1)
    return map_x, map_y


def sample_map(
    image: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    *,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """Sample an ERP map with horizontal wrap and polar-row clamping."""

    height, width = image.shape[:2]
    if interpolation == "nearest":
        xx = np.floor(map_x + 0.5).astype(np.int64) % width
        yy = np.clip(np.floor(map_y + 0.5).astype(np.int64), 0, height - 1)
        return image[yy, xx].astype(np.float64, copy=False)
    if interpolation != "bilinear":
        raise ValueError("interpolation must be 'nearest' or 'bilinear'")

    x0_raw = np.floor(map_x).astype(np.int64)
    y0 = np.floor(map_y).astype(np.int64)
    x0 = x0_raw % width
    x1 = (x0_raw + 1) % width
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    wx = map_x - x0_raw
    wy = map_y - y0
    if image.ndim == 3:
        wx = wx[..., None]
        wy = wy[..., None]
    top = (
        image[y0, x0].astype(np.float64) * (1.0 - wx)
        + image[y0, x1].astype(np.float64) * wx
    )
    bottom = (
        image[y1, x0].astype(np.float64) * (1.0 - wx)
        + image[y1, x1].astype(np.float64) * wx
    )
    return top * (1.0 - wy) + bottom * wy


def sample_tangent(
    image: np.ndarray,
    rows: slice,
    east_offset_rad: float | np.ndarray,
    north_offset_rad: float | np.ndarray,
    *,
    interpolation: str = "bilinear",
) -> np.ndarray:
    map_x, map_y = tangent_map(image.shape[:2], rows, east_offset_rad, north_offset_rad)
    return sample_map(image, map_x, map_y, interpolation=interpolation)


def sample_rays(
    image: np.ndarray, rays: np.ndarray, *, interpolation: str = "bilinear"
) -> np.ndarray:
    """Sample ``image`` at canonical Cartesian rays."""

    height, width = image.shape[:2]
    unit = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    lon = np.arctan2(unit[..., 0], unit[..., 2])
    lat = np.arcsin(np.clip(unit[..., 1], -1.0, 1.0))
    map_x = np.mod((lon + np.pi) / (2.0 * np.pi) * width - 0.5, width)
    map_y = np.clip((np.pi / 2.0 - lat) / np.pi * height - 0.5, 0, height - 1)
    return sample_map(image, map_x, map_y, interpolation=interpolation)
