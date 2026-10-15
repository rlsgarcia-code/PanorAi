"""Geometry-safe transformations of equirectangular images."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .._sampling import output_dtype, row_chunks, sample_rays, validate_image
from .._sampling import sample_map
from ._native import (
    native_rotation_available,
    native_sample_equirectangular_map,
    native_spherical_rotation_map,
    supports_native_rotation_sample,
)


def _validate_output_shape(output_shape_hw: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(output_shape_hw, tuple)
        or len(output_shape_hw) != 2
        or any(isinstance(value, bool) for value in output_shape_hw)
        or any(not isinstance(value, (int, np.integer)) for value in output_shape_hw)
    ):
        raise TypeError("output_shape_hw must be a (height, width) integer tuple")
    height, width = (int(value) for value in output_shape_hw)
    if height < 2 or width < 2:
        raise ValueError("output height and width must both be at least 2")
    return height, width


def _output_rays(shape_hw: tuple[int, int], rows: slice) -> np.ndarray:
    height, width = shape_hw
    y = np.arange(rows.start, rows.stop, dtype=np.float64)[:, None]
    x = np.arange(width, dtype=np.float64)[None, :]
    lon = ((x + 0.5) / width) * (2.0 * np.pi) - np.pi
    lat = (np.pi / 2.0) - ((y + 0.5) / height) * np.pi
    cos_lat = np.cos(lat)
    return np.stack(
        (
            np.sin(lon) * cos_lat,
            np.broadcast_to(np.sin(lat), (rows.stop - rows.start, width)),
            np.cos(lon) * cos_lat,
        ),
        axis=-1,
    )


def spherical_resize(
    image: np.ndarray,
    output_shape_hw: tuple[int, int],
    *,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """Resample an ERP at the pixel-centre rays of a new ERP raster."""

    source = validate_image(image)
    output_shape = _validate_output_shape(output_shape_hw)
    trailing = source.shape[2:]
    result = np.empty((*output_shape, *trailing), dtype=output_dtype(source))
    for rows in row_chunks(*output_shape):
        result[rows] = sample_rays(
            source, _output_rays(output_shape, rows), interpolation=interpolation
        )
    return result


def _validate_rotation(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    if not np.isfinite(matrix).all():
        raise ValueError("rotation must contain only finite values")
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-7, rtol=0.0):
        raise ValueError("rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-7):
        raise ValueError("rotation must be a proper rotation with determinant +1")
    return matrix


@dataclass(frozen=True, slots=True)
class SphericalRotationPlan:
    """Reusable inverse sampling map for one NumPy ERP rotation.

    The plan is interpolation-independent: continuous and categorical fields
    share the same geometric map and select bilinear or nearest sampling only
    when :meth:`apply` is called.
    """

    source_shape_hw: tuple[int, int]
    output_shape_hw: tuple[int, int]
    map_x: np.ndarray
    map_y: np.ndarray

    @classmethod
    def from_rotation(
        cls,
        source_shape_hw: tuple[int, int],
        rotation: np.ndarray,
        *,
        output_shape_hw: tuple[int, int] | None = None,
    ) -> "SphericalRotationPlan":
        source_shape = _validate_output_shape(source_shape_hw)
        output_shape = (
            source_shape
            if output_shape_hw is None
            else _validate_output_shape(output_shape_hw)
        )
        matrix = _validate_rotation(rotation)
        source_height, source_width = source_shape
        if native_rotation_available():
            map_x, map_y = native_spherical_rotation_map(
                source_shape, output_shape, matrix
            )
            return cls(source_shape, output_shape, map_x, map_y)
        map_x = np.empty(output_shape, dtype=np.float64)
        map_y = np.empty(output_shape, dtype=np.float64)
        for rows in row_chunks(*output_shape):
            output_rays = _output_rays(output_shape, rows)
            source_rays = output_rays @ matrix
            longitude = np.arctan2(source_rays[..., 0], source_rays[..., 2])
            latitude = np.arcsin(np.clip(source_rays[..., 1], -1.0, 1.0))
            map_x[rows] = np.mod(
                (longitude + np.pi) / (2.0 * np.pi) * source_width - 0.5,
                source_width,
            )
            map_y[rows] = np.clip(
                (np.pi / 2.0 - latitude) / np.pi * source_height - 0.5,
                0.0,
                source_height - 1.0,
            )
        map_x.flags.writeable = False
        map_y.flags.writeable = False
        return cls(source_shape, output_shape, map_x, map_y)

    def apply(
        self,
        image: np.ndarray,
        *,
        interpolation: str = "bilinear",
    ) -> np.ndarray:
        """Sample an aligned ERP field through this plan."""

        raw = np.asarray(image)
        if raw.dtype == np.bool_:
            if interpolation != "nearest":
                raise TypeError("boolean ERP rotation requires nearest interpolation")
            source = raw
        else:
            source = validate_image(image)
        if source.shape[:2] != self.source_shape_hw:
            raise ValueError(
                "image shape does not match rotation plan source_shape_hw"
            )
        if interpolation == "nearest":
            if supports_native_rotation_sample(source):
                return native_sample_equirectangular_map(
                    source,
                    self.map_x,
                    self.map_y,
                    interpolation="nearest",
                )
            height, width = self.source_shape_hw
            xx = np.floor(self.map_x + 0.5).astype(np.int64) % width
            yy = np.clip(
                np.floor(self.map_y + 0.5).astype(np.int64), 0, height - 1
            )
            return np.array(source[yy, xx], copy=True)
        if interpolation != "bilinear":
            raise ValueError("interpolation must be 'nearest' or 'bilinear'")
        if supports_native_rotation_sample(source):
            return native_sample_equirectangular_map(
                source,
                self.map_x,
                self.map_y,
                interpolation="bilinear",
            )
        sampled = sample_map(
            source, self.map_x, self.map_y, interpolation="bilinear"
        )
        return sampled.astype(output_dtype(source), copy=False)


def spherical_rotate(
    image: np.ndarray,
    rotation: np.ndarray,
    *,
    output_shape_hw: tuple[int, int] | None = None,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """Actively rotate a panorama by a canonical 3x3 ray rotation.

    ``rotation`` maps source rays to output rays in PanorAi's ``+X`` right,
    ``+Y`` up, ``+Z`` forward frame. Reverse mapping uses ``rotation.T`` so
    every output ERP pixel is sampled exactly once.
    """

    raw = np.asarray(image)
    if raw.dtype != np.bool_:
        validate_image(image)
    elif interpolation != "nearest":
        raise TypeError("boolean ERP rotation requires nearest interpolation")
    plan = SphericalRotationPlan.from_rotation(
        raw.shape[:2], rotation, output_shape_hw=output_shape_hw
    )
    return plan.apply(raw, interpolation=interpolation)


__all__ = ["SphericalRotationPlan", "spherical_resize", "spherical_rotate"]
