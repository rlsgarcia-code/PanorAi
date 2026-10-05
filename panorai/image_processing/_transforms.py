"""Geometry-safe transformations of equirectangular images."""

from __future__ import annotations

import math

import numpy as np

from ._sampling import output_dtype, row_chunks, sample_rays, validate_image


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

    source = validate_image(image)
    matrix = _validate_rotation(rotation)
    shape = (
        source.shape[:2]
        if output_shape_hw is None
        else _validate_output_shape(output_shape_hw)
    )
    trailing = source.shape[2:]
    result = np.empty((*shape, *trailing), dtype=output_dtype(source))
    for rows in row_chunks(*shape):
        output_rays = _output_rays(shape, rows)
        source_rays = output_rays @ matrix
        result[rows] = sample_rays(source, source_rays, interpolation=interpolation)
    return result
