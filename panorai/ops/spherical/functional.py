"""Functional sphere-native operations over canonical NumPy ERP rasters.

The local-filter implementations are the historical public
``panorai.image_processing`` functions. This module is a compact compositional
namespace; it does not own a second mathematical implementation.
"""

from __future__ import annotations

import math
from typing import Annotated

import numpy as np

from panorai.image_processing import (
    spherical_bilateral_filter as bilateral_filter,
    spherical_box_blur as box_blur,
    spherical_canny as canny,
    spherical_filter2d as filter2d,
    spherical_gaussian_blur as gaussian_blur,
    spherical_gradient as gradient,
    spherical_laplacian as laplacian,
    spherical_median_blur as median_blur,
    spherical_resize as resize,
    spherical_rotate as rotate,
    spherical_sobel as sobel,
)
from ._morphology import closing, dilate, erode, opening
from ._regions import (
    connected_components,
    extract_boundaries,
    fill_holes,
    geodesic_distance_transform,
    measure_regions,
    region_adjacency_graph,
    region_boundaries,
    skeletonize,
    trace_contours,
    watershed,
)


RasterArray = Annotated[np.ndarray, "panorai:raster"]
MaskArray = Annotated[np.ndarray, "panorai:mask"]


def resample(
    image: np.ndarray,
    rays_xyz: np.ndarray,
    *,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """Sample an ERP at arbitrary canonical Cartesian bearings.

    ``rays_xyz`` may have any leading shape and a final axis of length three.
    The result has the ray leading shape followed by the image component axes.
    """

    from panorai.core._sampling import sample_rays, validate_image

    source = validate_image(image)
    rays = np.asarray(rays_xyz)
    if rays.ndim < 1 or rays.shape[-1] != 3:
        raise ValueError("rays_xyz must have shape (..., 3)")
    if not np.issubdtype(rays.dtype, np.floating):
        raise TypeError("rays_xyz must use a floating dtype")
    if not np.isfinite(rays).all():
        raise ValueError("rays_xyz must contain only finite values")
    norms = np.linalg.norm(rays, axis=-1)
    if np.any(norms <= np.finfo(rays.dtype).eps):
        raise ValueError("rays_xyz cannot contain zero-length bearings")
    return sample_rays(source, rays, interpolation=interpolation)


def _solid_angle_weights(shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    latitude_edges = np.pi / 2.0 - np.arange(height + 1) * np.pi / height
    row_area = (np.sin(latitude_edges[:-1]) - np.sin(latitude_edges[1:])) * (
        2.0 * np.pi / width
    )
    return np.broadcast_to(row_area[:, None], (height, width))


def _area_inputs(
    values: np.ndarray, mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values)
    if array.ndim not in {2, 3} or min(array.shape[:2]) < 1:
        raise ValueError("values must use non-empty HW or HWC layout")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError("values must use a real numeric dtype")
    if mask is None:
        selected = np.ones(array.shape[:2], dtype=bool)
    else:
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
            raise TypeError("mask must be a boolean NumPy array")
        if mask.shape != array.shape[:2]:
            raise ValueError(f"mask must have shape {array.shape[:2]}")
        selected = mask
    if not selected.any():
        raise ValueError("mask must select at least one ERP cell")
    finite = np.isfinite(array)
    if array.ndim == 3:
        finite = finite.all(axis=-1)
    if np.any(selected & ~finite):
        raise ValueError("selected values must be finite")
    return array, selected


def area_average(
    values: RasterArray, *, mask: MaskArray | None = None
) -> float | np.ndarray:
    """Return the exact solid-angle-weighted mean of an ERP raster."""

    array, selected = _area_inputs(values, mask)
    weights = _solid_angle_weights(array.shape[:2]) * selected
    denominator = weights.sum(dtype=np.float64)
    if array.ndim == 2:
        return np.sum(array * weights, dtype=np.float64) / denominator
    return (
        np.sum(array * weights[..., None], axis=(0, 1), dtype=np.float64) / denominator
    )


def area_histogram(
    values: np.ndarray,
    bins: int | np.ndarray = 10,
    range: tuple[float, float] | None = None,
    *,
    mask: np.ndarray | None = None,
    density: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a solid-angle-weighted histogram for one-channel ERP values."""

    array, selected = _area_inputs(values, mask)
    if array.ndim != 2:
        raise ValueError("area_histogram requires single-channel HW values")
    weights = _solid_angle_weights(array.shape)[selected]
    histogram, edges = np.histogram(
        array[selected], bins=bins, range=range, weights=weights, density=density
    )
    if not density:
        samples = array[selected]
        included = (samples >= edges[0]) & (samples <= edges[-1])
        expected_area = math.fsum(float(value) for value in weights[included])
        occupied = np.flatnonzero(histogram)
        if occupied.size:
            actual_area = math.fsum(float(value) for value in histogram)
            histogram[occupied[-1]] += expected_area - actual_area
    return histogram, edges


__all__ = [
    "area_average",
    "area_histogram",
    "bilateral_filter",
    "box_blur",
    "canny",
    "closing",
    "connected_components",
    "dilate",
    "erode",
    "extract_boundaries",
    "fill_holes",
    "filter2d",
    "gaussian_blur",
    "geodesic_distance_transform",
    "gradient",
    "laplacian",
    "median_blur",
    "measure_regions",
    "opening",
    "region_adjacency_graph",
    "region_boundaries",
    "skeletonize",
    "trace_contours",
    "resample",
    "resize",
    "rotate",
    "sobel",
    "watershed",
]
