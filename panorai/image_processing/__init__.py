"""Spherical image processing for equirectangular panoramas.

The public functions mirror familiar OpenCV image-processing stages while
defining every neighbourhood in a local tangent plane on the unit sphere.
Stability is assigned per symbol: ``spherical_slic`` is Stable while the
remaining historical namespace retains its explicitly inventoried tier.
"""

from ._edges import spherical_canny
from ._filters import (
    SphericalGradient,
    spherical_bilateral_filter,
    spherical_box_blur,
    spherical_filter2d,
    spherical_gaussian_blur,
    spherical_gradient,
    spherical_laplacian,
    spherical_median_blur,
    spherical_sobel,
)
from ._histogram import spherical_equalize_histogram
from ._native import native_filter_available
from ._pyramids import (
    reconstruct_laplacian_pyramid,
    spherical_gaussian_pyramid,
    spherical_laplacian_pyramid,
)
from ._slic import spherical_slic
from ._transforms import spherical_resize, spherical_rotate

__all__ = [
    "SphericalGradient",
    "native_filter_available",
    "reconstruct_laplacian_pyramid",
    "spherical_bilateral_filter",
    "spherical_box_blur",
    "spherical_canny",
    "spherical_equalize_histogram",
    "spherical_filter2d",
    "spherical_gaussian_blur",
    "spherical_gaussian_pyramid",
    "spherical_gradient",
    "spherical_laplacian",
    "spherical_laplacian_pyramid",
    "spherical_median_blur",
    "spherical_resize",
    "spherical_rotate",
    "spherical_slic",
    "spherical_sobel",
]
