"""Sphere-aware operations for direct equirectangular execution.

Importing this module keeps Torch optional. Accessing ``nn`` explicitly loads
the optional Torch facade.
"""

from __future__ import annotations

from importlib import import_module

from . import functional
from ._operations import (
    AreaAverage,
    AreaHistogram,
    BilateralFilter,
    BoxBlur,
    Canny,
    Filter2D,
    GaussianBlur,
    Gradient,
    Laplacian,
    MedianBlur,
    Resample,
    Resize,
    Rotate,
    Sobel,
)
from ._morphology import Closing, Dilate, Erode, GeodesicDisk, Opening
from .functional import (
    area_average,
    area_histogram,
    bilateral_filter,
    box_blur,
    canny,
    closing,
    dilate,
    erode,
    filter2d,
    gaussian_blur,
    gradient,
    laplacian,
    median_blur,
    opening,
    resample,
    resize,
    rotate,
    sobel,
)

__all__ = [
    "AreaAverage",
    "AreaHistogram",
    "BilateralFilter",
    "BoxBlur",
    "Canny",
    "Closing",
    "Dilate",
    "Erode",
    "Filter2D",
    "GaussianBlur",
    "Gradient",
    "GeodesicDisk",
    "Laplacian",
    "MedianBlur",
    "Opening",
    "Resample",
    "Resize",
    "Rotate",
    "Sobel",
    "DoGDetector",
    "GaussianPyramid",
    "LaplacianPyramid",
    "SLIC",
    "SphericalNMS",
    "algorithms",
    "functional",
    "area_average",
    "area_histogram",
    "bilateral_filter",
    "box_blur",
    "canny",
    "closing",
    "dilate",
    "erode",
    "filter2d",
    "gaussian_blur",
    "gradient",
    "laplacian",
    "median_blur",
    "opening",
    "resample",
    "resize",
    "rotate",
    "sobel",
]


def __getattr__(name: str):
    if name == "nn":
        value = import_module(".nn", __name__)
    elif name == "algorithms":
        value = import_module(".algorithms", __name__)
    elif name in {
        "DoGDetector",
        "GaussianPyramid",
        "LaplacianPyramid",
        "SLIC",
        "SphericalNMS",
    }:
        value = getattr(import_module(".algorithms", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | {"nn"})
