"""Experimental pretrained FCN/CAM transfer to equirectangular images.

The benchmark package deliberately avoids importing Torch at package import
time. This lets dependency-light geometry and backprojection helpers remain
testable in the core environment.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DenseFCNOutput",
    "ImageNetFCN",
    "SphericalConv2d",
    "SphericalMaxPool2d",
    "class_activation_map",
    "spherical_area_average",
    "sphericalize",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from panorai.experimental import deep_learning

    value = getattr(deep_learning, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
