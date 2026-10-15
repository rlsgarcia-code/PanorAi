"""Sphere-aware operations for direct equirectangular execution.

Importing this module keeps Torch optional. Accessing ``nn`` explicitly loads
the optional Torch facade.
"""

from __future__ import annotations

from importlib import import_module

from . import functional
from ._operations import GaussianBlur, Resize, Rotate
from .functional import gaussian_blur, resize, rotate

__all__ = [
    "GaussianBlur",
    "Resize",
    "Rotate",
    "SphericalNMS",
    "algorithms",
    "functional",
    "gaussian_blur",
    "resize",
    "rotate",
]


def __getattr__(name: str):
    if name == "nn":
        value = import_module(".nn", __name__)
    elif name == "algorithms":
        value = import_module(".algorithms", __name__)
    elif name == "SphericalNMS":
        value = getattr(import_module(".algorithms", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | {"nn"})
