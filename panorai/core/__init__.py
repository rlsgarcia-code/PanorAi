"""Fundamental spherical mathematics and backend-independent contracts.

Importing :mod:`panorai.core` requires NumPy but never imports optional Torch.
Backend modules are selected explicitly through :mod:`panorai.core.numpy` and
:mod:`panorai.core.torch`.
"""

from .specs import (
    SphericalConv2dSpec,
    SphericalExecutionSpec,
    SphericalGridSpec,
    SphericalNeighborhoodSpec,
    SphericalPool2dSpec,
    SphericalResizeSpec,
)

SPHERICAL_CORE_INTERFACE = "panorai-core-operations/v1"

__all__ = [
    "SPHERICAL_CORE_INTERFACE",
    "SphericalConv2dSpec",
    "SphericalExecutionSpec",
    "SphericalGridSpec",
    "SphericalNeighborhoodSpec",
    "SphericalPool2dSpec",
    "SphericalResizeSpec",
]
