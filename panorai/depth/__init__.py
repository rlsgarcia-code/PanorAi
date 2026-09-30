"""Lightweight compatibility adapters for optional depth models.

The upstream model implementations are not distributed by PanorAi 3.1. All
loader names and registry keys remain discoverable without importing Torch,
Open3D, or an upstream model package.
"""

from ._adapters import (
    DepthAdapterUnavailableError,
    load_dav2_model,
    load_dust3r_model,
    load_m3dv2_model,
    load_zoe_model,
)
from .registry import ModelRegistry


__all__ = [
    "DepthAdapterUnavailableError",
    "ModelRegistry",
    "load_dav2_model",
    "load_dust3r_model",
    "load_m3dv2_model",
    "load_zoe_model",
]
