# panorai/depth/__init__.py
"""Model loading utilities for optional depth models."""

import logging

logger = logging.getLogger(__name__)

try:
    from .DepthAnythingV2.loader import load_dav2_model  # type: ignore
except ImportError as e:
    load_dav2_model = None
    logger.warning("DepthAnythingV2 is unavailable: %s", e)

try:
    from .Metric3D.loader import load_m3dv2_model  # type: ignore
except ImportError as e:
    load_m3dv2_model = None
    logger.warning("Metric3D is unavailable: %s", e)

try:
    from .zoe import load_zoe_model  # type: ignore
except ImportError as e:
    load_zoe_model = None
    logger.warning("ZoeDepth is unavailable: %s", e)

try:
    from .Dust3r.loader import load_dust3r_model  # type: ignore
except ImportError as e:
    load_dust3r_model = None
    logger.warning("Dust3r is unavailable: %s", e)



from .registry import ModelRegistry


__all__ = [
'ModelRegistry'
]
