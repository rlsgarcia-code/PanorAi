"""Additive geometry namespace preview for the 4.x architecture.

The established :mod:`panorai.geometry` import remains the stable ergonomic
facade. During the 3.x-to-4.x transition both paths intentionally resolve to
the same class and function objects; this preview does not move the stable
3.x implementation boundary.
"""

from panorai.geometry import *  # noqa: F403
from panorai.geometry import __all__ as _geometry_all

from .._tangent import tangent_offsets_to_rays

__all__ = [*_geometry_all, "tangent_offsets_to_rays"]
