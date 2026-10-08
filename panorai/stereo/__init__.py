"""Experimental dense stereo for calibrated central spherical images.

The returned depth quantity is radial range from the reference camera centre.
This package is Experimental: its API and numerical policy may change in a
future minor release as real-scene validation expands.
"""

from ._dense import (
    SphericalDenseStereo,
    SphericalStereoOptions,
    SphericalStereoResult,
    estimate_spherical_range,
)
from ._match_filter import (
    DenseMatchFilterOptions,
    DenseMatchFilterResult,
    filter_matches_by_dense_range,
)
from ._visualization import (
    colorize_spherical_range,
    render_spherical_stereo_result,
)

__all__ = [
    "DenseMatchFilterOptions",
    "DenseMatchFilterResult",
    "SphericalDenseStereo",
    "SphericalStereoOptions",
    "SphericalStereoResult",
    "colorize_spherical_range",
    "estimate_spherical_range",
    "filter_matches_by_dense_range",
    "render_spherical_stereo_result",
]
