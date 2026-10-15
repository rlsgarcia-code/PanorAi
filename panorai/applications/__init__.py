"""High-level spherical applications assembled from PanorAi primitives.

This namespace is Experimental until each application passes its documented
installed-wheel and prospective scientific validation gates.
"""

from ._base import (
    Application,
    ApplicationConfigurationError,
    PipelineApplication,
)
from ._features import SphericalDoGFeatures
from ._mask_rcnn import InstanceHeuristicDecision, SphericalMaskRCNNPaper
from ._slic import SphericalSLIC
from ._stereo import SphericalStereoDepth, SphericalStereoInput

__all__ = [
    "Application",
    "ApplicationConfigurationError",
    "PipelineApplication",
    "InstanceHeuristicDecision",
    "SphericalDoGFeatures",
    "SphericalSLIC",
    "SphericalMaskRCNNPaper",
    "SphericalStereoDepth",
    "SphericalStereoInput",
]
