"""High-level spherical applications assembled from PanorAi primitives.

Stability is assigned per exported wrapper. ``SphericalSLIC`` is Stable;
the generic application framework and the DoG, stereo, and Mask R-CNN paper
wrappers remain Experimental while their individual gates are open.
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
