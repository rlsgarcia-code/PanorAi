"""High-level spherical applications assembled from PanorAi primitives.

This namespace is Experimental until each application passes its documented
installed-wheel and prospective scientific validation gates.
"""

from ._base import (
    Application,
    ApplicationConfigurationError,
    PipelineApplication,
)
from ._classical_stereo import (
    ADCensusStereoDepth,
    CensusMGMStereoDepth,
    ELASStereoDepth,
    PatchMatchStereoDepth,
    StereoBMDepth,
    StereoSGBMDepth,
)
from ._features import SphericalDoGFeatures
from ._mask_rcnn import InstanceHeuristicDecision, SphericalMaskRCNNPaper
from ._panoramic_stereo import FullSphereDenseStereo, PanoramicDenseStereo
from ._slic import SphericalSLIC
from ._stereo import SphericalStereoDepth, SphericalStereoInput

__all__ = [
    "Application",
    "ApplicationConfigurationError",
    "ADCensusStereoDepth",
    "CensusMGMStereoDepth",
    "ELASStereoDepth",
    "FullSphereDenseStereo",
    "PipelineApplication",
    "PatchMatchStereoDepth",
    "PanoramicDenseStereo",
    "InstanceHeuristicDecision",
    "SphericalDoGFeatures",
    "SphericalSLIC",
    "SphericalMaskRCNNPaper",
    "SphericalStereoDepth",
    "SphericalStereoInput",
    "StereoBMDepth",
    "StereoSGBMDepth",
]
