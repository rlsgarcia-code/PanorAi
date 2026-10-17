"""High-level spherical applications assembled from PanorAi primitives.

Stability is assigned per exported wrapper. ``SphericalSLIC`` is Stable;
the generic application framework and the DoG, relative-pose, stereo, and
Mask R-CNN paper wrappers remain Experimental while their individual gates
are open.
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
from ._relative_pose import RelativePoseProfile, SphericalRelativePose
from ._relative_pose_capture import (
    AssessRelativePoseCapture,
    RelativePoseCaptureAssessment,
    RelativePoseCaptureError,
    RelativePoseCaptureInput,
    RelativePoseCaptureMetadata,
    RelativePoseCapturePolicy,
)
from ._slic import SphericalSLIC
from ._stereo import SphericalStereoDepth, SphericalStereoInput

__all__ = [
    "Application",
    "ApplicationConfigurationError",
    "ADCensusStereoDepth",
    "AssessRelativePoseCapture",
    "CensusMGMStereoDepth",
    "ELASStereoDepth",
    "FullSphereDenseStereo",
    "PipelineApplication",
    "PatchMatchStereoDepth",
    "PanoramicDenseStereo",
    "InstanceHeuristicDecision",
    "RelativePoseCaptureAssessment",
    "RelativePoseCaptureError",
    "RelativePoseCaptureInput",
    "RelativePoseCaptureMetadata",
    "RelativePoseCapturePolicy",
    "RelativePoseProfile",
    "SphericalDoGFeatures",
    "SphericalSLIC",
    "SphericalMaskRCNNPaper",
    "SphericalRelativePose",
    "SphericalStereoDepth",
    "SphericalStereoInput",
    "StereoBMDepth",
    "StereoSGBMDepth",
]
