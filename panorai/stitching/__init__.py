"""Experimental registration and panorama stitching components."""

from ._stages import (
    BlendStitchingPanorama,
    CalibrateAndRegister,
    CompensateExposure,
    DetectStitchingFeatures,
    DiscoverOverlaps,
    StitchingRegistrationError,
    WarpRegisteredImages,
)
from ._types import (
    ImagePairMatches,
    ImagePairRegistration,
    PanoramaStitchingResult,
    RegisteredImageSet,
    StitchingDiagnostics,
    StitchingFeatureSet,
    StitchingFeatures,
    StitchingImage,
    StitchingInput,
    StitchingMatches,
    StitchingOptions,
    WarpedImageSet,
)

__all__ = [
    "BlendStitchingPanorama",
    "CalibrateAndRegister",
    "CompensateExposure",
    "DetectStitchingFeatures",
    "DiscoverOverlaps",
    "ImagePairMatches",
    "ImagePairRegistration",
    "PanoramaStitchingResult",
    "RegisteredImageSet",
    "StitchingDiagnostics",
    "StitchingFeatureSet",
    "StitchingFeatures",
    "StitchingImage",
    "StitchingInput",
    "StitchingMatches",
    "StitchingOptions",
    "StitchingRegistrationError",
    "WarpRegisteredImages",
    "WarpedImageSet",
]
