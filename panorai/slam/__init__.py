"""Experimental calibrated-fisheye visual SLAM.

The first version is a visual-only, arbitrary-scale pipeline for one central
camera. It deliberately does not merge non-coincident dual-fisheye lenses into
a fictitious central panorama.
"""

from ._fisheye import EquidistantFisheyeCamera, FisheyeRayProjection
from ._incremental import SphericalIncrementalSLAM
from ._models import (
    SphericalImageFrame,
    SphericalIncrementalSLAMDiagnostics,
    SphericalIncrementalSLAMOptions,
    SphericalIncrementalSLAMResult,
    SphericalKeyframeSummary,
    SphericalLocalBAReport,
    SphericalMapObservation,
    SphericalMapPoint,
    SphericalSLAMDiagnostics,
    SphericalSLAMFrameSummary,
    SphericalSLAMOptions,
    SphericalSLAMPose,
    SphericalSLAMResult,
    SphericalTrackingResult,
)
from ._pipeline import SphericalVisualSLAM

__all__ = [
    "EquidistantFisheyeCamera",
    "FisheyeRayProjection",
    "SphericalImageFrame",
    "SphericalIncrementalSLAM",
    "SphericalIncrementalSLAMDiagnostics",
    "SphericalIncrementalSLAMOptions",
    "SphericalIncrementalSLAMResult",
    "SphericalKeyframeSummary",
    "SphericalLocalBAReport",
    "SphericalMapObservation",
    "SphericalMapPoint",
    "SphericalSLAMDiagnostics",
    "SphericalSLAMFrameSummary",
    "SphericalSLAMOptions",
    "SphericalSLAMPose",
    "SphericalSLAMResult",
    "SphericalTrackingResult",
    "SphericalVisualSLAM",
]
