"""Experimental PanorAi-owned global spherical reconstruction.

The implementation consumes panorama-frame bearings and uses NumPy/SciPy.
It does not call OpenCV, PyCOLMAP, COLMAP, or Torch for geometry.
"""

from ._mapper import SphericalGlobalMapper
from ._models import (
    SphericalBaselinePrior,
    SphericalCameraPose,
    SphericalGlobalMapperOptions,
    SphericalPairwisePoseEdge,
    SphericalRangePrior,
    SphericalReconstructionDiagnostics,
    SphericalReconstructionResult,
    SphericalTrack,
    SphericalTrackObservation,
)

__all__ = [
    "SphericalBaselinePrior",
    "SphericalCameraPose",
    "SphericalGlobalMapper",
    "SphericalGlobalMapperOptions",
    "SphericalPairwisePoseEdge",
    "SphericalRangePrior",
    "SphericalReconstructionDiagnostics",
    "SphericalReconstructionResult",
    "SphericalTrack",
    "SphericalTrackObservation",
]
