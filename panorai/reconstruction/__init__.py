"""Experimental PanorAi-owned global spherical reconstruction.

The implementation consumes panorama-frame bearings and uses NumPy/SciPy.
It does not call OpenCV, PyCOLMAP, COLMAP, or Torch for geometry.
"""

from ._mapper import SphericalGlobalMapper
from ._models import (
    SphericalCameraPose,
    SphericalGlobalMapperOptions,
    SphericalPairwisePoseEdge,
    SphericalReconstructionDiagnostics,
    SphericalReconstructionResult,
    SphericalTrack,
    SphericalTrackObservation,
)

__all__ = [
    "SphericalCameraPose",
    "SphericalGlobalMapper",
    "SphericalGlobalMapperOptions",
    "SphericalPairwisePoseEdge",
    "SphericalReconstructionDiagnostics",
    "SphericalReconstructionResult",
    "SphericalTrack",
    "SphericalTrackObservation",
]
