"""Experimental PanorAi-owned global spherical reconstruction.

The implementation consumes panorama-frame bearings and uses NumPy/SciPy.
It does not call OpenCV, PyCOLMAP, COLMAP, or Torch for geometry.
"""

from importlib import import_module
from typing import Any

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
    "BuildRegisteredReferenceSets",
    "ColorizeSurface",
    "DenseReconstructionWorkspace",
    "EstimateMultiViewRanges",
    "ExtractSurface",
    "FuseTSDF",
    "MultiViewRangeMap",
    "MultiViewRangeSet",
    "RegisteredReferenceSets",
    "RegisteredSphericalScene",
    "RegisteredSphericalView",
    "SphericalBaselinePrior",
    "SphericalCameraPose",
    "SphericalDenseReconstructionResult",
    "SphericalGlobalMapper",
    "SphericalGlobalMapperOptions",
    "SphericalPairwisePoseEdge",
    "SphericalRangePrior",
    "SphericalReconstructionDiagnostics",
    "SphericalReconstructionResult",
    "SphericalTrack",
    "SphericalTrackObservation",
    "SurfaceMesh",
    "TSDFSpec",
    "TSDFVolume",
    "ValidateDenseReconstruction",
    "colorize_surface",
    "extract_surface",
    "fuse_tsdf",
]

_DENSE_STAGE_EXPORTS = frozenset(
    {
        "BuildRegisteredReferenceSets",
        "ColorizeSurface",
        "DenseReconstructionWorkspace",
        "EstimateMultiViewRanges",
        "ExtractSurface",
        "FuseTSDF",
        "RegisteredReferenceSets",
        "ValidateDenseReconstruction",
        "colorize_surface",
        "extract_surface",
        "fuse_tsdf",
    }
)
_DENSE_MODEL_EXPORTS = frozenset(
    {
        "MultiViewRangeMap",
        "MultiViewRangeSet",
        "RegisteredSphericalScene",
        "RegisteredSphericalView",
        "SphericalDenseReconstructionResult",
        "SurfaceMesh",
        "TSDFSpec",
        "TSDFVolume",
    }
)


def __getattr__(name: str) -> Any:
    """Load dense reconstruction only when its public objects are requested."""

    if name in _DENSE_STAGE_EXPORTS:
        value = getattr(import_module("._dense", __name__), name)
    elif name in _DENSE_MODEL_EXPORTS:
        value = getattr(import_module("._dense_models", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
