"""
panorai_data_rewritten
======================

A reorganized version of the `panorai_data` module, containing:
 - Factory methods for creating spherical data objects
 - Equirectangular image class
 - Gnomonic face class
 - Gnomonic face set class
 - Shared spherical data abstraction and multi-channel handling
 - Utility functions and exceptions
"""

from .factory import DataFactory
from .equirectangular_image import EquirectangularImage, SphericalTranspose
from .gnomonic_image import GnomonicFace
from .gnomonic_imageset import GnomonicFaceSet
from .fields import RasterFieldResult, RasterFieldSpec
from .observations import (
    AggregatedViewEvidence,
    FacePointBatch,
    FaceRegionBatch,
    FaceViewOutput,
    ObservationFusionResult,
    SphericalPointObservation,
    SphericalPointSet,
    SphericalRegionObservation,
    SphericalRegionSet,
    SphericalViewObservation,
    SphericalViewSet,
)
from .spherical_mask import SphericalBinaryMask
from .regions import (
    EnclosingSphericalCap,
    RegionAdjacencyEdge,
    RegionAdjacencyGraph,
    RegionAnalysisResult,
    RegionMeasurement,
    RegionMeasurements,
    SphericalBoundary,
    SphericalBoundarySet,
    SphericalContour,
    SphericalContourSet,
)
from .xyz_image_dataset import XYZImageDataset

__all__ = [
    "DataFactory",
    "EquirectangularImage",
    "SphericalTranspose",
    "GnomonicFace",
    "GnomonicFaceSet",
    "RasterFieldResult",
    "RasterFieldSpec",
    "FacePointBatch",
    "SphericalPointObservation",
    "SphericalPointSet",
    "FaceRegionBatch",
    "SphericalRegionObservation",
    "SphericalRegionSet",
    "FaceViewOutput",
    "SphericalViewObservation",
    "SphericalViewSet",
    "AggregatedViewEvidence",
    "ObservationFusionResult",
    "SphericalBinaryMask",
    "EnclosingSphericalCap",
    "RegionAdjacencyEdge",
    "RegionAdjacencyGraph",
    "RegionAnalysisResult",
    "RegionMeasurement",
    "RegionMeasurements",
    "SphericalBoundary",
    "SphericalBoundarySet",
    "SphericalContour",
    "SphericalContourSet",
    "XYZImageDataset",
]
