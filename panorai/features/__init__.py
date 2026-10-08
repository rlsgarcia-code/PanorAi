"""Stable spherical-feature core with Experimental optional extensions.

OpenCV implements detection, description, and nearest-neighbour matching.
PanorAi owns the panorama geometry, result objects, masks, deduplication, and
provenance. No OpenCV result object crosses the normal public API. Multiscale
routing, virtual rigs, and PyCOLMAP export remain explicitly Experimental.
"""

from ._config import (
    FaceSetSpec,
    FeatureExtractorConfig,
    FeatureMatcherConfig,
    SphericalFeaturePipelineConfig,
)
from ._extractor import FeatureExtractor, extract_opencv_features
from ._geometry import deduplicate_spherical_keypoints, gnomonic_feature_mask
from ._matcher import FeatureMatcher, match_opencv_features
from ._multiscale import (
    MultiscaleEmbeddingConfig,
    MultiscaleFeatureMatches,
    MultiscaleFeatureSet,
    MultiscaleSphericalFeaturePipeline,
    OpenCVContextEmbedding,
    VisualContextNode,
    VisualEmbeddingProvider,
)
from ._models import (
    DeduplicationResult,
    FeatureProvenance,
    GnomonicRig,
    GnomonicRigCamera,
    MatchProvenance,
    PyCOLMAPExportResult,
    SphericalBearingCorrespondences,
    SphericalFeature,
    SphericalFeatureMatches,
    SphericalFeatureSet,
)
from ._pipeline import SphericalFeaturePipeline
from ._presets import available_presets
from ._resolution_selection import (
    ResolutionLevelReport,
    ResolutionSelectionObservation,
    ResolutionSelectionPolicy,
    ResolutionSelectionReport,
    ResolutionTransitionReport,
    select_feature_resolution,
)
from ._spherical_dog import (
    SphericalDoGSIFTConfig,
    SphericalDoGSIFTExtractor,
    SphericalDoGSIFTPipeline,
)
from ._spherical_detector import (
    SphericalCoarseDoGDetector,
    SphericalCoarseDoGDetectorConfig,
    SphericalDetectionDiagnostics,
    SphericalDoGDetector,
    SphericalDoGDetectorConfig,
    SphericalKeypoint,
    SphericalKeypointSet,
    SphericalOctaveDetectionDiagnostics,
)
from ._tangent_patches import (
    OpenCVTangentDescriptor,
    OpenCVTangentDescriptorConfig,
    TangentDescriptorSet,
    TangentPatch,
    TangentPatchGeometry,
    TangentPatchProvider,
    TangentPatchRequest,
    TangentPatchSet,
)
from .backends.opencv import OpenCVFeatureBackend

__all__ = [
    "DeduplicationResult",
    "FaceSetSpec",
    "FeatureExtractor",
    "FeatureExtractorConfig",
    "FeatureMatcher",
    "FeatureMatcherConfig",
    "FeatureProvenance",
    "GnomonicRig",
    "GnomonicRigCamera",
    "MatchProvenance",
    "MultiscaleEmbeddingConfig",
    "MultiscaleFeatureMatches",
    "MultiscaleFeatureSet",
    "MultiscaleSphericalFeaturePipeline",
    "OpenCVFeatureBackend",
    "OpenCVContextEmbedding",
    "OpenCVTangentDescriptor",
    "OpenCVTangentDescriptorConfig",
    "PyCOLMAPExportResult",
    "ResolutionLevelReport",
    "ResolutionSelectionObservation",
    "ResolutionSelectionPolicy",
    "ResolutionSelectionReport",
    "ResolutionTransitionReport",
    "SphericalBearingCorrespondences",
    "SphericalCoarseDoGDetector",
    "SphericalCoarseDoGDetectorConfig",
    "SphericalDoGSIFTConfig",
    "SphericalDoGSIFTExtractor",
    "SphericalDoGSIFTPipeline",
    "SphericalDetectionDiagnostics",
    "SphericalDoGDetector",
    "SphericalDoGDetectorConfig",
    "SphericalFeature",
    "SphericalFeatureMatches",
    "SphericalFeaturePipeline",
    "SphericalFeaturePipelineConfig",
    "SphericalFeatureSet",
    "SphericalKeypoint",
    "SphericalKeypointSet",
    "SphericalOctaveDetectionDiagnostics",
    "TangentDescriptorSet",
    "TangentPatch",
    "TangentPatchGeometry",
    "TangentPatchProvider",
    "TangentPatchRequest",
    "TangentPatchSet",
    "VisualContextNode",
    "VisualEmbeddingProvider",
    "available_presets",
    "deduplicate_spherical_keypoints",
    "extract_opencv_features",
    "gnomonic_feature_mask",
    "match_opencv_features",
    "select_feature_resolution",
]
