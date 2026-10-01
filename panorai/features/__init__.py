"""Experimental spherical features for panoramic computer vision.

OpenCV implements detection, description, and nearest-neighbour matching.
PanorAi owns the panorama geometry, result objects, masks, deduplication, and
provenance. No OpenCV result object crosses the normal public API.
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
    "OpenCVFeatureBackend",
    "PyCOLMAPExportResult",
    "SphericalBearingCorrespondences",
    "SphericalFeature",
    "SphericalFeatureMatches",
    "SphericalFeaturePipeline",
    "SphericalFeaturePipelineConfig",
    "SphericalFeatureSet",
    "available_presets",
    "deduplicate_spherical_keypoints",
    "extract_opencv_features",
    "gnomonic_feature_mask",
    "match_opencv_features",
]
