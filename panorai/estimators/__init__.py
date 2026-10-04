"""Experimental geometric estimators owned by PanorAi.

The package is intentionally independent from feature extraction backends.
Its relative-pose estimator consumes spherical bearings and does not import
OpenCV, PyCOLMAP, or Torch.
"""

from ._sampling import (
    FivePointSample,
    FivePointSampler,
    FivePointSamplingDiagnostics,
    SpatiallyWeightedFivePointSampler,
    UniformFivePointSampler,
)
from ._calibration import (
    CalibrationEvaluation,
    RelativePoseConfidenceCalibrator,
)
from ._five_point import solve_five_point_essential
from ._native import native_kernels_available
from ._quality import (
    ModelCompetitionReport,
    ModelEvidence,
    PoseStabilityReport,
    RelativePoseAcceptancePolicy,
    RelativePoseQualityReport,
    TranslationOrientationReport,
)

from .relative_pose import (
    RelativePoseOptions,
    RelativePoseResult,
    SphericalRelativePoseEstimator,
    estimate_relative_pose,
    spherical_tangent_sampson_error,
)

__all__ = [
    "CalibrationEvaluation",
    "FivePointSample",
    "FivePointSampler",
    "FivePointSamplingDiagnostics",
    "ModelCompetitionReport",
    "ModelEvidence",
    "PoseStabilityReport",
    "RelativePoseAcceptancePolicy",
    "RelativePoseConfidenceCalibrator",
    "RelativePoseOptions",
    "RelativePoseQualityReport",
    "RelativePoseResult",
    "TranslationOrientationReport",
    "SpatiallyWeightedFivePointSampler",
    "SphericalRelativePoseEstimator",
    "UniformFivePointSampler",
    "estimate_relative_pose",
    "native_kernels_available",
    "solve_five_point_essential",
    "spherical_tangent_sampson_error",
]
