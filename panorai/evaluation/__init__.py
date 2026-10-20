"""Geometry-aware evaluation for spherical computer vision.

The NumPy-only region, feature, pose, radial-range and surface-reconstruction
evaluators are Stable. None of the evaluators
infer support or validity from numeric values, download datasets, run
experiments, or promote the algorithm being measured.
"""

from ._metrics import (
    evaluate_binary_regions,
    evaluate_features,
    evaluate_pose,
    evaluate_radial_range,
)
from ._results import (
    BinaryRegionEvaluation,
    DepthEvaluation,
    EvaluationProtocol,
    EvaluationRecord,
    FeatureEvaluation,
    PoseEvaluation,
)
from ._surface import (
    SurfaceReconstructionEvaluation,
    SurfaceReconstructionEvaluationReport,
    aggregate_surface_reconstruction,
    evaluate_surface_reconstruction,
)
from ._validation import solid_angle_weights

__all__ = [
    "BinaryRegionEvaluation",
    "DepthEvaluation",
    "EvaluationProtocol",
    "EvaluationRecord",
    "FeatureEvaluation",
    "PoseEvaluation",
    "SurfaceReconstructionEvaluation",
    "SurfaceReconstructionEvaluationReport",
    "aggregate_surface_reconstruction",
    "evaluate_binary_regions",
    "evaluate_features",
    "evaluate_pose",
    "evaluate_radial_range",
    "evaluate_surface_reconstruction",
    "solid_angle_weights",
]
