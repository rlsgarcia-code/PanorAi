"""Experimental geometry-aware evaluation for spherical computer vision.

This NumPy-only namespace evaluates results without inferring support or
validity from numeric values.  It does not download datasets, run experiments,
or promote the algorithm being measured.
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
from ._validation import solid_angle_weights

__all__ = [
    "BinaryRegionEvaluation",
    "DepthEvaluation",
    "EvaluationProtocol",
    "EvaluationRecord",
    "FeatureEvaluation",
    "PoseEvaluation",
    "evaluate_binary_regions",
    "evaluate_features",
    "evaluate_pose",
    "evaluate_radial_range",
    "solid_angle_weights",
]
