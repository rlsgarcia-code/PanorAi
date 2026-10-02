"""Experimental geometric estimators owned by PanorAi.

The package is intentionally independent from feature extraction backends.
Its relative-pose estimator consumes spherical bearings and does not import
OpenCV, PyCOLMAP, or Torch.
"""

from .relative_pose import (
    RelativePoseOptions,
    RelativePoseResult,
    SphericalRelativePoseEstimator,
    estimate_relative_pose,
    spherical_tangent_sampson_error,
)

__all__ = [
    "RelativePoseOptions",
    "RelativePoseResult",
    "SphericalRelativePoseEstimator",
    "estimate_relative_pose",
    "spherical_tangent_sampson_error",
]
