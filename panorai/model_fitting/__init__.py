"""Reusable robust model fitting for geometric computer vision.

The module is intentionally independent from stitching.  A model family owns
its minimal solver and residual definition; :func:`robust_fit` owns sampling,
MSAC/RANSAC scoring, deterministic tie breaking and local refitting.
"""

from ._robust import (
    HomographyModel,
    ModelFamily,
    RobustFitOptions,
    RobustFitResult,
    RotationModel,
    fit_homography,
    fit_rotation,
    robust_fit,
)

__all__ = [
    "HomographyModel",
    "ModelFamily",
    "RobustFitOptions",
    "RobustFitResult",
    "RotationModel",
    "fit_homography",
    "fit_rotation",
    "robust_fit",
]
