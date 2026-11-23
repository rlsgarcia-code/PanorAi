"""Experimental metric relative pose from paired spherical RGB-D evidence.

The existing spherical relative-pose API deliberately returns only a unit
translation direction.  This module is a separate contract for callers that
already have metric *radial range* aligned with paired unit bearings.  It does
not interpret camera-Z depth, pixels, image projections, or application
semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Annotated, Any

import numpy as np

from panorai.model_fitting import RobustFitOptions, robust_fit

from ._sampling import FivePointSampler
from .relative_pose import (
    RelativePoseOptions,
    RelativePoseResult,
    estimate_relative_pose,
)


_INTERFACE = "panorai-metric-rgbd-relative-pose/v1-experimental"
_EPS = np.finfo(np.float64).eps

BearingArray = Annotated[np.ndarray, "panorai:bearings"]
DepthArray = Annotated[np.ndarray, "panorai:radial-depth"]
MaskArray = Annotated[np.ndarray, "panorai:mask"]

__all__ = [
    "MetricRgbdPoseAcceptancePolicy",
    "MetricRgbdPoseOptions",
    "MetricRgbdPoseQualityReport",
    "MetricRgbdPoseResult",
    "estimate_metric_rgbd_relative_pose",
]


@dataclass(frozen=True, slots=True)
class MetricRgbdPoseOptions:
    """Robust-estimation options in the caller's radial-depth unit.

    ``max_metric_error`` is intentionally required.  PanorAi cannot infer
    whether the supplied radial range is measured in metres, blocks, or
    another unit, so the caller must declare an appropriate residual scale.
    """

    max_metric_error: float
    confidence: float = 0.999
    min_trials: int = 32
    max_trials: int = 1000
    local_optimization_steps: int = 3
    min_inliers: int = 8
    random_seed: int = 0
    use_visual_consensus: bool = True

    def __post_init__(self) -> None:
        _positive_real("max_metric_error", self.max_metric_error)
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, Real)
            or not math.isfinite(float(self.confidence))
            or not 0.0 < float(self.confidence) < 1.0
        ):
            raise ValueError("confidence must be finite and in (0, 1)")
        for name in (
            "min_trials",
            "max_trials",
            "local_optimization_steps",
            "min_inliers",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            minimum = 0 if name == "local_optimization_steps" else 1
            if int(value) < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        if self.min_trials > self.max_trials:
            raise ValueError("min_trials must not exceed max_trials")
        if self.min_inliers < 3:
            raise ValueError("min_inliers must be at least 3")
        if isinstance(self.random_seed, bool) or not isinstance(
            self.random_seed, Integral
        ):
            raise TypeError("random_seed must be an integer")
        if not isinstance(self.use_visual_consensus, bool):
            raise TypeError("use_visual_consensus must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MetricRgbdPoseAcceptancePolicy:
    """Dimensionless evidence gate for a metric RGB-D pose estimate."""

    min_inliers: int = 8
    min_inlier_ratio: float = 0.5
    max_median_residual_ratio: float = 0.5
    max_p90_residual_ratio: float = 1.0
    max_translation_std_ratio: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.min_inliers, bool) or not isinstance(
            self.min_inliers, Integral
        ):
            raise TypeError("min_inliers must be an integer")
        if self.min_inliers < 3:
            raise ValueError("min_inliers must be at least 3")
        for name in (
            "min_inlier_ratio",
            "max_median_residual_ratio",
            "max_p90_residual_ratio",
            "max_translation_std_ratio",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.min_inlier_ratio > 1.0:
            raise ValueError("min_inlier_ratio must not exceed 1")

    def rejection_reasons(
        self,
        report: MetricRgbdPoseQualityReport,
        *,
        max_metric_error: float,
    ) -> tuple[str, ...]:
        reasons = []
        if report.num_inliers < self.min_inliers:
            reasons.append("insufficient-metric-inliers")
        if report.inlier_ratio < self.min_inlier_ratio:
            reasons.append("low-metric-inlier-ratio")
        if report.median_residual > self.max_median_residual_ratio * max_metric_error:
            reasons.append("high-median-metric-residual")
        if report.p90_residual > self.max_p90_residual_ratio * max_metric_error:
            reasons.append("high-p90-metric-residual")
        translation_variance = max(
            0.0,
            float(np.max(np.diag(report.covariance)[3:])),
        )
        translation_std = math.sqrt(translation_variance)
        if translation_std > self.max_translation_std_ratio * max_metric_error:
            reasons.append("weak-metric-translation-certainty")
        return tuple(reasons)


@dataclass(frozen=True, slots=True)
class MetricRgbdPoseQualityReport:
    """Inspectible metric consensus and local linearized uncertainty."""

    num_valid: int
    num_candidates: int
    num_inliers: int
    inlier_ratio: float
    median_residual: float
    p90_residual: float
    max_residual: float
    covariance: np.ndarray
    accepted: bool
    rejection_reasons: tuple[str, ...]
    interface: str = "panorai-metric-rgbd-pose-quality/v1-experimental"

    def __post_init__(self) -> None:
        covariance = _readonly_array(self.covariance, (6, 6))
        if not np.all(np.isfinite(covariance)):
            raise ValueError("covariance must be finite")
        object.__setattr__(self, "covariance", covariance)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["covariance"] = self.covariance.tolist()
        return payload


@dataclass(frozen=True, slots=True)
class MetricRgbdPoseResult:
    """One metric target-from-source SE(3) estimate.

    The convention is ``point_b = R @ point_a + t``.  Translation and
    residuals use exactly the unit of the supplied radial ranges.
    """

    rotation: np.ndarray
    translation: np.ndarray
    inlier_mask: np.ndarray
    residuals: np.ndarray
    depth_valid_mask: np.ndarray
    visual_pose: RelativePoseResult | None
    visual_consensus_used: bool
    visual_consensus_fallback_reason: str | None
    num_trials: int
    num_degenerate_samples: int
    quality_report: MetricRgbdPoseQualityReport
    acceptance_policy: MetricRgbdPoseAcceptancePolicy
    options: MetricRgbdPoseOptions
    interface: str = _INTERFACE
    robust_estimator: str = "panorai-deterministic-msac-rigid-3d-v1"

    def __post_init__(self) -> None:
        rotation = _readonly_array(self.rotation, (3, 3))
        translation = _readonly_array(self.translation, (3,))
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
            raise ValueError("rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
            raise ValueError("rotation must have determinant +1")
        inliers = _readonly_mask(self.inlier_mask)
        valid = _readonly_mask(self.depth_valid_mask)
        residuals = np.array(self.residuals, dtype=np.float64, copy=True)
        if residuals.shape != inliers.shape or valid.shape != inliers.shape:
            raise ValueError("masks and residuals must align with shape (N,)")
        if np.any(np.isnan(residuals)) or np.any(residuals < 0.0):
            raise ValueError("residuals must be non-negative and not NaN")
        residuals.setflags(write=False)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "inlier_mask", inliers)
        object.__setattr__(self, "depth_valid_mask", valid)
        object.__setattr__(self, "residuals", residuals)

    @property
    def R(self) -> np.ndarray:
        return self.rotation

    @property
    def t(self) -> np.ndarray:
        return self.translation

    @property
    def num_inliers(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "pose_convention": "frame-b-from-frame-a",
            "depth_semantics": "radial-range-along-unit-bearing",
            "translation": "metric-in-input-radial-depth-units",
            "residual": "euclidean-3d-in-input-radial-depth-units",
            "robust_estimator": self.robust_estimator,
            "visual_consensus_used": self.visual_consensus_used,
            "visual_consensus_fallback_reason": (self.visual_consensus_fallback_reason),
            "num_trials": self.num_trials,
            "num_degenerate_samples": self.num_degenerate_samples,
            "quality": self.quality_report.to_dict(),
            "acceptance_policy": asdict(self.acceptance_policy),
            "options": self.options.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class _RigidTransformModel:
    name: str = "rigid-3d"
    min_samples: int = 3
    residual_unit: str = "input-radial-depth-unit"

    def fit(
        self, source: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
            return None
        if len(source) < self.min_samples:
            return None
        centre_source = source.mean(axis=0)
        centre_target = target.mean(axis=0)
        source_zero = source - centre_source
        target_zero = target - centre_target
        if (
            np.linalg.matrix_rank(source_zero, tol=1e-10) < 2
            or np.linalg.matrix_rank(target_zero, tol=1e-10) < 2
        ):
            return None
        left, _, right = np.linalg.svd(target_zero.T @ source_zero)
        rotation = left @ np.diag((1.0, 1.0, np.linalg.det(left @ right))) @ right
        translation = centre_target - rotation @ centre_source
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            return None
        return rotation, translation

    def residuals(
        self,
        model: tuple[np.ndarray, np.ndarray],
        source: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        rotation, translation = model
        predicted = source @ rotation.T + translation
        return np.linalg.norm(predicted - target, axis=1)


def estimate_metric_rgbd_relative_pose(
    bearings_a: BearingArray,
    bearings_b: BearingArray,
    radial_depth_a: DepthArray,
    radial_depth_b: DepthArray,
    *,
    valid: MaskArray,
    options: MetricRgbdPoseOptions,
    visual_options: RelativePoseOptions | None = None,
    visual_sampler: FivePointSampler | None = None,
    acceptance_policy: MetricRgbdPoseAcceptancePolicy | None = None,
) -> MetricRgbdPoseResult | None:
    """Estimate metric target-from-source pose from paired spherical RGB-D.

    ``radial_depth_a`` and ``radial_depth_b`` are ranges measured from each
    camera centre along the corresponding unit bearing.  They are not pinhole
    camera-Z depth.  ``valid`` is mandatory so missing depth is never inferred
    from a magic numeric sentinel.

    When enabled, the existing PanorAi spherical relative-pose consensus is
    used as a correspondence prefilter.  Metric fitting falls back to all
    explicitly valid RGB-D pairs if visual estimation is unavailable or the
    filtered metric consensus fails.  The returned diagnostics expose which
    path was used.
    """

    b1, b2, d1, d2, valid_mask = _prepare_metric_inputs(
        bearings_a,
        bearings_b,
        radial_depth_a,
        radial_depth_b,
        valid,
    )
    if int(np.count_nonzero(valid_mask)) < options.min_inliers:
        return None
    points_a = b1 * d1[:, None]
    points_b = b2 * d2[:, None]

    visual_pose = None
    visual_used = False
    fallback_reason = None
    candidate_mask = valid_mask.copy()
    if options.use_visual_consensus:
        visual_pose = estimate_relative_pose(
            b1,
            b2,
            valid=valid_mask,
            options=visual_options or RelativePoseOptions(),
            sampler=visual_sampler,
        )
        if (
            visual_pose is not None
            and int(np.count_nonzero(visual_pose.inlier_mask & valid_mask))
            >= options.min_inliers
        ):
            candidate_mask &= visual_pose.inlier_mask
            visual_used = True
        else:
            fallback_reason = "visual-consensus-unavailable-or-too-small"

    fitted = _robust_metric_fit(points_a, points_b, candidate_mask, options)
    if fitted is None and visual_used:
        candidate_mask = valid_mask.copy()
        visual_used = False
        fallback_reason = "visual-consensus-metric-refit-failed"
        fitted = _robust_metric_fit(points_a, points_b, candidate_mask, options)
    if fitted is None:
        return None

    robust_result, candidate_indices = fitted
    rotation, translation = robust_result.model
    full_inliers = np.zeros(len(valid_mask), dtype=bool)
    full_inliers[candidate_indices] = robust_result.inlier_mask
    full_residuals = np.full(len(valid_mask), np.inf, dtype=np.float64)
    valid_indices = np.flatnonzero(valid_mask)
    full_residuals[valid_indices] = _RigidTransformModel().residuals(
        (rotation, translation),
        points_a[valid_indices],
        points_b[valid_indices],
    )
    inlier_residuals = full_residuals[full_inliers]
    covariance = _pose_covariance(
        rotation,
        points_a[full_inliers],
        inlier_residuals,
    )
    quality = MetricRgbdPoseQualityReport(
        num_valid=int(np.count_nonzero(valid_mask)),
        num_candidates=len(candidate_indices),
        num_inliers=int(np.count_nonzero(full_inliers)),
        inlier_ratio=float(
            np.count_nonzero(full_inliers) / np.count_nonzero(valid_mask)
        ),
        median_residual=float(np.median(inlier_residuals)),
        p90_residual=float(np.quantile(inlier_residuals, 0.9)),
        max_residual=float(np.max(inlier_residuals)),
        covariance=covariance,
        accepted=False,
        rejection_reasons=(),
    )
    policy = acceptance_policy or MetricRgbdPoseAcceptancePolicy(
        min_inliers=options.min_inliers
    )
    reasons = policy.rejection_reasons(
        quality,
        max_metric_error=options.max_metric_error,
    )
    quality = MetricRgbdPoseQualityReport(
        num_valid=quality.num_valid,
        num_candidates=quality.num_candidates,
        num_inliers=quality.num_inliers,
        inlier_ratio=quality.inlier_ratio,
        median_residual=quality.median_residual,
        p90_residual=quality.p90_residual,
        max_residual=quality.max_residual,
        covariance=quality.covariance,
        accepted=not reasons,
        rejection_reasons=reasons,
    )
    return MetricRgbdPoseResult(
        rotation=rotation,
        translation=translation,
        inlier_mask=full_inliers,
        residuals=full_residuals,
        depth_valid_mask=valid_mask,
        visual_pose=visual_pose,
        visual_consensus_used=visual_used,
        visual_consensus_fallback_reason=fallback_reason,
        num_trials=robust_result.num_trials,
        num_degenerate_samples=robust_result.num_degenerate_samples,
        quality_report=quality,
        acceptance_policy=policy,
        options=options,
    )


def _robust_metric_fit(
    points_a: np.ndarray,
    points_b: np.ndarray,
    candidate_mask: np.ndarray,
    options: MetricRgbdPoseOptions,
):
    indices = np.flatnonzero(candidate_mask)
    if len(indices) < options.min_inliers:
        return None
    try:
        result = robust_fit(
            _RigidTransformModel(),
            points_a[indices],
            points_b[indices],
            options=RobustFitOptions(
                residual_threshold=options.max_metric_error,
                method="msac",
                confidence=options.confidence,
                min_trials=options.min_trials,
                max_trials=options.max_trials,
                local_optimization_steps=options.local_optimization_steps,
                min_inliers=options.min_inliers,
                random_seed=options.random_seed,
            ),
        )
    except ValueError:
        return None
    return result, indices


def _prepare_metric_inputs(
    bearings_a: Any,
    bearings_b: Any,
    radial_depth_a: Any,
    radial_depth_b: Any,
    valid: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    b1 = np.array(bearings_a, dtype=np.float64, copy=True)
    b2 = np.array(bearings_b, dtype=np.float64, copy=True)
    d1 = np.array(radial_depth_a, dtype=np.float64, copy=True)
    d2 = np.array(radial_depth_b, dtype=np.float64, copy=True)
    if b1.ndim != 2 or b1.shape[1:] != (3,):
        raise ValueError("bearings_a must have shape (N, 3)")
    if b2.shape != b1.shape:
        raise ValueError("bearings_b must have the same shape as bearings_a")
    if d1.shape != (len(b1),) or d2.shape != (len(b1),):
        raise ValueError("radial depths must have shape (N,)")
    raw_valid = np.asarray(valid)
    if raw_valid.dtype != np.bool_:
        raise TypeError("valid must have boolean dtype")
    if raw_valid.shape != (len(b1),):
        raise ValueError("valid must have shape (N,)")
    valid_mask = np.array(raw_valid, dtype=bool, copy=True)
    if not np.all(np.isfinite(b1[valid_mask])) or not np.all(
        np.isfinite(b2[valid_mask])
    ):
        raise ValueError("valid bearings must be finite")
    if not np.all(np.isfinite(d1[valid_mask])) or not np.all(
        np.isfinite(d2[valid_mask])
    ):
        raise ValueError("valid radial depths must be finite")
    if np.any(d1[valid_mask] <= 0.0) or np.any(d2[valid_mask] <= 0.0):
        raise ValueError("valid radial depths must be positive")
    norms1 = np.linalg.norm(b1, axis=1)
    norms2 = np.linalg.norm(b2, axis=1)
    if np.any(norms1[valid_mask] <= 64 * _EPS) or np.any(
        norms2[valid_mask] <= 64 * _EPS
    ):
        raise ValueError("valid bearings must be non-zero")
    b1[valid_mask] /= norms1[valid_mask, None]
    b2[valid_mask] /= norms2[valid_mask, None]
    return b1, b2, d1, d2, valid_mask


def _pose_covariance(
    rotation: np.ndarray,
    points_a: np.ndarray,
    residuals: np.ndarray,
) -> np.ndarray:
    rotated = points_a @ rotation.T
    jacobian = np.empty((3 * len(points_a), 6), dtype=np.float64)
    for index, point in enumerate(rotated):
        x, y, z = point
        skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
        jacobian[3 * index : 3 * index + 3, :3] = -skew
        jacobian[3 * index : 3 * index + 3, 3:] = np.eye(3)
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    sigma = max(1.4826 * mad, 64 * _EPS)
    covariance = sigma * sigma * np.linalg.pinv(jacobian.T @ jacobian, rcond=1e-12)
    return 0.5 * (covariance + covariance.T)


def _readonly_array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape:
        raise ValueError(f"array must have shape {shape}")
    array.setflags(write=False)
    return array


def _readonly_mask(value: Any) -> np.ndarray:
    array = np.array(value, dtype=bool, copy=True)
    if array.ndim != 1:
        raise ValueError("mask must have shape (N,)")
    array.setflags(write=False)
    return array


def _positive_real(name: str, value: Any) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
