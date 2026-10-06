"""Experimental, auditable selection of spherical feature resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


_INTERFACE = "panorai-resolution-selection/v1"
_EPS: float = float(np.finfo(np.float64).eps)


@dataclass(frozen=True, slots=True)
class ResolutionSelectionPolicy:
    """Dataset-agnostic convergence thresholds for adjacent resolutions.

    Thresholds are ratios or angular task tolerances. No absolute feature or
    inlier count is embedded in this policy; absolute evidence admission stays
    with the relative-pose quality policy.
    """

    min_feature_repeatability: float = 0.85
    min_effective_inlier_retention: float = 0.80
    min_coverage_retention: float = 0.90
    target_rotation_accuracy_deg: float = 0.15
    stability_multiplier: float = 2.0
    max_repeatability_error_pixels: float = 2.0
    require_accepted_pose: bool = True
    interface: str = "panorai-resolution-selection-policy/v1"

    def __post_init__(self) -> None:
        for name in (
            "min_feature_repeatability",
            "min_effective_inlier_retention",
            "min_coverage_retention",
        ):
            _unit_interval(name, getattr(self, name))
        for name in (
            "target_rotation_accuracy_deg",
            "stability_multiplier",
            "max_repeatability_error_pixels",
        ):
            _nonnegative_finite(name, getattr(self, name))
        if not isinstance(self.require_accepted_pose, bool):
            raise TypeError("require_accepted_pose must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResolutionSelectionObservation:
    """Feature, match, and pose products measured at one ERP resolution."""

    erp_shape_hw: tuple[int, int]
    face_shape_hw: tuple[int, int]
    features_a: Any
    features_b: Any
    matches: Any
    pose: Any | None
    runtime_seconds: float | None = None
    inlier_weights: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "erp_shape_hw", _shape(self.erp_shape_hw, "erp_shape_hw")
        )
        object.__setattr__(
            self, "face_shape_hw", _shape(self.face_shape_hw, "face_shape_hw")
        )
        if self.runtime_seconds is not None:
            _nonnegative_finite("runtime_seconds", self.runtime_seconds)
        if self.inlier_weights is not None:
            weights = np.array(self.inlier_weights, dtype=np.float64, copy=True)
            if weights.ndim != 1 or weights.shape[0] != len(self.matches):
                raise ValueError("inlier_weights must have shape (len(matches),)")
            if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError("inlier_weights must be finite and non-negative")
            weights.setflags(write=False)
            object.__setattr__(self, "inlier_weights", weights)


@dataclass(frozen=True, slots=True)
class ResolutionLevelReport:
    """Compact evidence recorded for one evaluated resolution."""

    erp_shape_hw: tuple[int, int]
    face_shape_hw: tuple[int, int]
    feature_count_a: int
    feature_count_b: int
    match_count: int
    valid_match_count: int
    pose_returned: bool
    pose_accepted: bool
    rejection_reasons: tuple[str, ...]
    num_inliers: int
    effective_inlier_count: float
    coverage_entropy_a: float
    coverage_entropy_b: float
    rotation_stability_p90_deg: float | None
    rotation: tuple[tuple[float, float, float], ...] | None
    runtime_seconds: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResolutionTransitionReport:
    """Convergence evidence from one level to the next higher level."""

    lower_erp_shape_hw: tuple[int, int]
    higher_erp_shape_hw: tuple[int, int]
    angular_tolerance_deg: float
    repeatable_features_a: int
    repeatable_features_b: int
    feature_repeatability_a: float
    feature_repeatability_b: float
    effective_inlier_retention: float
    coverage_retention_a: float
    coverage_retention_b: float
    rotation_delta_deg: float | None
    rotation_tolerance_deg: float
    converged: bool
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResolutionSelectionReport:
    """Auditable result; selection never changes a pipeline silently."""

    levels: tuple[ResolutionLevelReport, ...]
    transitions: tuple[ResolutionTransitionReport, ...]
    selected_level_index: int | None
    decision: str
    converged_plateau: bool
    decision_reasons: tuple[str, ...]
    policy: ResolutionSelectionPolicy
    interface: str = _INTERFACE
    stability: str = "experimental"

    @property
    def selected_level(self) -> ResolutionLevelReport | None:
        if self.selected_level_index is None:
            return None
        return self.levels[self.selected_level_index]

    @property
    def selected_erp_shape_hw(self) -> tuple[int, int] | None:
        selected = self.selected_level
        return None if selected is None else selected.erp_shape_hw

    @property
    def selected_face_shape_hw(self) -> tuple[int, int] | None:
        selected = self.selected_level
        return None if selected is None else selected.face_shape_hw

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "stability": self.stability,
            "decision": self.decision,
            "converged_plateau": self.converged_plateau,
            "selected_level_index": self.selected_level_index,
            "selected_erp_shape_hw": self.selected_erp_shape_hw,
            "selected_face_shape_hw": self.selected_face_shape_hw,
            "decision_reasons": self.decision_reasons,
            "policy": self.policy.to_dict(),
            "levels": [level.to_dict() for level in self.levels],
            "transitions": [item.to_dict() for item in self.transitions],
        }


def select_feature_resolution(
    observations: Sequence[ResolutionSelectionObservation],
    *,
    policy: ResolutionSelectionPolicy | None = None,
) -> ResolutionSelectionReport:
    """Select the lowest spherical resolution on a converged level plateau.

    Observations may arrive in any order. They are sorted by ERP width, and
    every adjacent pair is compared. If no pair converges, the highest tested
    level is returned only when its pose passed its own quality gate; that
    fallback is explicitly marked as not adjacent-level converged.
    """

    policy = policy or ResolutionSelectionPolicy()
    if not observations:
        raise ValueError("observations must not be empty")
    ordered = tuple(sorted(observations, key=lambda item: item.erp_shape_hw[1]))
    widths = [item.erp_shape_hw[1] for item in ordered]
    if len(set(widths)) != len(widths):
        raise ValueError("observations must have unique ERP widths")
    if any(
        ordered[index].erp_shape_hw[0] >= ordered[index + 1].erp_shape_hw[0]
        for index in range(len(ordered) - 1)
    ):
        raise ValueError("ERP height and width must increase together")

    levels = tuple(_level_report(item) for item in ordered)
    transitions = tuple(
        _transition_report(lower, higher, levels[index], levels[index + 1], policy)
        for index, (lower, higher) in enumerate(zip(ordered, ordered[1:]))
    )
    for index, transition in enumerate(transitions):
        if transition.converged and all(item.converged for item in transitions[index:]):
            return ResolutionSelectionReport(
                levels=levels,
                transitions=transitions,
                selected_level_index=index,
                decision="converged-minimum",
                converged_plateau=True,
                decision_reasons=("first-converged-plateau",),
                policy=policy,
            )

    highest = levels[-1]
    usable = highest.pose_returned and (
        highest.pose_accepted or not policy.require_accepted_pose
    )
    if usable:
        return ResolutionSelectionReport(
            levels=levels,
            transitions=transitions,
            selected_level_index=len(levels) - 1,
            decision="highest-resolution-fallback",
            converged_plateau=False,
            decision_reasons=("no-converged-plateau",),
            policy=policy,
        )
    reasons = ["no-converged-plateau", "highest-resolution-unusable"]
    reasons.extend(highest.rejection_reasons)
    return ResolutionSelectionReport(
        levels=levels,
        transitions=transitions,
        selected_level_index=None,
        decision="no-selection",
        converged_plateau=False,
        decision_reasons=tuple(dict.fromkeys(reasons)),
        policy=policy,
    )


def _level_report(observation: ResolutionSelectionObservation) -> ResolutionLevelReport:
    matches = observation.matches
    valid = np.asarray(matches.valid, dtype=bool)
    if valid.shape != (len(matches),):
        raise ValueError("matches.valid must have shape (len(matches),)")
    if observation.pose is None:
        return ResolutionLevelReport(
            erp_shape_hw=observation.erp_shape_hw,
            face_shape_hw=observation.face_shape_hw,
            feature_count_a=len(observation.features_a),
            feature_count_b=len(observation.features_b),
            match_count=len(matches),
            valid_match_count=int(valid.sum()),
            pose_returned=False,
            pose_accepted=False,
            rejection_reasons=("pose-not-returned",),
            num_inliers=0,
            effective_inlier_count=0.0,
            coverage_entropy_a=0.0,
            coverage_entropy_b=0.0,
            rotation_stability_p90_deg=None,
            rotation=None,
            runtime_seconds=observation.runtime_seconds,
        )
    pose = observation.pose
    inliers = np.asarray(pose.inlier_mask, dtype=bool)
    if inliers.shape != (len(matches),):
        raise ValueError("pose.inlier_mask must have shape (len(matches),)")
    weights = (
        np.ones(len(matches), dtype=np.float64)
        if observation.inlier_weights is None
        else observation.inlier_weights
    )
    effective = _effective_count(np.asarray(weights, dtype=np.float64)[inliers])
    quality = pose.quality_report
    rotation = np.asarray(pose.rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("pose.rotation must be a finite 3x3 array")
    successful_stability_trials = int(
        getattr(quality.stability, "successful_trials", 0)
    )
    stability_p90 = (
        float(quality.stability.rotation_p90_deg)
        if successful_stability_trials > 0
        else None
    )
    if stability_p90 is not None:
        _nonnegative_finite("rotation stability p90", stability_p90)
    rotation_rows = tuple(
        (float(row[0]), float(row[1]), float(row[2])) for row in rotation
    )
    return ResolutionLevelReport(
        erp_shape_hw=observation.erp_shape_hw,
        face_shape_hw=observation.face_shape_hw,
        feature_count_a=len(observation.features_a),
        feature_count_b=len(observation.features_b),
        match_count=len(matches),
        valid_match_count=int(valid.sum()),
        pose_returned=True,
        pose_accepted=bool(quality.accepted),
        rejection_reasons=tuple(str(item) for item in quality.rejection_reasons),
        num_inliers=int(inliers.sum()),
        effective_inlier_count=effective,
        coverage_entropy_a=float(quality.coverage_entropy_a),
        coverage_entropy_b=float(quality.coverage_entropy_b),
        rotation_stability_p90_deg=stability_p90,
        rotation=rotation_rows,
        runtime_seconds=observation.runtime_seconds,
    )


def _transition_report(
    lower: ResolutionSelectionObservation,
    higher: ResolutionSelectionObservation,
    lower_level: ResolutionLevelReport,
    higher_level: ResolutionLevelReport,
    policy: ResolutionSelectionPolicy,
) -> ResolutionTransitionReport:
    tolerance = policy.max_repeatability_error_pixels * 360.0 / lower.erp_shape_hw[1]
    repeatable_a, repeatability_a = _bearing_repeatability(
        lower.features_a.bearings, higher.features_a.bearings, tolerance
    )
    repeatable_b, repeatability_b = _bearing_repeatability(
        lower.features_b.bearings, higher.features_b.bearings, tolerance
    )
    inlier_retention = _retention(
        lower_level.effective_inlier_count, higher_level.effective_inlier_count
    )
    coverage_a = _retention(
        lower_level.coverage_entropy_a, higher_level.coverage_entropy_a
    )
    coverage_b = _retention(
        lower_level.coverage_entropy_b, higher_level.coverage_entropy_b
    )
    high_stability = higher_level.rotation_stability_p90_deg or 0.0
    rotation_tolerance = max(
        policy.target_rotation_accuracy_deg,
        policy.stability_multiplier * high_stability,
    )
    rotation_delta = None
    if lower_level.rotation is not None and higher_level.rotation is not None:
        rotation_delta = _rotation_distance_deg(
            np.asarray(lower_level.rotation), np.asarray(higher_level.rotation)
        )

    reasons: list[str] = []
    if not lower_level.pose_returned:
        reasons.append("lower-pose-not-returned")
    if not higher_level.pose_returned:
        reasons.append("higher-pose-not-returned")
    if policy.require_accepted_pose and not lower_level.pose_accepted:
        reasons.append("lower-pose-rejected")
    if policy.require_accepted_pose and not higher_level.pose_accepted:
        reasons.append("higher-pose-rejected")
    if min(repeatability_a, repeatability_b) < policy.min_feature_repeatability:
        reasons.append("low-feature-repeatability")
    if inlier_retention < policy.min_effective_inlier_retention:
        reasons.append("low-effective-inlier-retention")
    if min(coverage_a, coverage_b) < policy.min_coverage_retention:
        reasons.append("low-coverage-retention")
    if rotation_delta is None:
        reasons.append("rotation-comparison-unavailable")
    elif rotation_delta > rotation_tolerance:
        reasons.append("rotation-not-converged")
    return ResolutionTransitionReport(
        lower_erp_shape_hw=lower.erp_shape_hw,
        higher_erp_shape_hw=higher.erp_shape_hw,
        angular_tolerance_deg=tolerance,
        repeatable_features_a=repeatable_a,
        repeatable_features_b=repeatable_b,
        feature_repeatability_a=repeatability_a,
        feature_repeatability_b=repeatability_b,
        effective_inlier_retention=inlier_retention,
        coverage_retention_a=coverage_a,
        coverage_retention_b=coverage_b,
        rotation_delta_deg=rotation_delta,
        rotation_tolerance_deg=rotation_tolerance,
        converged=not reasons,
        rejection_reasons=tuple(reasons),
    )


def _bearing_repeatability(
    lower: Any, higher: Any, tolerance_deg: float
) -> tuple[int, float]:
    a = _unit_bearings(lower, "lower feature bearings")
    b = _unit_bearings(higher, "higher feature bearings")
    if not len(a) or not len(b):
        return 0, 0.0
    a_to_b, a_cosine = _nearest_bearings(a, b)
    b_to_a, _ = _nearest_bearings(b, a)
    mutual = b_to_a[a_to_b] == np.arange(len(a))
    threshold = math.cos(math.radians(tolerance_deg))
    count = int(np.sum(mutual & (a_cosine >= threshold)))
    return count, count / min(len(a), len(b))


def _nearest_bearings(
    query: NDArray[np.float64], reference: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    indices = np.empty(len(query), dtype=np.int64)
    cosine = np.empty(len(query), dtype=np.float64)
    chunk_size = max(1, 2_000_000 // max(len(reference), 1))
    for start in range(0, len(query), chunk_size):
        stop = min(start + chunk_size, len(query))
        scores = query[start:stop] @ reference.T
        local = np.argmax(scores, axis=1)
        indices[start:stop] = local
        cosine[start:stop] = scores[np.arange(stop - start), local]
    clipped = np.asarray(np.clip(cosine, -1.0, 1.0), dtype=np.float64)
    return indices, clipped


def _unit_bearings(value: Any, name: str) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    norms = np.linalg.norm(result, axis=1)
    if np.any(norms <= _EPS):
        raise ValueError(f"{name} must be non-zero")
    return np.asarray(result / norms[:, None], dtype=np.float64)


def _effective_count(weights: NDArray[np.float64]) -> float:
    if not len(weights):
        return 0.0
    total = float(np.sum(weights))
    squared = float(np.dot(weights, weights))
    return 0.0 if squared <= _EPS else total * total / squared


def _retention(lower: float, higher: float) -> float:
    if higher <= _EPS:
        return 1.0
    return min(lower / higher, 1.0)


def _rotation_distance_deg(
    first: NDArray[np.float64], second: NDArray[np.float64]
) -> float:
    cosine = np.clip((np.trace(first @ second.T) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def _shape(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{name} must be a two-item sequence")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise TypeError(f"{name} entries must be integers")
        if int(item) <= 0:
            raise ValueError(f"{name} entries must be positive")
        result.append(int(item))
    return result[0], result[1]


def _unit_interval(name: str, value: Any) -> None:
    _nonnegative_finite(name, value)
    if float(value) > 1.0:
        raise ValueError(f"{name} must not exceed 1")


def _nonnegative_finite(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
