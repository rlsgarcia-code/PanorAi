"""Evidence objects and explicit acceptance policy for relative pose."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from numbers import Integral, Real
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelEvidence:
    """Comparable evidence for one geometric explanation."""

    model: str
    num_inliers: int
    inlier_ratio: float
    normalized_robust_score: float
    median_residual_deg: float


@dataclass(frozen=True, slots=True)
class ModelCompetitionReport:
    """Evidence for Essential, rotation-only and spherical homography models."""

    essential: ModelEvidence
    rotation_only: ModelEvidence
    spherical_homography: ModelEvidence
    preferred_model: str
    essential_score_margin: float


@dataclass(frozen=True, slots=True)
class PoseStabilityReport:
    """Dispersion after independent deterministic consensus re-estimation."""

    requested_trials: int
    successful_trials: int
    rotation_median_deg: float
    rotation_p90_deg: float
    translation_median_deg: float
    translation_p90_deg: float


@dataclass(frozen=True, slots=True)
class TranslationOrientationReport:
    """Cheirality evidence distinguishing the four Essential decompositions."""

    hypothesis_count: int
    provisional_correspondence_count: int
    best_positive_depth_count: int
    alternative_positive_depth_count: int
    positive_depth_fraction: float
    cheirality_margin: float
    median_triangulation_angle_deg: float
    ambiguous: bool


@dataclass(frozen=True, slots=True)
class RelativePoseQualityReport:
    """Inspectible pose evidence; ``raw_quality_score`` is not a probability."""

    num_correspondences: int
    num_inliers: int
    inlier_ratio: float
    occupied_cells_a: int
    occupied_cells_b: int
    coverage_entropy_a: float
    coverage_entropy_b: float
    median_residual_deg: float
    p90_residual_deg: float
    median_parallax_deg: float
    cheirality_ratio: float
    translation_orientation: TranslationOrientationReport
    stability: PoseStabilityReport
    model_competition: ModelCompetitionReport
    raw_quality_score: float
    accepted: bool
    rejection_reasons: tuple[str, ...]
    interface: str = "panorai-relative-pose-quality/v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_decision(
        self, policy: RelativePoseAcceptancePolicy
    ) -> RelativePoseQualityReport:
        reasons = policy.rejection_reasons(self)
        return replace(
            self,
            accepted=not reasons,
            rejection_reasons=reasons,
        )


@dataclass(frozen=True, slots=True)
class RelativePoseAcceptancePolicy:
    """Conservative, explicit evidence gate for an already estimated pose.

    This policy does not alter RANSAC or the returned pose.  It turns measured
    evidence into an auditable accept/reject decision.  Its output is not a
    calibrated probability.
    """

    min_inliers: int = 20
    min_inlier_ratio: float = 0.15
    min_occupied_cells: int = 4
    min_coverage_entropy: float = 0.25
    max_median_residual_deg: float = 0.5
    min_median_parallax_deg: float = 0.5
    min_cheirality_ratio: float = 0.6
    min_translation_orientation_margin: float = 0.05
    max_stability_rotation_p90_deg: float = 3.0
    max_stability_translation_p90_deg: float = 10.0
    min_essential_score_margin: float = 0.0
    require_essential_preferred: bool = True
    require_stability: bool = True
    interface: str = "panorai-relative-pose-acceptance/v1"

    def __post_init__(self) -> None:
        _positive_int("min_inliers", self.min_inliers)
        _unit_interval("min_inlier_ratio", self.min_inlier_ratio)
        _positive_int("min_occupied_cells", self.min_occupied_cells)
        _unit_interval("min_coverage_entropy", self.min_coverage_entropy)
        for name in (
            "max_median_residual_deg",
            "min_median_parallax_deg",
            "min_cheirality_ratio",
            "min_translation_orientation_margin",
            "max_stability_rotation_p90_deg",
            "max_stability_translation_p90_deg",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.min_cheirality_ratio > 1.0:
            raise ValueError("min_cheirality_ratio must not exceed 1")
        if self.min_translation_orientation_margin > 1.0:
            raise ValueError("min_translation_orientation_margin must not exceed 1")
        if (
            isinstance(self.min_essential_score_margin, bool)
            or not isinstance(self.min_essential_score_margin, Real)
            or not math.isfinite(float(self.min_essential_score_margin))
        ):
            raise ValueError("min_essential_score_margin must be finite")

    def rejection_reasons(self, report: RelativePoseQualityReport) -> tuple[str, ...]:
        reasons: list[str] = []
        if report.num_inliers < self.min_inliers:
            reasons.append("too-few-inliers")
        if report.inlier_ratio < self.min_inlier_ratio:
            reasons.append("low-inlier-ratio")
        if (
            min(report.occupied_cells_a, report.occupied_cells_b)
            < self.min_occupied_cells
        ):
            reasons.append("weak-angular-coverage")
        if (
            min(report.coverage_entropy_a, report.coverage_entropy_b)
            < self.min_coverage_entropy
        ):
            reasons.append("concentrated-inliers")
        if report.median_residual_deg > self.max_median_residual_deg:
            reasons.append("high-residual")
        if report.median_parallax_deg < self.min_median_parallax_deg:
            reasons.append("low-parallax")
        if report.cheirality_ratio < self.min_cheirality_ratio:
            reasons.append("weak-cheirality")
        if (
            report.translation_orientation.cheirality_margin
            < self.min_translation_orientation_margin
        ):
            reasons.append("ambiguous-translation-orientation")
        if self.require_stability and (
            report.stability.requested_trials == 0
            or report.stability.successful_trials < report.stability.requested_trials
        ):
            reasons.append("stability-unavailable")
        if report.stability.successful_trials > 0:
            if report.stability.rotation_p90_deg > self.max_stability_rotation_p90_deg:
                reasons.append("unstable-rotation")
            if (
                report.stability.translation_p90_deg
                > self.max_stability_translation_p90_deg
            ):
                reasons.append("unstable-translation")
        if (
            report.model_competition.essential_score_margin
            < self.min_essential_score_margin
        ):
            reasons.append("weak-essential-margin")
        if (
            self.require_essential_preferred
            and report.model_competition.preferred_model != "essential"
        ):
            reasons.append(
                f"competing-model:{report.model_competition.preferred_model}"
            )
        return tuple(reasons)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def raw_quality_score(
    *,
    inlier_ratio: float,
    coverage_entropy_a: float,
    coverage_entropy_b: float,
    median_residual_deg: float,
    residual_scale_deg: float,
    cheirality_ratio: float,
    translation_orientation_margin: float,
    stability_rotation_p90_deg: float,
    stability_translation_p90_deg: float,
    essential_score_margin: float,
) -> float:
    """Combine bounded evidence into a ranking score, not a probability."""

    residual_quality = math.exp(-median_residual_deg / max(residual_scale_deg, 1e-12))
    stability_quality = math.exp(
        -stability_rotation_p90_deg / 3.0 - stability_translation_p90_deg / 10.0
    )
    model_quality = 1.0 / (1.0 + math.exp(-8.0 * essential_score_margin))
    values = (
        min(max(inlier_ratio, 0.0), 1.0),
        min(max(coverage_entropy_a, 0.0), 1.0),
        min(max(coverage_entropy_b, 0.0), 1.0),
        residual_quality,
        min(max(cheirality_ratio, 0.0), 1.0),
        min(max(translation_orientation_margin, 0.0), 1.0),
        stability_quality,
        model_quality,
    )
    return float(sum(values) / len(values))


def _positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _unit_interval(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(float(value)) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
