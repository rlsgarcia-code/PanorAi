"""Experimental spherical relative-pose estimation.

This module implements a PanorAi-owned first version of the calibrated
five-correspondence / locally-optimized RANSAC structure.  It estimates one
central relative pose from bearings expressed in the two panorama frames.

The implementation is deliberately isolated from OpenCV and PyCOLMAP.  Its
primary minimal kernel builds the calibrated cubic constraints in the
four-dimensional epipolar nullspace and enumerates real roots through an
action matrix.  A separately identified numerical root search remains a
fallback for singular polynomial charts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any

import numpy as np

from ._five_point import solve_five_point_essential
from ._native import (
    native_sampson_residuals,
    resolve_compute_backend,
)
from ._quality import (
    ModelCompetitionReport,
    ModelEvidence,
    PoseStabilityReport,
    RelativePoseAcceptancePolicy,
    RelativePoseQualityReport,
    TranslationOrientationReport,
    raw_quality_score,
)
from ._sampling import (
    FivePointSampler,
    FivePointSamplingDiagnostics,
    SpatiallyWeightedFivePointSampler,
)

_INTERFACE = "panorai-spherical-relative-pose/v1"
_EPS = np.finfo(np.float64).eps


@dataclass(frozen=True, slots=True)
class RelativePoseOptions:
    """Configuration for :func:`estimate_relative_pose`.

    ``max_angular_error_deg`` thresholds the first-order spherical tangent
    Sampson residual.  Its unit is degrees, not ERP or face pixels.
    """

    max_angular_error_deg: float = 1.0
    confidence: float = 0.999
    min_inlier_ratio: float = 0.1
    min_num_trials: int = 32
    max_num_trials: int = 1000
    dynamic_trials_multiplier: float = 3.0
    min_inliers: int = 8
    local_optimization_steps: int = 3
    minimal_solver_starts: int = 16
    minimal_solver_max_nfev: int = 100
    refinement_max_nfev: int = 100
    random_seed: int = 0
    min_median_parallax_deg: float = 0.25
    minimal_solver: str = "polynomial"
    scale_marginal_levels: int = 8
    scale_marginal_min_fraction: float = 0.2
    robust_refinement_steps: int = 2
    quality_cell_count: int = 20
    stability_trials: int = 6
    stability_fraction: float = 0.8
    stability_ransac_trials: int = 24
    model_competition_trials: int = 128
    model_competition_tie_margin: float = 0.01
    translation_orientation_method: str = "parallax-weighted"
    translation_orientation_parallax_scale_deg: float = 1.0
    compute_backend: str = "auto"
    hypothesis_ranking: str = "count-first"
    nonminimal_refit_max_steps: int = 0
    pose_refinement_method: str = "joint"
    decoupled_rotation_trials: int = 512
    decoupled_refit_max_steps: int = 100
    decoupled_translation_min_score_margin: float = 0.15

    def __post_init__(self) -> None:
        _finite_between("max_angular_error_deg", self.max_angular_error_deg, 0.0, 90.0)
        _finite_between("confidence", self.confidence, 0.0, 1.0)
        _finite_between(
            "min_inlier_ratio", self.min_inlier_ratio, 0.0, 1.0, closed=True
        )
        _positive_int("min_num_trials", self.min_num_trials, allow_zero=True)
        _positive_int("max_num_trials", self.max_num_trials)
        if self.min_num_trials > self.max_num_trials:
            raise ValueError("min_num_trials must not exceed max_num_trials")
        if (
            isinstance(self.dynamic_trials_multiplier, bool)
            or not isinstance(self.dynamic_trials_multiplier, (int, float))
            or not math.isfinite(float(self.dynamic_trials_multiplier))
            or self.dynamic_trials_multiplier < 1.0
        ):
            raise ValueError("dynamic_trials_multiplier must be finite and >= 1")
        _positive_int("min_inliers", self.min_inliers)
        if self.min_inliers < 5:
            raise ValueError("min_inliers must be at least 5")
        _positive_int(
            "local_optimization_steps", self.local_optimization_steps, allow_zero=True
        )
        _positive_int("minimal_solver_starts", self.minimal_solver_starts)
        if self.minimal_solver_starts < 8:
            raise ValueError("minimal_solver_starts must be at least 8")
        _positive_int("minimal_solver_max_nfev", self.minimal_solver_max_nfev)
        _positive_int("refinement_max_nfev", self.refinement_max_nfev)
        if isinstance(self.random_seed, bool) or not isinstance(
            self.random_seed, (int, np.integer)
        ):
            raise TypeError("random_seed must be an integer")
        _finite_between(
            "min_median_parallax_deg",
            self.min_median_parallax_deg,
            0.0,
            180.0,
            lower_closed=True,
            upper_closed=True,
        )
        if self.minimal_solver not in {"polynomial", "numerical"}:
            raise ValueError("minimal_solver must be 'polynomial' or 'numerical'")
        _positive_int("scale_marginal_levels", self.scale_marginal_levels)
        _finite_between(
            "scale_marginal_min_fraction",
            self.scale_marginal_min_fraction,
            0.0,
            1.0,
        )
        _positive_int(
            "robust_refinement_steps", self.robust_refinement_steps, allow_zero=True
        )
        if self.hypothesis_ranking not in {
            "count-first",
            "msac-first",
            "scale-marginal-first",
        }:
            raise ValueError(
                "hypothesis_ranking must be 'count-first', 'msac-first', or "
                "'scale-marginal-first'"
            )
        _positive_int(
            "nonminimal_refit_max_steps",
            self.nonminimal_refit_max_steps,
            allow_zero=True,
        )
        if self.pose_refinement_method not in {"joint", "decoupled"}:
            raise ValueError("pose_refinement_method must be 'joint' or 'decoupled'")
        _positive_int("decoupled_rotation_trials", self.decoupled_rotation_trials)
        _positive_int("decoupled_refit_max_steps", self.decoupled_refit_max_steps)
        _finite_between(
            "decoupled_translation_min_score_margin",
            self.decoupled_translation_min_score_margin,
            0.0,
            1.0,
            lower_closed=True,
            upper_closed=True,
        )
        _positive_int("quality_cell_count", self.quality_cell_count)
        _positive_int("stability_trials", self.stability_trials, allow_zero=True)
        _finite_between(
            "stability_fraction", self.stability_fraction, 0.0, 1.0, upper_closed=True
        )
        _positive_int("stability_ransac_trials", self.stability_ransac_trials)
        _positive_int("model_competition_trials", self.model_competition_trials)
        _finite_between(
            "model_competition_tie_margin",
            self.model_competition_tie_margin,
            0.0,
            1.0,
            lower_closed=True,
            upper_closed=True,
        )
        if self.translation_orientation_method not in {
            "positive-depth-count",
            "parallax-weighted",
        }:
            raise ValueError(
                "translation_orientation_method must be "
                "'positive-depth-count' or 'parallax-weighted'"
            )
        _finite_between(
            "translation_orientation_parallax_scale_deg",
            self.translation_orientation_parallax_scale_deg,
            0.0,
            90.0,
        )
        resolve_compute_backend(self.compute_backend)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DecoupledPoseReport:
    """Evidence from the Experimental rotation/translation split."""

    attempted: bool
    applied: bool
    rotation_inliers: int
    rotation_inlier_ratio: float
    translation_pool_size: int
    translation_consensus_size: int
    translation_score_margin: float
    rotation_refit_steps: int
    translation_refit_steps: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RelativePoseResult:
    """One panorama-2-from-panorama-1 relative-pose estimate.

    Translation has unit norm.  Its metric magnitude is unobservable from two
    calibrated central views and is therefore intentionally absent.
    """

    rotation: np.ndarray
    translation_direction: np.ndarray
    essential_matrix: np.ndarray
    inlier_mask: np.ndarray
    residuals_rad: np.ndarray
    num_inliers: int
    num_trials: int
    median_parallax_deg: float
    cheirality_ratio: float
    degenerate: bool
    degeneracy_reasons: tuple[str, ...]
    sampling_diagnostics: FivePointSamplingDiagnostics
    quality_report: RelativePoseQualityReport
    acceptance_policy: RelativePoseAcceptancePolicy
    options: RelativePoseOptions
    compute_backend: str = "numpy"
    interface: str = _INTERFACE
    minimal_solver: str = "panorai-polynomial-action-matrix-v1+numerical-chart-fallback"
    robust_estimator: str = "panorai-scale-marginal-lo-ransac-v1"
    consensus_refit_steps: int = 0
    decoupled_pose_report: DecoupledPoseReport | None = None

    def __post_init__(self) -> None:
        rotation = _readonly_array(self.rotation, (3, 3))
        translation = _readonly_array(self.translation_direction, (3,))
        essential = _readonly_array(self.essential_matrix, (3, 3))
        inliers = np.array(self.inlier_mask, dtype=bool, copy=True)
        residuals = np.array(self.residuals_rad, dtype=np.float64, copy=True)
        if inliers.ndim != 1 or residuals.shape != inliers.shape:
            raise ValueError("inlier_mask and residuals_rad must have shape (N,)")
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            raise ValueError("pose arrays must be finite")
        if not np.all(np.isfinite(essential)):
            raise ValueError("essential_matrix must be finite")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
            raise ValueError("rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
            raise ValueError("rotation must have determinant +1")
        if not np.isclose(np.linalg.norm(translation), 1.0, atol=1e-8):
            raise ValueError("translation_direction must have unit norm")
        if self.num_inliers != int(inliers.sum()):
            raise ValueError("num_inliers must equal the inlier-mask count")
        if (
            isinstance(self.consensus_refit_steps, bool)
            or not isinstance(self.consensus_refit_steps, (int, np.integer))
            or self.consensus_refit_steps < 0
        ):
            raise ValueError("consensus_refit_steps must be a non-negative integer")
        if self.compute_backend not in {"numpy", "native"}:
            raise ValueError("compute_backend must resolve to 'numpy' or 'native'")
        inliers.setflags(write=False)
        residuals.setflags(write=False)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation_direction", translation)
        object.__setattr__(self, "essential_matrix", essential)
        object.__setattr__(self, "inlier_mask", inliers)
        object.__setattr__(self, "residuals_rad", residuals)

    @property
    def R(self) -> np.ndarray:
        """Alias for the panorama-2-from-panorama-1 rotation."""

        return self.rotation

    @property
    def t(self) -> np.ndarray:
        """Alias for the unit translation direction in panorama-2 coordinates."""

        return self.translation_direction

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "minimal_solver": self.minimal_solver,
            "robust_estimator": self.robust_estimator,
            "pose_convention": "panorama-2-from-panorama-1",
            "translation": "unit-direction-only",
            "residual": "spherical-tangent-sampson-radians",
            "compute_backend": self.compute_backend,
            "num_inliers": self.num_inliers,
            "num_trials": self.num_trials,
            "consensus_refit_steps": self.consensus_refit_steps,
            "decoupled_pose": (
                None
                if self.decoupled_pose_report is None
                else self.decoupled_pose_report.to_dict()
            ),
            "median_parallax_deg": self.median_parallax_deg,
            "cheirality_ratio": self.cheirality_ratio,
            "degenerate": self.degenerate,
            "degeneracy_reasons": self.degeneracy_reasons,
            "sampling": self.sampling_diagnostics.to_dict(),
            "quality": self.quality_report.to_dict(),
            "acceptance_policy": self.acceptance_policy.to_dict(),
            "options": self.options.to_dict(),
        }


class SphericalRelativePoseEstimator:
    """Reusable façade for the Experimental PanorAi relative-pose estimator."""

    def __init__(
        self,
        options: RelativePoseOptions | None = None,
        *,
        sampler: FivePointSampler | None = None,
        quality_policy: RelativePoseAcceptancePolicy | None = None,
    ) -> None:
        self.options = options or RelativePoseOptions()
        self.sampler = sampler or SpatiallyWeightedFivePointSampler()
        self.quality_policy = quality_policy or RelativePoseAcceptancePolicy()

    def estimate(
        self,
        bearings_a: Any,
        bearings_b: Any | None = None,
        *,
        valid: Any | None = None,
        sampling_weights: Any | None = None,
    ) -> RelativePoseResult | None:
        return estimate_relative_pose(
            bearings_a,
            bearings_b,
            valid=valid,
            sampling_weights=sampling_weights,
            options=self.options,
            sampler=self.sampler,
            quality_policy=self.quality_policy,
        )

    def __repr__(self) -> str:
        return (
            "SphericalRelativePoseEstimator("
            f"max_angular_error_deg={self.options.max_angular_error_deg}, "
            f"max_num_trials={self.options.max_num_trials}, "
            f"random_seed={self.options.random_seed}, "
            f"compute_backend={self.options.compute_backend!r}, "
            f"sampler={self.sampler.name!r})"
        )


@dataclass(slots=True)
class _Hypothesis:
    rotation: np.ndarray
    translation: np.ndarray
    essential: np.ndarray
    inlier_mask: np.ndarray
    residuals: np.ndarray
    num_inliers: int
    residual_sum: float
    robust_score: float
    cheirality_ratio: float
    msac_cost: float = math.inf
    consensus_refit_steps: int = 0


@dataclass(slots=True)
class _DecoupledTranslationCandidate:
    score: float
    translation: np.ndarray
    consensus_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class _EssentialPoseCandidate:
    """One of the four decompositions of an Essential matrix."""

    positive_depth_count: int
    weighted_positive_depth_support: float
    total_parallax_weight: float
    reliable_correspondence_count: int
    rotation: np.ndarray
    translation: np.ndarray


def estimate_relative_pose(
    bearings_a: Any,
    bearings_b: Any | None = None,
    *,
    valid: Any | None = None,
    sampling_weights: Any | None = None,
    options: RelativePoseOptions | None = None,
    sampler: FivePointSampler | None = None,
    quality_policy: RelativePoseAcceptancePolicy | None = None,
) -> RelativePoseResult | None:
    """Estimate one central relative pose from paired spherical bearings.

    Parameters
    ----------
    bearings_a, bearings_b:
        Arrays with shape ``(N, 3)``. Alternatively, ``bearings_a`` may be a
        PanorAi correspondence object exposing ``bearings_a``, ``bearings_b``
        and ``valid``, in which case ``bearings_b`` is omitted. Finite non-zero
        rows are normalized on private copies, so inputs are never modified.
    valid:
        Optional explicit boolean mask.  Invalid rows are excluded and remain
        false in the returned inlier mask; validity is never inferred from a
        bearing's numeric value beyond mandatory finite/non-zero validation.
    sampling_weights:
        Optional non-negative proposal weights with shape ``(N,)``. They only
        influence selection of five-point minimal sets; all explicitly valid
        correspondences remain in hypothesis scoring and refinement. A
        correspondence object's ``weights`` field is used when present unless
        this argument overrides it.
    options:
        Robust-estimation and numerical-solver configuration.
    sampler:
        Injectable minimal-set proposal component. The default spatial sampler
        chooses only the five correspondences used to generate each
        hypothesis; it never prefilters the full scoring, inlier or refinement
        set. Use :class:`UniformFivePointSampler` for classic uniform proposals.

    Returns
    -------
    RelativePoseResult | None
        ``None`` when no hypothesis reaches ``min_inliers``.  A successful
        result contains ``R`` and only the direction of ``t``.
    """

    options = options or RelativePoseOptions()
    resolved_backend = resolve_compute_backend(options.compute_backend)
    quality_policy = quality_policy or RelativePoseAcceptancePolicy()
    if bearings_b is None:
        correspondence_object = bearings_a
        required = ("bearings_a", "bearings_b", "valid")
        if not all(hasattr(correspondence_object, name) for name in required):
            raise TypeError(
                "one-argument estimation requires an object exposing "
                "bearings_a, bearings_b, and valid"
            )
        if valid is not None:
            raise ValueError(
                "valid must be omitted when a correspondence object is provided"
            )
        bearings_a = correspondence_object.bearings_a
        bearings_b = correspondence_object.bearings_b
        valid = correspondence_object.valid
        if sampling_weights is None and hasattr(correspondence_object, "weights"):
            sampling_weights = correspondence_object.weights
    b1, b2, valid_mask = _prepare_bearings(bearings_a, bearings_b, valid)
    active = np.flatnonzero(valid_mask)
    required_inliers = max(
        5,
        options.min_inliers,
        math.ceil(options.min_inlier_ratio * active.size),
    )
    if active.size < required_inliers:
        return None

    sampler = sampler or SpatiallyWeightedFivePointSampler()
    if not all(
        hasattr(sampler, name)
        for name in ("name", "supports_uniform_trial_bound", "prepare", "describe")
    ):
        raise TypeError(
            "sampler must provide name, supports_uniform_trial_bound, "
            "prepare(), and describe()"
        )
    prepared_sampler = sampler.prepare(
        b1,
        b2,
        active,
        None if sampling_weights is None else np.asarray(sampling_weights),
    )

    threshold = math.radians(options.max_angular_error_deg)
    rng = np.random.default_rng(options.random_seed)
    dynamic_limit = options.max_num_trials
    best: _Hypothesis | None = None
    num_trials = 0
    samples = []
    failed_draws = 0

    while num_trials < options.max_num_trials:
        if num_trials >= dynamic_limit and num_trials >= options.min_num_trials:
            break
        sample_result = prepared_sampler.draw(rng)
        if sample_result is None:
            failed_draws += 1
            break
        samples.append(sample_result)
        sample = sample_result.indices
        essentials = _solve_five_correspondence_essential(
            b1[sample], b2[sample], options
        )
        num_trials += 1
        for essential in essentials:
            candidate = _score_essential(
                essential, b1, b2, valid_mask, threshold, options
            )
            if candidate is None or not _is_better(
                candidate, best, options.hypothesis_ranking
            ):
                continue
            candidate = _locally_optimize(
                candidate, b1, b2, valid_mask, threshold, options
            )
            if _is_better(candidate, best, options.hypothesis_ranking):
                best = candidate
                if prepared_sampler.supports_uniform_trial_bound:
                    dynamic_limit = min(
                        dynamic_limit,
                        _dynamic_trial_limit(
                            candidate.num_inliers,
                            active.size,
                            options.confidence,
                            options.dynamic_trials_multiplier,
                            options.max_num_trials,
                        ),
                    )

    if best is None or best.num_inliers < required_inliers:
        return None

    final = _refine_hypothesis(best, b1, b2, valid_mask, threshold, options)
    if _is_better(final, best, options.hypothesis_ranking):
        best = final
    refitted = _refit_consensus(best, b1, b2, valid_mask, threshold, options)
    if _is_better(refitted, best, options.hypothesis_ranking):
        best = refitted
        final = _refine_hypothesis(best, b1, b2, valid_mask, threshold, options)
        if _is_better(final, best, options.hypothesis_ranking):
            best = final

    decoupled_report = None
    if options.pose_refinement_method == "decoupled":
        decoupled, decoupled_report = _decoupled_pose_refinement(
            b1,
            b2,
            valid_mask,
            threshold,
            options,
        )
        if decoupled is not None:
            best = decoupled

    parallax = _median_parallax_deg(
        best.rotation, b1[best.inlier_mask], b2[best.inlier_mask]
    )
    orientation = _translation_orientation_report(
        best, b1, b2, valid_mask, threshold, options
    )
    reasons = []
    if parallax < options.min_median_parallax_deg:
        reasons.append("low-parallax")
    if best.cheirality_ratio < 0.5:
        reasons.append("weak-cheirality")
    if orientation.ambiguous:
        reasons.append("ambiguous-translation-orientation")
    decoupled_rejection = None
    if decoupled_report is not None and decoupled_report.reason in {
        "translation-unobservable",
        "translation-ambiguous",
        "translation-candidate-unscorable",
    }:
        decoupled_rejection = f"decoupled-{decoupled_report.reason}"
        reasons.append(decoupled_rejection)

    full_residuals = np.full(b1.shape[0], np.inf, dtype=np.float64)
    finite_residuals = spherical_tangent_sampson_error(
        b1[valid_mask],
        b2[valid_mask],
        best.essential,
        squared=False,
        backend=resolved_backend,
    )
    full_residuals[valid_mask] = finite_residuals
    competition = _model_competition_report(
        best, b1, b2, valid_mask, threshold, options
    )
    stability = _pose_stability_report(best, b1, b2, valid_mask, threshold, options)
    quality = _pose_quality_report(
        best,
        b1,
        b2,
        valid_mask,
        threshold,
        parallax,
        stability,
        competition,
        orientation,
        options,
    ).with_decision(quality_policy)
    if decoupled_rejection is not None and decoupled_rejection not in (
        quality.rejection_reasons
    ):
        quality = replace(
            quality,
            accepted=False,
            rejection_reasons=quality.rejection_reasons + (decoupled_rejection,),
        )
    return RelativePoseResult(
        rotation=best.rotation,
        translation_direction=best.translation,
        essential_matrix=best.essential,
        inlier_mask=best.inlier_mask,
        residuals_rad=full_residuals,
        num_inliers=best.num_inliers,
        num_trials=num_trials,
        median_parallax_deg=parallax,
        cheirality_ratio=best.cheirality_ratio,
        degenerate=bool(reasons),
        degeneracy_reasons=tuple(reasons),
        sampling_diagnostics=_sampling_diagnostics(
            sampler,
            samples,
            failed_draws,
            prepared_sampler.supports_uniform_trial_bound,
        ),
        quality_report=quality,
        acceptance_policy=quality_policy,
        options=options,
        compute_backend=resolved_backend,
        minimal_solver=(
            "panorai-polynomial-action-matrix-v1+numerical-chart-fallback"
            if options.minimal_solver == "polynomial"
            else "panorai-numerical-five-correspondence-v1"
        ),
        robust_estimator=_robust_estimator_name(options),
        consensus_refit_steps=best.consensus_refit_steps,
        decoupled_pose_report=decoupled_report,
    )


def spherical_tangent_sampson_error(
    bearings_a: Any,
    bearings_b: Any,
    essential_matrix: Any,
    *,
    squared: bool = False,
    backend: str = "auto",
) -> np.ndarray:
    """Return intrinsic first-order epipolar errors for central-camera rays.

    The constraint gradient is projected into each unit sphere's tangent plane,
    giving an approximately angular residual.  The returned unit is radians
    (or squared radians when ``squared=True``).
    """

    b1, b2, valid = _prepare_bearings(bearings_a, bearings_b, None)
    if not valid.all():  # pragma: no cover - mandatory validation catches this
        raise ValueError("bearings must be valid")
    essential = np.asarray(essential_matrix, dtype=np.float64)
    if essential.shape != (3, 3) or not np.all(np.isfinite(essential)):
        raise ValueError("essential_matrix must be a finite array with shape (3, 3)")
    if resolve_compute_backend(backend) == "native":
        return native_sampson_residuals(b1, b2, essential, squared=bool(squared))
    eb1 = b1 @ essential.T
    etb2 = b2 @ essential
    numerator = np.einsum("ni,ni->n", b2, eb1)
    grad1 = etb2 - b1 * np.einsum("ni,ni->n", b1, etb2)[:, None]
    grad2 = eb1 - b2 * np.einsum("ni,ni->n", b2, eb1)[:, None]
    denominator = np.sqrt(
        np.einsum("ni,ni->n", grad1, grad1) + np.einsum("ni,ni->n", grad2, grad2)
    )
    errors = np.full(b1.shape[0], np.inf, dtype=np.float64)
    nonzero = denominator > 64 * _EPS
    errors[nonzero] = np.abs(numerator[nonzero]) / denominator[nonzero]
    return errors * errors if squared else errors


def _signed_tangent_sampson_error(
    b1: np.ndarray,
    b2: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    backend: str = "auto",
) -> np.ndarray:
    essential = _skew(translation) @ rotation
    if resolve_compute_backend(backend) == "native":
        unsigned = native_sampson_residuals(b1, b2, essential, squared=False)
        signs = np.sign(np.einsum("ni,ij,nj->n", b2, essential, b1))
        return signs * unsigned
    eb1 = b1 @ essential.T
    etb2 = b2 @ essential
    numerator = np.einsum("ni,ni->n", b2, eb1)
    grad1 = etb2 - b1 * np.einsum("ni,ni->n", b1, etb2)[:, None]
    grad2 = eb1 - b2 * np.einsum("ni,ni->n", b2, eb1)[:, None]
    denominator = np.sqrt(
        np.einsum("ni,ni->n", grad1, grad1) + np.einsum("ni,ni->n", grad2, grad2)
    )
    denominator = np.maximum(denominator, 64 * _EPS)
    return numerator / denominator


def _solve_five_correspondence_essential(
    b1: np.ndarray, b2: np.ndarray, options: RelativePoseOptions
) -> list[np.ndarray]:
    """Solve all available polynomial roots, with an explicit numeric fallback."""

    if options.minimal_solver == "polynomial":
        candidates = solve_five_point_essential(b1, b2, backend=options.compute_backend)
        if candidates:
            return candidates
    return _solve_five_correspondence_essential_numerically(b1, b2, options)


def _solve_five_correspondence_essential_numerically(
    b1: np.ndarray, b2: np.ndarray, options: RelativePoseOptions
) -> list[np.ndarray]:
    """Fallback numerical root search for singular polynomial charts."""

    from scipy.optimize import least_squares

    equations = np.einsum("ni,nj->nij", b2, b1).reshape(5, 9)
    if np.linalg.matrix_rank(equations, tol=1e-10) < 5:
        return []
    _, _, vh = np.linalg.svd(equations, full_matrices=True)
    nullspace = vh[-4:].T

    def constraints(coefficients: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(coefficients)
        if norm <= 64 * _EPS:
            return np.full(11, 1e3, dtype=np.float64)
        unit = coefficients / norm
        essential = (nullspace @ unit).reshape(3, 3)
        eet = essential @ essential.T
        cubic = 2.0 * eet @ essential - np.trace(eet) * essential
        return np.concatenate(
            (cubic.ravel(), [np.linalg.det(essential), 0.1 * (norm - 1.0)])
        )

    candidates: list[np.ndarray] = []
    for seed in _minimal_solver_seeds(options.minimal_solver_starts):
        result = least_squares(
            constraints,
            seed,
            method="lm",
            max_nfev=options.minimal_solver_max_nfev,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        coefficient_norm = np.linalg.norm(result.x)
        if not result.success or coefficient_norm <= 64 * _EPS:
            continue
        essential = (nullspace @ (result.x / coefficient_norm)).reshape(3, 3)
        essential = _project_to_essential(essential)
        epipolar = np.abs(np.einsum("ni,ij,nj->n", b2, essential, b1))
        cubic = 2.0 * (essential @ essential.T) @ essential
        cubic -= np.trace(essential @ essential.T) * essential
        if epipolar.max(initial=0.0) > 2e-5 or np.linalg.norm(cubic) > 2e-6:
            continue
        if any(
            min(
                np.linalg.norm(essential - previous),
                np.linalg.norm(essential + previous),
            )
            < 1e-4
            for previous in candidates
        ):
            continue
        candidates.append(essential)
    return candidates


def _minimal_solver_seeds(count: int) -> tuple[np.ndarray, ...]:
    seeds: list[np.ndarray] = []
    identity = np.eye(4, dtype=np.float64)
    for index in range(4):
        seeds.extend((identity[index], -identity[index]))
    for bits in range(8):
        seed = np.ones(4, dtype=np.float64)
        seed[1:] = [1.0 if bits & (1 << axis) else -1.0 for axis in range(3)]
        seeds.append(seed / 2.0)
    if count > len(seeds):
        rng = np.random.default_rng(0x50414E4F524149)
        for seed in rng.normal(size=(count - len(seeds), 4)):
            seeds.append(seed / np.linalg.norm(seed))
    return tuple(np.array(seed, copy=True) for seed in seeds[:count])


def _score_essential(
    essential: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    options: RelativePoseOptions,
) -> _Hypothesis | None:
    residuals = np.full(b1.shape[0], np.inf, dtype=np.float64)
    residuals[valid] = spherical_tangent_sampson_error(
        b1[valid], b2[valid], essential, backend=options.compute_backend
    )
    provisional = valid & (residuals <= threshold)
    if provisional.sum() < 5:
        return None
    pose = _pose_from_essential(essential, b1[provisional], b2[provisional], options)
    if pose is None:
        return None
    rotation, translation = pose
    cheiral = np.zeros(b1.shape[0], dtype=bool)
    cheiral[valid] = _cheirality_mask(rotation, translation, b1[valid], b2[valid])
    inliers = provisional & cheiral
    count = int(inliers.sum())
    if count < 5:
        return None
    essential = _normalized_essential(rotation, translation)
    ratio = count / max(1, int(provisional.sum()))
    return _Hypothesis(
        rotation=rotation,
        translation=translation,
        essential=essential,
        inlier_mask=inliers,
        residuals=residuals,
        num_inliers=count,
        residual_sum=float(residuals[inliers].sum()),
        robust_score=_scale_marginal_score(
            residuals, valid & cheiral, threshold, options
        ),
        cheirality_ratio=float(ratio),
        msac_cost=_msac_cost(residuals, valid, threshold),
    )


def _score_pose(
    rotation: np.ndarray,
    translation: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    options: RelativePoseOptions,
) -> _Hypothesis | None:
    translation = translation / np.linalg.norm(translation)
    essential = _normalized_essential(rotation, translation)
    residuals = np.full(b1.shape[0], np.inf, dtype=np.float64)
    residuals[valid] = spherical_tangent_sampson_error(
        b1[valid], b2[valid], essential, backend=options.compute_backend
    )
    provisional = valid & (residuals <= threshold)
    cheiral = np.zeros(b1.shape[0], dtype=bool)
    cheiral[valid] = _cheirality_mask(rotation, translation, b1[valid], b2[valid])
    inliers = provisional & cheiral
    count = int(inliers.sum())
    if count < 5:
        return None
    return _Hypothesis(
        rotation=rotation,
        translation=translation,
        essential=essential,
        inlier_mask=inliers,
        residuals=residuals,
        num_inliers=count,
        residual_sum=float(residuals[inliers].sum()),
        robust_score=_scale_marginal_score(
            residuals, valid & cheiral, threshold, options
        ),
        cheirality_ratio=count / max(1, int(provisional.sum())),
        msac_cost=_msac_cost(residuals, valid, threshold),
    )


def _locally_optimize(
    hypothesis: _Hypothesis,
    b1: np.ndarray,
    b2: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    options: RelativePoseOptions,
) -> _Hypothesis:
    best = _refit_consensus(hypothesis, b1, b2, valid, threshold, options)
    for _ in range(options.local_optimization_steps):
        refined = _refine_hypothesis(
            best,
            b1,
            b2,
            valid,
            threshold,
            options,
            max_nfev=min(options.refinement_max_nfev, 50),
        )
        if not _is_better(refined, best, options.hypothesis_ranking):
            break
        best = refined
    return best


def _fit_nonminimal_essential(
    b1: np.ndarray,
    b2: np.ndarray,
    inliers: np.ndarray,
) -> np.ndarray | None:
    """Fit one calibrated Essential matrix from eight or more inlier rays."""

    selected_b1 = b1[inliers]
    selected_b2 = b2[inliers]
    if len(selected_b1) < 8:
        return None
    equations = np.einsum("ni,nj->nij", selected_b2, selected_b1).reshape(-1, 9)
    if np.linalg.matrix_rank(equations, tol=1e-10) < 8:
        return None
    _, _, vh = np.linalg.svd(equations, full_matrices=True)
    matrix = vh[-1].reshape(3, 3)
    if not np.all(np.isfinite(matrix)) or np.linalg.norm(matrix) <= 64 * _EPS:
        return None
    return _project_to_essential(matrix)


def _refit_consensus(
    hypothesis: _Hypothesis,
    b1: np.ndarray,
    b2: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    options: RelativePoseOptions,
) -> _Hypothesis:
    """Repeat all-inlier Essential refits until stable or the bounded cap."""

    best = hypothesis
    current = hypothesis
    initial_steps = best.consensus_refit_steps
    remaining = options.nonminimal_refit_max_steps - initial_steps
    if remaining <= 0:
        return best
    seen_masks = {current.inlier_mask.tobytes()}
    completed_steps = 0
    for _ in range(remaining):
        essential = _fit_nonminimal_essential(b1, b2, current.inlier_mask)
        if essential is None:
            break
        candidate = _score_essential(essential, b1, b2, valid, threshold, options)
        if candidate is None:
            break
        completed_steps += 1
        candidate.consensus_refit_steps = initial_steps + completed_steps
        if _is_better(candidate, best, options.hypothesis_ranking):
            best = candidate
        stable = np.array_equal(candidate.inlier_mask, current.inlier_mask)
        key = candidate.inlier_mask.tobytes()
        if stable or key in seen_masks:
            break
        seen_masks.add(key)
        current = candidate
    best.consensus_refit_steps = initial_steps + completed_steps
    return best


def _fit_wahba_rotation(b1: np.ndarray, b2: np.ndarray) -> np.ndarray | None:
    """Return the proper rotation aligning paired unit bearings."""

    if len(b1) < 3 or b1.shape != b2.shape:
        return None
    cross_covariance = b2.T @ b1
    if np.linalg.matrix_rank(cross_covariance, tol=1e-12) < 2:
        return None
    u, _, vh = np.linalg.svd(cross_covariance)
    correction = np.diag((1.0, 1.0, np.linalg.det(u @ vh)))
    rotation = u @ correction @ vh
    if not np.all(np.isfinite(rotation)):
        return None
    return rotation


def _rotation_alignment_residuals(
    rotation: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    rotated = b1 @ rotation.T
    cosine = np.clip(np.einsum("ni,ni->n", rotated, b2), -1.0, 1.0)
    return np.arccos(cosine)


def _decoupled_rotation_consensus(
    b1: np.ndarray,
    b2: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    options: RelativePoseOptions,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Estimate far-scene rotation using MSAC-ranked three-ray Wahba fits."""

    active = np.flatnonzero(valid)
    if len(active) < 3:
        return None
    consensus_threshold = 2.5 * threshold
    rng = np.random.default_rng(options.random_seed ^ 0x5741484241)

    def score(
        rotation: np.ndarray,
    ) -> tuple[tuple[float, int], np.ndarray]:
        residuals = _rotation_alignment_residuals(rotation, b1[active], b2[active])
        inliers = residuals <= consensus_threshold
        cost = float(np.minimum((residuals / consensus_threshold) ** 2, 1.0).sum())
        global_mask = np.zeros(len(valid), dtype=bool)
        global_mask[active] = inliers
        return (-cost, int(inliers.sum())), global_mask

    best: tuple[tuple[float, int], np.ndarray, np.ndarray] | None = None
    for _ in range(options.decoupled_rotation_trials):
        sample = rng.choice(active, size=3, replace=False)
        rotation = _fit_wahba_rotation(b1[sample], b2[sample])
        if rotation is None:
            continue
        key, inliers = score(rotation)
        if best is None or key > best[0]:
            best = (key, rotation, inliers)
    if best is None:
        return None

    current = best
    seen_masks: set[bytes] = set()
    steps = 0
    for _ in range(options.decoupled_refit_max_steps):
        mask_key = current[2].tobytes()
        if mask_key in seen_masks:
            break
        seen_masks.add(mask_key)
        rotation = _fit_wahba_rotation(b1[current[2]], b2[current[2]])
        if rotation is None:
            break
        key, inliers = score(rotation)
        steps += 1
        candidate = (key, rotation, inliers)
        if key > best[0]:
            best = candidate
        if np.array_equal(inliers, current[2]):
            break
        current = candidate
    return best[1], best[2], steps


def _score_decoupled_translation(
    rotation: np.ndarray,
    translation: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    pool: np.ndarray,
    scales: np.ndarray,
    options: RelativePoseOptions,
) -> _DecoupledTranslationCandidate:
    essential = _normalized_essential(rotation, translation)
    residuals = spherical_tangent_sampson_error(
        b1[pool],
        b2[pool],
        essential,
        backend=options.compute_backend,
    )
    gains = np.maximum(
        1.0 - (residuals[None, :] / scales[:, None]) ** 2,
        0.0,
    )
    candidates = []
    for sign in (1.0, -1.0):
        oriented = sign * translation
        cheiral, weights = _cheirality_evidence(
            rotation,
            oriented,
            b1[pool],
            b2[pool],
            parallax_scale_deg=options.translation_orientation_parallax_scale_deg,
        )
        score = float((gains * weights[None, :] * cheiral[None, :]).sum())
        consensus = (residuals <= scales[-1]) & cheiral
        candidates.append(_DecoupledTranslationCandidate(score, oriented, consensus))
    return max(
        candidates,
        key=lambda item: (item.score, int(item.consensus_mask.sum())),
    )


def _decoupled_translation_consensus(
    rotation: np.ndarray,
    rotation_inliers: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    options: RelativePoseOptions,
) -> tuple[np.ndarray | None, int, int, float, int, str]:
    """Estimate oriented translation from rotation-unexplained rays."""

    pool = np.flatnonzero(valid & ~rotation_inliers)
    if len(pool) < 5:
        return None, len(pool), 0, 0.0, 0, "translation-unobservable"

    rotated = b1 @ rotation.T
    normals = np.cross(rotated, b2)
    strengths = np.linalg.norm(normals, axis=1)
    if len(pool) > 64:
        strongest = np.argpartition(strengths[pool], -64)[-64:]
        pool = pool[strongest]
    scales = threshold * np.asarray((0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.5))
    candidates: list[_DecoupledTranslationCandidate] = []
    for first_index in range(len(pool)):
        for second_index in range(first_index + 1, len(pool)):
            translation = np.cross(
                normals[pool[first_index]], normals[pool[second_index]]
            )
            norm = np.linalg.norm(translation)
            if norm <= 64 * _EPS:
                continue
            candidates.append(
                _score_decoupled_translation(
                    rotation,
                    translation / norm,
                    b1,
                    b2,
                    pool,
                    scales,
                    options,
                )
            )
    if not candidates:
        return None, len(pool), 0, 0.0, 0, "translation-unobservable"

    best = max(
        candidates,
        key=lambda item: (item.score, int(item.consensus_mask.sum())),
    )
    current = best
    seen_masks: set[bytes] = set()
    refit_steps = 0
    for _ in range(options.decoupled_refit_max_steps):
        mask_key = current.consensus_mask.tobytes()
        if mask_key in seen_masks:
            break
        seen_masks.add(mask_key)
        selected = pool[current.consensus_mask]
        if len(selected) < 2:
            break
        rows = normals[selected] / np.maximum(strengths[selected, None], 64 * _EPS)
        _, _, vh = np.linalg.svd(rows, full_matrices=False)
        candidate = _score_decoupled_translation(
            rotation,
            vh[-1],
            b1,
            b2,
            pool,
            scales,
            options,
        )
        candidates.append(candidate)
        refit_steps += 1
        if (candidate.score, int(candidate.consensus_mask.sum())) > (
            best.score,
            int(best.consensus_mask.sum()),
        ):
            best = candidate
        if np.array_equal(candidate.consensus_mask, current.consensus_mask):
            break
        current = candidate

    alternatives = [
        item
        for item in candidates
        if _direction_distance_deg(item.translation, best.translation) > 10.0
    ]
    alternative_score = max((item.score for item in alternatives), default=0.0)
    score_margin = (best.score - alternative_score) / max(best.score, _EPS)
    consensus_size = int(best.consensus_mask.sum())
    if (
        consensus_size < 3
        or score_margin < options.decoupled_translation_min_score_margin
    ):
        return (
            None,
            len(pool),
            consensus_size,
            float(score_margin),
            refit_steps,
            "translation-ambiguous",
        )
    return (
        best.translation,
        len(pool),
        consensus_size,
        float(score_margin),
        refit_steps,
        "applied",
    )


def _decoupled_pose_refinement(
    b1: np.ndarray,
    b2: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    options: RelativePoseOptions,
) -> tuple[_Hypothesis | None, DecoupledPoseReport]:
    """Separate far-scene rotation from high-parallax translation evidence."""

    rotation_result = _decoupled_rotation_consensus(b1, b2, valid, threshold, options)
    if rotation_result is None:
        report = DecoupledPoseReport(
            True, False, 0, 0.0, 0, 0, 0.0, 0, 0, "rotation-fit-failed"
        )
        return None, report
    rotation, rotation_inliers, rotation_steps = rotation_result
    rotation_count = int(rotation_inliers.sum())
    rotation_ratio = rotation_count / max(1, int(valid.sum()))
    if rotation_ratio < 0.5:
        report = DecoupledPoseReport(
            True,
            False,
            rotation_count,
            rotation_ratio,
            0,
            0,
            0.0,
            rotation_steps,
            0,
            "rotation-model-inapplicable",
        )
        return None, report

    (
        translation,
        pool_size,
        translation_count,
        score_margin,
        translation_steps,
        reason,
    ) = _decoupled_translation_consensus(
        rotation,
        rotation_inliers,
        b1,
        b2,
        valid,
        threshold,
        options,
    )
    if translation is None:
        report = DecoupledPoseReport(
            True,
            False,
            rotation_count,
            rotation_ratio,
            pool_size,
            translation_count,
            score_margin,
            rotation_steps,
            translation_steps,
            reason,
        )
        return None, report

    hypothesis = _score_pose(
        rotation,
        translation,
        b1,
        b2,
        valid,
        threshold,
        options,
    )
    if hypothesis is None:
        reason = "translation-candidate-unscorable"
    report = DecoupledPoseReport(
        True,
        hypothesis is not None,
        rotation_count,
        rotation_ratio,
        pool_size,
        translation_count,
        score_margin,
        rotation_steps,
        translation_steps,
        "applied" if hypothesis is not None else reason,
    )
    return hypothesis, report


def _refine_hypothesis(
    hypothesis: _Hypothesis,
    b1: np.ndarray,
    b2: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    options: RelativePoseOptions,
    *,
    max_nfev: int | None = None,
) -> _Hypothesis:
    from scipy.optimize import least_squares

    inliers = hypothesis.inlier_mask
    if int(inliers.sum()) < 5:
        return hypothesis
    rotation0 = hypothesis.rotation
    translation0 = hypothesis.translation
    basis = _tangent_basis(translation0)

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rotation = _rotation_exp(parameters[:3]) @ rotation0
        translation = translation0 + basis @ parameters[3:]
        norm = np.linalg.norm(translation)
        if norm <= 64 * _EPS:
            translation = translation0
        else:
            translation = translation / norm
        return rotation, translation

    best = hypothesis
    parameters = np.zeros(5, dtype=np.float64)
    iteration_count = max(1, options.robust_refinement_steps)
    for _ in range(iteration_count):
        signed = _signed_tangent_sampson_error(
            b1[inliers],
            b2[inliers],
            best.rotation,
            best.translation,
            backend=options.compute_backend,
        )
        weights = _scale_marginal_weights(np.abs(signed), threshold, options)

        def residual(current: np.ndarray) -> np.ndarray:
            rotation, translation = unpack(current)
            return np.sqrt(weights) * _signed_tangent_sampson_error(
                b1[inliers],
                b2[inliers],
                rotation,
                translation,
                backend=options.compute_backend,
            )

        optimized = least_squares(
            residual,
            parameters,
            method="trf",
            max_nfev=max_nfev or options.refinement_max_nfev,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        if not optimized.success or not np.all(np.isfinite(optimized.x)):
            break
        parameters = optimized.x
        rotation, translation = unpack(parameters)
        scored = _score_pose(rotation, translation, b1, b2, valid, threshold, options)
        if scored is not None:
            scored.consensus_refit_steps = best.consensus_refit_steps
        if scored is not None and _is_better(scored, best, options.hypothesis_ranking):
            best = scored
    return best


def _pose_from_essential(
    essential: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    options: RelativePoseOptions,
) -> tuple[np.ndarray, np.ndarray] | None:
    candidates = _essential_pose_candidates(
        essential,
        b1,
        b2,
        parallax_scale_deg=options.translation_orientation_parallax_scale_deg,
    )
    if not candidates:
        return None
    best, _ = _select_essential_pose_candidate(
        candidates, options.translation_orientation_method
    )
    return best.rotation, best.translation


def _essential_pose_candidates(
    essential: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    *,
    parallax_scale_deg: float,
) -> list[_EssentialPoseCandidate]:
    u, _, vh = np.linalg.svd(essential)
    if np.linalg.det(u) < 0:
        u[:, -1] *= -1
    if np.linalg.det(vh) < 0:
        vh[-1, :] *= -1
    w = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    rotations = (u @ w @ vh, u @ w.T @ vh)
    translation = u[:, 2]
    candidates: list[_EssentialPoseCandidate] = []
    for rotation in rotations:
        if np.linalg.det(rotation) < 0:
            rotation = -rotation
        for direction in (translation, -translation):
            direction = direction / np.linalg.norm(direction)
            cheiral, weights = _cheirality_evidence(
                rotation,
                direction,
                b1,
                b2,
                parallax_scale_deg=parallax_scale_deg,
            )
            candidates.append(
                _EssentialPoseCandidate(
                    positive_depth_count=int(cheiral.sum()),
                    weighted_positive_depth_support=float(weights[cheiral].sum()),
                    total_parallax_weight=float(weights.sum()),
                    reliable_correspondence_count=int((weights >= 0.5).sum()),
                    rotation=rotation,
                    translation=direction,
                )
            )
    return candidates


def _orientation_candidate_key(
    candidate: _EssentialPoseCandidate, method: str
) -> tuple[float, ...]:
    if method == "positive-depth-count":
        # Keep the historical ordering exactly: ties retain the first SVD
        # decomposition instead of being resolved by a new secondary score.
        return (float(candidate.positive_depth_count),)
    return (
        candidate.weighted_positive_depth_support,
        float(candidate.positive_depth_count),
    )


def _select_essential_pose_candidate(
    candidates: list[_EssentialPoseCandidate], method: str
) -> tuple[_EssentialPoseCandidate, str]:
    if method == "positive-depth-count":
        return (
            max(
                candidates,
                key=lambda item: _orientation_candidate_key(item, method),
            ),
            method,
        )
    best = max(
        candidates,
        key=lambda item: _orientation_candidate_key(item, method),
    )
    if best.reliable_correspondence_count >= 5:
        return best, method
    # Fewer than a minimal set of rays at or above the declared parallax scale
    # cannot support a new oriented-translation decision. Preserve the
    # historical axis representative for compatibility, but report zero
    # decision margin so the quality policy abstains.
    return (
        max(
            candidates,
            key=lambda item: _orientation_candidate_key(item, "positive-depth-count"),
        ),
        "positive-depth-count-fallback",
    )


def _translation_orientation_report(
    hypothesis: _Hypothesis,
    b1: np.ndarray,
    b2: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    options: RelativePoseOptions,
) -> TranslationOrientationReport:
    provisional = valid & (hypothesis.residuals <= threshold)
    count = int(provisional.sum())
    candidates = _essential_pose_candidates(
        hypothesis.essential,
        b1[provisional],
        b2[provisional],
        parallax_scale_deg=options.translation_orientation_parallax_scale_deg,
    )
    selected, applied_method = _select_essential_pose_candidate(
        candidates, options.translation_orientation_method
    )
    ranking_method = (
        "positive-depth-count"
        if applied_method == "positive-depth-count-fallback"
        else applied_method
    )
    ranked = sorted(
        candidates,
        key=lambda item: _orientation_candidate_key(item, ranking_method),
        reverse=True,
    )
    ranked = [selected, *(item for item in ranked if item is not selected)]
    best = ranked[0] if ranked else None
    alternative = ranked[1] if len(ranked) > 1 else None
    best_count = 0 if best is None else best.positive_depth_count
    alternative_count = 0 if alternative is None else alternative.positive_depth_count
    best_weighted = 0.0 if best is None else best.weighted_positive_depth_support
    alternative_weighted = (
        0.0 if alternative is None else alternative.weighted_positive_depth_support
    )
    raw_margin = (best_count - alternative_count) / max(1, count)
    effective_weight = max(
        0.0 if best is None else best.total_parallax_weight,
        0.0 if alternative is None else alternative.total_parallax_weight,
        np.finfo(np.float64).eps,
    )
    weighted_margin = (best_weighted - alternative_weighted) / effective_weight
    margin = (
        0.0
        if applied_method == "positive-depth-count-fallback"
        else raw_margin if applied_method == "positive-depth-count" else weighted_margin
    )
    parallax = _median_parallax_deg(
        hypothesis.rotation, b1[hypothesis.inlier_mask], b2[hypothesis.inlier_mask]
    )
    return TranslationOrientationReport(
        hypothesis_count=len(candidates),
        provisional_correspondence_count=count,
        best_positive_depth_count=best_count,
        alternative_positive_depth_count=alternative_count,
        positive_depth_fraction=best_count / max(1, count),
        cheirality_margin=float(margin),
        median_triangulation_angle_deg=parallax,
        ambiguous=margin < 0.05,
        selection_method=applied_method,
        parallax_weight_scale_deg=options.translation_orientation_parallax_scale_deg,
        best_weighted_positive_depth_support=best_weighted,
        alternative_weighted_positive_depth_support=alternative_weighted,
        weighted_cheirality_margin=float(weighted_margin),
        raw_cheirality_margin=float(raw_margin),
        effective_correspondence_weight=float(effective_weight),
        reliable_correspondence_count=(
            0 if best is None else best.reliable_correspondence_count
        ),
    )


def _cheirality_mask(
    rotation: np.ndarray,
    translation: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    mask, _ = _cheirality_evidence(
        rotation,
        translation,
        b1,
        b2,
        parallax_scale_deg=1.0,
    )
    return mask


def _cheirality_evidence(
    rotation: np.ndarray,
    translation: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    *,
    parallax_scale_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive-depth decisions and bounded parallax information.

    For two unit rays, triangulation sensitivity is proportional to
    ``1 / sin(theta)``.  The bounded weight below therefore suppresses votes
    whose sign is dominated by angular noise while capping every reliable
    correspondence at one vote.  It is used only to choose among the four
    decompositions; the public inlier mask retains the historical binary
    cheirality definition.
    """

    rotated = b1 @ rotation.T
    dot = np.einsum("ni,ni->n", rotated, b2)
    denominator = 1.0 - dot * dot
    at = rotated @ translation
    bt = b2 @ translation
    depth1 = np.full(dot.shape, -np.inf, dtype=np.float64)
    depth2 = np.full(dot.shape, -np.inf, dtype=np.float64)
    stable = denominator > 1e-12
    depth1[stable] = (-at[stable] + dot[stable] * bt[stable]) / denominator[stable]
    depth2[stable] = (-dot[stable] * at[stable] + bt[stable]) / denominator[stable]
    cheiral = stable & (depth1 > 1e-10) & (depth2 > 1e-10)
    scale_sine = math.sin(math.radians(parallax_scale_deg))
    weights = np.zeros(dot.shape, dtype=np.float64)
    weights[stable] = denominator[stable] / (
        denominator[stable] + scale_sine * scale_sine
    )
    return cheiral, weights


def _prepare_bearings(
    bearings_a: Any, bearings_b: Any, valid: Any | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b1 = np.array(bearings_a, dtype=np.float64, copy=True)
    b2 = np.array(bearings_b, dtype=np.float64, copy=True)
    if b1.ndim != 2 or b1.shape[1:] != (3,):
        raise ValueError("bearings_a must have shape (N, 3)")
    if b2.shape != b1.shape:
        raise ValueError("bearings_b must have the same shape as bearings_a")
    if valid is None:
        valid_mask = np.ones(b1.shape[0], dtype=bool)
    else:
        raw_valid = np.asarray(valid)
        if raw_valid.dtype != np.bool_:
            raise TypeError("valid must have boolean dtype")
        if raw_valid.shape != (b1.shape[0],):
            raise ValueError("valid must have shape (N,)")
        valid_mask = np.array(raw_valid, dtype=bool, copy=True)
    if not np.all(np.isfinite(b1[valid_mask])) or not np.all(
        np.isfinite(b2[valid_mask])
    ):
        raise ValueError("valid bearings must be finite")
    norms1 = np.linalg.norm(b1, axis=1)
    norms2 = np.linalg.norm(b2, axis=1)
    if np.any(norms1[valid_mask] <= 64 * _EPS) or np.any(
        norms2[valid_mask] <= 64 * _EPS
    ):
        raise ValueError("valid bearings must be non-zero")
    b1[valid_mask] /= norms1[valid_mask, None]
    b2[valid_mask] /= norms2[valid_mask, None]
    return b1, b2, valid_mask


def _project_to_essential(matrix: np.ndarray) -> np.ndarray:
    u, singular, vh = np.linalg.svd(matrix)
    if np.linalg.det(u @ vh) < 0:
        u[:, -1] *= -1
    value = 0.5 * (singular[0] + singular[1])
    essential = u @ np.diag((value, value, 0.0)) @ vh
    norm = np.linalg.norm(essential)
    return essential / norm


def _normalized_essential(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    essential = _skew(translation / np.linalg.norm(translation)) @ rotation
    return essential / np.linalg.norm(essential)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def _rotation_exp(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    angle = np.linalg.norm(vector)
    cross = _skew(vector)
    if angle < 1e-8:
        return np.eye(3) + cross + 0.5 * cross @ cross
    unit_cross = cross / angle
    return (
        np.eye(3)
        + math.sin(angle) * unit_cross
        + (1.0 - math.cos(angle)) * (unit_cross @ unit_cross)
    )


def _tangent_basis(direction: np.ndarray) -> np.ndarray:
    direction = direction / np.linalg.norm(direction)
    axis = np.zeros(3, dtype=np.float64)
    axis[int(np.argmin(np.abs(direction)))] = 1.0
    first = np.cross(direction, axis)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    return np.stack((first, second), axis=1)


def _median_parallax_deg(rotation: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> float:
    if b1.size == 0:
        return 0.0
    rotated = b1 @ rotation.T
    cosine = np.clip(np.einsum("ni,ni->n", rotated, b2), -1.0, 1.0)
    return float(np.degrees(np.median(np.arccos(cosine))))


def _scale_marginal_weights(
    residuals: np.ndarray,
    threshold: float,
    options: RelativePoseOptions,
) -> np.ndarray:
    scales = np.linspace(
        threshold * options.scale_marginal_min_fraction,
        threshold,
        options.scale_marginal_levels,
    )
    normalized = residuals[:, None] / scales[None, :]
    weights = np.mean(
        np.square(np.clip(1.0 - normalized * normalized, 0.0, 1.0)), axis=1
    )
    return np.maximum(weights, 1e-6)


def _scale_marginal_score(
    residuals: np.ndarray,
    eligible: np.ndarray,
    threshold: float,
    options: RelativePoseOptions,
) -> float:
    selected = residuals[eligible]
    if selected.size == 0:
        return 0.0
    return float(_scale_marginal_weights(selected, threshold, options).sum())


def _msac_cost(
    residuals: np.ndarray,
    eligible: np.ndarray,
    threshold: float,
) -> float:
    """Return the classic normalized truncated-quadratic MSAC cost."""

    normalized = residuals[eligible] / threshold
    return float(np.minimum(normalized * normalized, 1.0).sum())


def _model_competition_report(
    essential: _Hypothesis,
    b1: np.ndarray,
    b2: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    options: RelativePoseOptions,
) -> ModelCompetitionReport:
    valid_count = max(1, int(valid.sum()))
    essential_evidence = _model_evidence(
        "essential",
        essential.residuals,
        essential.inlier_mask,
        essential.robust_score,
        valid_count,
    )
    rotation_evidence = _rotation_model_evidence(b1, b2, valid, threshold, options)
    homography_evidence = _homography_model_evidence(b1, b2, valid, threshold, options)
    evidence = (essential_evidence, rotation_evidence, homography_evidence)
    best_score = max(item.normalized_robust_score for item in evidence)
    competitive = {
        item.model
        for item in evidence
        if item.normalized_robust_score
        >= best_score - options.model_competition_tie_margin
    }
    # A near-tie is itself evidence of ambiguity. Prefer the more restrictive
    # competing explanation so the acceptance policy cannot silently promote a
    # pure-rotation or projective-degenerate case as a trustworthy Essential
    # pose. The estimated R,t remains available for inspection.
    preferred_name = next(
        name
        for name in ("rotation-only", "spherical-homography", "essential")
        if name in competitive
    )
    competing_score = max(
        rotation_evidence.normalized_robust_score,
        homography_evidence.normalized_robust_score,
    )
    return ModelCompetitionReport(
        essential=essential_evidence,
        rotation_only=rotation_evidence,
        spherical_homography=homography_evidence,
        preferred_model=preferred_name,
        essential_score_margin=(
            essential_evidence.normalized_robust_score - competing_score
        ),
    )


def _model_evidence(
    name: str,
    residuals: np.ndarray,
    inliers: np.ndarray,
    robust_score: float,
    valid_count: int,
) -> ModelEvidence:
    count = int(inliers.sum())
    median = float(np.degrees(np.median(residuals[inliers]))) if count else 180.0
    return ModelEvidence(
        model=name,
        num_inliers=count,
        inlier_ratio=count / valid_count,
        normalized_robust_score=robust_score / valid_count,
        median_residual_deg=median,
    )


def _rotation_model_evidence(
    b1: np.ndarray,
    b2: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    options: RelativePoseOptions,
) -> ModelEvidence:
    active = np.flatnonzero(valid)
    if len(active) < 3:
        return ModelEvidence("rotation-only", 0, 0.0, 0.0, 180.0)
    rng = np.random.default_rng(options.random_seed ^ 0x524F5441)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for _ in range(options.model_competition_trials):
        indices = rng.choice(active, size=3, replace=False)
        rotation = _wahba_rotation(b1[indices], b2[indices])
        residuals = _rotation_residuals(rotation, b1, b2)
        score = _scale_marginal_score(residuals, valid, threshold, options)
        if best is None or score > best[0]:
            best = (score, rotation, residuals)
    assert best is not None
    inliers = valid & (best[2] <= threshold)
    if int(inliers.sum()) >= 3:
        rotation = _wahba_rotation(b1[inliers], b2[inliers])
        residuals = _rotation_residuals(rotation, b1, b2)
        score = _scale_marginal_score(residuals, valid, threshold, options)
        if score >= best[0]:
            best = (score, rotation, residuals)
            inliers = valid & (residuals <= threshold)
    return _model_evidence(
        "rotation-only", best[2], inliers, best[0], max(1, len(active))
    )


def _wahba_rotation(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(b2.T @ b1)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vh)
    return u @ correction @ vh


def _rotation_residuals(
    rotation: np.ndarray, b1: np.ndarray, b2: np.ndarray
) -> np.ndarray:
    cosine = np.clip(np.einsum("ni,ni->n", b1 @ rotation.T, b2), -1.0, 1.0)
    return np.arccos(cosine)


def _homography_model_evidence(
    b1: np.ndarray,
    b2: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    options: RelativePoseOptions,
) -> ModelEvidence:
    active = np.flatnonzero(valid)
    if len(active) < 4:
        return ModelEvidence("spherical-homography", 0, 0.0, 0.0, 180.0)
    rng = np.random.default_rng(options.random_seed ^ 0x484F4D4F)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for _ in range(options.model_competition_trials):
        indices = rng.choice(active, size=4, replace=False)
        homography = _fit_spherical_homography(b1[indices], b2[indices])
        if homography is None:
            continue
        residuals = _homography_residuals(homography, b1, b2)
        score = _scale_marginal_score(residuals, valid, threshold, options)
        if best is None or score > best[0]:
            best = (score, homography, residuals)
    if best is None:
        return ModelEvidence("spherical-homography", 0, 0.0, 0.0, 180.0)
    inliers = valid & (best[2] <= threshold)
    if int(inliers.sum()) >= 4:
        homography = _fit_spherical_homography(b1[inliers], b2[inliers])
        if homography is not None:
            residuals = _homography_residuals(homography, b1, b2)
            score = _scale_marginal_score(residuals, valid, threshold, options)
            if score >= best[0]:
                best = (score, homography, residuals)
                inliers = valid & (residuals <= threshold)
    return _model_evidence(
        "spherical-homography", best[2], inliers, best[0], max(1, len(active))
    )


def _fit_spherical_homography(b1: np.ndarray, b2: np.ndarray) -> np.ndarray | None:
    rows = []
    for source, target in zip(b1, b2, strict=True):
        rows.extend(np.kron(_skew(target), source).reshape(3, 9))
    design = np.asarray(rows, dtype=np.float64)
    if np.linalg.matrix_rank(design, tol=1e-10) < 8:
        return None
    _, _, vh = np.linalg.svd(design, full_matrices=False)
    homography = vh[-1].reshape(3, 3)
    predictions = b1 @ homography.T
    norms = np.linalg.norm(predictions, axis=1)
    if np.any(norms <= 64 * _EPS):
        return None
    predictions /= norms[:, None]
    if np.median(np.einsum("ni,ni->n", predictions, b2)) < 0.0:
        homography = -homography
    return homography / np.linalg.norm(homography)


def _homography_residuals(
    homography: np.ndarray, b1: np.ndarray, b2: np.ndarray
) -> np.ndarray:
    predictions = b1 @ homography.T
    norms = np.linalg.norm(predictions, axis=1)
    residuals = np.full(len(b1), math.inf, dtype=np.float64)
    stable = norms > 64 * _EPS
    predictions[stable] /= norms[stable, None]
    cosine = np.clip(np.einsum("ni,ni->n", predictions[stable], b2[stable]), -1.0, 1.0)
    residuals[stable] = np.arccos(cosine)
    return residuals


def _pose_stability_report(
    best: _Hypothesis,
    b1: np.ndarray,
    b2: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    options: RelativePoseOptions,
) -> PoseStabilityReport:
    if options.stability_trials == 0:
        return PoseStabilityReport(0, 0, 180.0, 180.0, 180.0, 180.0)
    # Re-estimate from the discovered consensus rather than starting each
    # stability trial from the winning pose. Drawing from every observation
    # would mostly measure whether the deliberately small auxiliary RANSAC
    # budget happened to draw five inliers, not whether the consensus supports
    # a repeatable pose. Candidates are still scored and refined against every
    # valid correspondence so a competing, better-supported solution can win.
    active = np.flatnonzero(best.inlier_mask)
    subset_size = max(5, int(math.ceil(options.stability_fraction * len(active))))
    rng = np.random.default_rng(options.random_seed ^ 0x53544142)
    rotation_errors: list[float] = []
    translation_errors: list[float] = []
    for _ in range(options.stability_trials):
        selected = rng.choice(active, size=subset_size, replace=False)
        trial_best: _Hypothesis | None = None
        for _ in range(options.stability_ransac_trials):
            sample = rng.choice(selected, size=5, replace=False)
            for essential in _solve_five_correspondence_essential(
                b1[sample], b2[sample], options
            ):
                candidate = _score_essential(
                    essential, b1, b2, valid, threshold, options
                )
                if _is_better(candidate, trial_best, options.hypothesis_ranking):
                    trial_best = candidate
        if trial_best is None:
            continue
        refined = _refine_hypothesis(
            trial_best,
            b1,
            b2,
            valid,
            threshold,
            options,
            max_nfev=min(40, options.refinement_max_nfev),
        )
        rotation_errors.append(_rotation_distance_deg(refined.rotation, best.rotation))
        translation_errors.append(
            _direction_distance_deg(refined.translation, best.translation)
        )
    if not rotation_errors:
        return PoseStabilityReport(
            options.stability_trials,
            0,
            180.0,
            180.0,
            180.0,
            180.0,
        )
    return PoseStabilityReport(
        requested_trials=options.stability_trials,
        successful_trials=len(rotation_errors),
        rotation_median_deg=float(np.median(rotation_errors)),
        rotation_p90_deg=float(np.quantile(rotation_errors, 0.9)),
        translation_median_deg=float(np.median(translation_errors)),
        translation_p90_deg=float(np.quantile(translation_errors, 0.9)),
    )


def _pose_quality_report(
    best: _Hypothesis,
    b1: np.ndarray,
    b2: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    parallax: float,
    stability: PoseStabilityReport,
    competition: ModelCompetitionReport,
    orientation: TranslationOrientationReport,
    options: RelativePoseOptions,
) -> RelativePoseQualityReport:
    cells_a, entropy_a = _spherical_coverage(
        b1[best.inlier_mask], options.quality_cell_count
    )
    cells_b, entropy_b = _spherical_coverage(
        b2[best.inlier_mask], options.quality_cell_count
    )
    residuals = np.degrees(best.residuals[best.inlier_mask])
    median_residual = float(np.median(residuals))
    p90_residual = float(np.quantile(residuals, 0.9))
    ratio = best.num_inliers / max(1, int(valid.sum()))
    score = raw_quality_score(
        inlier_ratio=ratio,
        coverage_entropy_a=entropy_a,
        coverage_entropy_b=entropy_b,
        median_residual_deg=median_residual,
        residual_scale_deg=math.degrees(threshold),
        cheirality_ratio=best.cheirality_ratio,
        translation_orientation_margin=orientation.cheirality_margin,
        stability_rotation_p90_deg=stability.rotation_p90_deg,
        stability_translation_p90_deg=stability.translation_p90_deg,
        essential_score_margin=competition.essential_score_margin,
    )
    return RelativePoseQualityReport(
        num_correspondences=int(valid.sum()),
        num_inliers=best.num_inliers,
        inlier_ratio=ratio,
        occupied_cells_a=cells_a,
        occupied_cells_b=cells_b,
        coverage_entropy_a=entropy_a,
        coverage_entropy_b=entropy_b,
        median_residual_deg=median_residual,
        p90_residual_deg=p90_residual,
        median_parallax_deg=parallax,
        cheirality_ratio=best.cheirality_ratio,
        translation_orientation=orientation,
        stability=stability,
        model_competition=competition,
        raw_quality_score=score,
        accepted=False,
        rejection_reasons=(),
    )


def _spherical_coverage(bearings: np.ndarray, cell_count: int) -> tuple[int, float]:
    index = np.arange(cell_count, dtype=np.float64)
    y = 1.0 - 2.0 * (index + 0.5) / cell_count
    radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    longitude = index * (math.pi * (3.0 - math.sqrt(5.0)))
    centers = np.stack(
        (radius * np.cos(longitude), y, radius * np.sin(longitude)), axis=1
    )
    cells = np.argmax(bearings @ centers.T, axis=1)
    counts = np.bincount(cells, minlength=cell_count)
    positive = counts[counts > 0] / max(1, len(bearings))
    entropy = (
        float(-np.sum(positive * np.log(positive)) / math.log(cell_count))
        if len(positive) > 1
        else 0.0
    )
    return int(len(positive)), entropy


def _rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = np.clip((np.trace(first @ second.T) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def _direction_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = np.clip(np.dot(first, second), -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def _dynamic_trial_limit(
    num_inliers: int,
    num_samples: int,
    confidence: float,
    multiplier: float,
    hard_limit: int,
) -> int:
    if num_inliers < 5 or num_samples < 5:
        return hard_limit
    probability = 1.0
    for index in range(5):
        probability *= (num_inliers - index) / (num_samples - index)
    probability = min(max(probability, 0.0), 1.0)
    if probability <= 0.0:
        return hard_limit
    if probability >= 1.0:
        return 1
    failure = max(1.0 - confidence, np.finfo(np.float64).tiny)
    trials = math.ceil(math.log(failure) / math.log1p(-probability) * multiplier)
    return min(hard_limit, max(1, trials))


def _is_better(
    candidate: _Hypothesis | None,
    current: _Hypothesis | None,
    ranking: str = "count-first",
) -> bool:
    if candidate is None:
        return False
    if current is None:
        return True
    if ranking == "count-first":
        if candidate.num_inliers != current.num_inliers:
            return candidate.num_inliers > current.num_inliers
        if abs(candidate.robust_score - current.robust_score) > 1e-12:
            return candidate.robust_score > current.robust_score
    elif ranking == "msac-first":
        if abs(candidate.msac_cost - current.msac_cost) > 1e-12:
            return candidate.msac_cost < current.msac_cost
        if candidate.num_inliers != current.num_inliers:
            return candidate.num_inliers > current.num_inliers
        if abs(candidate.robust_score - current.robust_score) > 1e-12:
            return candidate.robust_score > current.robust_score
    elif ranking == "scale-marginal-first":
        if abs(candidate.robust_score - current.robust_score) > 1e-12:
            return candidate.robust_score > current.robust_score
        if candidate.num_inliers != current.num_inliers:
            return candidate.num_inliers > current.num_inliers
    else:  # pragma: no cover - public options validation owns this boundary
        raise ValueError(f"unsupported hypothesis ranking: {ranking!r}")
    return candidate.residual_sum < current.residual_sum


def _robust_estimator_name(options: RelativePoseOptions) -> str:
    if (
        options.hypothesis_ranking == "count-first"
        and options.nonminimal_refit_max_steps == 0
        and options.pose_refinement_method == "joint"
    ):
        return "panorai-scale-marginal-lo-ransac-v1"
    suffix = (
        "+all-inlier-essential-refit-v1" if options.nonminimal_refit_max_steps else ""
    )
    if options.pose_refinement_method == "decoupled":
        suffix += "+decoupled-wahba-translation-v1"
    return f"panorai-{options.hypothesis_ranking}-lo-ransac-v1{suffix}"


def _sampling_diagnostics(
    sampler: FivePointSampler,
    samples: list[Any],
    failed_draws: int,
    adaptive_uniform_trial_bound_enabled: bool,
) -> FivePointSamplingDiagnostics:
    configuration = sampler.describe()
    uniform = sum(item.strategy.startswith("uniform") for item in samples)
    strict = sum(
        not item.strategy.startswith("uniform") and item.relaxation_level == 0
        for item in samples
    )
    relaxed = len(samples) - uniform - strict

    def median(attribute: str) -> float:
        if not samples:
            return math.nan
        return float(np.median([getattr(item, attribute) for item in samples]))

    return FivePointSamplingDiagnostics(
        sampler_name=str(sampler.name),
        sampler_configuration=tuple(sorted(configuration.items())),
        samples_drawn=len(samples),
        strict_spatial_samples=strict,
        relaxed_spatial_samples=relaxed,
        uniform_samples=uniform,
        failed_draws=failed_draws,
        median_min_separation_a_deg=median("min_separation_a_deg"),
        median_min_separation_b_deg=median("min_separation_b_deg"),
        median_design_condition_number=median("design_condition_number"),
        adaptive_uniform_trial_bound_enabled=adaptive_uniform_trial_bound_enabled,
    )


def _readonly_array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape:
        raise ValueError(f"array must have shape {shape}")
    array.setflags(write=False)
    return array


def _positive_int(name: str, value: Any, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < (0 if allow_zero else 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")


def _finite_between(
    name: str,
    value: Any,
    lower: float,
    upper: float,
    *,
    closed: bool = False,
    lower_closed: bool | None = None,
    upper_closed: bool | None = None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    lower_closed = closed if lower_closed is None else lower_closed
    upper_closed = closed if upper_closed is None else upper_closed
    lower_ok = number >= lower if lower_closed else number > lower
    upper_ok = number <= upper if upper_closed else number < upper
    if not (lower_ok and upper_ok):
        left = "[" if lower_closed else "("
        right = "]" if upper_closed else ")"
        raise ValueError(f"{name} must be in {left}{lower}, {upper}{right}")
