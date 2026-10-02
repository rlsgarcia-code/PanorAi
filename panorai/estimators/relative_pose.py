"""Experimental spherical relative-pose estimation.

This module implements a PanorAi-owned first version of the calibrated
five-correspondence / locally-optimized RANSAC structure.  It estimates one
central relative pose from bearings expressed in the two panorama frames.

The implementation is deliberately isolated from OpenCV and PyCOLMAP.  The
minimal kernel solves the five-point essential constraints numerically in the
four-dimensional epipolar nullspace; it is not a copy of Nister's polynomial
elimination implementation.  That distinction is part of the Experimental
contract and leaves room for a faster algebraic kernel later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


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
    options: RelativePoseOptions
    interface: str = _INTERFACE
    minimal_solver: str = "panorai-numerical-five-correspondence-v1"
    robust_estimator: str = "panorai-lo-ransac-v1"

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
            "num_inliers": self.num_inliers,
            "num_trials": self.num_trials,
            "median_parallax_deg": self.median_parallax_deg,
            "cheirality_ratio": self.cheirality_ratio,
            "degenerate": self.degenerate,
            "degeneracy_reasons": self.degeneracy_reasons,
            "options": self.options.to_dict(),
        }


class SphericalRelativePoseEstimator:
    """Reusable façade for the Experimental PanorAi relative-pose estimator."""

    def __init__(self, options: RelativePoseOptions | None = None) -> None:
        self.options = options or RelativePoseOptions()

    def estimate(
        self,
        bearings_a: Any,
        bearings_b: Any | None = None,
        *,
        valid: Any | None = None,
    ) -> RelativePoseResult | None:
        return estimate_relative_pose(
            bearings_a, bearings_b, valid=valid, options=self.options
        )

    def __repr__(self) -> str:
        return (
            "SphericalRelativePoseEstimator("
            f"max_angular_error_deg={self.options.max_angular_error_deg}, "
            f"max_num_trials={self.options.max_num_trials}, "
            f"random_seed={self.options.random_seed})"
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
    cheirality_ratio: float


def estimate_relative_pose(
    bearings_a: Any,
    bearings_b: Any | None = None,
    *,
    valid: Any | None = None,
    options: RelativePoseOptions | None = None,
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
    options:
        Robust-estimation and numerical-solver configuration.

    Returns
    -------
    RelativePoseResult | None
        ``None`` when no hypothesis reaches ``min_inliers``.  A successful
        result contains ``R`` and only the direction of ``t``.
    """

    options = options or RelativePoseOptions()
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
    b1, b2, valid_mask = _prepare_bearings(bearings_a, bearings_b, valid)
    active = np.flatnonzero(valid_mask)
    required_inliers = max(
        5,
        options.min_inliers,
        math.ceil(options.min_inlier_ratio * active.size),
    )
    if active.size < required_inliers:
        return None

    threshold = math.radians(options.max_angular_error_deg)
    rng = np.random.default_rng(options.random_seed)
    dynamic_limit = options.max_num_trials
    best: _Hypothesis | None = None
    num_trials = 0

    while num_trials < options.max_num_trials:
        if num_trials >= dynamic_limit and num_trials >= options.min_num_trials:
            break
        sample = rng.choice(active, size=5, replace=False)
        essentials = _solve_five_correspondence_essential(
            b1[sample], b2[sample], options
        )
        num_trials += 1
        for essential in essentials:
            candidate = _score_essential(essential, b1, b2, valid_mask, threshold)
            if candidate is None or not _is_better(candidate, best):
                continue
            candidate = _locally_optimize(
                candidate, b1, b2, valid_mask, threshold, options
            )
            if _is_better(candidate, best):
                best = candidate
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

    final = _refine_hypothesis(
        best, b1, b2, valid_mask, threshold, options.refinement_max_nfev
    )
    if _is_better(final, best):
        best = final

    parallax = _median_parallax_deg(
        best.rotation, b1[best.inlier_mask], b2[best.inlier_mask]
    )
    reasons = []
    if parallax < options.min_median_parallax_deg:
        reasons.append("low-parallax")
    if best.cheirality_ratio < 0.5:
        reasons.append("weak-cheirality")

    full_residuals = np.full(b1.shape[0], np.inf, dtype=np.float64)
    finite_residuals = spherical_tangent_sampson_error(
        b1[valid_mask], b2[valid_mask], best.essential, squared=False
    )
    full_residuals[valid_mask] = finite_residuals
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
        options=options,
    )


def spherical_tangent_sampson_error(
    bearings_a: Any,
    bearings_b: Any,
    essential_matrix: Any,
    *,
    squared: bool = False,
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
    b1: np.ndarray, b2: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    essential = _skew(translation) @ rotation
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
    """Numerically solve the calibrated five-correspondence constraints."""

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
) -> _Hypothesis | None:
    residuals = np.full(b1.shape[0], np.inf, dtype=np.float64)
    residuals[valid] = spherical_tangent_sampson_error(b1[valid], b2[valid], essential)
    provisional = valid & (residuals <= threshold)
    if provisional.sum() < 5:
        return None
    pose = _pose_from_essential(essential, b1[provisional], b2[provisional])
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
        cheirality_ratio=float(ratio),
    )


def _score_pose(
    rotation: np.ndarray,
    translation: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    valid: np.ndarray,
    threshold: float,
) -> _Hypothesis | None:
    translation = translation / np.linalg.norm(translation)
    essential = _normalized_essential(rotation, translation)
    residuals = np.full(b1.shape[0], np.inf, dtype=np.float64)
    residuals[valid] = spherical_tangent_sampson_error(b1[valid], b2[valid], essential)
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
        cheirality_ratio=count / max(1, int(provisional.sum())),
    )


def _locally_optimize(
    hypothesis: _Hypothesis,
    b1: np.ndarray,
    b2: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    options: RelativePoseOptions,
) -> _Hypothesis:
    best = hypothesis
    for _ in range(options.local_optimization_steps):
        refined = _refine_hypothesis(
            best,
            b1,
            b2,
            valid,
            threshold,
            min(options.refinement_max_nfev, 50),
        )
        if not _is_better(refined, best):
            break
        best = refined
    return best


def _refine_hypothesis(
    hypothesis: _Hypothesis,
    b1: np.ndarray,
    b2: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    max_nfev: int,
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

    def residual(parameters: np.ndarray) -> np.ndarray:
        rotation, translation = unpack(parameters)
        return _signed_tangent_sampson_error(
            b1[inliers], b2[inliers], rotation, translation
        )

    optimized = least_squares(
        residual,
        np.zeros(5, dtype=np.float64),
        method="trf",
        max_nfev=max_nfev,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    if not optimized.success or not np.all(np.isfinite(optimized.x)):
        return hypothesis
    rotation, translation = unpack(optimized.x)
    scored = _score_pose(rotation, translation, b1, b2, valid, threshold)
    return hypothesis if scored is None else scored


def _pose_from_essential(
    essential: np.ndarray, b1: np.ndarray, b2: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    u, _, vh = np.linalg.svd(essential)
    if np.linalg.det(u) < 0:
        u[:, -1] *= -1
    if np.linalg.det(vh) < 0:
        vh[-1, :] *= -1
    w = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    rotations = (u @ w @ vh, u @ w.T @ vh)
    translation = u[:, 2]
    best: tuple[np.ndarray, np.ndarray] | None = None
    best_count = -1
    for rotation in rotations:
        if np.linalg.det(rotation) < 0:
            rotation = -rotation
        for direction in (translation, -translation):
            count = int(_cheirality_mask(rotation, direction, b1, b2).sum())
            if count > best_count:
                best_count = count
                best = (rotation, direction / np.linalg.norm(direction))
    return best if best_count > 0 else None


def _cheirality_mask(
    rotation: np.ndarray,
    translation: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
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
    return stable & (depth1 > 1e-10) & (depth2 > 1e-10)


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


def _is_better(candidate: _Hypothesis | None, current: _Hypothesis | None) -> bool:
    if candidate is None:
        return False
    if current is None or candidate.num_inliers > current.num_inliers:
        return True
    return (
        candidate.num_inliers == current.num_inliers
        and candidate.residual_sum < current.residual_sum
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
