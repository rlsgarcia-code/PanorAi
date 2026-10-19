"""Deterministic RANSAC/MSAC with built-in rotation and homography models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np


ModelT = TypeVar("ModelT")


@runtime_checkable
class ModelFamily(Protocol[ModelT]):
    """Minimal public contract consumed by :func:`robust_fit`.

    Residuals must be finite, non-negative and expressed in the same unit as
    ``RobustFitOptions.residual_threshold``.  Implementations may return
    ``None`` from ``fit`` for a degenerate sample.
    """

    name: str
    min_samples: int
    residual_unit: str

    def fit(self, *observations: np.ndarray) -> ModelT | None:
        """Fit one model to aligned observation arrays."""

    def residuals(self, model: ModelT, *observations: np.ndarray) -> np.ndarray:
        """Return one residual for each observation row."""


@dataclass(frozen=True, slots=True)
class RobustFitOptions:
    """Configuration shared by deterministic RANSAC and MSAC."""

    residual_threshold: float
    method: str = "msac"
    confidence: float = 0.999
    min_trials: int = 32
    max_trials: int = 1000
    local_optimization_steps: int = 3
    min_inliers: int | None = None
    random_seed: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.residual_threshold, bool)
            or not math.isfinite(float(self.residual_threshold))
            or self.residual_threshold <= 0
        ):
            raise ValueError("residual_threshold must be finite and positive")
        if self.method not in {"ransac", "msac"}:
            raise ValueError("method must be 'ransac' or 'msac'")
        if (
            isinstance(self.confidence, bool)
            or not math.isfinite(float(self.confidence))
            or not 0.0 < self.confidence < 1.0
        ):
            raise ValueError("confidence must be finite and in (0, 1)")
        for name in ("min_trials", "max_trials", "local_optimization_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if value < (0 if name == "local_optimization_steps" else 1):
                raise ValueError(f"{name} is outside its valid range")
        if self.min_trials > self.max_trials:
            raise ValueError("min_trials must not exceed max_trials")
        if self.min_inliers is not None and (
            isinstance(self.min_inliers, bool)
            or not isinstance(self.min_inliers, (int, np.integer))
            or self.min_inliers < 1
        ):
            raise ValueError("min_inliers must be a positive integer or None")
        if isinstance(self.random_seed, bool) or not isinstance(
            self.random_seed, (int, np.integer)
        ):
            raise TypeError("random_seed must be an integer")


@dataclass(frozen=True, slots=True)
class RobustFitResult:
    """Immutable model, consensus and diagnostics from robust fitting."""

    model: Any
    inlier_mask: np.ndarray
    residuals: np.ndarray
    score: float
    num_trials: int
    num_degenerate_samples: int
    model_family: str
    residual_unit: str
    method: str
    interface: str = "panorai-robust-model-fitting/v1-experimental"
    stability: str = "experimental"

    def __post_init__(self) -> None:
        inliers = np.asarray(self.inlier_mask, dtype=bool)
        residuals = np.asarray(self.residuals, dtype=np.float64)
        if inliers.ndim != 1 or residuals.shape != inliers.shape:
            raise ValueError("inlier_mask and residuals must align with shape (N,)")
        if not np.all(np.isfinite(residuals)) or np.any(residuals < 0.0):
            raise ValueError("residuals must be finite and non-negative")
        if not math.isfinite(float(self.score)) or self.score < 0.0:
            raise ValueError("score must be finite and non-negative")
        for name in ("num_trials", "num_degenerate_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        inliers = inliers.copy()
        residuals = residuals.copy()
        inliers.setflags(write=False)
        residuals.setflags(write=False)
        object.__setattr__(self, "inlier_mask", inliers)
        object.__setattr__(self, "residuals", residuals)

    @property
    def num_inliers(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))

    @property
    def inlier_ratio(self) -> float:
        return float(np.mean(self.inlier_mask)) if len(self.inlier_mask) else 0.0


def _observations(values: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    if not values:
        raise ValueError("at least one observation array is required")
    arrays = tuple(np.asarray(value) for value in values)
    count = arrays[0].shape[0] if arrays[0].ndim else 0
    if count == 0 or any(
        array.ndim == 0 or array.shape[0] != count for array in arrays
    ):
        raise ValueError("observation arrays must have the same non-zero first axis")
    return arrays


def _score(
    residuals: np.ndarray, threshold: float, method: str
) -> tuple[tuple[float, float], np.ndarray, float]:
    inliers = residuals <= threshold
    count = int(np.count_nonzero(inliers))
    if method == "ransac":
        cost = float(len(residuals) - count)
    else:
        cost = float(np.minimum((residuals / threshold) ** 2, 1.0).sum())
    # Maximizing inlier count remains the first ordering rule.  MSAC then
    # distinguishes hypotheses whose consensus counts are identical.
    return (-float(count), cost), inliers, cost


def _required_trials(confidence: float, inlier_ratio: float, sample_size: int) -> int:
    probability = min(max(inlier_ratio**sample_size, 0.0), 1.0)
    if probability >= 1.0:
        return 1
    if probability <= np.finfo(np.float64).eps:
        return np.iinfo(np.int32).max
    return int(math.ceil(math.log1p(-confidence) / math.log1p(-probability)))


def robust_fit(
    family: ModelFamily[ModelT],
    *observations: np.ndarray,
    options: RobustFitOptions,
) -> RobustFitResult:
    """Fit ``family`` with deterministic RANSAC/MSAC and local refitting."""

    arrays = _observations(observations)
    count = arrays[0].shape[0]
    minimum = int(family.min_samples)
    required_inliers = minimum if options.min_inliers is None else options.min_inliers
    if minimum < 1:
        raise ValueError("model family min_samples must be positive")
    if count < minimum:
        raise ValueError(
            f"{family.name} requires at least {minimum} observations; got {count}"
        )
    if required_inliers < minimum or required_inliers > count:
        raise ValueError("min_inliers must be between model min_samples and N")

    random = np.random.default_rng(int(options.random_seed))
    best: tuple[tuple[float, float], ModelT, np.ndarray, np.ndarray, float] | None = (
        None
    )
    trial_limit = int(options.max_trials)
    degenerate = 0
    trial = 0
    while trial < trial_limit:
        indices = random.choice(count, size=minimum, replace=False)
        model = family.fit(*(array[indices] for array in arrays))
        trial += 1
        if model is None:
            degenerate += 1
            continue
        residuals = np.asarray(family.residuals(model, *arrays), dtype=np.float64)
        if residuals.shape != (count,) or not np.all(np.isfinite(residuals)):
            degenerate += 1
            continue
        ordering, inliers, cost = _score(
            residuals, float(options.residual_threshold), options.method
        )
        candidate = (ordering, model, inliers, residuals, cost)
        if best is None or candidate[0] < best[0]:
            best = candidate
            adaptive = _required_trials(options.confidence, inliers.mean(), minimum)
            trial_limit = min(
                options.max_trials,
                max(options.min_trials, trial, adaptive),
            )

    if best is None or int(np.count_nonzero(best[2])) < required_inliers:
        raise ValueError(
            f"robust {family.name} fitting found no consensus with "
            f"at least {required_inliers} inliers"
        )

    _, model, inliers, residuals, cost = best
    for _ in range(options.local_optimization_steps):
        refined = family.fit(*(array[inliers] for array in arrays))
        if refined is None:
            break
        refined_residuals = np.asarray(
            family.residuals(refined, *arrays), dtype=np.float64
        )
        ordering, refined_inliers, refined_cost = _score(
            refined_residuals,
            float(options.residual_threshold),
            options.method,
        )
        previous_ordering, _, _ = _score(
            residuals, float(options.residual_threshold), options.method
        )
        if ordering > previous_ordering:
            break
        model, inliers, residuals, cost = (
            refined,
            refined_inliers,
            refined_residuals,
            refined_cost,
        )
        if np.array_equal(refined_inliers, best[2]):
            break
        best = (ordering, model, inliers, residuals, cost)

    if int(np.count_nonzero(inliers)) < required_inliers:
        raise ValueError("local model refit fell below the required consensus")
    return RobustFitResult(
        model=model,
        inlier_mask=inliers,
        residuals=residuals,
        score=cost,
        num_trials=trial,
        num_degenerate_samples=degenerate,
        model_family=str(family.name),
        residual_unit=str(family.residual_unit),
        method=options.method,
    )


def _unit_vectors(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have finite shape (N, 3)")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{name} must not contain zero vectors")
    return array / norms[:, None]


@dataclass(frozen=True, slots=True)
class RotationModel:
    """Wahba/Kabsch model mapping source bearings into target bearings."""

    name: str = "rotation"
    min_samples: int = 3
    residual_unit: str = "radians"

    def fit(self, source: np.ndarray, target: np.ndarray) -> np.ndarray | None:
        source = _unit_vectors(source, name="source")
        target = _unit_vectors(target, name="target")
        if source.shape != target.shape or len(source) < self.min_samples:
            return None
        if np.linalg.matrix_rank(source) < 2 or np.linalg.matrix_rank(target) < 2:
            return None
        left, _, right = np.linalg.svd(target.T @ source)
        sign = np.linalg.det(left @ right)
        rotation = left @ np.diag((1.0, 1.0, sign)) @ right
        if not np.all(np.isfinite(rotation)) or np.linalg.det(rotation) < 0.0:
            return None
        return rotation

    def residuals(
        self, model: np.ndarray, source: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        rotation = np.asarray(model, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("rotation model must have shape (3, 3)")
        source = _unit_vectors(source, name="source")
        target = _unit_vectors(target, name="target")
        predicted = source @ rotation.T
        cosine = np.sum(predicted * target, axis=1)
        return np.arccos(np.clip(cosine, -1.0, 1.0))


def _normalized_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("points must have finite shape (N, 2)")
    centre = points.mean(axis=0)
    distance = np.linalg.norm(points - centre, axis=1).mean()
    if distance <= 1e-12:
        return None
    scale = math.sqrt(2.0) / distance
    transform = np.asarray(
        (
            (scale, 0.0, -scale * centre[0]),
            (0.0, scale, -scale * centre[1]),
            (0.0, 0.0, 1.0),
        )
    )
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (homogeneous @ transform.T)[:, :2], transform


@dataclass(frozen=True, slots=True)
class HomographyModel:
    """Normalized-DLT planar homography with symmetric transfer residuals."""

    name: str = "homography"
    min_samples: int = 4
    residual_unit: str = "pixels"

    def fit(self, source: np.ndarray, target: np.ndarray) -> np.ndarray | None:
        if len(source) < self.min_samples or len(target) != len(source):
            return None
        normalized_source = _normalized_points(source)
        normalized_target = _normalized_points(target)
        if normalized_source is None or normalized_target is None:
            return None
        source_xy, source_transform = normalized_source
        target_xy, target_transform = normalized_target
        rows = []
        for (x, y), (u, v) in zip(source_xy, target_xy, strict=True):
            rows.extend(
                (
                    (-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u),
                    (0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v),
                )
            )
        design = np.asarray(rows, dtype=np.float64)
        if np.linalg.matrix_rank(design) < 8:
            return None
        _, _, right = np.linalg.svd(design)
        normalized_model = right[-1].reshape(3, 3)
        try:
            model = (
                np.linalg.inv(target_transform) @ normalized_model @ source_transform
            )
        except np.linalg.LinAlgError:
            return None
        norm = float(np.linalg.norm(model))
        if norm <= 1e-12 or abs(np.linalg.det(model)) <= 1e-12:
            return None
        model /= norm
        if model[2, 2] < 0.0:
            model = -model
        return model

    def residuals(
        self, model: np.ndarray, source: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        model = np.asarray(model, dtype=np.float64)
        if model.shape != (3, 3):
            raise ValueError("homography model must have shape (3, 3)")
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        source_h = np.column_stack((source, np.ones(len(source))))
        target_h = np.column_stack((target, np.ones(len(target))))
        try:
            inverse = np.linalg.inv(model)
        except np.linalg.LinAlgError:
            return np.full(len(source), np.finfo(np.float64).max)
        forward_h = source_h @ model.T
        backward_h = target_h @ inverse.T
        good = (np.abs(forward_h[:, 2]) > 1e-12) & (np.abs(backward_h[:, 2]) > 1e-12)
        residuals = np.full(len(source), np.finfo(np.float64).max)
        forward = forward_h[good, :2] / forward_h[good, 2:]
        backward = backward_h[good, :2] / backward_h[good, 2:]
        residuals[good] = np.sqrt(
            0.5
            * (
                np.sum((forward - target[good]) ** 2, axis=1)
                + np.sum((backward - source[good]) ** 2, axis=1)
            )
        )
        return residuals


def fit_rotation(
    source_bearings: np.ndarray,
    target_bearings: np.ndarray,
    *,
    max_angular_error_deg: float = 1.0,
    options: RobustFitOptions | None = None,
) -> RobustFitResult:
    """Robustly estimate one target-from-source SO(3) rotation."""

    resolved = options or RobustFitOptions(
        residual_threshold=math.radians(float(max_angular_error_deg))
    )
    return robust_fit(
        RotationModel(), source_bearings, target_bearings, options=resolved
    )


def fit_homography(
    source_pixels_xy: np.ndarray,
    target_pixels_xy: np.ndarray,
    *,
    max_transfer_error_px: float = 3.0,
    options: RobustFitOptions | None = None,
) -> RobustFitResult:
    """Robustly estimate a normalized-DLT planar homography."""

    resolved = options or RobustFitOptions(
        residual_threshold=float(max_transfer_error_px)
    )
    return robust_fit(
        HomographyModel(), source_pixels_xy, target_pixels_xy, options=resolved
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
