"""Quality-gated image-pose adapters for metric depth controls."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .p74 import RegisteredPose


INTERFACE = "panorai-experimental-estimated-metric-pose-control/v1"


@dataclass(frozen=True, slots=True)
class EstimatedMetricPose:
    """Image-estimated pose with an explicitly external metric baseline."""

    rotation_source_from_target: np.ndarray
    translation_source_from_target_m: np.ndarray
    translation_direction_source_from_target: np.ndarray
    metric_baseline_m: float
    registered_baseline_m: float
    rotation_error_deg: float
    translation_direction_error_deg: float
    source_id: str
    quality: dict[str, Any]
    scale_source: str = "registered-baseline-norm-only"
    convention: str = "X_source = R_source_from_target @ X_target + t"
    interface: str = INTERFACE

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "source_id": self.source_id,
            "convention": self.convention,
            "scale_source": self.scale_source,
            "metric_baseline_m": self.metric_baseline_m,
            "registered_baseline_m": self.registered_baseline_m,
            "rotation_error_deg": self.rotation_error_deg,
            "translation_direction_error_deg": self.translation_direction_error_deg,
            "rotation_source_from_target": self.rotation_source_from_target.tolist(),
            "translation_direction_source_from_target": (
                self.translation_direction_source_from_target.tolist()
            ),
            "translation_source_from_target_m": (
                self.translation_source_from_target_m.tolist()
            ),
            "quality": self.quality,
        }


@dataclass(frozen=True, slots=True)
class EstimatedTranslationScale:
    """Robust metric baseline inferred from prior depth and bearing matches."""

    scale_m: float
    candidate_count: int
    retained_count: int
    median_absolute_deviation_m: float
    epipolar_threshold_deg: float
    interface: str = INTERFACE

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "source": "target-depth-prior-and-bearing-correspondences",
            "scale_m": self.scale_m,
            "candidate_count": self.candidate_count,
            "retained_count": self.retained_count,
            "median_absolute_deviation_m": self.median_absolute_deviation_m,
            "epipolar_threshold_deg": self.epipolar_threshold_deg,
        }


def load_quality_accepted_metric_pose(
    results_path: Path,
    source_id: str,
    registered_pose: RegisteredPose,
) -> EstimatedMetricPose:
    """Load accepted image pose and attach only the registered baseline norm."""

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise ValueError("pose results must contain a sources list")
    matches = [item for item in sources if item.get("source_id") == source_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one pose record for {source_id}")
    record = matches[0]
    diagnostic = record.get("dog_pose_diagnostic")
    if not isinstance(diagnostic, dict):
        raise ValueError("source record has no dog_pose_diagnostic")
    if not diagnostic.get("returned"):
        raise ValueError("estimated pose was not returned")
    if not diagnostic.get("quality_accepted"):
        raise ValueError("estimated pose did not pass the quality gate")
    rotation = np.asarray(
        diagnostic.get("rotation_source_from_target"), dtype=np.float64
    )
    direction = np.asarray(
        diagnostic.get("translation_direction_source_from_target"),
        dtype=np.float64,
    )
    _validate_rotation(rotation)
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError("estimated translation direction must be finite shape (3,)")
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-12:
        raise ValueError("estimated translation direction must be nonzero")
    direction = direction / direction_norm

    registered_translation = np.asarray(
        registered_pose.translation_source_from_target_m, dtype=np.float64
    )
    baseline = float(np.linalg.norm(registered_translation))
    if not math.isfinite(baseline) or baseline <= 0.0:
        raise ValueError("registered pose must provide a positive metric baseline")
    recorded_baseline = float(record.get("registered_baseline_m", baseline))
    if not math.isclose(recorded_baseline, baseline, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("recorded and current registered baselines disagree")
    registered_direction = registered_translation / baseline
    rotation_error = rotation_distance_deg(
        rotation,
        np.asarray(registered_pose.rotation_source_from_target, dtype=np.float64),
    )
    direction_error = direction_distance_deg(direction, registered_direction)
    recorded_rotation_error = diagnostic.get("rotation_error_deg")
    recorded_direction_error = diagnostic.get("translation_direction_error_deg")
    if recorded_rotation_error is not None and not math.isclose(
        float(recorded_rotation_error), rotation_error, abs_tol=1e-7
    ):
        raise ValueError("recorded rotation error disagrees with the oracle control")
    if recorded_direction_error is not None and not math.isclose(
        float(recorded_direction_error), direction_error, abs_tol=1e-7
    ):
        raise ValueError(
            "recorded translation-direction error disagrees with the oracle control"
        )
    description = diagnostic.get("description")
    quality = description.get("quality", {}) if isinstance(description, dict) else {}
    return EstimatedMetricPose(
        rotation_source_from_target=rotation,
        translation_source_from_target_m=direction * baseline,
        translation_direction_source_from_target=direction,
        metric_baseline_m=baseline,
        registered_baseline_m=baseline,
        rotation_error_deg=rotation_error,
        translation_direction_error_deg=direction_error,
        source_id=source_id,
        quality=quality,
    )


def estimate_translation_scale_from_depth_prior(
    target_bearings: Any,
    source_bearings: Any,
    target_prior_range_m: Any,
    rotation_source_from_target: Any,
    translation_direction_source_from_target: Any,
    *,
    maximum_epipolar_error_deg: float = 1.0,
    minimum_candidates: int = 8,
    min_scale_m: float = 0.05,
    max_scale_m: float = 50.0,
) -> EstimatedTranslationScale:
    """Infer translation magnitude from prior range and epipolar correspondences.

    For each target point ``d*u``, solve the scalar translation magnitude that
    minimizes its component perpendicular to the matched source bearing. The
    final scale is a robust median and does not inspect registered scale.
    """

    target = np.asarray(target_bearings, dtype=np.float64)
    source = np.asarray(source_bearings, dtype=np.float64)
    prior = np.asarray(target_prior_range_m, dtype=np.float64)
    if target.ndim != 2 or target.shape[1:] != (3,):
        raise ValueError("target_bearings must have shape (N, 3)")
    if source.shape != target.shape:
        raise ValueError("source_bearings must match target_bearings")
    if prior.shape != (target.shape[0],):
        raise ValueError("target_prior_range_m must have shape (N,)")
    if minimum_candidates < 3:
        raise ValueError("minimum_candidates must be at least 3")
    if (
        not math.isfinite(maximum_epipolar_error_deg)
        or maximum_epipolar_error_deg <= 0.0
    ):
        raise ValueError("maximum_epipolar_error_deg must be positive")
    if not 0.0 < min_scale_m < max_scale_m:
        raise ValueError("scale bounds must satisfy 0 < min < max")
    rotation = np.asarray(rotation_source_from_target, dtype=np.float64)
    _validate_rotation(rotation)
    direction = np.asarray(translation_direction_source_from_target, dtype=np.float64)
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError("translation direction must be finite shape (3,)")
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-12:
        raise ValueError("translation direction must be nonzero")
    direction = direction / direction_norm
    target_norm = np.linalg.norm(target, axis=1)
    source_norm = np.linalg.norm(source, axis=1)
    finite = (
        np.isfinite(target).all(axis=1)
        & np.isfinite(source).all(axis=1)
        & np.isfinite(prior)
        & (prior > 0.0)
        & (target_norm > 1e-12)
        & (source_norm > 1e-12)
    )
    target = target / np.maximum(target_norm[:, None], 1e-12)
    source = source / np.maximum(source_norm[:, None], 1e-12)
    rotated = target @ rotation.T
    epipolar_normal = np.cross(direction[None], rotated)
    normal_norm = np.linalg.norm(epipolar_normal, axis=1)
    sine_error = np.divide(
        np.abs(np.sum(source * epipolar_normal, axis=1)),
        normal_norm,
        out=np.full(normal_norm.shape, np.inf),
        where=normal_norm > 1e-12,
    )
    epipolar_error = np.degrees(np.arcsin(np.clip(sine_error, 0.0, 1.0)))
    point = prior[:, None] * rotated
    perpendicular_direction = direction[None] - (
        np.sum(source * direction[None], axis=1)[:, None] * source
    )
    perpendicular_point = point - np.sum(source * point, axis=1)[:, None] * source
    denominator = np.sum(perpendicular_direction * perpendicular_direction, axis=1)
    scale = np.divide(
        -np.sum(perpendicular_direction * perpendicular_point, axis=1),
        denominator,
        out=np.full(denominator.shape, np.nan),
        where=denominator > 1e-8,
    )
    selected = (
        finite
        & np.isfinite(scale)
        & (scale >= min_scale_m)
        & (scale <= max_scale_m)
        & (epipolar_error <= maximum_epipolar_error_deg)
    )
    candidates = scale[selected]
    if candidates.size < minimum_candidates:
        raise ValueError(
            f"only {candidates.size} scale candidates passed; need {minimum_candidates}"
        )
    initial_median = float(np.median(candidates))
    mad = float(np.median(np.abs(candidates - initial_median)))
    robust_radius = max(3.0 * 1.4826 * mad, 0.05 * initial_median)
    retained = candidates[np.abs(candidates - initial_median) <= robust_radius]
    if retained.size < minimum_candidates:
        retained = candidates
    return EstimatedTranslationScale(
        scale_m=float(np.median(retained)),
        candidate_count=int(candidates.size),
        retained_count=int(retained.size),
        median_absolute_deviation_m=mad,
        epipolar_threshold_deg=maximum_epipolar_error_deg,
    )


def attach_estimated_translation_scale(
    pose: EstimatedMetricPose,
    scale: EstimatedTranslationScale,
) -> EstimatedMetricPose:
    """Replace the control baseline by a prior-derived metric estimate."""

    return replace(
        pose,
        translation_source_from_target_m=(
            pose.translation_direction_source_from_target * scale.scale_m
        ),
        metric_baseline_m=scale.scale_m,
        scale_source="convnext-large-depth-prior-correspondence-median",
    )


def rotation_distance_deg(first: Any, second: Any) -> float:
    """Return the geodesic SO(3) distance in degrees."""

    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    _validate_rotation(first_array)
    _validate_rotation(second_array)
    relative = first_array @ second_array.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def direction_distance_deg(first: Any, second: Any) -> float:
    """Return the unsigned angular distance between two oriented directions."""

    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.shape != (3,) or second_array.shape != (3,):
        raise ValueError("directions must have shape (3,)")
    first_norm = float(np.linalg.norm(first_array))
    second_norm = float(np.linalg.norm(second_array))
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        raise ValueError("directions must be nonzero")
    cosine = np.clip(
        float(first_array @ second_array) / (first_norm * second_norm),
        -1.0,
        1.0,
    )
    return math.degrees(math.acos(cosine))


def _validate_rotation(rotation: np.ndarray) -> None:
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation must be finite shape (3, 3)")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise ValueError("rotation must have determinant +1")


__all__ = [
    "EstimatedMetricPose",
    "EstimatedTranslationScale",
    "INTERFACE",
    "attach_estimated_translation_scale",
    "direction_distance_deg",
    "estimate_translation_scale_from_depth_prior",
    "load_quality_accepted_metric_pose",
    "rotation_distance_deg",
]
