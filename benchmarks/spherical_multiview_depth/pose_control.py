"""Quality-gated image-pose adapters for metric depth controls."""

from __future__ import annotations

from dataclasses import dataclass
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
        registered_baseline_m=baseline,
        rotation_error_deg=rotation_error,
        translation_direction_error_deg=direction_error,
        source_id=source_id,
        quality=quality,
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
    "INTERFACE",
    "direction_distance_deg",
    "load_quality_accepted_metric_pose",
    "rotation_distance_deg",
]
