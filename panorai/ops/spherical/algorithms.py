"""Registered multi-stage algorithms operating on spherical results."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SphericalNMS:
    """Deterministic angular NMS for :class:`SphericalObservations`.

    The implementation delegates grouping and tie-breaking to the existing
    spherical-feature deduplication utility. Invalid observations and scores
    below ``min_score`` are removed before grouping. Labels are isolated by
    default so detections from different classes do not suppress one another.
    """

    angle_deg: float = 1.5
    class_aware: bool = True
    min_score: float | None = None

    spherical_algorithm = True

    def __post_init__(self) -> None:
        if isinstance(self.angle_deg, bool) or not isinstance(self.angle_deg, Real):
            raise TypeError("angle_deg must be a real number")
        if not math.isfinite(float(self.angle_deg)) or not 0 <= self.angle_deg <= 180:
            raise ValueError("angle_deg must be finite and in [0, 180]")
        if not isinstance(self.class_aware, bool):
            raise TypeError("class_aware must be boolean")
        if self.min_score is not None:
            if isinstance(self.min_score, bool) or not isinstance(self.min_score, Real):
                raise TypeError("min_score must be a real number or None")
            if not math.isfinite(float(self.min_score)):
                raise ValueError("min_score must be finite")

    def __call__(self, observations: Any) -> Any:
        from panorai.features import deduplicate_spherical_keypoints
        from panorai.pipeline import SphericalObservations

        if not isinstance(observations, SphericalObservations):
            raise TypeError("SphericalNMS requires SphericalObservations")
        eligible = observations.valid.copy()
        if self.min_score is not None:
            eligible &= observations.scores >= float(self.min_score)
        eligible_indices = np.flatnonzero(eligible)
        if eligible_indices.size == 0:
            return observations.take(np.empty(0, dtype=np.int64))

        if self.class_aware and observations.labels is not None:
            groups = [
                eligible_indices[observations.labels[eligible_indices] == label]
                for label in np.unique(observations.labels[eligible_indices])
            ]
        else:
            groups = [eligible_indices]
        selected = []
        threshold = math.radians(float(self.angle_deg))
        for group in groups:
            deduplication = deduplicate_spherical_keypoints(
                observations.bearings_xyz[group],
                observations.scores[group],
                angular_threshold_rad=threshold,
            )
            selected.extend(group[deduplication.selected_indices].tolist())
        return observations.take(np.asarray(sorted(selected), dtype=np.int64))

    def describe(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "domain": "sphere",
            "angle_deg": float(self.angle_deg),
            "class_aware": self.class_aware,
            "min_score": self.min_score,
            "selection": "highest-score-then-source-order",
        }


__all__ = ["SphericalNMS"]
