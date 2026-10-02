"""Leakage-aware calibration for relative-pose quality scores."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from ._quality import RelativePoseQualityReport


@dataclass(frozen=True, slots=True)
class CalibrationEvaluation:
    count: int
    brier_score: float
    expected_calibration_error: float
    accuracy_at_half: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RelativePoseConfidenceCalibrator:
    """Isotonic post-hoc calibrator with mandatory sample-ID separation."""

    score_upper_bounds: tuple[float, ...]
    probabilities: tuple[float, ...]
    calibration_sample_ids: frozenset[str]
    interface: str = "panorai-relative-pose-confidence-calibration/v1"

    @classmethod
    def fit(
        cls,
        reports: Sequence[RelativePoseQualityReport],
        successes: Sequence[bool],
        *,
        sample_ids: Sequence[str],
    ) -> RelativePoseConfidenceCalibrator:
        scores, labels, identifiers = _calibration_inputs(
            reports, successes, sample_ids
        )
        order = np.argsort(scores, kind="stable")
        scores = scores[order]
        labels = labels[order]
        blocks: list[list[float]] = []
        for score in np.unique(scores):
            selected = scores == score
            blocks.append(
                [
                    float(score),
                    float(score),
                    float(labels[selected].sum()),
                    float(selected.sum()),
                ]
            )
            while len(blocks) >= 2:
                previous = blocks[-2][2] / blocks[-2][3]
                current = blocks[-1][2] / blocks[-1][3]
                if previous <= current:
                    break
                right = blocks.pop()
                left = blocks.pop()
                blocks.append(
                    [left[0], right[1], left[2] + right[2], left[3] + right[3]]
                )
        return cls(
            score_upper_bounds=tuple(block[1] for block in blocks),
            probabilities=tuple(block[2] / block[3] for block in blocks),
            calibration_sample_ids=frozenset(identifiers),
        )

    def predict_proba(
        self, report: RelativePoseQualityReport | Iterable[RelativePoseQualityReport]
    ) -> float | np.ndarray:
        if isinstance(report, RelativePoseQualityReport):
            return self._predict_one(report.raw_quality_score)
        return np.asarray(
            [self._predict_one(item.raw_quality_score) for item in report],
            dtype=np.float64,
        )

    def evaluate(
        self,
        reports: Sequence[RelativePoseQualityReport],
        successes: Sequence[bool],
        *,
        sample_ids: Sequence[str],
        bins: int = 5,
    ) -> CalibrationEvaluation:
        if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)):
            raise TypeError("bins must be an integer")
        if bins <= 0:
            raise ValueError("bins must be positive")
        _, labels, identifiers = _calibration_inputs(reports, successes, sample_ids)
        overlap = self.calibration_sample_ids.intersection(identifiers)
        if overlap:
            raise ValueError(
                "evaluation sample_ids overlap calibration data: "
                + ", ".join(sorted(overlap)[:5])
            )
        probabilities = np.asarray(self.predict_proba(reports), dtype=np.float64)
        brier = float(np.mean((probabilities - labels) ** 2))
        ece = 0.0
        edges = np.linspace(0.0, 1.0, bins + 1)
        for index in range(bins):
            selected = (probabilities >= edges[index]) & (
                probabilities <= edges[index + 1]
                if index == bins - 1
                else probabilities < edges[index + 1]
            )
            if selected.any():
                ece += float(selected.mean()) * abs(
                    float(probabilities[selected].mean() - labels[selected].mean())
                )
        accuracy = float(np.mean((probabilities >= 0.5) == labels.astype(bool)))
        return CalibrationEvaluation(len(labels), brier, ece, accuracy)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["calibration_sample_ids"] = sorted(self.calibration_sample_ids)
        return result

    def _predict_one(self, score: float) -> float:
        index = int(np.searchsorted(self.score_upper_bounds, score, side="left"))
        index = min(index, len(self.probabilities) - 1)
        return float(self.probabilities[index])


def _calibration_inputs(
    reports: Sequence[RelativePoseQualityReport],
    successes: Sequence[bool],
    sample_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if not reports or len(reports) != len(successes) or len(reports) != len(sample_ids):
        raise ValueError(
            "reports, successes, and sample_ids must have equal non-zero length"
        )
    identifiers = tuple(str(value) for value in sample_ids)
    if any(not value for value in identifiers) or len(set(identifiers)) != len(
        identifiers
    ):
        raise ValueError("sample_ids must be non-empty and unique")
    scores = np.asarray([item.raw_quality_score for item in reports], dtype=np.float64)
    labels = np.asarray(successes)
    if labels.dtype != np.bool_:
        raise TypeError("successes must have boolean dtype")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("raw quality scores must be finite and in [0, 1]")
    return scores, labels.astype(np.float64), identifiers
