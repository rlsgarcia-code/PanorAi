"""Outcome-separated confidence calibration for stereo benchmark evidence.

This module is intentionally benchmark-only.  It does not define public
confidence semantics for :mod:`panorai.stereo` and never interprets the score
as a probability.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class ConfidenceSample:
    """Flattened prediction evidence for one fixed evaluation population."""

    pair_id: str
    dataset_id: str
    confidence: np.ndarray
    correct: np.ndarray
    weights: np.ndarray
    population_weight: float

    def __post_init__(self) -> None:
        confidence = np.asarray(self.confidence, dtype=np.float64)
        correct = np.asarray(self.correct)
        weights = np.asarray(self.weights, dtype=np.float64)
        if confidence.ndim != 1:
            raise ValueError("confidence must have shape N")
        if correct.dtype != np.bool_ or correct.shape != confidence.shape:
            raise ValueError("correct must be a boolean array with shape N")
        if weights.shape != confidence.shape:
            raise ValueError("weights must have shape N")
        if not np.all(np.isfinite(confidence)):
            raise ValueError("confidence must be finite")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("weights must be finite and positive")
        population_weight = float(self.population_weight)
        if not math.isfinite(population_weight) or population_weight <= 0.0:
            raise ValueError("population_weight must be finite and positive")
        if float(weights.sum()) > population_weight * (1.0 + 1e-12):
            raise ValueError("prediction weight cannot exceed population weight")
        for name, value in (
            ("confidence", confidence),
            ("correct", correct),
            ("weights", weights),
        ):
            copied = np.array(value, copy=True)
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        object.__setattr__(self, "population_weight", population_weight)


@dataclass(frozen=True, slots=True)
class ConfidenceOperatingPoint:
    """One absolute score threshold selected only from calibration samples."""

    threshold: float
    target_conditional_accuracy: float
    conditional_accuracy: float
    fixed_population_coverage: float
    fixed_population_accuracy: float
    selected_weight: float
    population_weight: float
    calibration_pair_ids: tuple[str, ...]


def make_confidence_sample(
    *,
    pair_id: str,
    dataset_id: str,
    truth_range: np.ndarray,
    predicted_range: np.ndarray,
    confidence: np.ndarray,
    population: np.ndarray,
    prediction_validity: np.ndarray,
    sample_weights: np.ndarray,
) -> ConfidenceSample:
    """Build one sample without deriving validity or sky support from values."""

    truth = np.asarray(truth_range, dtype=np.float64)
    prediction = np.asarray(predicted_range, dtype=np.float64)
    score = np.asarray(confidence, dtype=np.float64)
    fixed_population = np.asarray(population)
    validity = np.asarray(prediction_validity)
    weights = np.asarray(sample_weights, dtype=np.float64)
    shape = truth.shape
    if truth.ndim != 2:
        raise ValueError("truth_range must have shape HW")
    if prediction.shape != shape or score.shape != shape or weights.shape != shape:
        raise ValueError("range, confidence, and weights must share shape HW")
    if fixed_population.dtype != np.bool_ or fixed_population.shape != shape:
        raise ValueError("population must be a boolean mask with shape HW")
    if validity.dtype != np.bool_ or validity.shape != shape:
        raise ValueError("prediction_validity must be a boolean mask with shape HW")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("sample_weights must be finite and non-negative")
    if np.any(
        fixed_population & (~np.isfinite(truth) | (truth <= 0.0) | (weights <= 0.0))
    ):
        raise ValueError(
            "population must contain finite positive truth and positive weights"
        )
    population_weight = float(weights[fixed_population].sum())
    eligible = (
        fixed_population
        & validity
        & np.isfinite(truth)
        & (truth > 0.0)
        & np.isfinite(prediction)
        & (prediction > 0.0)
        & np.isfinite(score)
        & (weights > 0.0)
    )
    selected_truth = truth[eligible]
    selected_prediction = prediction[eligible]
    ratio = np.maximum(
        selected_prediction / selected_truth,
        selected_truth / selected_prediction,
    )
    return ConfidenceSample(
        pair_id=str(pair_id),
        dataset_id=str(dataset_id),
        confidence=score[eligible],
        correct=ratio < 1.25,
        weights=weights[eligible],
        population_weight=population_weight,
    )


def evaluate_confidence_threshold(
    samples: Iterable[ConfidenceSample],
    threshold: float,
) -> dict[str, float | int | list[str]]:
    """Evaluate one unchanged absolute threshold on explicit target samples."""

    items = tuple(samples)
    if not items:
        raise ValueError("at least one sample is required")
    value = float(threshold)
    if not math.isfinite(value):
        raise ValueError("threshold must be finite")
    population_weight = sum(item.population_weight for item in items)
    prediction_weight = sum(float(item.weights.sum()) for item in items)
    selected_weight = 0.0
    correct_weight = 0.0
    selected_pixels = 0
    for item in items:
        selected = item.confidence >= value
        selected_pixels += int(selected.sum())
        selected_weights = item.weights[selected]
        selected_weight += float(selected_weights.sum())
        correct_weight += float(selected_weights[item.correct[selected]].sum())
    return {
        "pair_ids": [item.pair_id for item in items],
        "pair_count": len(items),
        "threshold": value,
        "selected_pixels": selected_pixels,
        "population_weight": population_weight,
        "prediction_weight": prediction_weight,
        "selected_weight": selected_weight,
        "fixed_population_coverage": selected_weight / population_weight,
        "fixed_population_accuracy": correct_weight / population_weight,
        "conditional_accuracy": (
            correct_weight / selected_weight if selected_weight > 0.0 else 0.0
        ),
        "valid_prediction_retention": (
            selected_weight / prediction_weight if prediction_weight > 0.0 else 0.0
        ),
    }


def select_confidence_threshold(
    samples: Iterable[ConfidenceSample],
    *,
    target_conditional_accuracy: float = 0.75,
    minimum_fixed_population_coverage: float = 0.10,
) -> ConfidenceOperatingPoint | None:
    """Select the widest calibrated prefix meeting frozen operating targets.

    Tied scores are always admitted or rejected together.  The held-out sample
    is not an argument and therefore cannot affect the selected threshold.
    """

    items = tuple(samples)
    if not items:
        raise ValueError("at least one calibration sample is required")
    target = float(target_conditional_accuracy)
    minimum_coverage = float(minimum_fixed_population_coverage)
    if not 0.0 < target <= 1.0:
        raise ValueError("target_conditional_accuracy must be in (0, 1]")
    if not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("minimum_fixed_population_coverage must be in [0, 1]")
    confidence = np.concatenate([item.confidence for item in items])
    correct = np.concatenate([item.correct for item in items])
    weights = np.concatenate([item.weights for item in items])
    if confidence.size == 0:
        return None
    # Stable input order makes tied-score behavior reproducible, while the
    # unique-boundary selection prevents a threshold from splitting a tie.
    order = np.argsort(-confidence, kind="stable")
    ordered_confidence = confidence[order]
    ordered_weights = weights[order]
    ordered_correct = correct[order]
    cumulative_weight = np.cumsum(ordered_weights)
    cumulative_correct = np.cumsum(ordered_weights * ordered_correct.astype(np.float64))
    group_end = np.r_[
        np.flatnonzero(ordered_confidence[:-1] != ordered_confidence[1:]),
        ordered_confidence.size - 1,
    ]
    selected_weight = cumulative_weight[group_end]
    selected_correct = cumulative_correct[group_end]
    population_weight = sum(item.population_weight for item in items)
    coverage = selected_weight / population_weight
    conditional_accuracy = selected_correct / selected_weight
    acceptable = (conditional_accuracy >= target) & (coverage >= minimum_coverage)
    if not np.any(acceptable):
        return None
    # Prefixes are ordered from highest to lowest threshold.  The last
    # acceptable prefix has maximum fixed-population coverage.
    selected_group = int(np.flatnonzero(acceptable)[-1])
    end = int(group_end[selected_group])
    threshold = float(ordered_confidence[end])
    selected = float(selected_weight[selected_group])
    selected_right = float(selected_correct[selected_group])
    return ConfidenceOperatingPoint(
        threshold=threshold,
        target_conditional_accuracy=target,
        conditional_accuracy=selected_right / selected,
        fixed_population_coverage=selected / population_weight,
        fixed_population_accuracy=selected_right / population_weight,
        selected_weight=selected,
        population_weight=population_weight,
        calibration_pair_ids=tuple(item.pair_id for item in items),
    )
