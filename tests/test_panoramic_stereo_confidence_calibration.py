from __future__ import annotations

import numpy as np
import pytest

from benchmarks.panoramic_stereo.confidence_calibration import (
    ConfidenceSample,
    evaluate_confidence_threshold,
    make_confidence_sample,
    select_confidence_threshold,
)


def _sample(
    pair_id: str,
    confidence: tuple[float, ...],
    correct: tuple[bool, ...],
    *,
    population_weight: float | None = None,
) -> ConfidenceSample:
    weights = np.ones(len(confidence), dtype=np.float64)
    return ConfidenceSample(
        pair_id=pair_id,
        dataset_id="fixture",
        confidence=np.asarray(confidence, dtype=np.float64),
        correct=np.asarray(correct, dtype=bool),
        weights=weights,
        population_weight=(
            float(len(confidence)) if population_weight is None else population_weight
        ),
    )


def test_threshold_is_selected_from_calibration_only_and_transfers_unchanged():
    calibration = _sample(
        "calibration",
        (0.9, 0.8, 0.7, 0.6),
        (True, True, True, False),
    )
    held_out = _sample(
        "held-out",
        (0.85, 0.75, 0.65, 0.55),
        (True, True, False, False),
    )

    operating_point = select_confidence_threshold(
        (calibration,),
        target_conditional_accuracy=0.80,
        minimum_fixed_population_coverage=0.5,
    )

    assert operating_point is not None
    assert operating_point.threshold == pytest.approx(0.7)
    result = evaluate_confidence_threshold((held_out,), operating_point.threshold)
    assert result["threshold"] == pytest.approx(0.7)
    assert result["selected_pixels"] == 2
    assert result["conditional_accuracy"] == 1.0
    assert result["fixed_population_coverage"] == 0.5


def test_threshold_never_splits_equal_confidence_scores():
    calibration = _sample(
        "ties",
        (0.9, 0.8, 0.8, 0.2),
        (True, True, False, False),
    )

    operating_point = select_confidence_threshold(
        (calibration,),
        target_conditional_accuracy=0.75,
        minimum_fixed_population_coverage=0.2,
    )

    assert operating_point is not None
    assert operating_point.threshold == pytest.approx(0.9)
    assert operating_point.fixed_population_coverage == pytest.approx(0.25)


def test_unavailable_calibration_target_fails_closed():
    calibration = _sample(
        "bad-ranking",
        (0.9, 0.8, 0.7),
        (False, False, True),
    )

    assert (
        select_confidence_threshold(
            (calibration,),
            target_conditional_accuracy=0.75,
            minimum_fixed_population_coverage=0.5,
        )
        is None
    )


def test_sample_builder_keeps_missing_predictions_in_population_denominator():
    truth = np.full((1, 4), 2.0, dtype=np.float32)
    prediction = truth.copy()
    confidence = np.asarray(((0.9, 0.8, 0.7, 0.0),), dtype=np.float32)
    population = np.ones_like(truth, dtype=bool)
    validity = population.copy()
    validity[0, 3] = False
    prediction[0, 3] = np.nan
    weights = np.asarray(((1.0, 1.0, 1.0, 5.0),), dtype=np.float64)

    sample = make_confidence_sample(
        pair_id="fixture",
        dataset_id="fixture",
        truth_range=truth,
        predicted_range=prediction,
        confidence=confidence,
        population=population,
        prediction_validity=validity,
        sample_weights=weights,
    )
    result = evaluate_confidence_threshold((sample,), 0.0)

    assert sample.population_weight == 8.0
    assert result["selected_weight"] == 3.0
    assert result["fixed_population_coverage"] == 3.0 / 8.0


def test_confidence_sample_rejects_prediction_weight_above_population():
    with pytest.raises(ValueError, match="cannot exceed"):
        ConfidenceSample(
            pair_id="invalid",
            dataset_id="fixture",
            confidence=np.asarray((0.9,)),
            correct=np.asarray((True,)),
            weights=np.asarray((2.0,)),
            population_weight=1.0,
        )


def test_sample_builder_rejects_invalid_truth_inside_fixed_population():
    truth = np.asarray(((2.0, np.nan),), dtype=np.float32)
    mask = np.ones_like(truth, dtype=bool)

    with pytest.raises(ValueError, match="finite positive truth"):
        make_confidence_sample(
            pair_id="invalid",
            dataset_id="fixture",
            truth_range=truth,
            predicted_range=np.ones_like(truth),
            confidence=np.ones_like(truth),
            population=mask,
            prediction_validity=mask,
            sample_weights=np.ones_like(truth, dtype=np.float64),
        )
