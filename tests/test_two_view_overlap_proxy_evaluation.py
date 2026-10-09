"""Regression tests for retrospective post-policy evaluation alignment."""

from __future__ import annotations

import pytest

from benchmarks.two_view_overlap_proxy.evaluate_post_policy import (
    MODEL_ID,
    _metrics,
    _validated_predictions,
)


def _outcome(pair_id: str) -> dict[str, object]:
    return {
        "dataset_id": "domain",
        "pair_id": pair_id,
        "accepted": True,
        "precise": True,
    }


def _prediction(pair_id: str, probability: float = 0.95) -> dict[str, object]:
    return {
        "dataset_id": "domain",
        "pair_id": pair_id,
        "split": "evaluation",
        "model_id": MODEL_ID,
        "probability": probability,
    }


def test_complete_unique_predictions_are_evaluated() -> None:
    outcomes = [_outcome("a"), _outcome("b")]
    predictions = _validated_predictions(
        outcomes,
        [_prediction("a"), _prediction("b", 0.1)],
    )
    metrics = _metrics(outcomes, predictions)
    assert metrics["selected"] == 1
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5


def test_missing_evaluation_prediction_fails_instead_of_abstaining() -> None:
    with pytest.raises(ValueError, match="1 missing"):
        _validated_predictions(
            [_outcome("a"), _outcome("b")],
            [_prediction("a")],
        )


def test_duplicate_evaluation_prediction_fails() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _validated_predictions(
            [_outcome("a")],
            [_prediction("a"), _prediction("a")],
        )


def test_invalid_evaluation_probability_fails() -> None:
    with pytest.raises(ValueError, match="finite and in"):
        _validated_predictions([_outcome("a")], [_prediction("a", float("nan"))])
