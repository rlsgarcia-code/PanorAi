#!/usr/bin/env python3
"""Evaluate the deployed public-quality + post-probability policy."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

from scipy.stats import beta

MODEL_ID = "post-precise-aligned-orientation"


def _pair_key(row: dict[str, Any]) -> tuple[str, str]:
    dataset_id = row["dataset_id"]
    pair_id = row["pair_id"]
    if (
        not isinstance(dataset_id, str)
        or not dataset_id
        or not isinstance(pair_id, str)
        or not pair_id
    ):
        raise ValueError("dataset_id and pair_id must be non-empty strings")
    return dataset_id, pair_id


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _metrics(
    rows: list[dict[str, Any]], predictions: dict[tuple[str, str], float]
) -> dict[str, Any]:
    _validated_outcome_keys(rows)
    selected = [
        row
        for row in rows
        if row["accepted"]
        and predictions[_pair_key(row)] >= 0.9
    ]
    true_positive = sum(row["precise"] for row in selected)
    precise = sum(row["precise"] for row in rows)
    count = len(selected)
    return {
        "pairs": len(rows),
        "selected": count,
        "true_positive": true_positive,
        "false_positive": count - true_positive,
        "precision": true_positive / count if count else None,
        "precision_exact_one_sided_95_lower": (
            float(beta.ppf(0.05, true_positive, count - true_positive + 1))
            if true_positive
            else 0.0
        ),
        "recall": true_positive / precise if precise else None,
        "coverage": count / len(rows) if rows else None,
    }


def _validated_outcome_keys(
    outcomes: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for row in outcomes:
        key = _pair_key(row)
        if type(row["accepted"]) is not bool or type(row["precise"]) is not bool:
            raise ValueError("accepted and precise must be JSON booleans")
        keys.append(key)
    return keys


def _validated_predictions(
    outcomes: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]
) -> dict[tuple[str, str], float]:
    """Require a one-to-one prediction for every evaluated outcome."""

    outcome_keys = _validated_outcome_keys(outcomes)
    duplicate_outcomes = [
        key for key, count in Counter(outcome_keys).items() if count != 1
    ]
    if duplicate_outcomes:
        raise ValueError(
            "outcomes contain duplicate (dataset_id, pair_id) keys: "
            f"{len(duplicate_outcomes)} duplicate key(s)"
        )

    evaluation_rows = [
        row
        for row in prediction_rows
        if row["split"] == "evaluation" and row["model_id"] == MODEL_ID
    ]
    prediction_keys = [_pair_key(row) for row in evaluation_rows]
    duplicate_predictions = [
        key for key, count in Counter(prediction_keys).items() if count != 1
    ]
    if duplicate_predictions:
        raise ValueError(
            "predictions contain duplicate (dataset_id, pair_id) keys: "
            f"{len(duplicate_predictions)} duplicate key(s)"
        )

    expected = set(outcome_keys)
    observed = set(prediction_keys)
    missing = expected - observed
    unexpected = observed - expected
    if missing or unexpected:
        raise ValueError(
            "evaluation prediction coverage must match outcomes exactly: "
            f"{len(missing)} missing and {len(unexpected)} unexpected key(s)"
        )
    predictions: dict[tuple[str, str], float] = {}
    for key, row in zip(prediction_keys, evaluation_rows, strict=True):
        probability = float(row["probability"])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("evaluation probabilities must be finite and in [0, 1]")
        predictions[key] = probability
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    outcomes = _read_jsonl(args.outcomes)
    prediction_rows = _read_jsonl(args.predictions)
    predictions = _validated_predictions(outcomes, prediction_rows)
    datasets = sorted({row["dataset_id"] for row in outcomes})
    domain_aliases = {
        dataset: f"domain-{index}" for index, dataset in enumerate(datasets, start=1)
    }
    result = {
        "schema": "panorai-experimental-post-policy-evaluation/v1",
        "status": "retrospective held-out evaluation; not prospective release evidence",
        "rule": "public quality accepted AND post probability >= 0.90",
        "precision_target": 0.90,
        "pooled": _metrics(outcomes, predictions),
        "datasets": {
            domain_aliases[dataset]: _metrics(
                [row for row in outcomes if row["dataset_id"] == dataset],
                predictions,
            )
            for dataset in datasets
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
