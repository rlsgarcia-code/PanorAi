#!/usr/bin/env python3
"""Evaluate the deployed public-quality + post-probability policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scipy.stats import beta

MODEL_ID = "post-precise-aligned-orientation"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _metrics(
    rows: list[dict[str, Any]], predictions: dict[tuple[str, str], float]
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["accepted"]
        and predictions.get((row["dataset_id"], row["pair_id"]), -1.0) >= 0.9
    ]
    true_positive = sum(bool(row["precise"]) for row in selected)
    precise = sum(bool(row["precise"]) for row in rows)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    outcomes = _read_jsonl(args.outcomes)
    prediction_rows = _read_jsonl(args.predictions)
    predictions = {
        (row["dataset_id"], row["pair_id"]): float(row["probability"])
        for row in prediction_rows
        if row["split"] == "evaluation" and row["model_id"] == MODEL_ID
    }
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
