#!/usr/bin/env python3
"""Evaluate source-label agreement as a continuous stereo confidence factor."""

# ruff: noqa: E402 -- source-checkout benchmark intentionally precedes imports

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.panoramic_stereo.confidence_calibration import (
    ConfidenceSample,
    make_confidence_sample,
)
from benchmarks.panoramic_stereo.run_confidence_transfer_validation import (
    run_validation,
)
from benchmarks.panoramic_stereo.run_full_sphere_stability_validation import (
    solid_angle_weights,
)
from benchmarks.panoramic_stereo.run_real_domain_validation import json_safe
from benchmarks.panoramic_stereo.stability_protocol import (
    summarize_confidence_selectivity,
)


SCHEMA = "panorai-soft-source-consensus-confidence/v1"
SOURCE_SCHEMA = "panorai-public-multiview-full-sphere-stereo/v1"
FROZEN_MANIFEST_SHA256 = (
    "f731882028feb17221e6eb88632fa563160951f6000ebc0bfd222bf95b03534e"
)
DATASETS = ("stanford2d3d", "matterport360")
CONFIGURATION = "two-source"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def soft_source_consensus_score(
    confidence: np.ndarray,
    source_consensus_count: np.ndarray,
    source_count: np.ndarray,
    validity: np.ndarray,
) -> np.ndarray:
    """Apply one fixed, bounded and parameter-free agreement factor.

    The factor is ``(1 + consensus_count / source_count) / 2``. It retains a
    non-zero floor for a valid one-source-supported match and reaches one only
    when every selected source independently prefers a nearby range label.
    """

    score = np.asarray(confidence, dtype=np.float64)
    consensus = np.asarray(source_consensus_count)
    count = np.asarray(source_count)
    valid = np.asarray(validity)
    shape = score.shape
    if score.ndim != 2:
        raise ValueError("confidence must have shape HW")
    if consensus.shape != shape or count.shape != shape or valid.shape != shape:
        raise ValueError("confidence, counts and validity must share shape HW")
    if valid.dtype != np.bool_:
        raise ValueError("validity must be boolean")
    if np.any(~np.isfinite(score)) or np.any((score < 0.0) | (score > 1.0)):
        raise ValueError("confidence must be finite and in [0, 1]")
    if np.any(consensus < 0) or np.any(count < 0) or np.any(consensus > count):
        raise ValueError("source consensus must be bounded by source count")
    if np.any(valid & (count == 0)):
        raise ValueError("a valid prediction must have source evidence")
    fraction = np.divide(
        consensus.astype(np.float64),
        count.astype(np.float64),
        out=np.zeros(shape, dtype=np.float64),
        where=count > 0,
    )
    result = np.where(valid, score * (1.0 + fraction) * 0.5, 0.0).astype(np.float32)
    result.setflags(write=False)
    return result


def _selectivity_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset_id in DATASETS:
        selected = [item for item in records if item["dataset_id"] == dataset_id]
        output[dataset_id] = {}
        for score_name in ("original", "soft_source_consensus"):
            values = [item["selectivity"][score_name] for item in selected]
            output[dataset_id][score_name] = {
                "pair_count": len(values),
                "monotonic_pair_count": sum(
                    item["monotonic_conditional_delta_1_25"] is True for item in values
                ),
                "median_top_quartile_conditional_delta_gain": float(
                    np.median(
                        [item["top_quartile_conditional_delta_gain"] for item in values]
                    )
                ),
                "median_top_quartile_conditional_delta_1_25": float(
                    np.median(
                        [
                            item["retention"]["0.25"]["conditional_delta_1_25"]
                            for item in values
                        ]
                    )
                ),
                "median_top_quartile_fixed_population_coverage": float(
                    np.median(
                        [
                            item["retention"]["0.25"]["fixed_population_coverage"]
                            for item in values
                        ]
                    )
                ),
            }
    return output


def load_and_evaluate(
    root: Path,
) -> tuple[dict[str, Any], tuple[ConfidenceSample, ...], tuple[ConfidenceSample, ...]]:
    evidence_path = root / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    if evidence.get("schema") != SOURCE_SCHEMA:
        raise ValueError("unexpected multiview source schema")
    if evidence.get("selection", {}).get("manifest_sha256") != FROZEN_MANIFEST_SHA256:
        raise ValueError("source evidence does not use the frozen manifest")
    selected = [
        item
        for item in evidence.get("records", [])
        if item.get("configuration") == CONFIGURATION
    ]
    if len(selected) != 6:
        raise ValueError("frozen protocol requires six two-source records")
    records = []
    original_samples = []
    soft_samples = []
    for record in selected:
        dataset_id = str(record["dataset_id"])
        triplet_id = str(record["triplet_id"])
        if dataset_id not in DATASETS:
            raise ValueError(f"unexpected dataset: {dataset_id}")
        arrays_path = root / triplet_id / "two-source.npz"
        with np.load(arrays_path, allow_pickle=False) as arrays:
            truth = np.asarray(arrays["reference_radial_range_m"])
            prediction = np.asarray(arrays["estimated_radial_range_m"])
            validity = np.asarray(arrays["prediction_validity"])
            original = np.asarray(arrays["confidence"])
            population = np.asarray(arrays["population_observable_union"])
            soft = soft_source_consensus_score(
                original,
                arrays["source_consensus_count"],
                arrays["source_count"],
                validity,
            )
            weights = solid_angle_weights(truth.shape)
            original_samples.append(
                make_confidence_sample(
                    pair_id=triplet_id,
                    dataset_id=dataset_id,
                    truth_range=truth,
                    predicted_range=prediction,
                    confidence=original,
                    population=population,
                    prediction_validity=validity,
                    sample_weights=weights,
                )
            )
            soft_samples.append(
                make_confidence_sample(
                    pair_id=triplet_id,
                    dataset_id=dataset_id,
                    truth_range=truth,
                    predicted_range=prediction,
                    confidence=soft,
                    population=population,
                    prediction_validity=validity,
                    sample_weights=weights,
                )
            )
            records.append(
                {
                    "dataset_id": dataset_id,
                    "triplet_id": triplet_id,
                    "source_array_payload_sha256": record["array_payload_sha256"],
                    "selectivity": {
                        "original": summarize_confidence_selectivity(
                            truth,
                            prediction,
                            original,
                            population=population,
                            prediction_validity=validity,
                            sample_weights=weights,
                        ),
                        "soft_source_consensus": summarize_confidence_selectivity(
                            truth,
                            prediction,
                            soft,
                            population=population,
                            prediction_validity=validity,
                            sample_weights=weights,
                        ),
                    },
                }
            )
    original_tuple = tuple(original_samples)
    soft_tuple = tuple(soft_samples)
    result = {
        "source": {
            "evidence_sha256": sha256(evidence_path),
            "manifest_sha256": FROZEN_MANIFEST_SHA256,
            "configuration": CONFIGURATION,
            "pair_count": len(records),
        },
        "score": {
            "formula": (
                "confidence * (1 + source_consensus_count / max(source_count, 1)) / 2"
            ),
            "tuned_parameters": 0,
            "changes_prediction_validity": False,
            "changes_radial_range": False,
        },
        "selectivity": _selectivity_summary(records),
        "threshold_transfer": {
            "original": run_validation(original_tuple),
            "soft_source_consensus": run_validation(soft_tuple),
        },
        "records": records,
    }
    return result, original_tuple, soft_tuple


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result, _, _ = load_and_evaluate(args.input_dir)
    output = {"schema": SCHEMA, **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_safe(output), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
