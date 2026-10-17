#!/usr/bin/env python3
"""Audit absolute stereo-confidence thresholds on held-out public pairs."""

# ruff: noqa: E402 -- source-checkout runner intentionally precedes imports

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.panoramic_stereo.confidence_calibration import (
    ConfidenceSample,
    evaluate_confidence_threshold,
    make_confidence_sample,
    select_confidence_threshold,
)
from benchmarks.panoramic_stereo.run_full_sphere_stability_validation import (
    solid_angle_weights,
)
from benchmarks.panoramic_stereo.run_real_domain_validation import json_safe


SCHEMA = "panorai-full-sphere-confidence-transfer/v1"
SOURCE_SCHEMA = "panorai-public-multipair-full-sphere-stereo/v1"
FROZEN_MANIFEST_SHA256 = (
    "4910a0285aa6693881b84c1f3b58839154d1032f230f0478da3dfcae93dd0896"
)
DATASETS = ("stanford2d3d", "matterport360")
TARGET_CONDITIONAL_ACCURACY = 0.75
MINIMUM_FIXED_POPULATION_COVERAGE = 0.10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_samples(root: Path) -> tuple[dict[str, Any], tuple[ConfidenceSample, ...]]:
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("schema") != SOURCE_SCHEMA:
        raise ValueError("unexpected source summary schema")
    if summary.get("selection", {}).get("sha256") != FROZEN_MANIFEST_SHA256:
        raise ValueError("source summary does not use the frozen public manifest")
    samples = []
    for record in summary.get("records", []):
        dataset_id = str(record["dataset_id"])
        pair_id = str(record["pair_id"])
        if dataset_id not in DATASETS:
            raise ValueError(f"unexpected dataset: {dataset_id}")
        arrays_path = root / pair_id / Path(record["arrays"]).name
        with np.load(arrays_path) as arrays:
            shape = arrays["reference_radial_range_m"].shape
            samples.append(
                make_confidence_sample(
                    pair_id=pair_id,
                    dataset_id=dataset_id,
                    truth_range=arrays["reference_radial_range_m"],
                    predicted_range=arrays["estimated_radial_range_m"],
                    confidence=arrays["confidence"],
                    population=arrays["evaluation_population"],
                    prediction_validity=arrays["prediction_validity"],
                    sample_weights=solid_angle_weights(shape),
                )
            )
    if len(samples) != 11:
        raise ValueError("frozen protocol requires eleven samples")
    return summary, tuple(samples)


def _calibrate(
    calibration: tuple[ConfidenceSample, ...],
    target: tuple[ConfidenceSample, ...],
) -> dict[str, Any]:
    operating_point = select_confidence_threshold(
        calibration,
        target_conditional_accuracy=TARGET_CONDITIONAL_ACCURACY,
        minimum_fixed_population_coverage=MINIMUM_FIXED_POPULATION_COVERAGE,
    )
    if operating_point is None:
        return {
            "status": "calibration-target-unavailable",
            "calibration_pair_ids": [item.pair_id for item in calibration],
            "target_pair_ids": [item.pair_id for item in target],
        }
    evaluation = evaluate_confidence_threshold(target, operating_point.threshold)
    return {
        "status": "completed",
        "calibration": asdict(operating_point),
        "evaluation": evaluation,
        "gate": {
            "conditional_accuracy_at_least_target": (
                evaluation["conditional_accuracy"] >= TARGET_CONDITIONAL_ACCURACY
            ),
            "fixed_population_coverage_at_least_minimum": (
                evaluation["fixed_population_coverage"]
                >= MINIMUM_FIXED_POPULATION_COVERAGE
            ),
        },
    }


def run_validation(samples: tuple[ConfidenceSample, ...]) -> dict[str, Any]:
    within_domain = []
    all_other_pairs = []
    for target in samples:
        same_domain = tuple(
            item
            for item in samples
            if item.dataset_id == target.dataset_id and item.pair_id != target.pair_id
        )
        all_others = tuple(item for item in samples if item.pair_id != target.pair_id)
        within_domain.append(
            {
                "dataset_id": target.dataset_id,
                "target_pair_id": target.pair_id,
                **_calibrate(same_domain, (target,)),
            }
        )
        all_other_pairs.append(
            {
                "dataset_id": target.dataset_id,
                "target_pair_id": target.pair_id,
                **_calibrate(all_others, (target,)),
            }
        )
    cross_domain = []
    for source_dataset, target_dataset in (
        ("stanford2d3d", "matterport360"),
        ("matterport360", "stanford2d3d"),
    ):
        calibration = tuple(
            item for item in samples if item.dataset_id == source_dataset
        )
        target = tuple(item for item in samples if item.dataset_id == target_dataset)
        cross_domain.append(
            {
                "calibration_dataset": source_dataset,
                "target_dataset": target_dataset,
                **_calibrate(calibration, target),
            }
        )

    def summarize(folds: list[dict[str, Any]]) -> dict[str, int]:
        completed = [item for item in folds if item["status"] == "completed"]
        return {
            "fold_count": len(folds),
            "completed_count": len(completed),
            "conditional_accuracy_gate_pass_count": sum(
                item["gate"]["conditional_accuracy_at_least_target"]
                for item in completed
            ),
            "coverage_gate_pass_count": sum(
                item["gate"]["fixed_population_coverage_at_least_minimum"]
                for item in completed
            ),
            "joint_gate_pass_count": sum(
                all(item["gate"].values()) for item in completed
            ),
        }

    return {
        "within_domain_leave_one_pair_out": {
            "summary": summarize(within_domain),
            "folds": within_domain,
        },
        "all_other_pairs_leave_one_pair_out": {
            "summary": summarize(all_other_pairs),
            "folds": all_other_pairs,
        },
        "cross_domain": {
            "summary": summarize(cross_domain),
            "folds": cross_domain,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, samples = load_samples(args.input_dir)
    output = {
        "schema": SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(),
        "evidence_target": "external-frozen-val099-arrays",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "source": {
            "directory": str(args.input_dir.resolve()),
            "summary_sha256": sha256(args.input_dir / "summary.json"),
            "schema": source["schema"],
            "manifest_sha256": source["selection"]["sha256"],
            "pair_count": len(samples),
        },
        "protocol": {
            "target_conditional_accuracy": TARGET_CONDITIONAL_ACCURACY,
            "minimum_fixed_population_coverage": (MINIMUM_FIXED_POPULATION_COVERAGE),
            "threshold_semantics": "absolute score; ties retained together",
            "selection": (
                "lowest threshold / maximum fixed-population coverage meeting "
                "both calibration constraints"
            ),
            "held_out_outcomes_used_for_calibration": False,
            "population": "explicit non-sky, observable, fixed, solid-angle weighted",
        },
        "results": run_validation(samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_safe(output), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
