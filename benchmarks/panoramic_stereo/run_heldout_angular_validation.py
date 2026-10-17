#!/usr/bin/env python3
"""Run the frozen VAL-121 angular-neighbourhood comparison."""

# ruff: noqa: E402 -- the source-checkout root precedes PanorAi imports

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import platform
import sys
from typing import Any
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.panoramic_stereo.run_angular_neighborhood_validation import (
    AngularCell,
    ERP_FEATURE_TAP_DEG,
    FACE_COST_TAP_DEG,
    _run,
)
from benchmarks.panoramic_stereo.run_full_sphere_stability_validation import (
    solid_angle_weights,
)
from benchmarks.panoramic_stereo.run_real_domain_validation import json_safe, sha256


SCHEMA = "panorai-heldout-angular-neighborhood-validation/v1"
SELECTION_SCHEMA = "panorai-public-angular-neighborhood-selection/v1"
DEFAULT_MANIFEST = Path(__file__).with_name("public_angular_neighborhood_manifest.json")
EXPECTED_MANIFEST_SHA256 = (
    "0c59dde87819e5342296d75455fcd39b0c8aabe241f7993704170709d72def39"
)
DATASETS = ("stanford2d3d", "matterport360")
LEGACY = AngularCell("legacy-pixel-taps", None, None)
ANGULAR = AngularCell(
    "erp-and-face-angular",
    ERP_FEATURE_TAP_DEG,
    FACE_COST_TAP_DEG,
)
CELLS = (LEGACY, ANGULAR)
MAXIMUM_COVERAGE_REGRESSION = 0.02
MAXIMUM_CONDITIONAL_GROSS_REGRESSION = 0.02
MAXIMUM_RUNTIME_RATIO = 1.35


def load_manifest(path: Path) -> dict[str, Any]:
    if sha256(path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("held-out angular-neighbourhood manifest hash changed")
    payload = json.loads(path.read_text())
    if payload.get("schema") != SELECTION_SCHEMA:
        raise ValueError("unexpected angular-neighbourhood selection schema")
    if payload.get("protocol_id") != "VAL-121-heldout-angular-neighborhoods":
        raise ValueError("unexpected angular-neighbourhood protocol")
    if payload.get("frozen_before_dense_evaluation") is not True:
        raise ValueError("selection must be frozen before dense evaluation")
    policy = payload.get("selection_policy")
    if not isinstance(policy, dict) or any(
        policy.get(name) is not False
        for name in (
            "dense_output_used_for_selection",
            "rgb_match_used_for_selection",
            "range_reference_used_for_selection",
        )
    ):
        raise ValueError("selection must be output, RGB-match and range blind")
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 12:
        raise ValueError("frozen protocol requires exactly twelve pairs")
    dataset_counts = Counter(str(item.get("dataset_id")) for item in pairs)
    if dataset_counts != Counter({name: 6 for name in DATASETS}):
        raise ValueError("frozen protocol requires six pairs per dataset")
    stanford_groups = Counter(
        str(item["group"]) for item in pairs if item["dataset_id"] == "stanford2d3d"
    )
    if stanford_groups != Counter({"area_2": 2, "area_3": 2, "area_4": 2}):
        raise ValueError("Stanford pairs must be balanced across areas 2, 3 and 4")
    matterport_groups = {
        str(item["group"]) for item in pairs if item["dataset_id"] == "matterport360"
    }
    if len(matterport_groups) != 6:
        raise ValueError("Matterport pairs must use six distinct groups")
    views = [str(item[role]) for item in pairs for role in ("reference", "source")]
    if len(views) != len(set(views)):
        raise ValueError("held-out angular manifest reuses a view")
    return payload


def _gross_relative_metrics(record: dict[str, Any]) -> dict[str, float]:
    with np.load(record["arrays"]) as archive:
        truth = np.asarray(archive["reference_radial_range_m"], dtype=np.float64)
        prediction = np.asarray(archive["estimated_radial_range_m"], dtype=np.float64)
        valid = np.asarray(archive["prediction_validity"], dtype=bool)
        population = np.asarray(archive["evaluation_population"], dtype=bool)
    weights = solid_angle_weights(population.shape)
    population_weight = float(weights[population].sum())
    accepted = population & valid & np.isfinite(prediction) & (prediction > 0.0)
    accepted_weight = float(weights[accepted].sum())
    ratio = np.maximum(
        prediction[accepted] / truth[accepted], truth[accepted] / prediction[accepted]
    )
    gross_weight = float(weights[accepted][ratio > 2.0].sum())
    return {
        "fixed_population_fraction": gross_weight / population_weight,
        "conditional_fraction": gross_weight / accepted_weight
        if accepted_weight
        else 0.0,
    }


def _micro(records: list[dict[str, Any]]) -> dict[str, Any]:
    overall = [item["metrics"]["overall"] for item in records]
    population = sum(float(item["population_weight"]) for item in overall)
    predicted = sum(float(item["predicted_valid_weight"]) for item in overall)
    correct = sum(
        float(item["delta_1_25"]) * float(item["population_weight"]) for item in overall
    )
    gross = sum(
        float(item["relative_gross_error"]["fixed_population_fraction"])
        * float(overall_item["population_weight"])
        for item, overall_item in zip(records, overall, strict=True)
    )
    return {
        "pair_count": len(records),
        "population_weight": population,
        "coverage": predicted / population,
        "fixed_delta_1_25": correct / population,
        "conditional_delta_1_25": correct / predicted,
        "fixed_gross_gt_2x": gross / population,
        "conditional_gross_gt_2x": gross / predicted,
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for dataset in DATASETS:
        cells = {
            cell.name: _micro(
                [
                    item
                    for item in records
                    if item["dataset_id"] == dataset and item["cell"] == cell.name
                ]
            )
            for cell in CELLS
        }
        baseline = cells[LEGACY.name]
        candidate = cells[ANGULAR.name]
        criteria = {
            "fixed_delta_strictly_improves": candidate["fixed_delta_1_25"]
            > baseline["fixed_delta_1_25"],
            "coverage_regression_within_limit": candidate["coverage"]
            >= baseline["coverage"] - MAXIMUM_COVERAGE_REGRESSION,
            "conditional_gross_regression_within_limit": candidate[
                "conditional_gross_gt_2x"
            ]
            <= baseline["conditional_gross_gt_2x"]
            + MAXIMUM_CONDITIONAL_GROSS_REGRESSION,
        }
        output[dataset] = {
            "cells": cells,
            "candidate_minus_legacy": {
                "coverage": candidate["coverage"] - baseline["coverage"],
                "fixed_delta_1_25": candidate["fixed_delta_1_25"]
                - baseline["fixed_delta_1_25"],
                "conditional_gross_gt_2x": candidate["conditional_gross_gt_2x"]
                - baseline["conditional_gross_gt_2x"],
            },
            "criteria": criteria,
            "scientific_acceptance": all(criteria.values()),
        }
    return output


def performance_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    cells = {}
    for cell in CELLS:
        values = np.asarray(
            [
                float(item["elapsed_seconds"])
                for item in records
                if item["cell"] == cell.name
            ],
            dtype=np.float64,
        )
        cells[cell.name] = {
            "raw_seconds": values.tolist(),
            "median_seconds": float(np.median(values)),
            "p95_seconds": float(np.percentile(values, 95.0)),
        }
    ratio = cells[ANGULAR.name]["median_seconds"] / cells[LEGACY.name]["median_seconds"]
    return {
        "cells": cells,
        "candidate_median_runtime_ratio": ratio,
        "maximum_runtime_ratio": MAXIMUM_RUNTIME_RATIO,
        "accepted": ratio <= MAXIMUM_RUNTIME_RATIO,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--stanford-root",
        type=Path,
        default=Path("data/stanford2d3d"),
    )
    parser.add_argument(
        "--matterport-root",
        type=Path,
        default=Path("data/matterport360"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--filter-backend", choices=("auto", "numpy", "native"), default="auto"
    )
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roots = {
        "stanford2d3d": args.stanford_root,
        "matterport360": args.matterport_root,
    }
    records = []
    for index, selection in enumerate(manifest["pairs"]):
        dataset = str(selection["dataset_id"])
        pair_id = str(selection["pair_id"])
        order = CELLS if index % 2 == 0 else CELLS[::-1]
        print(f"[{dataset}] {pair_id}", flush=True)
        for cell in order:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                record = _run(
                    cell=cell,
                    selection=selection,
                    root=roots[dataset],
                    output_dir=args.output_dir / pair_id / cell.name,
                    filter_backend=args.filter_backend,
                )
            record["execution_order"] = [item.name for item in order]
            record["relative_gross_error"] = _gross_relative_metrics(record)
            records.append(record)
            overall = record["metrics"]["overall"]
            print(
                f"  {cell.name}: coverage={overall['coverage']:.4f} "
                f"delta={overall['delta_1_25']:.4f} "
                f"gross={record['relative_gross_error']['conditional_fraction']:.4f} "
                f"seconds={record['elapsed_seconds']:.3f}",
                flush=True,
            )
    for selection in manifest["pairs"]:
        pair_records = [
            item for item in records if item["pair_id"] == selection["pair_id"]
        ]
        hashes = {item["evaluation_population_sha256"] for item in pair_records}
        if len(hashes) != 1:
            raise RuntimeError(
                f"evaluation population drift for {selection['pair_id']}"
            )
    aggregates = aggregate_records(records)
    evidence = {
        "schema": SCHEMA,
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256(args.manifest),
            "protocol_id": manifest["protocol_id"],
        },
        "configuration": {
            "erp_shape_hw": [512, 1024],
            "face_size": 256,
            "face_fov_deg": 100.0,
            "hypotheses": 96,
            "range_m": [0.3, 12.0],
            "matching_cost": "ad-census",
            "local_aggregation": "box",
            "regularizer": "sgm4",
            "legacy": {
                "erp_feature_tap_deg": None,
                "face_cost_tap_deg": None,
            },
            "candidate": {
                "erp_feature_tap_deg": ERP_FEATURE_TAP_DEG,
                "face_cost_tap_deg": FACE_COST_TAP_DEG,
            },
            "fixed_denominator": True,
            "solid_angle_weighted": True,
            "sky_and_no_return_excluded": True,
            "maximum_coverage_regression": MAXIMUM_COVERAGE_REGRESSION,
            "maximum_conditional_gross_regression": MAXIMUM_CONDITIONAL_GROSS_REGRESSION,
        },
        "records": [
            {
                key: record[key]
                for key in (
                    "cell",
                    "dataset_id",
                    "pair_id",
                    "metrics",
                    "population_audit",
                    "relative_gross_error",
                    "array_payload_sha256",
                    "evaluation_population_sha256",
                    "execution_order",
                )
            }
            for record in records
        ],
        "aggregates": aggregates,
        "scientific_acceptance": all(
            value["scientific_acceptance"] for value in aggregates.values()
        ),
    }
    (args.output_dir / "evidence.json").write_text(
        json.dumps(json_safe(evidence), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    performance = performance_summary(records)
    summary = {
        **evidence,
        "created_at": datetime.now().astimezone().isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "performance": performance,
        "accepted": bool(evidence["scientific_acceptance"])
        and bool(performance["accepted"]),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
