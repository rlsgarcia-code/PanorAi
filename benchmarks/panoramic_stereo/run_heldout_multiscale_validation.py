#!/usr/bin/env python3
"""Prospectively compare single- and two-scale AD-Census stereo costs."""

# ruff: noqa: E402 -- the source-checkout root precedes benchmark imports

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Iterable
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.panoramic_stereo.run_angular_neighborhood_validation import (
    CELLS,
    ERP_SHAPE_HW,
    FACE_FOV_DEG,
    FACE_SIZE,
    HYPOTHESES,
    MAX_RANGE_M,
    MIN_RANGE_M,
    _run,
)
from benchmarks.panoramic_stereo.run_heldout_angular_validation import (
    _gross_relative_metrics,
)
from benchmarks.panoramic_stereo.run_real_domain_validation import (
    json_safe,
    load_public_pair_selection,
    sha256,
)


SCHEMA = "panorai-heldout-multiscale-ad-census-validation/v1"
SELECTION_SCHEMA = "panorai-public-multiscale-cost-selection/v1"
DEFAULT_MANIFEST = Path(__file__).with_name("public_multiscale_cost_manifest.json")
EXPECTED_MANIFEST_SHA256 = (
    "a0f32be4461ab3f26ed1fc980ea65f03d3817985a60fb26dd331b1c3ef053603"
)
DATASETS = ("stanford2d3d", "matterport360")
COSTS = ("ad-census", "multiscale-ad-census")
MAXIMUM_COVERAGE_REGRESSION = 0.02
MAXIMUM_CONDITIONAL_GROSS_REGRESSION = 0.02
MAXIMUM_PAIR_FIXED_DELTA_REGRESSION = 0.02
MAXIMUM_RUNTIME_RATIO = 1.35
BASELINE_AUDIT_TOLERANCE_M = 5e-6
ANGULAR_CELL = next(item for item in CELLS if item.name == "erp-and-face-angular")


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the exact output-blind selection and fail closed on drift."""

    if sha256(path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("held-out multiscale manifest hash changed")
    payload = json.loads(path.read_text())
    if payload.get("schema") != SELECTION_SCHEMA:
        raise ValueError("unexpected multiscale selection schema")
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
    datasets = Counter(str(item.get("dataset_id")) for item in pairs)
    if datasets != Counter({name: 6 for name in DATASETS}):
        raise ValueError("frozen protocol requires six pairs per dataset")
    pair_ids: set[str] = set()
    view_ids: set[str] = set()
    stanford_groups: Counter[str] = Counter()
    matterport_groups: set[str] = set()
    for index, item in enumerate(pairs):
        if not isinstance(item, dict):
            raise TypeError(f"pair {index} must be an object")
        dataset = str(item.get("dataset_id"))
        pair_id = str(item.get("pair_id"))
        group = str(item.get("group"))
        if dataset not in DATASETS or pair_id in pair_ids:
            raise ValueError(f"invalid or duplicate pair: {pair_id}")
        pair_ids.add(pair_id)
        for role in ("reference", "source"):
            if role not in item:
                raise ValueError(f"pair {index} is missing {role}")
            view_id = f"{dataset}::{group}::{item[role]}"
            if view_id in view_ids:
                raise ValueError(f"reused view identifier: {view_id}")
            view_ids.add(view_id)
        baseline = float(item["baseline_m"])
        overlap = float(item["minimum_directional_cloud_overlap"])
        if not 0.2 <= baseline <= 5.0 or not 0.35 <= overlap <= 1.0:
            raise ValueError(f"pair outside frozen geometry bounds: {pair_id}")
        if dataset == "stanford2d3d":
            stanford_groups[group] += 1
        else:
            matterport_groups.add(group)
    if stanford_groups != Counter({"area_2": 2, "area_3": 2, "area_4": 2}):
        raise ValueError("Stanford selection must contain two pairs per area")
    if len(matterport_groups) != 6:
        raise ValueError("Matterport selection must use six distinct groups")
    return payload


def _cost(record: dict[str, Any]) -> str:
    return str(record["configuration"]["matching_cost"])


def _micro(records: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    items = list(records)
    overall = [item["metrics"]["overall"] for item in items]
    population = sum(float(item["population_weight"]) for item in overall)
    predicted = sum(float(item["predicted_valid_weight"]) for item in overall)
    correct = sum(
        float(item["delta_1_25"]) * float(item["population_weight"]) for item in overall
    )
    gross = sum(
        float(item["relative_gross_error"]["fixed_population_fraction"])
        * float(overall_item["population_weight"])
        for item, overall_item in zip(items, overall, strict=True)
    )
    return {
        "pair_count": len(items),
        "population_weight": population,
        "coverage": predicted / population,
        "fixed_delta_1_25": correct / population,
        "conditional_delta_1_25": correct / predicted,
        "fixed_gross_gt_2x": gross / population,
        "conditional_gross_gt_2x": gross / predicted,
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen domain and per-pair scientific acceptance gates."""

    output: dict[str, Any] = {}
    pair_regressions: dict[str, float] = {}
    for dataset in DATASETS:
        costs = {
            cost: _micro(
                item
                for item in records
                if item["dataset_id"] == dataset and _cost(item) == cost
            )
            for cost in COSTS
        }
        baseline = costs["ad-census"]
        candidate = costs["multiscale-ad-census"]
        changes = {
            "coverage": candidate["coverage"] - baseline["coverage"],
            "fixed_delta_1_25": (
                candidate["fixed_delta_1_25"] - baseline["fixed_delta_1_25"]
            ),
            "conditional_gross_gt_2x": (
                candidate["conditional_gross_gt_2x"]
                - baseline["conditional_gross_gt_2x"]
            ),
        }
        checks = {
            "fixed_delta_strictly_improves": changes["fixed_delta_1_25"] > 0.0,
            "coverage_regression_within_limit": (
                changes["coverage"] >= -MAXIMUM_COVERAGE_REGRESSION
            ),
            "conditional_gross_regression_within_limit": (
                changes["conditional_gross_gt_2x"]
                <= MAXIMUM_CONDITIONAL_GROSS_REGRESSION
            ),
        }
        output[dataset] = {
            "costs": costs,
            "candidate_minus_baseline": changes,
            "checks": checks,
            "accepted": all(checks.values()),
        }
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for item in records:
        by_pair.setdefault(str(item["pair_id"]), {})[_cost(item)] = item
    for pair_id, items in by_pair.items():
        baseline = float(items["ad-census"]["metrics"]["overall"]["delta_1_25"])
        candidate = float(
            items["multiscale-ad-census"]["metrics"]["overall"]["delta_1_25"]
        )
        pair_regressions[pair_id] = candidate - baseline
    worst_pair_change = min(pair_regressions.values())
    output["per_pair_fixed_delta_changes"] = pair_regressions
    output["maximum_pair_regression_check"] = {
        "worst_change": worst_pair_change,
        "minimum_allowed": -MAXIMUM_PAIR_FIXED_DELTA_REGRESSION,
        "accepted": worst_pair_change >= -MAXIMUM_PAIR_FIXED_DELTA_REGRESSION,
    }
    output["scientific_acceptance"] = all(
        bool(output[dataset]["accepted"]) for dataset in DATASETS
    ) and bool(output["maximum_pair_regression_check"]["accepted"])
    return output


def performance_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare paired timings while alternating execution order by pair."""

    by_pair: dict[str, dict[str, float]] = {}
    for item in records:
        by_pair.setdefault(str(item["pair_id"]), {})[_cost(item)] = float(
            item["elapsed_seconds"]
        )
    ratios = [
        values["multiscale-ad-census"] / values["ad-census"]
        for values in by_pair.values()
    ]
    median_ratio = float(np.median(ratios))
    return {
        "paired_runtime_ratios": ratios,
        "median_candidate_over_baseline": median_ratio,
        "maximum_allowed_ratio": MAXIMUM_RUNTIME_RATIO,
        "accepted": median_ratio <= MAXIMUM_RUNTIME_RATIO,
    }


def verify_common_populations(records: list[dict[str, Any]]) -> None:
    hashes: dict[str, set[str]] = {}
    for record in records:
        hashes.setdefault(str(record["pair_id"]), set()).add(
            str(record["evaluation_population_sha256"])
        )
    changed = [pair_id for pair_id, values in hashes.items() if len(values) != 1]
    if changed:
        raise RuntimeError("matching cost changed populations: " + ", ".join(changed))


def deterministic_evidence(
    manifest_path: Path,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build path- and timing-independent evidence suitable for replay."""

    stable_records = []
    for record in records:
        stable_records.append(
            {
                key: value
                for key, value in record.items()
                if key not in {"arrays", "elapsed_seconds", "execution_order"}
            }
        )
    aggregate = aggregate_records(records)
    return {
        "schema": SCHEMA,
        "selection": {
            "protocol_id": manifest["protocol_id"],
            "manifest_sha256": sha256(manifest_path),
            "pair_count": len(manifest["pairs"]),
        },
        "configuration": {
            "erp_shape_hw": ERP_SHAPE_HW,
            "face_size": FACE_SIZE,
            "face_fov_deg": FACE_FOV_DEG,
            "hypotheses": HYPOTHESES,
            "range_m": [MIN_RANGE_M, MAX_RANGE_M],
            "angular_cell": ANGULAR_CELL.name,
            "matching_costs": COSTS,
            "local_aggregation": "box",
            "regularizer": "sgm4",
            "sky_and_no_return_excluded": True,
            "fixed_denominator": True,
            "solid_angle_weighted": True,
        },
        "frozen_acceptance": {
            "strict_fixed_delta_improvement_in_both_domains": True,
            "maximum_coverage_regression": MAXIMUM_COVERAGE_REGRESSION,
            "maximum_conditional_gross_regression": (
                MAXIMUM_CONDITIONAL_GROSS_REGRESSION
            ),
            "maximum_pair_fixed_delta_regression": (
                MAXIMUM_PAIR_FIXED_DELTA_REGRESSION
            ),
            "maximum_median_runtime_ratio": MAXIMUM_RUNTIME_RATIO,
        },
        "aggregates": aggregate,
        "records": stable_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stanford-root", type=Path, required=True)
    parser.add_argument("--matterport-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--filter-backend", choices=("auto", "numpy", "native"), default="auto"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    roots = {
        "stanford2d3d": args.stanford_root,
        "matterport360": args.matterport_root,
    }
    records: list[dict[str, Any]] = []
    for pair_index, selection in enumerate(manifest["pairs"]):
        dataset = str(selection["dataset_id"])
        pair_id = str(selection["pair_id"])
        pair = load_public_pair_selection(
            dataset,
            roots[dataset],
            ERP_SHAPE_HW,
            selection,
        )
        baseline = float(np.linalg.norm(pair.translation_source_from_reference))
        if not math.isclose(
            baseline,
            float(selection["baseline_m"]),
            rel_tol=0.0,
            abs_tol=BASELINE_AUDIT_TOLERANCE_M,
        ):
            raise RuntimeError(f"baseline audit failed for {pair_id}")
        order = COSTS if pair_index % 2 == 0 else tuple(reversed(COSTS))
        for cost in order:
            record_dir = args.output_dir / cost / pair_id
            record_path = record_dir / "record.json"
            if args.resume and record_path.exists():
                record = json.loads(record_path.read_text())
                print(f"[resume] {dataset} / {pair_id} / {cost}", flush=True)
            else:
                print(f"[run] {dataset} / {pair_id} / {cost}", flush=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    record = _run(
                        cell=ANGULAR_CELL,
                        selection=selection,
                        root=roots[dataset],
                        output_dir=record_dir,
                        filter_backend=args.filter_backend,
                        matching_cost=cost,
                        regularizer="sgm4",
                    )
            record["execution_order"] = list(order)
            record["relative_gross_error"] = _gross_relative_metrics(record)
            records.append(record)
            overall = record["metrics"]["overall"]
            print(
                f"  coverage={overall['coverage']:.4f} "
                f"delta={overall['delta_1_25']:.4f} "
                f"gross={record['relative_gross_error']['conditional_fraction']:.4f} "
                f"seconds={record['elapsed_seconds']:.3f}",
                flush=True,
            )
    verify_common_populations(records)
    evidence = deterministic_evidence(args.manifest, manifest, records)
    performance = performance_summary(records)
    accepted = bool(evidence["aggregates"]["scientific_acceptance"]) and bool(
        performance["accepted"]
    )
    payload = {
        **evidence,
        "performance": performance,
        "accepted": accepted,
        "created_at": datetime.now().astimezone().isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_text = (
        json.dumps(json_safe(evidence), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    (args.output_dir / "evidence.json").write_text(evidence_text)
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "accepted": accepted,
                "evidence_sha256": hashlib.sha256(evidence_text.encode()).hexdigest(),
                "aggregates": evidence["aggregates"],
                "performance": performance,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
