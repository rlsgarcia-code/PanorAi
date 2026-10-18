#!/usr/bin/env python3
"""Run the frozen VAL-117 full-sphere matching-cost comparison."""

# ruff: noqa: E402 -- the source-checkout root precedes PanorAi imports

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Mapping
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.panoramic_stereo.run_full_sphere_stability_validation import (
    make_candidate,
    run_dataset,
    solid_angle_weights,
)
from benchmarks.panoramic_stereo.run_public_multiview_validation import (
    _array_payload_sha256,
    _label_error_metrics,
)
from benchmarks.panoramic_stereo.run_real_domain_validation import (
    json_safe,
    load_public_pair_selection,
    sha256,
)
from panorai.applications import FullSphereDenseStereo


SCHEMA = "panorai-heldout-matching-cost-default-validation/v1"
SELECTION_SCHEMA = "panorai-public-matching-cost-selection/v1"
DEFAULT_MANIFEST = Path(__file__).with_name("public_cost_default_manifest.json")
EXPECTED_MANIFEST_SHA256 = (
    "a9c778e87363be6cb7fbf14e8bade493bccd2347607e3f77b95fac505a89e9cc"
)
DATASETS = ("stanford2d3d", "matterport360")
COSTS = ("intensity-gradient", "ad-census")
BASELINE_AUDIT_TOLERANCE_M = 5e-6
MAXIMUM_COVERAGE_REGRESSION = 0.02
MAXIMUM_CONDITIONAL_GROSS_REGRESSION = 0.02
MAXIMUM_RUNTIME_RATIO = 1.25


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the exact frozen selection and reject any drift."""

    if sha256(path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("held-out matching-cost manifest hash changed")
    payload = json.loads(path.read_text())
    if payload.get("schema") != SELECTION_SCHEMA:
        raise ValueError("unexpected matching-cost selection schema")
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
    dataset_counts: Counter[str] = Counter()
    stanford_groups: Counter[str] = Counter()
    matterport_groups: set[str] = set()
    pair_ids: set[str] = set()
    view_ids: set[str] = set()
    for index, item in enumerate(pairs):
        if not isinstance(item, dict):
            raise TypeError(f"pair {index} must be an object")
        dataset_id = str(item.get("dataset_id"))
        pair_id = str(item.get("pair_id"))
        group = str(item.get("group"))
        if dataset_id not in DATASETS:
            raise ValueError(f"unsupported manifest dataset: {dataset_id}")
        if pair_id in pair_ids:
            raise ValueError(f"duplicate pair_id: {pair_id}")
        pair_ids.add(pair_id)
        for role in ("reference", "source"):
            if role not in item:
                raise ValueError(f"pair {index} is missing {role}")
            view_id = f"{dataset_id}::{group}::{item[role]}"
            if view_id in view_ids:
                raise ValueError(f"reused view identifier: {view_id}")
            view_ids.add(view_id)
        baseline = float(item["baseline_m"])
        overlap = float(item["minimum_directional_cloud_overlap"])
        if not 0.2 <= baseline <= 5.0:
            raise ValueError(f"baseline outside frozen interval for {pair_id}")
        if not 0.35 <= overlap <= 1.0:
            raise ValueError(f"overlap outside frozen interval for {pair_id}")
        dataset_counts[dataset_id] += 1
        if dataset_id == "stanford2d3d":
            stanford_groups[group] += 1
        else:
            matterport_groups.add(group)
    if dataset_counts != Counter({name: 6 for name in DATASETS}):
        raise ValueError("frozen protocol requires six pairs per dataset")
    if stanford_groups != Counter({"area_2": 2, "area_3": 2, "area_4": 2}):
        raise ValueError("Stanford selection must contain two pairs per area")
    if len(matterport_groups) != 6:
        raise ValueError("Matterport selection must use six distinct groups")
    return payload


def candidate_for_cost(
    base: FullSphereDenseStereo,
    matching_cost: str,
) -> FullSphereDenseStereo:
    """Change only the public unary-cost option."""

    if matching_cost not in COSTS:
        raise ValueError(f"unsupported matching cost: {matching_cost}")
    return FullSphereDenseStereo(replace(base.options, matching_cost=matching_cost))


def _micro(records: list[dict[str, Any]]) -> dict[str, Any]:
    overall = [item["metrics"]["overall"] for item in records]
    population_weight = sum(float(item["population_weight"]) for item in overall)
    predicted_weight = sum(float(item["predicted_valid_weight"]) for item in overall)
    correct_weight = sum(
        float(item["delta_1_25"]) * float(item["population_weight"]) for item in overall
    )
    gross_weight = sum(
        float(item["label_error"]["gross_gt_2_steps_fixed_population_fraction"])
        * float(overall_item["population_weight"])
        for item, overall_item in zip(records, overall, strict=True)
    )
    return {
        "pair_count": len(records),
        "population_weight": population_weight,
        "micro_coverage": predicted_weight / population_weight,
        "micro_delta_1_25": correct_weight / population_weight,
        "micro_gross_gt_2_steps_fixed_population_fraction": gross_weight
        / population_weight,
        "micro_gross_gt_2_steps_conditional_fraction": gross_weight / predicted_weight,
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate scientific metrics and apply the frozen domain criteria."""

    output: dict[str, Any] = {}
    for dataset_id in DATASETS:
        costs = {
            cost: _micro(
                [
                    item
                    for item in records
                    if item["dataset_id"] == dataset_id
                    and item["matching_cost"] == cost
                ]
            )
            for cost in COSTS
        }
        baseline = costs["intensity-gradient"]
        candidate = costs["ad-census"]
        coverage_change = float(candidate["micro_coverage"]) - float(
            baseline["micro_coverage"]
        )
        delta_change = float(candidate["micro_delta_1_25"]) - float(
            baseline["micro_delta_1_25"]
        )
        gross_change = float(
            candidate["micro_gross_gt_2_steps_conditional_fraction"]
        ) - float(baseline["micro_gross_gt_2_steps_conditional_fraction"])
        checks = {
            "strict_fixed_delta_1_25_improvement": delta_change > 0.0,
            "coverage_regression_at_most_0_02": (
                coverage_change >= -MAXIMUM_COVERAGE_REGRESSION
            ),
            "conditional_gross_regression_at_most_0_02": (
                gross_change <= MAXIMUM_CONDITIONAL_GROSS_REGRESSION
            ),
        }
        output[dataset_id] = {
            "costs": costs,
            "ad_census_minus_intensity_gradient": {
                "micro_coverage": coverage_change,
                "micro_delta_1_25": delta_change,
                "micro_gross_gt_2_steps_conditional_fraction": gross_change,
            },
            "acceptance_checks": checks,
            "accepted": all(checks.values()),
        }
    output["scientific_acceptance"] = all(
        bool(output[dataset_id]["accepted"]) for dataset_id in DATASETS
    )
    return output


def performance_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return paired runtime evidence outside deterministic output hashes."""

    medians = {
        cost: float(
            np.median(
                [
                    float(item["elapsed_seconds"])
                    for item in records
                    if item["matching_cost"] == cost
                ]
            )
        )
        for cost in COSTS
    }
    ratio = medians["ad-census"] / medians["intensity-gradient"]
    return {
        "median_seconds": medians,
        "ad_census_over_intensity_gradient": ratio,
        "maximum_allowed_ratio": MAXIMUM_RUNTIME_RATIO,
        "accepted": ratio <= MAXIMUM_RUNTIME_RATIO,
    }


def _verify_identical_populations(records: list[dict[str, Any]]) -> None:
    by_pair: dict[str, set[str]] = {}
    for record in records:
        by_pair.setdefault(str(record["pair_id"]), set()).add(
            str(record["evaluation_population_sha256"])
        )
    changed = [pair_id for pair_id, hashes in by_pair.items() if len(hashes) != 1]
    if changed:
        raise RuntimeError(
            "matching cost changed fixed evaluation populations: " + ", ".join(changed)
        )


def deterministic_evidence(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    configuration: Mapping[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return replay-comparable numerical evidence without paths or timings."""

    _verify_identical_populations(records)
    stable_records = [
        {
            key: value
            for key, value in record.items()
            if key not in {"arrays", "panel", "elapsed_seconds"}
        }
        for record in records
    ]
    return {
        "schema": SCHEMA,
        "selection": {
            "protocol_id": manifest["protocol_id"],
            "manifest_sha256": sha256(manifest_path),
            "pair_count": len(manifest["pairs"]),
        },
        "frozen_acceptance": {
            "strict_fixed_delta_1_25_improvement_in_both_domains": True,
            "maximum_coverage_regression": MAXIMUM_COVERAGE_REGRESSION,
            "maximum_conditional_gross_regression": (
                MAXIMUM_CONDITIONAL_GROSS_REGRESSION
            ),
            "maximum_median_runtime_ratio": MAXIMUM_RUNTIME_RATIO,
        },
        "configuration": dict(configuration),
        "aggregates": aggregate_records(stable_records),
        "records": stable_records,
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
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--face-size", type=int, default=128)
    parser.add_argument("--face-fov-deg", type=float, default=100.0)
    parser.add_argument("--min-range-m", type=float, default=0.3)
    parser.add_argument("--max-range-m", type=float, default=12.0)
    parser.add_argument("--hypotheses", type=int, default=48)
    parser.add_argument(
        "--filter-backend", choices=("auto", "numpy", "native"), default="auto"
    )
    args = parser.parse_args()
    if args.width != 2 * args.height:
        raise ValueError("ERP width must be twice its height")
    manifest = load_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roots = {
        "stanford2d3d": args.stanford_root,
        "matterport360": args.matterport_root,
    }
    base = make_candidate(
        face_size=args.face_size,
        face_fov_deg=args.face_fov_deg,
        min_range_m=args.min_range_m,
        max_range_m=args.max_range_m,
        hypotheses=args.hypotheses,
        filter_backend=args.filter_backend,
    )
    records: list[dict[str, Any]] = []
    for pair_index, selection in enumerate(manifest["pairs"]):
        dataset_id = str(selection["dataset_id"])
        pair = load_public_pair_selection(
            dataset_id,
            roots[dataset_id],
            (args.height, args.width),
            selection,
        )
        baseline = float(np.linalg.norm(pair.translation_source_from_reference))
        if not math.isclose(
            baseline,
            float(selection["baseline_m"]),
            rel_tol=0.0,
            abs_tol=BASELINE_AUDIT_TOLERANCE_M,
        ):
            raise RuntimeError(f"baseline mismatch for {pair.pair_id}")
        order = COSTS if pair_index % 2 == 0 else COSTS[::-1]
        print(f"[{dataset_id}] {pair.pair_id}", flush=True)
        for matching_cost in order:
            candidate = candidate_for_cost(base, matching_cost)
            cost_dir = args.output_dir / pair.pair_id / matching_cost
            cost_dir.mkdir(parents=True, exist_ok=True)
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                record = run_dataset(
                    pair,
                    candidate,
                    cost_dir,
                    min_range_m=args.min_range_m,
                    max_range_m=args.max_range_m,
                )
            with np.load(record["arrays"]) as archive:
                arrays = {name: archive[name] for name in archive.files}
            population = np.asarray(arrays["evaluation_population"], dtype=bool)
            record["matching_cost"] = matching_cost
            record["execution_order"] = list(order)
            record["selection"] = dict(selection)
            record["array_payload_sha256"] = _array_payload_sha256(arrays)
            record["evaluation_population_sha256"] = _array_payload_sha256(
                {"evaluation_population": population}
            )
            record["label_error"] = _label_error_metrics(
                np.asarray(arrays["reference_radial_range_m"]),
                np.asarray(arrays["estimated_radial_range_m"]),
                np.asarray(arrays["prediction_validity"], dtype=bool),
                population,
                solid_angle_weights(population.shape),
                min_range_m=args.min_range_m,
                max_range_m=args.max_range_m,
                hypotheses=args.hypotheses,
            )
            records.append(record)
            overall = record["metrics"]["overall"]
            print(
                f"  {matching_cost}: coverage={overall['coverage']:.3f} "
                f"delta_1_25={overall['delta_1_25']:.3f} "
                f"gross={record['label_error']['gross_gt_2_steps_conditional_fraction']:.3f} "
                f"seconds={record['elapsed_seconds']:.3f}",
                flush=True,
            )
    configuration = {
        "erp_shape_hw": [args.height, args.width],
        "face_size": args.face_size,
        "face_fov_deg": args.face_fov_deg,
        "min_range_m": args.min_range_m,
        "max_range_m": args.max_range_m,
        "hypotheses": args.hypotheses,
        "filter_backend": args.filter_backend,
        "matching_costs": list(COSTS),
        "local_aggregation": "box",
        "regularizer": "sgm4",
        "paired_execution_order": "alternating by pair index",
        "solid_angle_weighted": True,
        "fixed_denominator": True,
        "sky_and_no_return_excluded": True,
    }
    evidence = deterministic_evidence(
        manifest_path=args.manifest,
        manifest=manifest,
        configuration=configuration,
        records=records,
    )
    (args.output_dir / "evidence.json").write_text(
        json.dumps(json_safe(evidence), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    performance = performance_summary(records)
    summary = {
        **evidence,
        "created_at": datetime.now().astimezone().isoformat(),
        "evidence_target": "dirty-source-checkout-licensed-local-public-datasets",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "baseline_options": asdict(base.options),
        "performance": performance,
        "accepted": bool(evidence["aggregates"]["scientific_acceptance"])
        and bool(performance["accepted"]),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
