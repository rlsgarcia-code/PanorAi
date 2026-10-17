#!/usr/bin/env python3
"""Decompose acceptance-gate effects when adding a third stereo source.

This is a benchmark-only diagnostic.  It deliberately disables one existing
gate at a time under fixed extreme values; it does not tune a new threshold or
change the Experimental runtime defaults.
"""

# ruff: noqa: E402 -- source-checkout benchmark intentionally precedes imports

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
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
)
from benchmarks.panoramic_stereo.run_public_three_source_validation import (
    DATASETS,
    DEFAULT_MANIFEST,
    POPULATIONS,
    load_manifest,
    load_public_source_set,
    run_configuration,
)
from benchmarks.panoramic_stereo.run_real_domain_validation import (
    json_safe,
    sha256,
)
from panorai.applications import FullSphereDenseStereo


SCHEMA = "panorai-source-count-gate-decomposition/v1"
ROUTES = ("two-source", "three-source")
GATE_POLICIES: Mapping[str, Mapping[str, float]] = {
    "baseline": {},
    "no-matching-cost-gate": {"max_matching_cost": 1.0},
    "no-confidence-gate": {"min_confidence": 0.0},
    "no-matching-cost-or-confidence-gates": {
        "max_matching_cost": 1.0,
        "min_confidence": 0.0,
    },
    "no-texture-gate": {"min_texture_std": 0.0},
}


def candidate_for_policy(
    base: FullSphereDenseStereo,
    policy: str,
) -> FullSphereDenseStereo:
    """Apply one frozen gate ablation without altering other options."""

    try:
        overrides = GATE_POLICIES[policy]
    except KeyError as error:
        raise ValueError(f"unsupported gate policy: {policy}") from error
    stereo = replace(base.options.stereo, **overrides)
    return FullSphereDenseStereo(replace(base.options, stereo=stereo))


def _micro(entries: list[dict[str, Any]]) -> dict[str, float | int | None]:
    overall = [item["metrics"]["overall"] for item in entries]
    population_weight = sum(float(item["population_weight"]) for item in overall)
    predicted_weight = sum(float(item["predicted_valid_weight"]) for item in overall)
    correct_weight = sum(
        float(item["delta_1_25"]) * float(item["population_weight"]) for item in overall
    )
    gross_weight = sum(
        float(entry["label_error"]["gross_gt_2_steps_fixed_population_fraction"])
        * float(overall_item["population_weight"])
        for entry, overall_item in zip(entries, overall, strict=True)
    )
    return {
        "set_count": len(entries),
        "micro_coverage": predicted_weight / population_weight,
        "micro_delta_1_25": correct_weight / population_weight,
        "micro_gross_gt_2_steps_fixed_population_fraction": (
            gross_weight / population_weight
        ),
        "micro_gross_gt_2_steps_conditional_fraction": (
            gross_weight / predicted_weight if predicted_weight > 0.0 else None
        ),
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate every policy/route on the same fixed evaluation population."""

    output: dict[str, Any] = {}
    for dataset_id in DATASETS:
        output[dataset_id] = {}
        for policy in GATE_POLICIES:
            output[dataset_id][policy] = {}
            for route in ROUTES:
                matching = [
                    record
                    for record in records
                    if record["dataset_id"] == dataset_id
                    and record["gate_policy"] == policy
                    and record["configuration"] == route
                ]
                if not matching:
                    continue
                output[dataset_id][policy][route] = {
                    population_name: _micro(
                        [item["populations"][population_name] for item in matching]
                    )
                    for population_name in POPULATIONS
                }
            routes = output[dataset_id][policy]
            if not all(route in routes for route in ROUTES):
                continue
            comparisons: dict[str, Any] = {}
            for population_name in POPULATIONS:
                two = routes["two-source"][population_name]
                three = routes["three-source"][population_name]
                comparisons[population_name] = {
                    "three_minus_two_coverage": (
                        float(three["micro_coverage"]) - float(two["micro_coverage"])
                    ),
                    "three_minus_two_delta_1_25": (
                        float(three["micro_delta_1_25"])
                        - float(two["micro_delta_1_25"])
                    ),
                    "three_minus_two_gross_fixed_fraction": (
                        float(three["micro_gross_gt_2_steps_fixed_population_fraction"])
                        - float(two["micro_gross_gt_2_steps_fixed_population_fraction"])
                    ),
                }
            routes["three-minus-two"] = comparisons
    return output


def deterministic_evidence(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    configuration: Mapping[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return replay-comparable evidence without host paths or clocks."""

    stable_records = [
        {
            key: value
            for key, value in record.items()
            if key not in {"arrays", "elapsed_seconds"}
        }
        for record in records
    ]
    return {
        "schema": SCHEMA,
        "selection": {
            "protocol_id": manifest["protocol_id"],
            "manifest_sha256": sha256(manifest_path),
        },
        "gate_policies": {
            name: dict(overrides) for name, overrides in GATE_POLICIES.items()
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
    for selection in manifest["sets"]:
        source_set = load_public_source_set(
            selection,
            roots[str(selection["dataset_id"])],
            (args.height, args.width),
        )
        print(f"[{source_set.dataset_id}] {source_set.set_id}", flush=True)
        for policy in GATE_POLICIES:
            candidate = candidate_for_policy(base, policy)
            policy_dir = args.output_dir / source_set.set_id / policy
            policy_dir.mkdir(parents=True, exist_ok=True)
            for route in ROUTES:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    record = run_configuration(
                        source_set,
                        candidate,
                        route,
                        policy_dir,
                        min_range_m=args.min_range_m,
                        max_range_m=args.max_range_m,
                        hypotheses=args.hypotheses,
                    )
                record.update(
                    dataset_id=source_set.dataset_id,
                    set_id=source_set.set_id,
                    gate_policy=policy,
                )
                records.append(record)
                overall = record["populations"]["observable-union"]["metrics"][
                    "overall"
                ]
                print(
                    f"  {policy}/{route}: coverage={overall['coverage']:.3f} "
                    f"delta_1_25={overall['delta_1_25']:.3f}",
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
        "matching_cost": "intensity-gradient",
        "local_aggregation": "box",
        "regularizer": "sgm4",
        "source_aggregation": "median",
        "solid_angle_weighted": True,
        "fixed_denominator": True,
        "sky_excluded": True,
        "routes": list(ROUTES),
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
        "runtime_seconds": sum(float(item["elapsed_seconds"]) for item in records),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
