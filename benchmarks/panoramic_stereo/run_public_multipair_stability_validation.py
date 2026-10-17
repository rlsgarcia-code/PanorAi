#!/usr/bin/env python3
"""Run the frozen public multipair full-sphere stereo promotion protocol."""

# ruff: noqa: E402 -- the source-checkout root precedes PanorAi imports

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.panoramic_stereo.run_full_sphere_stability_validation import (
    make_candidate,
    run_dataset,
)
from benchmarks.panoramic_stereo.run_real_domain_validation import (
    json_safe,
    load_public_pair_selection,
    sha256,
)


SCHEMA = "panorai-public-multipair-full-sphere-stereo/v1"
SELECTION_SCHEMA = "panorai-public-multipair-stereo-selection/v1"
DEFAULT_MANIFEST = Path(__file__).with_name("public_multipair_manifest.json")
DATASETS = ("stanford2d3d", "matterport360")
BASELINE_AUDIT_TOLERANCE_M = 5e-6


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and fail closed on changes to the frozen selection contract."""

    payload = json.loads(path.read_text())
    if payload.get("schema") != SELECTION_SCHEMA:
        raise ValueError("unexpected public multipair selection schema")
    if payload.get("frozen_before_dense_evaluation") is not True:
        raise ValueError("selection must be frozen before dense evaluation")
    policy = payload.get("selection_policy")
    if (
        not isinstance(policy, dict)
        or policy.get("dense_output_used_for_selection") is not False
    ):
        raise ValueError("selection must be explicitly dense-output blind")
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("selection manifest must contain pairs")
    pair_ids: set[str] = set()
    view_ids: set[str] = set()
    group_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    for index, item in enumerate(pairs):
        if not isinstance(item, dict):
            raise TypeError(f"pair {index} must be an object")
        required = (
            "dataset_id",
            "pair_id",
            "group",
            "reference",
            "source",
            "baseline_m",
            "minimum_directional_cloud_overlap",
            "spatial_group_id",
        )
        missing = [name for name in required if name not in item]
        if missing:
            raise ValueError(f"pair {index} is missing: {', '.join(missing)}")
        dataset_id = str(item["dataset_id"])
        if dataset_id not in DATASETS:
            raise ValueError(f"unsupported manifest dataset: {dataset_id}")
        pair_id = str(item["pair_id"])
        if pair_id in pair_ids:
            raise ValueError(f"duplicate pair_id: {pair_id}")
        pair_ids.add(pair_id)
        group = str(item["group"])
        for role in ("reference", "source"):
            view_id = f"{dataset_id}::{group}::{item[role]}"
            if view_id in view_ids:
                raise ValueError(f"reused view identifier: {view_id}")
            view_ids.add(view_id)
        baseline = float(item["baseline_m"])
        overlap = float(item["minimum_directional_cloud_overlap"])
        if not math.isfinite(baseline) or baseline <= 0.0:
            raise ValueError(f"invalid baseline for {pair_id}")
        if not math.isfinite(overlap) or not 0.0 <= overlap <= 1.0:
            raise ValueError(f"invalid overlap for {pair_id}")
        spatial_group = str(item["spatial_group_id"])
        group_counts[spatial_group] += 1
        dataset_counts[dataset_id] += 1
    if any(count > 3 for count in group_counts.values()):
        raise ValueError("manifest exceeds three pairs per spatial group")
    if dict(dataset_counts) != {"stanford2d3d": 5, "matterport360": 6}:
        raise ValueError(
            "frozen protocol requires five Stanford and six Matterport pairs"
        )
    return payload


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate fixed-denominator metrics without pooling pair medians."""

    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in DATASETS}
    for record in records:
        grouped[str(record["dataset_id"])].append(record)
    output: dict[str, Any] = {}
    for dataset_id, items in grouped.items():
        if not items:
            continue
        overall = [item["metrics"]["overall"] for item in items]
        population_weight = sum(float(item["population_weight"]) for item in overall)
        predicted_weight = sum(
            float(item["predicted_valid_weight"]) for item in overall
        )
        correct_weight = sum(
            float(item["delta_1_25"]) * float(item["population_weight"])
            for item in overall
        )
        coverage = np.asarray([item["coverage"] for item in overall], dtype=float)
        delta = np.asarray([item["delta_1_25"] for item in overall], dtype=float)
        median_error = np.asarray(
            [item["median_abs_error_m_valid"] for item in overall], dtype=float
        )
        output[dataset_id] = {
            "pair_count": len(items),
            "population": sum(int(item["population"]) for item in overall),
            "population_weight": population_weight,
            "micro_coverage": predicted_weight / population_weight,
            "micro_delta_1_25": correct_weight / population_weight,
            "pair_macro": {
                "median_coverage": float(np.median(coverage)),
                "median_delta_1_25": float(np.median(delta)),
                "median_of_pair_median_abs_error_m": float(np.median(median_error)),
            },
            "worst_pair": {
                "minimum_coverage": float(np.min(coverage)),
                "minimum_delta_1_25": float(np.min(delta)),
                "maximum_pair_median_abs_error_m": float(np.max(median_error)),
            },
        }
        selectivity = [item["confidence_selectivity"] for item in items]
        keys = tuple(selectivity[0]["retention"])
        if any(tuple(item["retention"]) != keys for item in selectivity[1:]):
            raise ValueError("confidence retention keys differ between pairs")
        retention: dict[str, dict[str, float]] = {}
        for key in keys:
            selected_weight = sum(
                float(item["retention"][key]["selected_weight"]) for item in selectivity
            )
            correct_weight = sum(
                float(item["retention"][key]["selected_correct_weight"])
                for item in selectivity
            )
            retention[key] = {
                "micro_fixed_population_coverage": selected_weight / population_weight,
                "micro_fixed_population_delta_1_25": correct_weight / population_weight,
                "micro_conditional_delta_1_25": correct_weight / selected_weight,
                "median_pair_conditional_delta_1_25": float(
                    np.median(
                        [
                            item["retention"][key]["conditional_delta_1_25"]
                            for item in selectivity
                        ]
                    )
                ),
            }
        output[dataset_id]["confidence_selectivity"] = {
            "retention": retention,
            "monotonic_pair_count": sum(
                item["monotonic_conditional_delta_1_25"] is True for item in selectivity
            ),
            "median_top_quartile_conditional_delta_gain": float(
                np.median(
                    [
                        item["top_quartile_conditional_delta_gain"]
                        for item in selectivity
                    ]
                )
            ),
        }
    return output


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
    candidate = make_candidate(
        face_size=args.face_size,
        face_fov_deg=args.face_fov_deg,
        min_range_m=args.min_range_m,
        max_range_m=args.max_range_m,
        hypotheses=args.hypotheses,
        filter_backend=args.filter_backend,
    )
    records: list[dict[str, Any]] = []
    for selection in manifest["pairs"]:
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
        pair_dir = args.output_dir / pair.pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{dataset_id}] {pair.pair_id}", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            record = run_dataset(
                pair,
                candidate,
                pair_dir,
                min_range_m=args.min_range_m,
                max_range_m=args.max_range_m,
            )
        record["selection"] = dict(selection)
        records.append(record)
        overall = record["metrics"]["overall"]
        print(
            f"  coverage={overall['coverage']:.3f} "
            f"delta_1_25={overall['delta_1_25']:.3f} "
            f"median_abs_error_m={overall['median_abs_error_m_valid']:.3f}",
            flush=True,
        )
    summary = {
        "schema": SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(),
        "evidence_target": "dirty-source-checkout-licensed-local-public-datasets",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "selection": {
            "manifest": str(args.manifest.resolve()),
            "sha256": sha256(args.manifest),
            "protocol_id": manifest["protocol_id"],
            "pair_count": len(manifest["pairs"]),
        },
        "configuration": {
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
            "fixed_denominator": True,
            "solid_angle_weighted": True,
            "runtime_warnings": "error",
            "baseline_audit_tolerance_m": BASELINE_AUDIT_TOLERANCE_M,
        },
        "options": asdict(candidate.options),
        "aggregates": aggregate_records(records),
        "records": records,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
