#!/usr/bin/env python3
"""Evaluate resolution-invariant angular stereo neighbourhoods.

The runner uses the already observed VAL-098 development pairs.  It is an
ablation for API-120, not held-out promotion evidence.
"""

# ruff: noqa: E402 -- the source-checkout root precedes PanorAi imports

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.panoramic_stereo.run_full_sphere_stability_validation import (
    evaluate_result,
    make_candidate,
    registered_pair,
    solid_angle_weights,
)
from benchmarks.panoramic_stereo.run_public_multipair_stability_validation import (
    load_manifest,
)
from benchmarks.panoramic_stereo.run_public_multiview_validation import (
    _array_payload_sha256,
)
from benchmarks.panoramic_stereo.run_real_domain_validation import (
    json_safe,
    load_public_pair_selection,
    sha256,
)
from benchmarks.panoramic_stereo.stability_protocol import (
    summarize_confidence_selectivity,
)
from panorai.stereo.panoramic import OverlappingCubemapPlanner
from panorai.stereo.panoramic._full_sphere import _face_center_pixel_step_deg


SCHEMA = "panorai-angular-stereo-neighborhood-validation/v1"
DEFAULT_MANIFEST = Path(__file__).with_name("public_multipair_manifest.json")
DATASETS = ("stanford2d3d", "matterport360")
ERP_SHAPE_HW = (512, 1024)
FACE_SIZE = 256
HYPOTHESES = 96
MIN_RANGE_M = 0.3
MAX_RANGE_M = 12.0
FACE_FOV_DEG = 100.0
ERP_FEATURE_TAP_DEG = 180.0 / 256.0
FACE_COST_TAP_DEG = _face_center_pixel_step_deg(
    OverlappingCubemapPlanner(
        face_size=128,
        face_fov_deg=FACE_FOV_DEG,
    ).plan()[0][1]
)


@dataclass(frozen=True, slots=True)
class AngularCell:
    name: str
    erp_feature_tap_deg: float | None
    face_cost_tap_deg: float | None


CELLS = (
    AngularCell("legacy-pixel-taps", None, None),
    AngularCell("erp-angular-only", ERP_FEATURE_TAP_DEG, None),
    AngularCell("face-angular-only", None, FACE_COST_TAP_DEG),
    AngularCell("erp-and-face-angular", ERP_FEATURE_TAP_DEG, FACE_COST_TAP_DEG),
)


def _micro(records: Iterable[dict[str, Any]]) -> dict[str, float | int | None]:
    items = list(records)
    if not items:
        return {
            "pair_count": 0,
            "coverage": None,
            "fixed_delta_1_25": None,
            "conditional_delta_1_25": None,
            "median_pair_absolute_error_m": None,
            "median_runtime_seconds": None,
        }
    overall = [item["metrics"]["overall"] for item in items]
    population = sum(float(item["population_weight"]) for item in overall)
    predicted = sum(float(item["predicted_valid_weight"]) for item in overall)
    correct = sum(
        float(item["delta_1_25"]) * float(item["population_weight"]) for item in overall
    )
    errors = [
        float(item["median_abs_error_m_valid"])
        for item in overall
        if item["median_abs_error_m_valid"] is not None
    ]
    return {
        "pair_count": len(items),
        "coverage": predicted / population,
        "fixed_delta_1_25": correct / population,
        "conditional_delta_1_25": correct / predicted if predicted else None,
        "median_pair_absolute_error_m": (float(np.median(errors)) if errors else None),
        "median_runtime_seconds": float(
            np.median([float(item["elapsed_seconds"]) for item in items])
        ),
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        cell.name: {
            dataset: _micro(
                item
                for item in records
                if item["cell"] == cell.name and item["dataset_id"] == dataset
            )
            for dataset in DATASETS
        }
        for cell in CELLS
    }


def classify_candidate(aggregates: dict[str, Any]) -> dict[str, Any]:
    baseline = aggregates["legacy-pixel-taps"]
    effects: dict[str, Any] = {}
    eligible: list[tuple[float, str]] = []
    for cell in CELLS[1:]:
        deltas = []
        coverage_ok = True
        domain_effects = {}
        for dataset in DATASETS:
            candidate = aggregates[cell.name][dataset]
            base = baseline[dataset]
            coverage_change = float(candidate["coverage"]) - float(base["coverage"])
            delta_change = float(candidate["fixed_delta_1_25"]) - float(
                base["fixed_delta_1_25"]
            )
            domain_effects[dataset] = {
                "coverage_change": coverage_change,
                "fixed_delta_1_25_change": delta_change,
            }
            coverage_ok &= coverage_change >= -0.02
            deltas.append(delta_change)
        mean_delta = float(np.mean(deltas))
        effects[cell.name] = {
            "by_domain": domain_effects,
            "coverage_gate_passed": coverage_ok,
            "mean_fixed_delta_1_25_change": mean_delta,
        }
        if coverage_ok and all(value > 0.0 for value in deltas):
            eligible.append((mean_delta, cell.name))
    return {
        "baseline": "legacy-pixel-taps",
        "candidate": max(eligible)[1] if eligible else None,
        "rule": (
            "largest mean fixed-delta improvement that is strictly positive "
            "in both domains and loses at most 0.02 coverage per domain"
        ),
        "effects": effects,
    }


def verify_common_populations(records: list[dict[str, Any]]) -> None:
    hashes: dict[tuple[str, str], set[str]] = {}
    for record in records:
        key = (record["dataset_id"], record["pair_id"])
        hashes.setdefault(key, set()).add(record["evaluation_population_sha256"])
    failures = {key: values for key, values in hashes.items() if len(values) != 1}
    if failures:
        raise RuntimeError(f"angular-neighbourhood population drift: {failures}")


def _run(
    *,
    cell: AngularCell,
    selection: dict[str, Any],
    root: Path,
    output_dir: Path,
    filter_backend: str,
) -> dict[str, Any]:
    pair = load_public_pair_selection(
        str(selection["dataset_id"]),
        root,
        ERP_SHAPE_HW,
        selection,
    )
    candidate = make_candidate(
        face_size=FACE_SIZE,
        face_fov_deg=FACE_FOV_DEG,
        min_range_m=MIN_RANGE_M,
        max_range_m=MAX_RANGE_M,
        hypotheses=HYPOTHESES,
        filter_backend=filter_backend,
        matching_cost="ad-census",
        local_aggregation="box",
        regularizer="sgm4",
        erp_feature_tap_deg=cell.erp_feature_tap_deg,
        face_cost_tap_deg=cell.face_cost_tap_deg,
    )
    started = time.perf_counter()
    result = candidate.run(registered_pair(pair))
    elapsed = time.perf_counter() - started
    evaluation, explicit_non_sky, population = evaluate_result(
        pair,
        result,
        min_range_m=MIN_RANGE_M,
        max_range_m=MAX_RANGE_M,
    )
    selectivity = summarize_confidence_selectivity(
        pair.reference_range,
        result.radial_range,
        result.confidence,
        population=population,
        prediction_validity=result.validity,
        sample_weights=solid_angle_weights(pair.reference_range.shape),
    )
    arrays = {
        "estimated_radial_range_m": result.radial_range,
        "prediction_validity": result.validity,
        "confidence": result.confidence,
        "sampling_support": result.sampling_support,
        "application_observable": result.observable,
        "reference_radial_range_m": pair.reference_range,
        "explicit_non_sky": explicit_non_sky,
        "evaluation_population": population,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "result.npz"
    np.savez_compressed(arrays_path, **arrays)
    record = {
        "cell": cell.name,
        "dataset_id": pair.dataset_id,
        "pair_id": pair.pair_id,
        "selection": dict(selection),
        "configuration": asdict(cell),
        "elapsed_seconds": elapsed,
        "metrics": evaluation["metrics"],
        "population_audit": evaluation["population_audit"],
        "confidence_selectivity": selectivity,
        "array_payload_sha256": _array_payload_sha256(arrays),
        "evaluation_population_sha256": _array_payload_sha256(
            {"evaluation_population": population}
        ),
        "arrays": str(arrays_path.resolve()),
        "diagnostics": dict(result.diagnostics),
    }
    (output_dir / "record.json").write_text(
        json.dumps(json_safe(record), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return record


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
    parser.add_argument("--pair-limit", type=int)
    parser.add_argument("--pair-ids", nargs="*")
    parser.add_argument(
        "--cells",
        nargs="+",
        choices=tuple(cell.name for cell in CELLS),
        default=tuple(cell.name for cell in CELLS),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    selections = list(manifest["pairs"])
    if args.pair_ids:
        requested = set(args.pair_ids)
        selections = [item for item in selections if item["pair_id"] in requested]
        if {item["pair_id"] for item in selections} != requested:
            raise ValueError("one or more requested pair IDs are absent from manifest")
    if args.pair_limit is not None:
        if args.pair_limit < 1:
            raise ValueError("pair-limit must be positive")
        selections = selections[: args.pair_limit]
    selected_cells = [cell for cell in CELLS if cell.name in args.cells]
    roots = {
        "stanford2d3d": args.stanford_root,
        "matterport360": args.matterport_root,
    }
    records = []
    for selection in selections:
        dataset = str(selection["dataset_id"])
        pair_id = str(selection["pair_id"])
        for cell in selected_cells:
            record_dir = args.output_dir / cell.name / pair_id
            record_path = record_dir / "record.json"
            if args.resume and record_path.exists():
                records.append(json.loads(record_path.read_text()))
                print(f"[resume] {cell.name} / {dataset} / {pair_id}", flush=True)
                continue
            print(f"[run] {cell.name} / {dataset} / {pair_id}", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                record = _run(
                    cell=cell,
                    selection=selection,
                    root=roots[dataset],
                    output_dir=record_dir,
                    filter_backend=args.filter_backend,
                )
            records.append(record)
            overall = record["metrics"]["overall"]
            print(
                f"  coverage={overall['coverage']:.4f} "
                f"delta={overall['delta_1_25']:.4f} "
                f"seconds={record['elapsed_seconds']:.3f}",
                flush=True,
            )
    if {cell.name for cell in selected_cells} == {cell.name for cell in CELLS}:
        verify_common_populations(records)
    aggregates = aggregate_records(records)
    complete = len(selections) == len(manifest["pairs"]) and {
        cell.name for cell in selected_cells
    } == {cell.name for cell in CELLS}
    conclusion = classify_candidate(aggregates) if complete else None
    evidence = {
        "schema": SCHEMA,
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256(args.manifest),
            "protocol_id": manifest["protocol_id"],
        },
        "configuration": {
            "erp_shape_hw": ERP_SHAPE_HW,
            "face_size": FACE_SIZE,
            "hypotheses": HYPOTHESES,
            "range_m": [MIN_RANGE_M, MAX_RANGE_M],
            "face_fov_deg": FACE_FOV_DEG,
            "historical_erp_feature_tap_deg": ERP_FEATURE_TAP_DEG,
            "historical_face_cost_tap_deg": FACE_COST_TAP_DEG,
            "cells": [asdict(cell) for cell in selected_cells],
            "matching_cost": "ad-census",
            "local_aggregation": "box",
            "regularizer": "sgm4",
            "sky_and_no_return_excluded": True,
            "fixed_denominator": True,
            "solid_angle_weighted": True,
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
                    "array_payload_sha256",
                    "evaluation_population_sha256",
                )
            }
            for record in records
        ],
        "aggregates": aggregates,
        "conclusion": conclusion,
        "complete": complete,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evidence.json").write_text(
        json.dumps(json_safe(evidence), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            json_safe(
                {
                    **evidence,
                    "created_at": datetime.now().astimezone().isoformat(),
                    "environment": {
                        "python": platform.python_version(),
                        "numpy": np.__version__,
                        "platform": platform.platform(),
                    },
                }
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
