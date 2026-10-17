#!/usr/bin/env python3
"""Diagnose spatial-resolution and inverse-range-lattice limits.

This is a development diagnostic over the already observed VAL-098 pairs.  It
cannot authorize a public default or a stability promotion.  The controlled
factorial uses a common 512x1024 evaluation grid while varying only cube-face
resolution and inverse-range hypothesis count.  A 256x512 anchor records the
relationship to the historical validation configuration.
"""

# ruff: noqa: E402 -- the source-checkout root precedes PanorAi imports

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
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


SCHEMA = "panorai-resolution-lattice-diagnostic/v1"
DEFAULT_MANIFEST = Path(__file__).with_name("public_multipair_manifest.json")
DATASETS = ("stanford2d3d", "matterport360")
MIN_RANGE_M = 0.3
MAX_RANGE_M = 12.0
FACE_FOV_DEG = 100.0


@dataclass(frozen=True, slots=True)
class DiagnosticCell:
    """One fixed spatial/range sampling cell."""

    name: str
    erp_shape_hw: tuple[int, int]
    face_size: int
    hypotheses: int
    factorial: bool


CELLS = (
    DiagnosticCell("historical-anchor", (256, 512), 128, 48, False),
    DiagnosticCell("spatial-low_range-low", (512, 1024), 128, 48, True),
    DiagnosticCell("spatial-low_range-high", (512, 1024), 128, 96, True),
    DiagnosticCell("spatial-high_range-low", (512, 1024), 256, 48, True),
    DiagnosticCell("spatial-high_range-high", (512, 1024), 256, 96, True),
)


def _micro(records: Iterable[dict[str, Any]]) -> dict[str, float | int | None]:
    items = list(records)
    if not items:
        return {
            "pair_count": 0,
            "population_weight": 0.0,
            "coverage": None,
            "fixed_delta_1_25": None,
            "conditional_delta_1_25": None,
            "median_pair_absolute_error_m": None,
            "median_runtime_seconds": None,
        }
    overall = [item["metrics"]["overall"] for item in items]
    population_weight = sum(float(item["population_weight"]) for item in overall)
    predicted_weight = sum(float(item["predicted_valid_weight"]) for item in overall)
    correct_weight = sum(
        float(item["delta_1_25"]) * float(item["population_weight"]) for item in overall
    )
    errors = [
        float(item["median_abs_error_m_valid"])
        for item in overall
        if item["median_abs_error_m_valid"] is not None
    ]
    return {
        "pair_count": len(items),
        "population_weight": population_weight,
        "coverage": predicted_weight / population_weight,
        "fixed_delta_1_25": correct_weight / population_weight,
        "conditional_delta_1_25": (
            correct_weight / predicted_weight if predicted_weight > 0.0 else None
        ),
        "median_pair_absolute_error_m": (float(np.median(errors)) if errors else None),
        "median_runtime_seconds": float(
            np.median([float(item["elapsed_seconds"]) for item in items])
        ),
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate fixed-denominator metrics by cell and public domain."""

    output: dict[str, Any] = {}
    for cell in CELLS:
        output[cell.name] = {
            dataset: _micro(
                item
                for item in records
                if item["cell"] == cell.name and item["dataset_id"] == dataset
            )
            for dataset in DATASETS
        }
    return output


def _metric(aggregates: dict[str, Any], cell: str, dataset: str, name: str) -> float:
    value = aggregates[cell][dataset][name]
    if value is None:
        raise ValueError(f"missing {name} for {cell}/{dataset}")
    return float(value)


def classify_ceiling(aggregates: dict[str, Any]) -> dict[str, Any]:
    """Classify the strongest controlled factor without a promotion claim."""

    baseline = "spatial-low_range-low"
    candidates = {
        "range-discretization": "spatial-low_range-high",
        "spatial-sampling": "spatial-high_range-low",
        "interaction": "spatial-high_range-high",
    }
    effects: dict[str, Any] = {}
    eligible: list[tuple[float, str]] = []
    for label, cell in candidates.items():
        by_domain: dict[str, Any] = {}
        deltas: list[float] = []
        acceptable = True
        for dataset in DATASETS:
            coverage_change = _metric(aggregates, cell, dataset, "coverage") - _metric(
                aggregates, baseline, dataset, "coverage"
            )
            accuracy_change = _metric(
                aggregates, cell, dataset, "fixed_delta_1_25"
            ) - _metric(aggregates, baseline, dataset, "fixed_delta_1_25")
            by_domain[dataset] = {
                "coverage_change": coverage_change,
                "fixed_delta_1_25_change": accuracy_change,
            }
            deltas.append(accuracy_change)
            acceptable &= coverage_change >= -0.02
        mean_delta = float(np.mean(deltas))
        effects[label] = {
            "cell": cell,
            "by_domain": by_domain,
            "mean_fixed_delta_1_25_change": mean_delta,
            "coverage_gate_passed": acceptable,
        }
        if acceptable and mean_delta >= 0.01 and all(value > 0.0 for value in deltas):
            eligible.append((mean_delta, label))
    conclusion = max(eligible)[1] if eligible else "neither"
    return {
        "baseline_cell": baseline,
        "classification": conclusion,
        "rule": (
            "largest mean fixed-delta improvement that is positive in both "
            "domains, at least 0.01 on average, and loses at most 0.02 "
            "coverage per domain; otherwise neither"
        ),
        "effects": effects,
    }


def verify_common_factorial_populations(records: list[dict[str, Any]]) -> None:
    """Require identical evaluated directions within the controlled 2x2 grid."""

    hashes: dict[tuple[str, str], set[str]] = {}
    for record in records:
        if not record["factorial"]:
            continue
        key = (record["dataset_id"], record["pair_id"])
        hashes.setdefault(key, set()).add(record["evaluation_population_sha256"])
    failures = {key: value for key, value in hashes.items() if len(value) != 1}
    if failures:
        raise RuntimeError(f"factorial population drift: {failures}")


def _run_cell_pair(
    *,
    cell: DiagnosticCell,
    selection: dict[str, Any],
    root: Path,
    output_dir: Path,
    filter_backend: str,
) -> dict[str, Any]:
    pair = load_public_pair_selection(
        str(selection["dataset_id"]),
        root,
        cell.erp_shape_hw,
        selection,
    )
    candidate = make_candidate(
        face_size=cell.face_size,
        face_fov_deg=FACE_FOV_DEG,
        min_range_m=MIN_RANGE_M,
        max_range_m=MAX_RANGE_M,
        hypotheses=cell.hypotheses,
        filter_backend=filter_backend,
        matching_cost="ad-census",
        local_aggregation="box",
        regularizer="sgm4",
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
    confidence = summarize_confidence_selectivity(
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
        "factorial": cell.factorial,
        "dataset_id": pair.dataset_id,
        "pair_id": pair.pair_id,
        "selection": dict(selection),
        "configuration": asdict(cell),
        "elapsed_seconds": elapsed,
        "metrics": evaluation["metrics"],
        "population_audit": evaluation["population_audit"],
        "confidence_selectivity": confidence,
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
    parser.add_argument(
        "--cells",
        nargs="+",
        choices=tuple(cell.name for cell in CELLS),
        default=tuple(cell.name for cell in CELLS),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.pair_limit is not None and args.pair_limit < 1:
        raise ValueError("pair-limit must be positive")
    manifest = load_manifest(args.manifest)
    selections = list(manifest["pairs"])
    if args.pair_limit is not None:
        selections = selections[: args.pair_limit]
    selected_cells = [cell for cell in CELLS if cell.name in args.cells]
    roots = {
        "stanford2d3d": args.stanford_root,
        "matterport360": args.matterport_root,
    }
    records: list[dict[str, Any]] = []
    for selection in selections:
        dataset = str(selection["dataset_id"])
        pair_id = str(selection["pair_id"])
        baseline = float(selection["baseline_m"])
        if not math.isfinite(baseline) or baseline <= 0.0:
            raise ValueError(f"invalid baseline for {pair_id}")
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
                record = _run_cell_pair(
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
    expected_factorial = {cell.name for cell in selected_cells if cell.factorial}
    if expected_factorial == {cell.name for cell in CELLS if cell.factorial}:
        verify_common_factorial_populations(records)
    aggregates = aggregate_records(records)
    complete = len(selections) == len(manifest["pairs"]) and {
        cell.name for cell in selected_cells
    } == {cell.name for cell in CELLS}
    conclusion = classify_ceiling(aggregates) if complete else None
    deterministic = {
        "schema": SCHEMA,
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256(args.manifest),
            "protocol_id": manifest["protocol_id"],
        },
        "configuration": {
            "cells": [asdict(cell) for cell in selected_cells],
            "matching_cost": "ad-census",
            "local_aggregation": "box",
            "regularizer": "sgm4",
            "range_m": [MIN_RANGE_M, MAX_RANGE_M],
            "face_fov_deg": FACE_FOV_DEG,
            "sky_and_no_return_excluded": True,
            "fixed_denominator": True,
            "solid_angle_weighted": True,
        },
        "records": [
            {
                key: record[key]
                for key in (
                    "cell",
                    "factorial",
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
        json.dumps(json_safe(deterministic), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    summary = {
        **deterministic,
        "created_at": datetime.now().astimezone().isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "performance": {
            cell.name: {
                dataset: aggregates[cell.name][dataset]["median_runtime_seconds"]
                for dataset in DATASETS
            }
            for cell in selected_cells
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
