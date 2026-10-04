#!/usr/bin/env python3
"""Run causal post-hoc ablations for multiscale spherical matching.

The ``run`` command reads only the frozen method-input JSONL.  It extracts a
single multiscale feature representation and a single globally ratio-tested
match set per pair, then changes only the RANSAC proposal weights.  The
``evaluate`` command is deliberately separate and is the only command that
opens evaluation-only metadata.

This is exploratory source-checkout evidence.  The references were inspected
in earlier work, so no result from this script is described as blind.
"""

# ruff: noqa: E402 -- cap native pools before NumPy/SciPy/OpenCV imports

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import csv
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any

import numpy as np

try:
    from scripts import run_baseline_depth_multiscale as _study
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    import run_baseline_depth_multiscale as _study


OUTPUT_SCHEMA = "panorai-multiscale-ablation-predictions/v1"
SUMMARY_SCHEMA = "panorai-multiscale-ablation-comparison/v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "representation-uniform",
    "persistence-proposal",
    "embedding-proposal",
    "persistence-embedding-proposal",
)
PERSISTENCE_FACTORS = {
    "wide-only": 1.0,
    "local-birth": 1.0,
    "local-split-support": 1.20,
    "persistent-multiscale": 1.35,
}


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _normalize_positive(weights: np.ndarray) -> np.ndarray:
    result = np.asarray(weights, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError("proposal weights must have shape (N,)")
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError("proposal weights must be finite and positive")
    if result.size:
        result = result / float(result.mean())
    return result.astype(np.float32)


def _persistence_weights(features: Any, indices: np.ndarray) -> np.ndarray:
    states = np.asarray(features.feature_scale_states, dtype=object)[indices]
    unknown = sorted({str(item) for item in states} - PERSISTENCE_FACTORS.keys())
    if unknown:
        raise ValueError(f"unknown scale states: {unknown}")
    return np.asarray(
        [PERSISTENCE_FACTORS[str(item)] for item in states], dtype=np.float64
    )


def _embedding_weights(
    features_a: Any,
    features_b: Any,
    indices_a: np.ndarray,
    indices_b: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return proposal-only visual graph evidence for global matches.

    Only mutual top-k node pairs above the configured threshold contribute.
    No descriptor candidate is added, removed, or rematched here.
    """

    from panorai.features._multiscale import _candidate_node_pairs

    config = features_a.config
    if features_b.config.to_dict() != config.to_dict():
        raise ValueError("multiscale feature configs differ")
    node_pairs = _candidate_node_pairs(
        features_a.nodes,
        features_b.nodes,
        threshold=config.node_match_similarity_threshold,
        top_k=config.node_match_top_k,
    )
    similarity = np.zeros(indices_a.shape[0], dtype=np.float64)
    for node_a, node_b, value in node_pairs:
        in_a = np.isin(indices_a, node_a.feature_indices, assume_unique=False)
        in_b = np.isin(indices_b, node_b.feature_indices, assume_unique=False)
        active = in_a & in_b
        similarity[active] = np.maximum(similarity[active], max(float(value), 0.0))
    raw = config.fallback_weight + config.regional_weight_gain * similarity
    return raw, {
        "candidate_node_pairs": len(node_pairs),
        "matches_with_embedding_evidence": int(np.count_nonzero(similarity)),
        "embedding_evidence_fraction": float(np.mean(similarity > 0.0))
        if similarity.size
        else 0.0,
        "node_similarity_min": float(similarity.min()) if similarity.size else 0.0,
        "node_similarity_max": float(similarity.max()) if similarity.size else 0.0,
    }


def proposal_variants(
    features_a: Any,
    features_b: Any,
    matches: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build predeclared proposal interventions over one fixed match set."""

    indices_a = np.asarray(matches.feature_indices_a, dtype=np.int64)
    indices_b = np.asarray(matches.feature_indices_b, dtype=np.int64)
    count = len(matches)
    if indices_a.shape != (count,) or indices_b.shape != (count,):
        raise ValueError("match indices must have shape (match_count,)")
    persistence = np.sqrt(
        _persistence_weights(features_a, indices_a)
        * _persistence_weights(features_b, indices_b)
    )
    embedding, diagnostics = _embedding_weights(
        features_a, features_b, indices_a, indices_b
    )
    weights = {
        "representation-uniform": _normalize_positive(np.ones(count)),
        "persistence-proposal": _normalize_positive(persistence),
        "embedding-proposal": _normalize_positive(embedding),
        "persistence-embedding-proposal": _normalize_positive(
            persistence * embedding
        ),
    }
    if tuple(weights) != VARIANTS:
        raise RuntimeError("variant order changed")
    diagnostics.update(
        persistence_state_counts_a={
            str(name): int(np.sum(features_a.feature_scale_states == name))
            for name in np.unique(features_a.feature_scale_states)
        },
        persistence_state_counts_b={
            str(name): int(np.sum(features_b.feature_scale_states == name))
            for name in np.unique(features_b.feature_scale_states)
        },
        interventions={
            name: {
                "minimum": float(value.min()) if value.size else 1.0,
                "maximum": float(value.max()) if value.size else 1.0,
                "mean": float(value.mean()) if value.size else 1.0,
            }
            for name, value in weights.items()
        },
    )
    return weights, diagnostics


def _correspondences(matches: Any, weights: np.ndarray) -> Any:
    from panorai.features import SphericalBearingCorrespondences

    valid = np.asarray(matches.valid, dtype=bool)
    return SphericalBearingCorrespondences(
        np.array(matches.bearings_a, copy=True),
        np.array(matches.bearings_b, copy=True),
        np.asarray(weights, dtype=np.float32) * valid.astype(np.float32),
        np.array(valid, copy=True),
    )


def _pose_record(pose: Any | None) -> dict[str, Any]:
    if pose is None:
        return {
            "status": "no-pose",
            "pose_returned": False,
            "method_valid": False,
            "inlier_count": 0,
            "failure_reason": "no-pose",
        }
    quality = pose.quality_report
    return {
        "status": "estimated",
        "pose_returned": True,
        "method_valid": True,
        "quality_accepted": bool(quality.accepted),
        "R_to_from_panorai": pose.R.tolist(),
        "t_direction_to_from_panorai": pose.t.tolist(),
        "inlier_count": int(pose.num_inliers),
        "num_trials": int(pose.num_trials),
        "median_parallax_deg": float(pose.median_parallax_deg),
        "cheirality_ratio": float(pose.cheirality_ratio),
        "degenerate": bool(pose.degenerate),
        "degeneracy_reasons": list(pose.degeneracy_reasons),
        "quality_report": quality.to_dict(),
        "quality_rejection_reasons": list(quality.rejection_reasons),
        "failure_reason": "",
        "minimal_solver": pose.minimal_solver,
        "robust_estimator": pose.robust_estimator,
        "sampling_diagnostics": pose.sampling_diagnostics.to_dict(),
    }


def _process_group(
    group_id: str,
    pairs: list[dict[str, Any]],
    run_id: str,
    method_version: str,
) -> dict[str, Any]:
    import cv2

    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    pipeline = _study.make_pipeline()
    feature_cache: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    started = time.perf_counter()

    def features(view_id: str, path: str) -> Any:
        if view_id not in feature_cache:
            feature_cache[view_id] = pipeline.extract(
                _study._read_rgb(path), panorama_id=view_id
            )
        return feature_cache[view_id]

    for item in pairs:
        pair_started = time.perf_counter()
        common = {
            "pair_id": item["pair_id"],
            "dataset_id": item["dataset_id"],
            "spatial_group_id": item["spatial_group_id"],
            "from_view_id": item["from_view_id"],
            "to_view_id": item["to_view_id"],
            "run_id": run_id,
            "method_version": method_version,
        }
        record: dict[str, Any] = {**common, "variants": {}}
        try:
            first = features(item["from_view_id"], item["from_rgb_path"])
            second = features(item["to_view_id"], item["to_rgb_path"])
            # This is the sole descriptor-matching call.  It applies the normal
            # global FLANN ratio test over every multiscale descriptor.
            matches = pipeline.base_pipeline.match(first.base, second.base)
            weights, diagnostics = proposal_variants(first, second, matches)
            record["shared"] = {
                "match_count": len(matches),
                "feature_count_a": len(first),
                "feature_count_b": len(second),
                "selected_nodes_a": sum(node.selected for node in first.nodes),
                "selected_nodes_b": sum(node.selected for node in second.nodes),
                "proposal_diagnostics": diagnostics,
            }
            for variant in VARIANTS:
                variant_started = time.perf_counter()
                try:
                    # Recreate the estimator so every intervention starts from
                    # the same configured random seed and stopping state.
                    estimator = _study.make_estimator()
                    pose = estimator.estimate(
                        _correspondences(matches, weights[variant])
                    )
                    variant_record = _pose_record(pose)
                except Exception as error:  # preserve every pair and variant
                    variant_record = {
                        "status": "failed",
                        "pose_returned": False,
                        "method_valid": False,
                        "inlier_count": 0,
                        "failure_reason": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                variant_record.update(
                    match_count=len(matches),
                    elapsed_seconds=time.perf_counter() - variant_started,
                )
                record["variants"][variant] = variant_record
        except Exception as error:
            record["shared"] = {
                "match_count": 0,
                "failure_reason": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            for variant in VARIANTS:
                record["variants"][variant] = {
                    "status": "failed",
                    "pose_returned": False,
                    "method_valid": False,
                    "match_count": 0,
                    "inlier_count": 0,
                    "failure_reason": record["shared"]["failure_reason"],
                }
        record["elapsed_seconds"] = time.perf_counter() - pair_started
        records.append(record)
    return {
        "schema": OUTPUT_SCHEMA,
        "run_id": run_id,
        "method_version": method_version,
        "group_id": group_id,
        "unique_views": len(feature_cache),
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }


def _group_filename(group_id: str) -> str:
    return "group-" + hashlib.sha256(group_id.encode()).hexdigest()[:16] + ".json"


def run_method(args: argparse.Namespace) -> None:
    pairs = _study.load_inputs(args.inputs, args.limit)
    import panorai.features._multiscale as multiscale_module
    import panorai.estimators.relative_pose as pose_module

    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for item in pairs:
        groups.setdefault(item["spatial_group_id"], []).append(item)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    group_dir = args.output_dir / "groups"
    group_dir.mkdir(exist_ok=True)
    manifest = {
        "schema": OUTPUT_SCHEMA,
        "study_status": "post-hoc-exploratory",
        "reference_access_during_run": False,
        "run_id": args.run_id,
        "method_version": args.method_version,
        "input_path": str(args.inputs.resolve()),
        "input_sha256": _study.sha256(args.inputs),
        "pair_count": len(pairs),
        "group_count": len(groups),
        "variants": list(VARIANTS),
        "workers": args.workers,
        "evidence_target": "source-checkout"
        if args.source_checkout
        else "installed-environment",
        "source_checkout": str(REPOSITORY_ROOT) if args.source_checkout else None,
        "intervention_contract": {
            "shared_representation": True,
            "shared_global_matches": True,
            "regional_matches_added": False,
            "full_scoring_set_preserved": True,
            "persistence_factors": PERSISTENCE_FACTORS,
        },
        "implementation_hashes": {
            "multiscale_module": _study.sha256(Path(multiscale_module.__file__)),
            "pose_module": _study.sha256(Path(pose_module.__file__)),
            "pipeline_runner": _study.sha256(Path(_study.__file__)),
            "ablation_runner": _study.sha256(Path(__file__).resolve()),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "pipeline": _study.make_pipeline().describe(),
        "estimator": _study.make_estimator().options.to_dict(),
        "created_unix": time.time(),
    }
    manifest_path = args.output_dir / "method-manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        stable = ("schema", "run_id", "method_version", "input_sha256", "pair_count")
        if any(previous.get(key) != manifest[key] for key in stable):
            raise RuntimeError("output directory belongs to another run")
    else:
        _write_json(manifest_path, manifest)

    finished: dict[str, dict[str, Any]] = {}
    pending: list[tuple[str, list[dict[str, Any]]]] = []
    for group_id, items in groups.items():
        path = group_dir / _group_filename(group_id)
        if path.exists():
            result = json.loads(path.read_text())
            if [x["pair_id"] for x in result["records"]] != [
                x["pair_id"] for x in items
            ]:
                raise RuntimeError(f"invalid cached group: {path}")
            finished.update((x["pair_id"], x) for x in result["records"])
        else:
            pending.append((group_id, items))
    started = time.perf_counter()
    completed = len(finished)
    if pending:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            futures = {
                pool.submit(
                    _process_group, group_id, items, args.run_id, args.method_version
                ): (group_id, items)
                for group_id, items in pending
            }
            for future in as_completed(futures):
                group_id, items = futures[future]
                result = future.result()
                _write_json(group_dir / _group_filename(group_id), result)
                finished.update((x["pair_id"], x) for x in result["records"])
                completed += len(items)
                rate = completed / max(time.perf_counter() - started, 1e-9)
                print(
                    f"group={group_id} pairs={completed}/{len(pairs)} "
                    f"group_seconds={result['elapsed_seconds']:.1f} "
                    f"eta_seconds={(len(pairs) - completed) / rate:.0f}",
                    flush=True,
                )

    ordered = [finished[item["pair_id"]] for item in pairs]
    summary: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "pair_count": len(ordered),
        "variants": {},
    }
    for variant in VARIANTS:
        predictions = args.output_dir / f"predictions-{variant}.jsonl"
        temporary = predictions.with_suffix(".jsonl.tmp")
        with temporary.open("w") as stream:
            stream.write(json.dumps({"manifest": manifest}, sort_keys=True) + "\n")
            for source in ordered:
                value = {
                    key: source[key]
                    for key in (
                        "pair_id",
                        "dataset_id",
                        "spatial_group_id",
                        "from_view_id",
                        "to_view_id",
                        "run_id",
                    )
                }
                value["method_version"] = f"{args.method_version}:{variant}"
                value.update(source["variants"][variant])
                value["ablation_variant"] = variant
                value["shared_diagnostics"] = source.get("shared", {})
                stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        os.replace(temporary, predictions)
        records = [item["variants"][variant] for item in ordered]
        summary["variants"][variant] = {
            "prediction_path": str(predictions),
            "prediction_sha256": _study.sha256(predictions),
            "pose_returned": sum(bool(x["pose_returned"]) for x in records),
            "quality_accepted": sum(bool(x.get("quality_accepted")) for x in records),
            "failed": sum(x["status"] == "failed" for x in records),
        }
    _write_json(args.output_dir / "method-summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _load_evaluated(path: Path, label: str) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(path.open(newline="")))
    for row in rows:
        row["method_label"] = label
        row["primary_success"] = _bool(row["primary_success"])
        row["strict_success"] = _bool(row["strict_success"])
        row["quality_accepted"] = _bool(row.get("quality_accepted", False))
        row["method_valid"] = _bool(row.get("method_valid", False))
        row["baseline_m"] = float(row["baseline_m"])
    return rows


def _paired(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    right_by_id = {item["pair_id"]: item for item in right}
    if set(right_by_id) != {item["pair_id"] for item in left}:
        raise RuntimeError("paired result IDs differ")
    gains = losses = ties_success = ties_failure = 0
    for item in left:
        before = bool(right_by_id[item["pair_id"]]["primary_success"])
        after = bool(item["primary_success"])
        gains += after and not before
        losses += before and not after
        ties_success += before and after
        ties_failure += not before and not after
    from scipy.stats import binomtest

    return {
        "gains": int(gains),
        "losses": int(losses),
        "net_change": int(gains - losses),
        "ties_success": int(ties_success),
        "ties_failure": int(ties_failure),
        "mcnemar_exact_p": float(binomtest(gains, gains + losses, 0.5).pvalue)
        if gains + losses
        else 1.0,
    }


def _quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [item for item in rows if item["quality_accepted"]]
    correct = sum(item["primary_success"] for item in accepted)
    all_correct = sum(item["primary_success"] for item in rows)
    return {
        "pair_count": len(rows),
        "method_valid": sum(item["method_valid"] for item in rows),
        "primary_success": int(all_correct),
        "strict_success": int(sum(item["strict_success"] for item in rows)),
        "quality_accepted": len(accepted),
        "accepted_primary_correct": int(correct),
        "accepted_primary_precision": float(correct / len(accepted))
        if accepted
        else None,
        "accepted_primary_recall": float(correct / all_correct) if all_correct else None,
    }


def _stratum(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = sorted({item[key] for item in rows})
    return {name: _quality([item for item in rows if item[key] == name]) for name in values}


def evaluate(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variant_rows: dict[str, list[dict[str, Any]]] = {}
    variant_comparisons: dict[str, Any] = {}
    for variant in VARIANTS:
        predictions = args.run_dir / f"predictions-{variant}.jsonl"
        destination = args.output_dir / variant
        _study.evaluate(
            SimpleNamespace(
                predictions=predictions,
                evaluation=args.evaluation,
                baseline=args.baseline_outcomes,
                output_dir=destination,
            )
        )
        variant_rows[variant] = _load_evaluated(
            destination / "evaluated-pairs.csv", variant
        )
        variant_comparisons[variant] = json.loads(
            (destination / "comparison.json").read_text()
        )

    expected_ids = [
        item["pair_id"] for item in variant_rows["representation-uniform"]
    ]

    def aligned_control(path: Path, label: str) -> list[dict[str, Any]]:
        available = {item["pair_id"]: item for item in _load_evaluated(path, label)}
        missing = [item for item in expected_ids if item not in available]
        if missing:
            raise RuntimeError(f"{label} is missing pair IDs: {missing[:3]}")
        return [available[item] for item in expected_ids]

    controls = {
        "baseline": aligned_control(args.baseline_evaluated, "baseline"),
        "regional-union-control": aligned_control(
            args.regional_control, "regional-union-control"
        ),
    }
    all_methods = {**controls, **variant_rows}
    for name, rows in all_methods.items():
        if [item["pair_id"] for item in rows] != expected_ids:
            raise RuntimeError(f"pair order differs for {name}")

    pairwise_vs_baseline = {
        name: _paired(rows, controls["baseline"])
        for name, rows in all_methods.items()
        if name != "baseline"
    }
    pairwise_incremental = {
        "persistence-vs-uniform": _paired(
            variant_rows["persistence-proposal"],
            variant_rows["representation-uniform"],
        ),
        "embedding-vs-uniform": _paired(
            variant_rows["embedding-proposal"],
            variant_rows["representation-uniform"],
        ),
        "combined-vs-uniform": _paired(
            variant_rows["persistence-embedding-proposal"],
            variant_rows["representation-uniform"],
        ),
        "combined-vs-persistence": _paired(
            variant_rows["persistence-embedding-proposal"],
            variant_rows["persistence-proposal"],
        ),
        "regional-vs-combined-safe": _paired(
            controls["regional-union-control"],
            variant_rows["persistence-embedding-proposal"],
        ),
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "study_status": "post-hoc-exploratory",
        "warning": "Evaluation references were previously inspected; this is not blind validation.",
        "run_manifest": json.loads((args.run_dir / "method-manifest.json").read_text()),
        "hashes": {
            "evaluation": _study.sha256(args.evaluation),
            "baseline_outcomes": _study.sha256(args.baseline_outcomes),
            "baseline_evaluated": _study.sha256(args.baseline_evaluated),
            "regional_control": _study.sha256(args.regional_control),
            **{
                f"predictions_{name}": _study.sha256(
                    args.run_dir / f"predictions-{name}.jsonl"
                )
                for name in VARIANTS
            },
        },
        "methods": {
            name: {
                "overall": _quality(rows),
                "by_dataset": _stratum(rows, "dataset_id"),
                "by_baseline_bin": _stratum(rows, "baseline_bin_m"),
            }
            for name, rows in all_methods.items()
        },
        "pairwise_vs_baseline": pairwise_vs_baseline,
        "pairwise_incremental": pairwise_incremental,
        "variant_comparisons": variant_comparisons,
    }
    _write_json(args.output_dir / "ablation-summary.json", summary)
    print(json.dumps(summary["methods"], indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--inputs", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--run-id", required=True)
    run.add_argument("--method-version", required=True)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--limit", type=int)
    run.add_argument("--source-checkout", action="store_true")
    run.set_defaults(handler=run_method)
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--run-dir", required=True, type=Path)
    evaluation.add_argument("--evaluation", required=True, type=Path)
    evaluation.add_argument("--baseline-outcomes", required=True, type=Path)
    evaluation.add_argument("--baseline-evaluated", required=True, type=Path)
    evaluation.add_argument("--regional-control", required=True, type=Path)
    evaluation.add_argument("--output-dir", required=True, type=Path)
    evaluation.set_defaults(handler=evaluate)
    args = parser.parse_args()
    if getattr(args, "workers", 1) <= 0:
        raise ValueError("workers must be positive")
    if getattr(args, "source_checkout", False):
        source = str(REPOSITORY_ROOT)
        if source not in sys.path:
            sys.path.insert(0, source)
        existing = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = (
            source if not existing else source + os.pathsep + existing
        )
    args.handler(args)


if __name__ == "__main__":
    main()
