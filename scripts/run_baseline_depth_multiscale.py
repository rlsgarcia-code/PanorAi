#!/usr/bin/env python3
"""Run and evaluate the Experimental multiscale matcher study.

``run`` reads only the frozen method-input JSONL.  ``evaluate`` is a separate
step that opens the evaluation-only JSONL after predictions exist.  Because
the dataset references were inspected during earlier PanorAi work, all output
is labelled post-hoc/exploratory rather than blind validation.
"""

# ruff: noqa: E402 -- thread-pool limits must be set before NumPy/SciPy import

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
import math
import multiprocessing
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any

import numpy as np


INPUT_SCHEMA = "wp1-essential-pair-method-input/v1"
OUTPUT_SCHEMA = "panorai-multiscale-posthoc-predictions/v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FRAME_FROM_PANORAI = {
    "matterport360": np.diag([1.0, 1.0, -1.0]),
    "stanford2d3d": np.diag([1.0, -1.0, 1.0]),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _group_filename(group_id: str) -> str:
    return "group-" + hashlib.sha256(group_id.encode()).hexdigest()[:16] + ".json"


def _read_rgb(path_value: str) -> np.ndarray:
    import cv2

    image = cv2.imread(str(Path(path_value)), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to open RGB panorama: {path_value}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def make_pipeline():
    from panorai.features import (
        MultiscaleEmbeddingConfig,
        MultiscaleSphericalFeaturePipeline,
    )

    return MultiscaleSphericalFeaturePipeline.from_preset(
        "sift-flann",
        face_sampler="icosahedron",
        face_fov_deg=80.0,
        face_shape_hw=(512, 512),
        face_overlap_deg=15.0,
        edge_margin_px=16,
        ratio_test=0.75,
        max_features=4096,
        angular_dedup_threshold_deg=0.15,
        multiscale_config=MultiscaleEmbeddingConfig(
            local_fov_deg=(42.0, 42.0),
            local_grid_size=1,
            max_local_views_per_root=1,
            context_similarity_threshold=0.82,
            scale_similarity_threshold=0.94,
            node_match_similarity_threshold=0.55,
            node_match_top_k=2,
            cross_scale_nms_threshold_deg=0.20,
            fallback_weight=0.35,
            regional_weight_gain=1.0,
            embedding_shape_hw=(64, 64),
        ),
    )


def make_estimator():
    from panorai.estimators import RelativePoseOptions, SphericalRelativePoseEstimator

    return SphericalRelativePoseEstimator(
        RelativePoseOptions(
            max_angular_error_deg=0.5,
            confidence=0.999,
            min_inlier_ratio=0.05,
            min_num_trials=32,
            max_num_trials=500,
            dynamic_trials_multiplier=3.0,
            min_inliers=8,
            local_optimization_steps=3,
            minimal_solver_starts=16,
            minimal_solver_max_nfev=100,
            refinement_max_nfev=100,
            random_seed=7,
            min_median_parallax_deg=0.25,
        )
    )


def load_inputs(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if limit is not None:
        records = records[:limit]
    identifiers: set[str] = set()
    for record in records:
        if record.get("schema_version") != INPUT_SCHEMA:
            raise RuntimeError(f"unexpected input schema for {record.get('pair_id')}")
        if record.get("allowed_method_inputs") != ["from_rgb_path", "to_rgb_path"]:
            raise RuntimeError(f"unexpected method input contract: {record['pair_id']}")
        if record["pair_id"] in identifiers:
            raise RuntimeError(f"duplicate pair_id: {record['pair_id']}")
        identifiers.add(record["pair_id"])
    return records


def _process_group(
    group_id: str,
    pairs: list[dict[str, Any]],
    run_id: str,
    method_version: str,
) -> dict[str, Any]:
    import cv2

    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    pipeline = make_pipeline()
    estimator = make_estimator()
    feature_cache: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    started = time.perf_counter()

    def features(view_id: str, path: str):
        if view_id not in feature_cache:
            feature_cache[view_id] = pipeline.extract(
                _read_rgb(path), panorama_id=view_id
            )
        return feature_cache[view_id]

    for item in pairs:
        pair_started = time.perf_counter()
        record: dict[str, Any] = {
            "pair_id": item["pair_id"],
            "dataset_id": item["dataset_id"],
            "spatial_group_id": item["spatial_group_id"],
            "from_view_id": item["from_view_id"],
            "to_view_id": item["to_view_id"],
            "run_id": run_id,
            "method_version": method_version,
            "status": "failed",
            "pose_returned": False,
            "method_valid": False,
            "match_count": 0,
            "inlier_count": 0,
        }
        try:
            first = features(item["from_view_id"], item["from_rgb_path"])
            second = features(item["to_view_id"], item["to_rgb_path"])
            matches = pipeline.match(first, second)
            record["match_count"] = len(matches)
            record["multiscale"] = {
                "selected_nodes_a": sum(node.selected for node in first.nodes),
                "selected_nodes_b": sum(node.selected for node in second.nodes),
                "feature_count_a": len(first),
                "feature_count_b": len(second),
                "matching": matches.diagnostics,
                "source_counts": matches.describe()["source_counts"],
            }
            pose = estimator.estimate(matches.to_bearing_correspondences())
            if pose is None:
                record.update(status="no-pose", failure_reason="no-pose")
            else:
                quality = pose.quality_report
                record.update(
                    status="estimated",
                    pose_returned=True,
                    method_valid=True,
                    quality_accepted=bool(quality.accepted),
                    R_to_from_panorai=pose.R.tolist(),
                    t_direction_to_from_panorai=pose.t.tolist(),
                    inlier_count=int(pose.num_inliers),
                    num_trials=int(pose.num_trials),
                    median_parallax_deg=float(pose.median_parallax_deg),
                    cheirality_ratio=float(pose.cheirality_ratio),
                    degenerate=bool(pose.degenerate),
                    degeneracy_reasons=list(pose.degeneracy_reasons),
                    quality_report=quality.to_dict(),
                    quality_rejection_reasons=list(quality.rejection_reasons),
                    failure_reason="",
                    minimal_solver=pose.minimal_solver,
                    robust_estimator=pose.robust_estimator,
                    sampling_diagnostics=pose.sampling_diagnostics.to_dict(),
                )
        except Exception as error:  # preserve every pair and failure reason
            record["failure_reason"] = f"{type(error).__name__}: {error}"
            record["traceback"] = traceback.format_exc()
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


def run_method(args: argparse.Namespace) -> None:
    pairs = load_inputs(args.inputs, args.limit)
    import panorai.features._multiscale as multiscale_module

    implementation_path = Path(multiscale_module.__file__).resolve()
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for item in pairs:
        groups.setdefault(item["spatial_group_id"], []).append(item)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    group_dir = args.output_dir / "groups"
    group_dir.mkdir(exist_ok=True)
    manifest = {
        "schema": OUTPUT_SCHEMA,
        "study_status": "post-hoc-exploratory",
        "run_id": args.run_id,
        "method_version": args.method_version,
        "input_path": str(args.inputs.resolve()),
        "input_sha256": sha256(args.inputs),
        "pair_count": len(pairs),
        "group_count": len(groups),
        "workers": args.workers,
        "evidence_target": "source-checkout"
        if args.source_checkout
        else "installed-environment",
        "source_checkout": str(REPOSITORY_ROOT) if args.source_checkout else None,
        "implementation_hashes": {
            "multiscale_module": sha256(implementation_path),
            "study_runner": sha256(Path(__file__).resolve()),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "pipeline": make_pipeline().describe(),
        "created_unix": time.time(),
    }
    manifest_path = args.output_dir / "method-manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ("schema", "run_id", "method_version", "input_sha256", "pair_count"):
            if previous.get(key) != manifest[key]:
                raise RuntimeError("output directory belongs to another run")
    else:
        _write_json(manifest_path, manifest)

    finished: dict[str, dict[str, Any]] = {}
    pending = []
    for group_id, items in groups.items():
        path = group_dir / _group_filename(group_id)
        if path.exists():
            result = json.loads(path.read_text())
            if [x["pair_id"] for x in result["records"]] != [x["pair_id"] for x in items]:
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
                    f"eta_seconds={(len(pairs)-completed)/rate:.0f}",
                    flush=True,
                )
    ordered = [finished[item["pair_id"]] for item in pairs]
    predictions = args.output_dir / "predictions.jsonl"
    temporary = predictions.with_suffix(".jsonl.tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps({"manifest": manifest}, sort_keys=True) + "\n")
        for item in ordered:
            stream.write(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, predictions)
    summary = {
        "prediction_path": str(predictions),
        "prediction_sha256": sha256(predictions),
        "pair_count": len(ordered),
        "pose_returned": sum(bool(x["pose_returned"]) for x in ordered),
        "quality_accepted": sum(bool(x.get("quality_accepted")) for x in ordered),
        "failed": sum(x["status"] == "failed" for x in ordered),
    }
    _write_json(args.output_dir / "method-summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _rotation_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    cosine = np.clip((np.trace(estimate @ reference.T) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def _direction_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate = estimate / np.linalg.norm(estimate)
    reference = reference / np.linalg.norm(reference)
    return math.degrees(math.acos(float(np.clip(estimate @ reference, -1.0, 1.0))))


def _finite_pose(record: dict[str, Any]) -> bool:
    if not record.get("method_valid") or not record.get("pose_returned"):
        return False
    try:
        rotation = np.asarray(record["R_to_from_panorai"], dtype=np.float64)
        translation = np.asarray(record["t_direction_to_from_panorai"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return False
    return (
        rotation.shape == (3, 3)
        and translation.shape == (3,)
        and np.isfinite(rotation).all()
        and np.isfinite(translation).all()
        and abs(np.linalg.det(rotation) - 1.0) < 1e-5
        and abs(np.linalg.norm(translation) - 1.0) < 1e-5
    )


def evaluate(args: argparse.Namespace) -> None:
    prediction_lines = args.predictions.read_text().splitlines()
    manifest = json.loads(prediction_lines[0])["manifest"]
    predictions = [json.loads(line) for line in prediction_lines[1:] if line]
    all_references = [
        json.loads(line) for line in args.evaluation.read_text().splitlines() if line
    ]
    reference_by_id = {item["pair_id"]: item for item in all_references}
    references = [reference_by_id[item["pair_id"]] for item in predictions]
    baseline_rows = list(csv.DictReader(args.baseline.open(newline="")))
    baseline = {row["pair_id"]: row for row in baseline_rows}
    if len(reference_by_id) != len(all_references):
        raise RuntimeError("evaluation contains duplicate pair IDs")
    if set(baseline) != {x["pair_id"] for x in predictions} and len(predictions) == 2340:
        raise RuntimeError("baseline pair IDs differ from full prediction set")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    for prediction, reference in zip(predictions, references, strict=True):
        adapter = FRAME_FROM_PANORAI[reference["dataset_id"]]
        valid = _finite_pose(prediction)
        rotation_error = translation_error = None
        if valid:
            rotation = adapter @ np.asarray(prediction["R_to_from_panorai"]) @ adapter.T
            translation = adapter @ np.asarray(prediction["t_direction_to_from_panorai"])
            rotation_error = _rotation_error(
                rotation, np.asarray(reference["reference"]["R_to_from"])
            )
            translation_error = _direction_error(
                translation,
                np.asarray(reference["reference"]["t_direction_to_from"]),
            )
        primary = bool(valid and rotation_error <= 15.0 and translation_error <= 30.0)
        strict = bool(valid and rotation_error <= 5.0 and translation_error <= 10.0)
        baseline_row = baseline.get(prediction["pair_id"])
        baseline_valid = baseline_row is not None and baseline_row["method_valid"] == "true"
        baseline_r = float(baseline_row["rotation_error_deg"]) if baseline_valid else None
        baseline_t = (
            float(baseline_row["translation_direction_error_deg"])
            if baseline_valid
            else None
        )
        baseline_primary = bool(
            baseline_valid and baseline_r <= 15.0 and baseline_t <= 30.0
        )
        baseline_strict = bool(
            baseline_valid and baseline_r <= 5.0 and baseline_t <= 10.0
        )
        covariates = reference["covariates"]
        row = {
            "pair_id": prediction["pair_id"],
            "dataset_id": reference["dataset_id"],
            "spatial_group_id": reference["spatial_group_id"],
            "baseline_m": float(covariates["baseline_m"]),
            "baseline_bin_m": covariates["absolute_baseline_bin_m"],
            "method_valid": valid,
            "rotation_error_deg": rotation_error,
            "translation_direction_error_deg": translation_error,
            "primary_success": primary,
            "strict_success": strict,
            "quality_accepted": bool(prediction.get("quality_accepted")),
            "match_count": int(prediction.get("match_count", 0)),
            "inlier_count": int(prediction.get("inlier_count", 0)),
            "baseline_primary_success": baseline_primary,
            "baseline_strict_success": baseline_strict,
            "primary_delta": int(primary) - int(baseline_primary),
            "strict_delta": int(strict) - int(baseline_strict),
            "failure_reason": prediction.get("failure_reason", ""),
        }
        rows.append(row)
        outcomes.append(
            {
                "pair_id": prediction["pair_id"],
                "method_valid": str(valid).lower(),
                "rotation_error_deg": "" if rotation_error is None else rotation_error,
                "translation_direction_error_deg": ""
                if translation_error is None
                else translation_error,
                "match_count": row["match_count"],
                "inlier_count": row["inlier_count"],
                "failure_reason": row["failure_reason"],
                "method_version": prediction["method_version"],
                "run_id": prediction["run_id"],
            }
        )
    evaluated_path = args.output_dir / "evaluated-pairs.csv"
    with evaluated_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    outcomes_path = args.output_dir / "outcomes.csv"
    with outcomes_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(outcomes[0]))
        writer.writeheader()
        writer.writerows(outcomes)

    def aggregate(subset: list[dict[str, Any]]) -> dict[str, Any]:
        if not subset:
            return {
                "pair_count": 0,
                "multiscale_primary_success": 0,
                "baseline_primary_success": 0,
                "primary_gains": 0,
                "primary_losses": 0,
                "primary_net_change": 0,
                "paired_mcnemar_exact_p": None,
                "multiscale_strict_success": 0,
                "baseline_strict_success": 0,
                "method_valid": 0,
                "quality_accepted": 0,
                "median_match_count": None,
                "median_inlier_count": None,
            }
        gains = sum(x["primary_delta"] == 1 for x in subset)
        losses = sum(x["primary_delta"] == -1 for x in subset)
        try:
            from scipy.stats import binomtest

            p_value = float(binomtest(gains, gains + losses, 0.5).pvalue) if gains + losses else 1.0
        except ImportError:  # pragma: no cover
            p_value = None
        return {
            "pair_count": len(subset),
            "multiscale_primary_success": int(
                sum(x["primary_success"] for x in subset)
            ),
            "baseline_primary_success": int(
                sum(x["baseline_primary_success"] for x in subset)
            ),
            "primary_gains": gains,
            "primary_losses": losses,
            "primary_net_change": gains - losses,
            "paired_mcnemar_exact_p": p_value,
            "multiscale_strict_success": int(
                sum(x["strict_success"] for x in subset)
            ),
            "baseline_strict_success": int(
                sum(x["baseline_strict_success"] for x in subset)
            ),
            "method_valid": int(sum(x["method_valid"] for x in subset)),
            "quality_accepted": int(sum(x["quality_accepted"] for x in subset)),
            "median_match_count": float(np.median([x["match_count"] for x in subset])),
            "median_inlier_count": float(np.median([x["inlier_count"] for x in subset])),
        }

    comparison = {
        "schema": "panorai-multiscale-posthoc-comparison/v1",
        "study_status": "post-hoc-exploratory",
        "warning": "Evaluation references were previously inspected; this is not blind validation.",
        "prediction_manifest": manifest,
        "hashes": {
            "predictions": sha256(args.predictions),
            "evaluation": sha256(args.evaluation),
            "baseline": sha256(args.baseline),
            "evaluated_pairs": sha256(evaluated_path),
            "outcomes": sha256(outcomes_path),
        },
        "overall": aggregate(rows),
        "by_dataset": {
            name: aggregate([x for x in rows if x["dataset_id"] == name])
            for name in FRAME_FROM_PANORAI
        },
        "by_baseline_bin": {
            name: aggregate([x for x in rows if x["baseline_bin_m"] == name])
            for name in sorted({x["baseline_bin_m"] for x in rows})
        },
    }
    _write_json(args.output_dir / "comparison.json", comparison)
    print(json.dumps(comparison["by_dataset"], indent=2, sort_keys=True))


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
    run.add_argument(
        "--source-checkout",
        action="store_true",
        help="import PanorAi from this repository and record source-tree evidence",
    )
    run.set_defaults(handler=run_method)
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--predictions", required=True, type=Path)
    evaluation.add_argument("--evaluation", required=True, type=Path)
    evaluation.add_argument("--baseline", required=True, type=Path)
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
        os.environ["PYTHONPATH"] = source if not existing else source + os.pathsep + existing
    args.handler(args)


if __name__ == "__main__":
    main()
