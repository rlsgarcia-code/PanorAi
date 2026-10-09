#!/usr/bin/env python3
"""Evaluate P74 pose re-estimation after dense match filtering/refinement."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panorai.features import MatchProvenance, SphericalFeatureMatches  # noqa: E402
from panorai.image_processing import spherical_resize  # noqa: E402
from panorai.stereo import (  # noqa: E402
    DenseMatchFilterOptions,
    SphericalMatchRefinementOptions,
    SphericalStereoOptions,
    estimate_spherical_range,
    filter_matches_by_dense_range,
    refine_matches_on_sphere,
)


SCHEMA = "panorai-p74-dense-pose-reestimation/v1"
PROFILE_NAME = "direct-spherical-dog-rootsift-d1p5"
FRONTEND_RUNNER = ROOT / "benchmarks/relative_pose_frontend/run_benchmark.py"
REGRESSION_LIMIT_R_DEG = 0.25
REGRESSION_LIMIT_T_DEG = 1.0


@dataclass(frozen=True)
class ExperimentConfig:
    dense_height: int = 256
    min_range: float = 0.5
    max_range: float = 20.0
    num_hypotheses: int = 64
    window_size: int = 5
    pole_margin_fraction: float = 0.06
    min_texture_std: float = 0.005
    max_matching_cost: float = 0.9
    filter_max_angular_error_deg: float = 2.5
    filter_min_valid_weight: float = 0.75
    patch_radius_px: int = 2
    search_radius_px: float = 0.6
    search_step_px: float = 0.1
    min_patch_support: float = 0.9
    min_patch_std: float = 0.005
    min_cost_improvement: float = 0.02
    max_angular_shift_deg: float = 1.5

    @property
    def dense_shape_hw(self) -> tuple[int, int]:
        return self.dense_height, 2 * self.dense_height


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frontend_runner():
    name = "panorai_geo027_relative_pose_frontend"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, FRONTEND_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frontend runner: {FRONTEND_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_prediction_cells(cells_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(cells_dir.glob("cell-*.json")):
        row = json.loads(path.read_text())
        if row.get("configuration", {}).get("name") != PROFILE_NAME:
            continue
        pair_id = str(row["pair_id"])
        if pair_id in result:
            raise ValueError(f"duplicate direct-spherical cell: {pair_id}")
        result[pair_id] = row
    return result


def join_inputs(
    methods: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    *,
    strata: set[str],
    pair_id: str | None = None,
) -> list[dict[str, Any]]:
    method_by_id = _unique_by_pair(methods, "methods")
    evaluation_by_id = _unique_by_pair(evaluations, "evaluations")
    sample_by_id = _unique_by_pair(samples, "samples")
    selected = []
    for current in sorted(method_by_id):
        sample = sample_by_id.get(current)
        if sample is None or sample.get("overlap_bin") not in strata:
            continue
        if pair_id is not None and current != pair_id:
            continue
        if current not in evaluation_by_id or current not in predictions:
            raise ValueError(f"missing evaluation or prediction: {current}")
        selected.append(
            {
                "pair_id": current,
                "method": method_by_id[current],
                "evaluation": evaluation_by_id[current],
                "sample": sample,
                "prediction": predictions[current],
            }
        )
    if pair_id is not None and not selected:
        raise ValueError(f"requested pair not found in selected strata: {pair_id}")
    if not selected:
        raise ValueError("selection contains no pairs")
    return selected


def _unique_by_pair(
    rows: list[dict[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = str(row["pair_id"])
        if pair_id in result:
            raise ValueError(f"duplicate pair in {label}: {pair_id}")
        result[pair_id] = row
    return result


def reference_pose_panorai(
    evaluation: dict[str, Any], frontend: Any
) -> tuple[np.ndarray, np.ndarray]:
    adapter = np.asarray(
        frontend._PANORAI_FROM_DATASET[evaluation["dataset_id"]], dtype=np.float64
    )
    rotation_dataset = np.asarray(
        evaluation["reference"]["R_to_from"], dtype=np.float64
    )
    translation_dataset = np.asarray(
        evaluation["reference"]["t_to_from_m"], dtype=np.float64
    )
    return (
        adapter.T @ rotation_dataset @ adapter,
        adapter.T @ translation_dataset,
    )


def build_matches(
    method: dict[str, Any], prediction: dict[str, Any]
) -> SphericalFeatureMatches:
    bearings_a = np.asarray(prediction["match_bearings_a"], dtype=np.float64)
    bearings_b = np.asarray(prediction["match_bearings_b"], dtype=np.float64)
    if bearings_a.shape != bearings_b.shape or bearings_a.ndim != 2:
        raise ValueError("serialized match bearings must have equal shape (N, 3)")
    count = bearings_a.shape[0]
    if bearings_a.shape[1:] != (3,):
        raise ValueError("serialized match bearings must have shape (N, 3)")
    checksums = (
        hashlib.sha256(np.ascontiguousarray(bearings_a).tobytes()).hexdigest(),
        hashlib.sha256(np.ascontiguousarray(bearings_b).tobytes()).hexdigest(),
    )
    face_pair = ("direct-sphere", "direct-sphere")
    return SphericalFeatureMatches(
        panorama_id_a=method["from_view_id"],
        panorama_id_b=method["to_view_id"],
        feature_indices_a=np.arange(count),
        feature_indices_b=np.arange(count),
        bearings_a=bearings_a,
        bearings_b=bearings_b,
        descriptor_distances=np.ones(count, dtype=np.float32),
        ratio_scores=None,
        mutual=None,
        valid=np.ones(count, dtype=bool),
        matcher_name="imported-direct-spherical-cell",
        matcher_config={
            **prediction["configuration"]["matcher"],
            "serialized_distances_available": False,
        },
        backend_name="p74-benchmark-import",
        backend_version="1",
        provenance=MatchProvenance(
            interface="panorai-spherical-features/v1",
            source_checksums=checksums,
            face_pairs=tuple(face_pair for _ in range(count)),
            face_pair_groups=tuple((face_pair,) for _ in range(count)),
            deduplicated=True,
            selection_reason="preserve-frozen-direct-spherical-match-order",
        ),
        keypoint_responses=np.ones((count, 2), dtype=np.float32),
        face_ids_a=np.full(count, "direct-sphere", dtype=object),
        face_ids_b=np.full(count, "direct-sphere", dtype=object),
        stability="experimental",
    )


def _resize_uint8(image: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    resized = spherical_resize(image, shape_hw)
    return np.clip(np.rint(resized), 0.0, 255.0).astype(np.uint8)


def _stereo_options(config: ExperimentConfig) -> SphericalStereoOptions:
    return SphericalStereoOptions(
        min_range=config.min_range,
        max_range=config.max_range,
        num_hypotheses=config.num_hypotheses,
        window_size=config.window_size,
        pole_margin_fraction=config.pole_margin_fraction,
        min_texture_std=config.min_texture_std,
        min_confidence=0.0,
        max_matching_cost=config.max_matching_cost,
        bidirectional_consistency=False,
        filter_backend="native",
    )


def _filter_options(config: ExperimentConfig) -> DenseMatchFilterOptions:
    return DenseMatchFilterOptions(
        max_angular_error_deg=config.filter_max_angular_error_deg,
        min_dense_confidence=0.0,
        min_valid_weight=config.filter_min_valid_weight,
    )


def _refinement_options(config: ExperimentConfig) -> SphericalMatchRefinementOptions:
    return SphericalMatchRefinementOptions(
        patch_radius_px=config.patch_radius_px,
        search_radius_px=config.search_radius_px,
        search_step_px=config.search_step_px,
        min_patch_support=config.min_patch_support,
        min_patch_std=config.min_patch_std,
        min_cost_improvement=config.min_cost_improvement,
        max_angular_shift_deg=config.max_angular_shift_deg,
    )


def _rotation_error_deg(estimated: np.ndarray, reference: np.ndarray) -> float:
    cosine = (np.trace(estimated @ reference.T) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _direction_error_deg(estimated: np.ndarray, reference: np.ndarray) -> float:
    left = estimated / np.linalg.norm(estimated)
    right = reference / np.linalg.norm(reference)
    return math.degrees(math.acos(float(np.clip(left @ right, -1.0, 1.0))))


def pose_metrics(
    rotation: np.ndarray | None,
    translation_direction: np.ndarray | None,
    quality_accepted: bool,
    reference_rotation: np.ndarray,
    reference_translation: np.ndarray,
) -> dict[str, Any]:
    if rotation is None or translation_direction is None:
        return {
            "returned": False,
            "quality_accepted": False,
            "rotation_error_deg": None,
            "translation_direction_error_deg": None,
            "broad": False,
            "strict": False,
            "precise": False,
        }
    r_error = _rotation_error_deg(rotation, reference_rotation)
    t_error = _direction_error_deg(translation_direction, reference_translation)
    return {
        "returned": True,
        "quality_accepted": bool(quality_accepted),
        "rotation_error_deg": r_error,
        "translation_direction_error_deg": t_error,
        "broad": r_error <= 15.0 and t_error <= 30.0,
        "strict": r_error <= 5.0 and t_error <= 10.0,
        "precise": r_error <= 2.0 and t_error <= 5.0,
    }


def result_pose_metrics(
    pose: Any, reference_rotation: np.ndarray, reference_translation: np.ndarray
) -> dict[str, Any]:
    if pose is None:
        return pose_metrics(
            None, None, False, reference_rotation, reference_translation
        )
    return pose_metrics(
        np.asarray(pose.rotation),
        np.asarray(pose.translation_direction),
        bool(pose.quality_report.accepted),
        reference_rotation,
        reference_translation,
    )


def run_pair(
    joined: dict[str, Any], config: ExperimentConfig, frontend: Any
) -> dict[str, Any]:
    pair_id = joined["pair_id"]
    method = joined["method"]
    evaluation = joined["evaluation"]
    sample = joined["sample"]
    prediction = joined["prediction"]
    reference_rotation, reference_translation = reference_pose_panorai(
        evaluation, frontend
    )
    initial_rotation = (
        np.asarray(prediction["rotation"], dtype=np.float64)
        if prediction.get("pose_returned")
        else None
    )
    initial_direction = (
        np.asarray(prediction["translation_direction"], dtype=np.float64)
        if prediction.get("pose_returned")
        else None
    )
    initial = pose_metrics(
        initial_rotation,
        initial_direction,
        bool(prediction.get("quality_accepted", False)),
        reference_rotation,
        reference_translation,
    )
    row: dict[str, Any] = {
        "schema": SCHEMA,
        "pair_id": pair_id,
        "overlap_bin": sample["overlap_bin"],
        "minimum_directional_cloud_overlap": sample[
            "minimum_directional_cloud_overlap"
        ],
        "initial": initial,
        "filtered": pose_metrics(
            None, None, False, reference_rotation, reference_translation
        ),
        "refined": pose_metrics(
            None, None, False, reference_rotation, reference_translation
        ),
        "configuration": asdict(config),
        "dense_scale_source": "norm(reference t_to_from_m); direction from initial pose",
    }
    if not initial["returned"]:
        row["status"] = "initial-no-pose"
        return row

    started = time.perf_counter()
    variant = frontend.Variant(name="p74-stage3-input-adapter", family="dog")
    image_a, _support_a = frontend._read_pair_side(method, "from", variant)
    image_b, _support_b = frontend._read_pair_side(method, "to", variant)
    image_a = _resize_uint8(image_a, config.dense_shape_hw)
    image_b = _resize_uint8(image_b, config.dense_shape_hw)
    row["load_resize_seconds"] = time.perf_counter() - started

    baseline = float(np.linalg.norm(reference_translation))
    translation = initial_direction * baseline
    matches = build_matches(method, prediction)

    dense_started = time.perf_counter()
    dense = estimate_spherical_range(
        image_a,
        image_b,
        initial_rotation,
        translation,
        options=_stereo_options(config),
    )
    row["dense_seconds"] = time.perf_counter() - dense_started
    row["dense_valid_fraction"] = float(dense.validity_mask.mean())

    filter_started = time.perf_counter()
    filtered = filter_matches_by_dense_range(
        matches,
        dense,
        initial_rotation,
        translation,
        options=_filter_options(config),
    )
    row["filter_seconds"] = time.perf_counter() - filter_started
    row["match_count"] = len(matches)
    row["dense_supported_matches"] = int(filtered.dense_supported_mask.sum())
    row["dense_accepted_matches"] = int(filtered.accepted_mask.sum())

    estimator = frontend._estimator(
        frontend.Variant(name="p74-stage3-full-estimator", family="dog")
    )
    pose_started = time.perf_counter()
    filtered_pose = estimator.estimate(filtered.to_bearing_correspondences())
    row["filtered_pose_seconds"] = time.perf_counter() - pose_started
    row["filtered"] = result_pose_metrics(
        filtered_pose, reference_rotation, reference_translation
    )

    refinement_started = time.perf_counter()
    refinement = refine_matches_on_sphere(
        image_a,
        image_b,
        matches,
        filtered,
        initial_rotation,
        translation,
        options=_refinement_options(config),
    )
    row["refinement_seconds"] = time.perf_counter() - refinement_started
    row["refinement_evaluated_matches"] = int(refinement.evaluated_mask.sum())
    row["refinement_applied_matches"] = int(refinement.applied_mask.sum())
    applied_gain = refinement.cost_improvement[refinement.applied_mask]
    row["median_applied_cost_improvement"] = (
        float(np.median(applied_gain)) if applied_gain.size else None
    )

    pose_started = time.perf_counter()
    refined_pose = estimator.estimate(refinement.to_bearing_correspondences())
    row["refined_pose_seconds"] = time.perf_counter() - pose_started
    row["refined"] = result_pose_metrics(
        refined_pose, reference_rotation, reference_translation
    )
    row["total_seconds"] = time.perf_counter() - started
    row["status"] = "evaluated"
    return row


def aggregate_pose(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    poses = [row[key] for row in rows]
    returned = [pose for pose in poses if pose["returned"]]
    strict = [pose for pose in poses if pose["strict"]]
    return {
        "returned": len(returned),
        "broad": sum(pose["broad"] for pose in poses),
        "strict": len(strict),
        "precise": sum(pose["precise"] for pose in poses),
        "quality_accepted": sum(pose["quality_accepted"] for pose in poses),
        "quality_accepted_wrong": sum(
            pose["quality_accepted"] and not pose["broad"] for pose in poses
        ),
        "median_rotation_error_deg_returned": _median(
            pose["rotation_error_deg"] for pose in returned
        ),
        "median_translation_error_deg_returned": _median(
            pose["translation_direction_error_deg"] for pose in returned
        ),
        "median_rotation_error_deg_strict": _median(
            pose["rotation_error_deg"] for pose in strict
        ),
        "median_translation_error_deg_strict": _median(
            pose["translation_direction_error_deg"] for pose in strict
        ),
    }


def _median(values: Iterable[float]) -> float | None:
    sequence = list(values)
    return float(np.median(sequence)) if sequence else None


def stage_gate(
    rows: list[dict[str, Any]], candidate: str = "refined"
) -> dict[str, Any]:
    initial_summary = aggregate_pose(rows, "initial")
    candidate_summary = aggregate_pose(rows, candidate)
    recovered = [
        row["pair_id"]
        for row in rows
        if not row["initial"]["strict"] and row[candidate]["strict"]
    ]
    strict_losses = [
        row["pair_id"]
        for row in rows
        if row["initial"]["strict"] and not row[candidate]["strict"]
    ]
    regressions = []
    for row in rows:
        if not row["initial"]["strict"]:
            continue
        if not row[candidate]["returned"]:
            regressions.append((math.inf, math.inf))
            continue
        regressions.append(
            (
                row[candidate]["rotation_error_deg"]
                - row["initial"]["rotation_error_deg"],
                row[candidate]["translation_direction_error_deg"]
                - row["initial"]["translation_direction_error_deg"],
            )
        )
    maximum_r = max((item[0] for item in regressions), default=0.0)
    maximum_t = max((item[1] for item in regressions), default=0.0)
    checks = {
        "strict_count_increased": candidate_summary["strict"]
        > initial_summary["strict"],
        "recovered_at_least_one_pair": bool(recovered),
        "no_initially_strict_pair_lost": not strict_losses,
        "rotation_regression_bounded": maximum_r <= REGRESSION_LIMIT_R_DEG,
        "translation_regression_bounded": maximum_t <= REGRESSION_LIMIT_T_DEG,
        "no_wrong_pose_quality_accepted": candidate_summary["quality_accepted_wrong"]
        == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "recovered_pairs": recovered,
        "strict_loss_pairs": strict_losses,
        "max_initially_strict_rotation_regression_deg": maximum_r,
        "max_initially_strict_translation_regression_deg": maximum_t,
    }


def summarize(rows: list[dict[str, Any]], config: ExperimentConfig) -> dict[str, Any]:
    evaluated = [row for row in rows if row["status"] == "evaluated"]
    return {
        "schema": SCHEMA,
        "evidence_target": "source-checkout-development",
        "configuration": asdict(config),
        "pair_count": len(rows),
        "evaluated_pair_count": len(evaluated),
        "initial": aggregate_pose(rows, "initial"),
        "filtered": aggregate_pose(rows, "filtered"),
        "refined": aggregate_pose(rows, "refined"),
        "median_dense_valid_fraction": _median(
            row["dense_valid_fraction"] for row in evaluated
        ),
        "median_dense_supported_matches": _median(
            row["dense_supported_matches"] for row in evaluated
        ),
        "median_dense_accepted_matches": _median(
            row["dense_accepted_matches"] for row in evaluated
        ),
        "median_refinement_applied_matches": _median(
            row["refinement_applied_matches"] for row in evaluated
        ),
        "median_dense_seconds": _median(row["dense_seconds"] for row in evaluated),
        "median_total_seconds": _median(row["total_seconds"] for row in evaluated),
        "filtered_stage_gate": stage_gate(rows, "filtered"),
        "refined_stage_gate": stage_gate(rows, "refined"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--cells-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--strata", default="0.50_to_0.70,ge_0.70")
    parser.add_argument("--pair-id")
    args = parser.parse_args()

    config = ExperimentConfig()
    strata = set(args.strata.split(","))
    joined = join_inputs(
        read_jsonl(args.inputs),
        read_jsonl(args.evaluation),
        read_jsonl(args.sample),
        load_prediction_cells(args.cells_dir),
        strata=strata,
        pair_id=args.pair_id,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": SCHEMA,
        "source_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "inputs_sha256": sha256(args.inputs),
        "evaluation_sha256": sha256(args.evaluation),
        "sample_sha256": sha256(args.sample),
        "profile": PROFILE_NAME,
        "strata": sorted(strata),
        "pair_count": len(joined),
        "configuration": asdict(config),
        "reference_pose_used_by_candidate": False,
        "reference_translation_norm_used_by_dense": True,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    frontend = load_frontend_runner()
    rows = []
    for index, item in enumerate(joined, start=1):
        digest = hashlib.sha256(
            f"{item['pair_id']}\0{canonical_json(asdict(config))}".encode()
        ).hexdigest()[:20]
        cell_path = args.output_dir / "cells" / f"cell-{digest}.json"
        if cell_path.is_file():
            row = json.loads(cell_path.read_text())
        else:
            row = run_pair(item, config, frontend)
            write_json(cell_path, row)
        rows.append(row)
        print(
            f"completed={index}/{len(joined)} pair={item['pair_id']} "
            f"status={row['status']} accepted={row.get('dense_accepted_matches', 0)} "
            f"initial_strict={row['initial']['strict']} "
            f"refined_strict={row['refined']['strict']}",
            flush=True,
        )
    write_jsonl(args.output_dir / "evaluated-cells.jsonl", rows)
    summary = summarize(rows, config)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["refined_stage_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
