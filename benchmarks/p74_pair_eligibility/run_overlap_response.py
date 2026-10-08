#!/usr/bin/env python3
"""Prepare and summarize the P74 frontend response over cloud-overlap bins."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import numpy as np


METHOD_SCHEMA = "panorai-relative-pose-frontend-method-input/v1"
EVALUATION_SCHEMA = "panorai-relative-pose-frontend-evaluation/v1"
ADAPTER = "eq-native-polar-0-150-endpoint-inclusive/v1"
BIN_ORDER = (
    "lt_0.10",
    "0.10_to_0.25",
    "0.25_to_0.50",
    "0.50_to_0.70",
    "ge_0.70",
)
EXECUTION_BIN_ORDER = tuple(reversed(BIN_ORDER))
BIN_LABELS = {
    "lt_0.10": "<10%",
    "0.10_to_0.25": "10-25%",
    "0.25_to_0.50": "25-50%",
    "0.50_to_0.70": "50-70%",
    "ge_0.70": ">=70%",
}
ROOT = Path(__file__).resolve().parents[2]
FRONTEND_RUNNER = ROOT / "benchmarks/relative_pose_frontend/run_benchmark.py"
BEST_PROFILE_NAME = "coarse-spherical-tangent-dog-rootsift-d1p5"
BEST_PROFILE = {
    "name": BEST_PROFILE_NAME,
    "detector": {
        "family": "coarse-dog",
        "octaves": 3,
        "contrast_threshold": 0.012,
        "max_keypoints": 4096,
        "proposal_height": 512,
        "proposal_width": 1024,
        "proposal_resampling": "spherical-gaussian",
        "proposal_prefilter_sigma_px": 1.0,
        "proposal_prefilter_intermediate_height": 1024,
        "fine_verification": "tangent-dog",
        "fine_candidate_multiplier": 2.0,
        "convolution_backend": "native",
    },
    "descriptor": {
        "method": "sift",
        "patch_size": 48,
        "descriptor_radius_sigmas": 6.0,
        "keypoint_diameter_in_scales": 1.5,
        "scale_multipliers": [1.0],
        "orientation_policy": "fixed-zero",
        "photometric_normalization": "local-standardization",
        "root_sift": True,
    },
    "matcher": {
        "method": "flann",
        "ratio_test": 0.72,
        "cross_check": False,
        "deduplicate_matches": True,
        "angular_dedup_threshold_deg": 0.15,
    },
    "estimator": {
        "profile": "full",
        "ranking": "msac-first",
        "refit_steps": 100,
        "sampler": "spatial",
        "max_error_deg": 1.0,
        "max_trials": 1000,
        "stability_trials": 6,
        "model_competition_trials": 128,
    },
}


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def overlap_bin(value: float) -> str:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"overlap outside [0, 1]: {value}")
    if value < 0.10:
        return "lt_0.10"
    if value < 0.25:
        return "0.10_to_0.25"
    if value < 0.50:
        return "0.25_to_0.50"
    if value < 0.70:
        return "0.50_to_0.70"
    return "ge_0.70"


def stable_hash(value: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def select_balanced(pairs: list[dict], *, per_family_bin: int, seed: str) -> list[dict]:
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in pairs:
        difficulty = float(row["difficulty_reference"]["value"])
        buckets[(row["family_id"], overlap_bin(difficulty))].append(row)
    families = sorted({row["family_id"] for row in pairs})
    selected = []
    for family in families:
        for bin_name in BIN_ORDER:
            candidates = buckets[(family, bin_name)]
            if len(candidates) < per_family_bin:
                raise RuntimeError(
                    f"{family}/{bin_name} has {len(candidates)} pairs, "
                    f"needs {per_family_bin}"
                )
            candidates.sort(key=lambda row: stable_hash(row["pair_id"], seed))
            selected.extend(candidates[:per_family_bin])
    return sorted(
        selected,
        key=lambda row: (
            EXECUTION_BIN_ORDER.index(
                overlap_bin(row["difficulty_reference"]["value"])
            ),
            row["family_id"],
            row["pair_id"],
        ),
    )


def load_pose(npz_root: Path, view_id: str) -> tuple[np.ndarray, np.ndarray]:
    path = npz_root / f"{view_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        rotation = np.asarray(data["rotation_matrix"], dtype=np.float64)
        translation = np.asarray(data["translation"], dtype=np.float64).reshape(3)
    return rotation, translation


def relative_transform(
    rotation_from: np.ndarray,
    translation_from: np.ndarray,
    rotation_to: np.ndarray,
    translation_to: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = rotation_to.T @ rotation_from
    translation = rotation_to.T @ (translation_from - translation_to)
    return rotation, translation


def prepare(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.pairs)
    selected = select_balanced(rows, per_family_bin=args.per_family_bin, seed=args.seed)
    requested_bins = set(BIN_ORDER) if args.bins == "all" else set(args.bins.split(","))
    unknown_bins = requested_bins - set(BIN_ORDER)
    if unknown_bins:
        raise ValueError(f"unknown overlap bins: {sorted(unknown_bins)}")
    selected = [
        row
        for row in selected
        if overlap_bin(row["difficulty_reference"]["value"]) in requested_bins
    ]
    methods = []
    evaluations = []
    sample = []
    pose_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in selected:
        left_id = row["from_view_id"]
        right_id = row["to_view_id"]
        for view_id in (left_id, right_id):
            if view_id not in pose_cache:
                pose_cache[view_id] = load_pose(args.npz_root, view_id)
        rotation, translation = relative_transform(
            *pose_cache[left_id], *pose_cache[right_id]
        )
        baseline = float(np.linalg.norm(translation))
        if baseline <= 1e-9:
            raise ValueError(f"zero baseline: {row['pair_id']}")
        difficulty = float(row["difficulty_reference"]["value"])
        bin_name = overlap_bin(difficulty)
        common = {
            "pair_id": row["pair_id"],
            "dataset_id": "p74_native_polar",
            "partition": "development",
            "spatial_group_id": f"p74-overlap::{row['family_id']}",
            "from_view_id": left_id,
            "to_view_id": right_id,
            "stratum": bin_name,
        }
        methods.append(
            {
                "schema": METHOD_SCHEMA,
                **common,
                "from_rgb_path": row["from_rgb_path"],
                "to_rgb_path": row["to_rgb_path"],
                "from_depth_path": None,
                "to_depth_path": None,
                "from_mask_path": None,
                "to_mask_path": None,
                "input_adapter": ADAPTER,
                "structural_support": "30deg-south-scanner-shadow",
            }
        )
        covariates = {
            "baseline_m": baseline,
            "absolute_baseline_bin_m": "not_used",
            "minimum_directional_cloud_overlap": difficulty,
            "symmetric_cloud_overlap": row["scene_cloud_intersection"][
                "symmetric_fraction"
            ],
            "symmetric_covisible_fraction": row["symmetric_covisible_fraction"],
            "overlap_bin": bin_name,
            "eligibility_status": row["selection"]["status"],
            "source": "VAL-013-frozen-cloud-overlap-census",
        }
        evaluations.append(
            {
                "schema": EVALUATION_SCHEMA,
                **common,
                "reference": {
                    "R_to_from": rotation.tolist(),
                    "t_to_from_m": translation.tolist(),
                    "t_direction_to_from": (translation / baseline).tolist(),
                    "translation_units": "meter",
                    "coordinate_equation": ("x_to = R_to_from @ x_from + t_to_from"),
                },
                "covariates": covariates,
            }
        )
        sample.append(
            {
                "pair_id": row["pair_id"],
                "family_id": row["family_id"],
                "from_view_id": left_id,
                "to_view_id": right_id,
                **covariates,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    method_path = args.output_dir / "pairs-method-inputs.jsonl"
    evaluation_path = args.output_dir / "pairs-evaluation.jsonl"
    sample_path = args.output_dir / "overlap-sample.jsonl"
    write_jsonl(method_path, methods)
    write_jsonl(evaluation_path, evaluations)
    write_jsonl(sample_path, sample)
    manifest = {
        "schema": "panorai-p74-overlap-response-sample/v1",
        "seed": args.seed,
        "per_family_bin": args.per_family_bin,
        "selected_bins": [
            bin_name for bin_name in EXECUTION_BIN_ORDER if bin_name in requested_bins
        ],
        "pair_count": len(selected),
        "bin_order": list(BIN_ORDER),
        "execution_bin_order": list(EXECUTION_BIN_ORDER),
        "counts_by_bin": dict(
            sorted(Counter(row["overlap_bin"] for row in sample).items())
        ),
        "counts_by_family": dict(
            sorted(Counter(row["family_id"] for row in sample).items())
        ),
        "input_pairs_sha256": sha256(args.pairs),
        "method_sha256": sha256(method_path),
        "evaluation_sha256": sha256(evaluation_path),
        "sample_sha256": sha256(sample_path),
        "selection_reads_frontend_predictions": False,
    }
    (args.output_dir / "sample-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def load_frontend_runner():
    module_name = "panorai_val014_relative_pose_frontend"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, FRONTEND_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frontend runner: {FRONTEND_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def best_variant(frontend):
    return frontend.Variant(
        name=BEST_PROFILE_NAME,
        family="coarse-dog",
        dog_octaves=3,
        dog_max_features=4096,
        dog_contrast=0.012,
        dog_patch_size=48,
        dog_radius_sigmas=6.0,
        dog_proposal_height=512,
        dog_proposal_resampling="spherical-gaussian",
        dog_fine_verification="tangent-dog",
        dog_keypoint_diameter_sigmas=1.5,
        dog_orientation_policy="fixed-zero",
        dog_photometric_normalization="local-standardization",
        dog_root_sift=True,
        matcher_method="flann",
        ratio_test=0.72,
        cross_check=False,
        deduplicate_matches=True,
        angular_dedup_deg=0.15,
    )


def exact_best_pipeline(_variant):
    from panorai.features import (
        FeatureMatcher,
        FeatureMatcherConfig,
        OpenCVTangentDescriptorV2Config,
        SphericalCoarseDoGDetector,
        SphericalCoarseDoGDetectorConfig,
        SphericalDoGSIFTConfig,
        SphericalDoGSIFTExtractor,
    )

    detector = SphericalCoarseDoGDetector(
        SphericalCoarseDoGDetectorConfig(
            octaves=3,
            max_keypoints=4096,
            contrast_threshold=0.012,
            convolution_backend="native",
            proposal_height=512,
            proposal_resampling="spherical-gaussian",
            proposal_prefilter_sigma_px=1.0,
            proposal_prefilter_intermediate_height=1024,
            fine_verification="tangent-dog",
            fine_candidate_multiplier=2.0,
        )
    )
    descriptor = OpenCVTangentDescriptorV2Config(
        keypoint_diameter_in_scales=1.5,
        scale_multipliers=(1.0,),
        orientation_policy="fixed-zero",
        photometric_normalization="local-standardization",
        root_sift=True,
    )
    extractor = SphericalDoGSIFTExtractor(
        SphericalDoGSIFTConfig(
            octaves=3,
            max_features=4096,
            contrast_threshold=0.012,
            patch_size=48,
            descriptor_radius_sigmas=6.0,
            descriptor_config=descriptor,
            convolution_backend="native",
        )
    )
    matcher = FeatureMatcher(
        FeatureMatcherConfig(
            method="flann",
            ratio_test=0.72,
            cross_check=False,
            deduplicate_matches=True,
            angular_dedup_threshold_deg=0.15,
        )
    )
    return detector, extractor, matcher


def run_best_cell(pair: dict) -> dict:
    frontend = load_frontend_runner()
    frontend._pipeline = exact_best_pipeline
    result = frontend._run_cell(pair, best_variant(frontend))
    result["configuration"] = BEST_PROFILE
    return result


def run_best(args: argparse.Namespace) -> None:
    pairs = read_jsonl(args.inputs)
    if args.strata:
        strata = set(args.strata.split(","))
        pairs = [row for row in pairs if row.get("stratum") in strata]
    pair_order = {row["pair_id"]: index for index, row in enumerate(pairs)}
    cell_dir = args.output_dir / "cells"
    cell_dir.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict] = {}
    jobs: list[tuple[dict, Path]] = []
    for pair in pairs:
        digest = hashlib.sha256(
            f"{pair['pair_id']}\0{BEST_PROFILE_NAME}".encode()
        ).hexdigest()[:20]
        path = cell_dir / f"cell-{digest}.json"
        if path.is_file():
            value = json.loads(path.read_text())
            if value.get("pair_id") != pair["pair_id"]:
                raise RuntimeError(f"invalid cached pair: {path}")
            if canonical_json(value.get("configuration")) != canonical_json(
                BEST_PROFILE
            ):
                raise RuntimeError(f"cached configuration differs: {path}")
            if args.retry_failed and value.get("status") == "failed":
                jobs.append((pair, path))
            else:
                completed[pair["pair_id"]] = value
        else:
            jobs.append((pair, path))
    manifest = {
        "schema": "panorai-p74-best-spherical-overlap-run/v1",
        "source_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "input_sha256": sha256(args.inputs),
        "pair_count": len(pairs),
        "strata": args.strata or "all",
        "execution_direction": "highest-overlap-to-lowest-overlap",
        "profile": BEST_PROFILE,
    }
    write_json(args.output_dir / "run-manifest.json", manifest)
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_best_cell, pair): (pair, path)
                for pair, path in jobs
            }
            for future in as_completed(futures):
                pair, path = futures[future]
                value = future.result()
                write_json(path, value)
                completed[pair["pair_id"]] = value
                print(
                    f"completed={len(completed)}/{len(pairs)} "
                    f"pair={pair['pair_id']} status={value['status']} "
                    f"matches={value.get('match_count', 0)} "
                    f"seconds={value['elapsed_seconds']:.2f}",
                    flush=True,
                )
    ordered = sorted(completed.values(), key=lambda row: pair_order[row["pair_id"]])
    prediction_path = args.output_dir / "predictions.jsonl"
    write_jsonl(prediction_path, ordered)
    summary = {
        "prediction_sha256": sha256(prediction_path),
        "cell_count": len(ordered),
        "pose_returned": sum(row["pose_returned"] for row in ordered),
        "failed": sum(row["status"] == "failed" for row in ordered),
    }
    write_json(args.output_dir / "run-summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def finite_median(rows: list[dict], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]
    return float(np.median(values)) if values else None


def wilson_interval(successes: int, count: int) -> tuple[float, float]:
    if count == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / count + z * z / (4 * count**2))
        / denominator
    )
    return center - margin, center + margin


def aggregate_bin(rows: list[dict]) -> dict:
    count = len(rows)
    result = {
        "pair_count": count,
        "overlap_min": min(row["minimum_directional_cloud_overlap"] for row in rows),
        "overlap_median": finite_median(rows, "minimum_directional_cloud_overlap"),
        "overlap_max": max(row["minimum_directional_cloud_overlap"] for row in rows),
        "median_matches": finite_median(rows, "match_count"),
        "median_inliers": finite_median(rows, "inlier_count"),
        "median_rotation_error_deg_returned": finite_median(
            [row for row in rows if row["pose_returned"]], "rotation_error_deg"
        ),
        "median_translation_error_deg_returned": finite_median(
            [row for row in rows if row["pose_returned"]],
            "translation_direction_error_deg",
        ),
        "median_reference_match_fraction_2deg": finite_median(
            rows, "reference_match_fraction_2deg"
        ),
        "median_elapsed_seconds": finite_median(rows, "elapsed_seconds"),
    }
    for field in (
        "pose_returned",
        "primary_success",
        "strict_success",
        "precise_success",
    ):
        successes = sum(bool(row[field]) for row in rows)
        low, high = wilson_interval(successes, count)
        result[field] = successes
        result[f"{field}_rate"] = successes / count
        result[f"{field}_wilson95"] = [low, high]
    return result


def joined_rows(evaluated: list[dict], sample: list[dict]) -> list[dict]:
    by_pair = {row["pair_id"]: row for row in sample}
    if len(by_pair) != len(sample):
        raise ValueError("duplicate pair in overlap sample")
    result = []
    for row in evaluated:
        if row["pair_id"] not in by_pair:
            raise ValueError(f"evaluated pair absent from sample: {row['pair_id']}")
        result.append({**row, **by_pair[row["pair_id"]]})
    if {row["pair_id"] for row in result} != set(by_pair):
        raise ValueError("evaluated/sample pair sets differ")
    return result


def render_plot(summary_rows: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray([row["overlap_median"] for row in summary_rows]) * 100.0
    labels = [BIN_LABELS[row["overlap_bin"]] for row in summary_rows]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for field, label, color in (
        ("pose_returned", "pose returned", "#2563eb"),
        ("primary_success", "broad (15/30 deg)", "#d97706"),
        ("strict_success", "strict (5/10 deg)", "#059669"),
    ):
        rates = np.asarray([row[f"{field}_rate"] for row in summary_rows])
        intervals = np.asarray([row[f"{field}_wilson95"] for row in summary_rows])
        axes[0].errorbar(
            x,
            rates * 100.0,
            yerr=np.vstack(
                ((rates - intervals[:, 0]) * 100, (intervals[:, 1] - rates) * 100)
            ),
            marker="o",
            capsize=3,
            label=label,
            color=color,
        )
    axes[0].axvline(50, color="#111827", linestyle="--", linewidth=1)
    axes[0].set(ylabel="rate (%)", ylim=(-3, 103), title="Pose recovery")
    axes[0].legend(fontsize=8)

    axes[1].plot(
        x,
        [row["median_matches"] for row in summary_rows],
        marker="o",
        label="matches",
    )
    axes[1].plot(
        x,
        [row["median_inliers"] for row in summary_rows],
        marker="o",
        label="pose inliers",
    )
    axes[1].axvline(50, color="#111827", linestyle="--", linewidth=1)
    axes[1].set(ylabel="median count", title="Correspondence support")
    axes[1].legend(fontsize=8)

    axes[2].plot(
        x,
        [row["median_rotation_error_deg_returned"] for row in summary_rows],
        marker="o",
        label="rotation",
    )
    axes[2].plot(
        x,
        [row["median_translation_error_deg_returned"] for row in summary_rows],
        marker="o",
        label="translation direction",
    )
    axes[2].axvline(50, color="#111827", linestyle="--", linewidth=1)
    axes[2].set(ylabel="median error (deg)", title="Returned-pose error")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.set_xlabel("minimum bidirectional cloud overlap (%)")
        axis.grid(alpha=0.2)
        axis.set_xticks(x, labels, rotation=20)
    figure.suptitle("P74 best spherical-convolution frontend vs spatial overlap")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def summarize(args: argparse.Namespace) -> None:
    evaluated = read_jsonl(args.evaluated)
    sample = read_jsonl(args.sample)
    rows = joined_rows(evaluated, sample)
    bins: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        bins[row["overlap_bin"]].append(row)
    if set(bins) != set(BIN_ORDER):
        raise RuntimeError(f"unexpected bins: {sorted(bins)}")
    summary_rows = [
        {"overlap_bin": bin_name, **aggregate_bin(bins[bin_name])}
        for bin_name in BIN_ORDER
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joined_path = args.output_dir / "evaluated-with-overlap.jsonl"
    summary_path = args.output_dir / "overlap-response.json"
    csv_path = args.output_dir / "overlap-response.csv"
    plot_path = args.output_dir / "overlap-response.png"
    write_jsonl(joined_path, rows)
    summary_path.write_text(
        json.dumps(summary_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    render_plot(summary_rows, plot_path)

    lines = [
        "# P74 frontend response by spatial overlap",
        "",
        "| Min cloud overlap | n | Returned | Broad | Strict | Median matches | Median inliers | Median R deg | Median t deg |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        count = row["pair_count"]
        lines.append(
            f"| {BIN_LABELS[row['overlap_bin']]} | {count} | "
            f"{row['pose_returned']}/{count} | {row['primary_success']}/{count} | "
            f"{row['strict_success']}/{count} | {row['median_matches']:.1f} | "
            f"{row['median_inliers']:.1f} | "
            f"{row['median_rotation_error_deg_returned']!s} | "
            f"{row['median_translation_error_deg_returned']!s} |"
        )
    (args.output_dir / "REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary_rows, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--pairs", required=True, type=Path)
    prepare_parser.add_argument("--npz-root", required=True, type=Path)
    prepare_parser.add_argument("--output-dir", required=True, type=Path)
    prepare_parser.add_argument("--per-family-bin", type=int, default=3)
    prepare_parser.add_argument(
        "--bins",
        default="all",
        help="all or comma-separated exact overlap-bin names",
    )
    prepare_parser.add_argument("--seed", default="panorai-val014-overlap-v1")
    prepare_parser.set_defaults(handler=prepare)
    run_parser = commands.add_parser("run-best-spherical")
    run_parser.add_argument("--inputs", required=True, type=Path)
    run_parser.add_argument("--output-dir", required=True, type=Path)
    run_parser.add_argument(
        "--strata", help="optional comma-separated exact overlap bins"
    )
    run_parser.add_argument("--workers", type=int, default=1)
    run_parser.add_argument("--retry-failed", action="store_true")
    run_parser.set_defaults(handler=run_best)
    summarize_parser = commands.add_parser("summarize")
    summarize_parser.add_argument("--evaluated", required=True, type=Path)
    summarize_parser.add_argument("--sample", required=True, type=Path)
    summarize_parser.add_argument("--output-dir", required=True, type=Path)
    summarize_parser.set_defaults(handler=summarize)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
