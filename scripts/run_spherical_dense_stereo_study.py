#!/usr/bin/env python3
"""Select and evaluate ten real spherical dense-stereo cases.

Selection uses frozen five-point/RANSAC predictions, reference pose only to
verify strict pose success, and reference depth only to measure geometric
visibility.  It never reads a dense-stereo output.  The run phase consumes the
frozen selection and compares estimated-pose/reference-baseline plane sweep
against a full-reference-pose upper bound.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import cv2
import numpy as np

from panorai.geometry import erp_pixels_to_rays, rays_to_erp_pixels
from panorai.stereo import SphericalStereoOptions, estimate_spherical_range

SELECTION_SCHEMA = "panorai-spherical-dense-stereo-selection/v1"
RESULT_SCHEMA = "panorai-spherical-dense-stereo-study/v1"
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def prediction_records(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = read_jsonl(path)
    if not records or "manifest" not in records[0]:
        raise RuntimeError("predictions must begin with a manifest record")
    return records[0]["manifest"], records[1:]


def rotation_error_deg(estimate: np.ndarray, reference: np.ndarray) -> float:
    delta = estimate @ reference.T
    cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def direction_error_deg(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate = estimate / np.linalg.norm(estimate)
    reference = reference / np.linalg.norm(reference)
    cosine = float(np.clip(estimate @ reference, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def reference_pose_panorai(evaluation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    adapter = FRAME_FROM_PANORAI[evaluation["dataset_id"]]
    reference = evaluation["reference"]
    rotation_source = np.asarray(reference["R_to_from"], dtype=np.float64)
    translation_source = np.asarray(reference["t_to_from_m"], dtype=np.float64)
    return (
        adapter.T @ rotation_source @ adapter,
        adapter.T @ translation_source,
    )


def _resize_depth(depth: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    return cv2.resize(
        np.asarray(depth, dtype=np.float32),
        (width, height),
        interpolation=cv2.INTER_NEAREST_EXACT,
    )


def _sample_wrapped(image: np.ndarray, pixels_xy: np.ndarray) -> np.ndarray:
    height, _ = image.shape
    padded = np.concatenate((image[:, -1:], image, image[:, :1]), axis=1)
    map_x = np.asarray(pixels_xy[..., 0] + 1.0, dtype=np.float32)
    map_y = np.asarray(pixels_xy[..., 1], dtype=np.float32)
    support = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_y >= 0.0)
        & (map_y <= height - 1.0)
    )
    sampled = cv2.remap(
        padded,
        np.where(support, map_x, 0.0).astype(np.float32),
        np.where(support, map_y, 0.0).astype(np.float32),
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.where(support, sampled, np.nan)


def geometric_visibility_fraction(
    depth_a_path: Path,
    depth_b_path: Path,
    rotation_b_from_a: np.ndarray,
    translation_b_from_a: np.ndarray,
    shape_hw: tuple[int, int] = (64, 128),
) -> float:
    depth_a = _resize_depth(np.load(depth_a_path, mmap_mode="r"), shape_hw)
    depth_b = _resize_depth(np.load(depth_b_path, mmap_mode="r"), shape_hw)
    height, width = shape_hw
    y, x = np.indices(shape_hw, dtype=np.float32)
    rays = erp_pixels_to_rays(np.stack((x, y), axis=-1), shape_hw)
    points_a = rays * depth_a[..., None]
    points_b = points_a @ rotation_b_from_a.T + translation_b_from_a
    projection = rays_to_erp_pixels(points_b, shape_hw)
    expected_b = np.asarray(projection.ranges, dtype=np.float32)
    observed_b = _sample_wrapped(depth_b, projection.pixels_xy)
    valid_a = np.isfinite(depth_a) & (depth_a > 0.0)
    valid_b = np.isfinite(observed_b) & (observed_b > 0.0)
    tolerance = np.maximum(0.15, 0.05 * observed_b)
    visible = valid_a & valid_b & (np.abs(expected_b - observed_b) <= tolerance)
    return float(visible.sum() / max(int(valid_a.sum()), 1))


def select_cases(args: argparse.Namespace) -> None:
    manifest, predictions = prediction_records(args.predictions)
    method_inputs = read_jsonl(args.method_inputs)
    evaluations = read_jsonl(args.pair_evaluation)
    views = {item["view_id"]: item for item in read_jsonl(args.views_evaluation)}
    by_prediction = {item["pair_id"]: item for item in predictions}
    by_input = {item["pair_id"]: item for item in method_inputs}
    if not (
        len(by_prediction) == len(by_input) == len(evaluations)
        and len(evaluations) == len({item["pair_id"] for item in evaluations})
    ):
        raise RuntimeError("prediction, method-input and evaluation joins differ")

    candidates: dict[str, list[dict[str, Any]]] = {
        "matterport360": [],
        "stanford2d3d": [],
    }
    for evaluation in evaluations:
        pair_id = evaluation["pair_id"]
        prediction = by_prediction[pair_id]
        method_input = by_input[pair_id]
        quality = bool(prediction.get("quality_accepted", False))
        if not (
            prediction.get("method_valid")
            and prediction.get("pose_returned")
            and quality
        ):
            continue
        estimated_rotation = np.asarray(
            prediction["R_to_from_panorai"], dtype=np.float64
        )
        estimated_translation = np.asarray(
            prediction["t_direction_to_from_panorai"], dtype=np.float64
        )
        reference_rotation, reference_translation = reference_pose_panorai(evaluation)
        rotation_error = rotation_error_deg(estimated_rotation, reference_rotation)
        translation_error = direction_error_deg(
            estimated_translation, reference_translation
        )
        covariates = evaluation["covariates"]
        ratio = float(covariates["baseline_over_harmonic_median_depth"])
        baseline = float(covariates["baseline_m"])
        if not (
            rotation_error <= 5.0
            and translation_error <= 10.0
            and 0.10 <= ratio <= 0.90
            and baseline >= 0.15
        ):
            continue
        from_view = views[evaluation["from_view_id"]]
        to_view = views[evaluation["to_view_id"]]
        paths = (
            Path(method_input["from_rgb_path"]),
            Path(method_input["to_rgb_path"]),
            Path(from_view["depth"]["path"]),
            Path(to_view["depth"]["path"]),
        )
        if not all(path.is_file() for path in paths):
            continue
        overlap = geometric_visibility_fraction(
            paths[2], paths[3], reference_rotation, reference_translation
        )
        candidates[evaluation["dataset_id"]].append(
            {
                "pair_id": pair_id,
                "dataset_id": evaluation["dataset_id"],
                "spatial_group_id": evaluation["spatial_group_id"],
                "from_view_id": evaluation["from_view_id"],
                "to_view_id": evaluation["to_view_id"],
                "rotation_error_deg": rotation_error,
                "translation_direction_error_deg": translation_error,
                "baseline_m": baseline,
                "baseline_over_harmonic_median_depth": ratio,
                "rgb_similarity": float(covariates["rgb_similarity"]),
                "reference_geometric_visibility_fraction": overlap,
                "inlier_count": int(prediction["inlier_count"]),
            }
        )

    selected: list[dict[str, Any]] = []
    per_dataset = args.count // 2
    for dataset_id in ("matterport360", "stanford2d3d"):
        ordered = sorted(
            candidates[dataset_id],
            key=lambda item: (
                -item["reference_geometric_visibility_fraction"],
                -item["rgb_similarity"],
                item["rotation_error_deg"] + item["translation_direction_error_deg"],
                item["pair_id"],
            ),
        )
        chosen: list[dict[str, Any]] = []
        used_groups: set[str] = set()
        used_views: set[str] = set()
        for require_new_group in (True, False):
            for item in ordered:
                if item in chosen:
                    continue
                if item["reference_geometric_visibility_fraction"] < args.min_overlap:
                    continue
                if (
                    item["from_view_id"] in used_views
                    or item["to_view_id"] in used_views
                ):
                    continue
                if require_new_group and item["spatial_group_id"] in used_groups:
                    continue
                chosen.append(item)
                used_groups.add(item["spatial_group_id"])
                used_views.update((item["from_view_id"], item["to_view_id"]))
                if len(chosen) == per_dataset:
                    break
            if len(chosen) == per_dataset:
                break
        if len(chosen) != per_dataset:
            raise RuntimeError(f"could not select {per_dataset} cases for {dataset_id}")
        selected.extend(chosen)

    selection = {
        "schema": SELECTION_SCHEMA,
        "selection_is_independent_of_dense_stereo_outputs": True,
        "selection_policy": {
            "count": args.count,
            "count_per_dataset": per_dataset,
            "requires_frozen_pose_quality_accepted": True,
            "maximum_rotation_error_deg": 5.0,
            "maximum_translation_direction_error_deg": 10.0,
            "baseline_over_depth_interval": [0.10, 0.90],
            "minimum_baseline_m": 0.15,
            "minimum_reference_geometric_visibility_fraction": args.min_overlap,
            "prefer_distinct_spatial_groups_then_fill": True,
            "prohibit_reused_views": True,
        },
        "source_hashes": {
            "predictions": sha256(args.predictions),
            "method_inputs": sha256(args.method_inputs),
            "pair_evaluation": sha256(args.pair_evaluation),
            "views_evaluation": sha256(args.views_evaluation),
        },
        "prediction_manifest": manifest,
        "cases": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(selection, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text() != encoded:
        raise RuntimeError("refusing to replace a different frozen selection")
    args.output.write_text(encoded)
    print(encoded, end="")


def _read_rgb(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read RGB panorama: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = shape_hw
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def depth_metrics(
    estimate: np.ndarray,
    validity: np.ndarray,
    reference: np.ndarray,
    options: SphericalStereoOptions,
) -> dict[str, float | int]:
    height, _ = reference.shape
    pole_margin = int(math.ceil(height * options.pole_margin_fraction))
    support = (
        np.isfinite(reference)
        & (reference >= options.min_range)
        & (reference <= options.max_range)
    )
    if pole_margin:
        support[:pole_margin] = False
        support[-pole_margin:] = False
    evaluated = support & validity & np.isfinite(estimate)
    count = int(evaluated.sum())
    support_count = int(support.sum())
    if count == 0:
        return {
            "search_support_pixels": support_count,
            "evaluated_pixels": 0,
            "coverage": 0.0,
            "abs_rel": math.nan,
            "rmse_m": math.nan,
            "median_abs_error_m": math.nan,
            "delta_1_25": math.nan,
        }
    truth = reference[evaluated].astype(np.float64)
    prediction = estimate[evaluated].astype(np.float64)
    absolute = np.abs(prediction - truth)
    ratio = np.maximum(prediction / truth, truth / prediction)
    return {
        "search_support_pixels": support_count,
        "evaluated_pixels": count,
        "coverage": count / max(support_count, 1),
        "abs_rel": float(np.mean(absolute / truth)),
        "rmse_m": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "median_abs_error_m": float(np.median(absolute)),
        "delta_1_25": float(np.mean(ratio < 1.25)),
    }


def _render_case(
    output_path: Path,
    pair_id: str,
    first: np.ndarray,
    second: np.ndarray,
    reference: np.ndarray,
    estimated: Any,
    oracle: Any,
    estimated_metrics: dict[str, Any],
    oracle_metrics: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid_reference = reference[np.isfinite(reference) & (reference > 0.0)]
    lower, upper = np.percentile(valid_reference, (2.0, 98.0))
    color_map = matplotlib.colormaps["turbo"].copy()
    color_map.set_bad("black")
    estimated_error = np.full_like(reference, np.nan, dtype=np.float32)
    oracle_error = np.full_like(reference, np.nan, dtype=np.float32)
    positive = np.isfinite(reference) & (reference > 0.0)
    np.divide(
        np.abs(estimated.range - reference),
        reference,
        out=estimated_error,
        where=positive & np.isfinite(estimated.range),
    )
    np.divide(
        np.abs(oracle.range - reference),
        reference,
        out=oracle_error,
        where=positive & np.isfinite(oracle.range),
    )
    figure, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    axes[0, 0].imshow(first)
    axes[0, 0].set_title("ERP A (reference)")
    axes[0, 1].imshow(second)
    axes[0, 1].set_title("ERP B")
    axes[0, 2].imshow(reference, cmap=color_map, vmin=lower, vmax=upper)
    axes[0, 2].set_title("Reference radial range [m]")
    axes[0, 3].imshow(estimated.range, cmap=color_map, vmin=lower, vmax=upper)
    axes[0, 3].set_title(
        "Estimated R,t-direction\n"
        f"AbsRel={estimated_metrics['abs_rel']:.3f}, "
        f"coverage={estimated_metrics['coverage']:.2f}"
    )
    axes[1, 0].imshow(
        np.ma.masked_invalid(estimated_error), cmap="magma", vmin=0, vmax=1
    )
    axes[1, 0].set_title("Estimated-pose relative error")
    axes[1, 1].imshow(estimated.confidence, cmap="viridis", vmin=0, vmax=0.25)
    axes[1, 1].set_title("Matching confidence")
    axes[1, 2].imshow(oracle.range, cmap=color_map, vmin=lower, vmax=upper)
    axes[1, 2].set_title(
        "Reference-pose upper bound\n"
        f"AbsRel={oracle_metrics['abs_rel']:.3f}, "
        f"coverage={oracle_metrics['coverage']:.2f}"
    )
    axes[1, 3].imshow(np.ma.masked_invalid(oracle_error), cmap="magma", vmin=0, vmax=1)
    axes[1, 3].set_title("Reference-pose relative error")
    for axis in axes.flat:
        axis.axis("off")
    figure.suptitle(pair_id)
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def _aggregate(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    metrics = [record[key] for record in records]
    weights = np.asarray([item["evaluated_pixels"] for item in metrics], dtype=float)
    result: dict[str, float] = {}
    for name in ("coverage", "abs_rel", "rmse_m", "median_abs_error_m", "delta_1_25"):
        values = np.asarray([item[name] for item in metrics], dtype=float)
        finite = np.isfinite(values)
        result[f"median_case_{name}"] = float(np.median(values[finite]))
        if name != "coverage" and np.any(finite & (weights > 0)):
            result[f"pixel_weighted_{name}"] = float(
                np.average(values[finite], weights=weights[finite])
            )
    return result


def _write_report(
    path: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Ten-case spherical dense-stereo study",
        "",
        (
            "The selected pairs had successful frozen five-point/RANSAC poses "
            "and metric radial-range ground truth. The main run uses the "
            "estimated rotation and translation direction, scaled by the "
            "reference baseline magnitude. Therefore it evaluates dense "
            "matching conditional on known metric scale; it does not claim "
            "that two-view geometry recovered that scale. The reference-pose "
            "run is an upper-bound diagnostic."
        ),
        "",
        (
            "| Pair | Dataset | Visibility | R error | t error | Main coverage "
            "| Main AbsRel | Oracle AbsRel |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        selection = record["selection"]
        main = record["estimated_pose_reference_scale"]
        oracle = record["reference_pose"]
        lines.append(
            f"| {record['pair_id']} | {record['dataset_id']} | "
            f"{selection['reference_geometric_visibility_fraction']:.3f} | "
            f"{selection['rotation_error_deg']:.2f}° | "
            f"{selection['translation_direction_error_deg']:.2f}° | "
            f"{main['coverage']:.3f} | {main['abs_rel']:.3f} | "
            f"{oracle['abs_rel']:.3f} |"
        )
    lines.extend(
        (
            "",
            "## Aggregate",
            "",
            "```json",
            json.dumps(summary["aggregate"], indent=2, sort_keys=True),
            "```",
            "",
            "Every per-case panel and NumPy result array is adjacent to this report.",
        )
    )
    path.write_text("\n".join(lines) + "\n")


def run_study(args: argparse.Namespace) -> None:
    selection = json.loads(args.selection.read_text())
    if selection.get("schema") != SELECTION_SCHEMA or len(selection["cases"]) != 10:
        raise RuntimeError("selection must contain exactly ten frozen v1 cases")
    for label, path in (
        ("predictions", args.predictions),
        ("method_inputs", args.method_inputs),
        ("pair_evaluation", args.pair_evaluation),
        ("views_evaluation", args.views_evaluation),
    ):
        if sha256(path) != selection["source_hashes"][label]:
            raise RuntimeError(f"{label} does not match the frozen selection hash")
    _, predictions = prediction_records(args.predictions)
    by_prediction = {item["pair_id"]: item for item in predictions}
    by_input = {item["pair_id"]: item for item in read_jsonl(args.method_inputs)}
    by_evaluation = {item["pair_id"]: item for item in read_jsonl(args.pair_evaluation)}
    views = {item["view_id"]: item for item in read_jsonl(args.views_evaluation)}
    shape_hw = (args.height, args.width)
    options = SphericalStereoOptions(
        min_range=args.min_range,
        max_range=args.max_range,
        num_hypotheses=args.hypotheses,
        window_size=args.window_size,
        pole_margin_fraction=args.pole_margin_fraction,
        min_texture_std=args.min_texture_std,
        min_confidence=args.min_confidence,
        max_matching_cost=args.max_matching_cost,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".matplotlib-cache"))
    case_dir = args.output_dir / "cases"
    case_dir.mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []

    for case_index, selected in enumerate(selection["cases"], start=1):
        pair_id = selected["pair_id"]
        prediction = by_prediction[pair_id]
        method_input = by_input[pair_id]
        evaluation = by_evaluation[pair_id]
        first = _read_rgb(Path(method_input["from_rgb_path"]), shape_hw)
        second = _read_rgb(Path(method_input["to_rgb_path"]), shape_hw)
        reference = _resize_depth(
            np.load(views[evaluation["from_view_id"]]["depth"]["path"], mmap_mode="r"),
            shape_hw,
        )
        estimated_rotation = np.asarray(
            prediction["R_to_from_panorai"], dtype=np.float64
        )
        estimated_direction = np.asarray(
            prediction["t_direction_to_from_panorai"], dtype=np.float64
        )
        reference_rotation, reference_translation = reference_pose_panorai(evaluation)
        baseline = float(np.linalg.norm(reference_translation))
        estimated_translation = baseline * estimated_direction

        started = time.perf_counter()
        estimated_result = estimate_spherical_range(
            first,
            second,
            estimated_rotation,
            estimated_translation,
            options=options,
        )
        estimated_seconds = time.perf_counter() - started
        started = time.perf_counter()
        oracle_result = estimate_spherical_range(
            first,
            second,
            reference_rotation,
            reference_translation,
            options=options,
        )
        oracle_seconds = time.perf_counter() - started
        estimated_metrics = depth_metrics(
            estimated_result.range,
            estimated_result.validity_mask,
            reference,
            options,
        )
        oracle_metrics = depth_metrics(
            oracle_result.range,
            oracle_result.validity_mask,
            reference,
            options,
        )
        stem = f"{case_index:02d}-{pair_id}"
        np.savez_compressed(
            case_dir / f"{stem}.npz",
            reference_range_m=reference,
            estimated_range_m=estimated_result.range,
            estimated_validity=estimated_result.validity_mask,
            estimated_confidence=estimated_result.confidence,
            oracle_range_m=oracle_result.range,
            oracle_validity=oracle_result.validity_mask,
            oracle_confidence=oracle_result.confidence,
        )
        _render_case(
            case_dir / f"{stem}.png",
            pair_id,
            first,
            second,
            reference,
            estimated_result,
            oracle_result,
            estimated_metrics,
            oracle_metrics,
        )
        record = {
            "pair_id": pair_id,
            "dataset_id": selected["dataset_id"],
            "selection": selected,
            "pose_input": {
                "rotation": "frozen-five-point-ransac-estimate",
                "translation_direction": "frozen-five-point-ransac-estimate",
                "translation_magnitude": "reference-baseline-evaluation-only",
                "baseline_m": baseline,
            },
            "estimated_pose_reference_scale": {
                **estimated_metrics,
                "elapsed_seconds": estimated_seconds,
            },
            "reference_pose": {
                **oracle_metrics,
                "elapsed_seconds": oracle_seconds,
            },
            "panel": str((case_dir / f"{stem}.png").resolve()),
            "arrays": str((case_dir / f"{stem}.npz").resolve()),
        }
        records.append(record)
        print(
            f"[{case_index:02d}/10] {pair_id} "
            f"coverage={estimated_metrics['coverage']:.3f} "
            f"AbsRel={estimated_metrics['abs_rel']:.3f}"
        )

    aggregate = {
        "estimated_pose_reference_scale": _aggregate(
            records, "estimated_pose_reference_scale"
        ),
        "reference_pose": _aggregate(records, "reference_pose"),
    }
    summary = {
        "schema": RESULT_SCHEMA,
        "evidence_target": "source-checkout-real-local-datasets",
        "selection_path": str(args.selection.resolve()),
        "selection_sha256": sha256(args.selection),
        "source_hashes": selection["source_hashes"],
        "shape_hw": list(shape_hw),
        "options": options.to_dict(),
        "pose_scale_limitation": (
            "The main run uses estimated R and translation direction with the "
            "reference baseline magnitude; metric-scale recovery is not evaluated."
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "aggregate": aggregate,
        "cases": records,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    csv_path = args.output_dir / "metrics.csv"
    with csv_path.open("w", newline="") as stream:
        fields = [
            "pair_id",
            "dataset_id",
            "mode",
            "coverage",
            "abs_rel",
            "rmse_m",
            "median_abs_error_m",
            "delta_1_25",
            "evaluated_pixels",
            "elapsed_seconds",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for mode in ("estimated_pose_reference_scale", "reference_pose"):
                writer.writerow(
                    {
                        "pair_id": record["pair_id"],
                        "dataset_id": record["dataset_id"],
                        "mode": mode,
                        **{
                            key: record[mode][key]
                            for key in fields
                            if key in record[mode]
                        },
                    }
                )
    _write_report(args.output_dir / "REPORT.md", records, summary)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--predictions", required=True, type=Path)
    select.add_argument("--method-inputs", required=True, type=Path)
    select.add_argument("--pair-evaluation", required=True, type=Path)
    select.add_argument("--views-evaluation", required=True, type=Path)
    select.add_argument("--output", required=True, type=Path)
    select.add_argument("--count", type=int, default=10, choices=(10,))
    select.add_argument("--min-overlap", type=float, default=0.35)
    select.set_defaults(handler=select_cases)

    run = subparsers.add_parser("run")
    run.add_argument("--selection", required=True, type=Path)
    run.add_argument("--predictions", required=True, type=Path)
    run.add_argument("--method-inputs", required=True, type=Path)
    run.add_argument("--pair-evaluation", required=True, type=Path)
    run.add_argument("--views-evaluation", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--height", type=int, default=128)
    run.add_argument("--width", type=int, default=256)
    run.add_argument("--min-range", type=float, default=0.30)
    run.add_argument("--max-range", type=float, default=12.0)
    run.add_argument("--hypotheses", type=int, default=128)
    run.add_argument("--window-size", type=int, default=7)
    run.add_argument("--pole-margin-fraction", type=float, default=0.04)
    run.add_argument("--min-texture-std", type=float, default=0.015)
    run.add_argument("--min-confidence", type=float, default=0.015)
    run.add_argument("--max-matching-cost", type=float, default=0.65)
    run.set_defaults(handler=run_study)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
