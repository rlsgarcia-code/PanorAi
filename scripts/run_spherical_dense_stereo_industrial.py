#!/usr/bin/env python3
"""Evaluate Experimental spherical dense stereo on ten industrial pairs.

The outcome-blind selection uses scanner overlap and baseline only. RGB is
adapted from the native centered partial panorama to a canonical 2:1 ERP with
the independently versioned EQ adapter. Reference pose and XYZ are loaded only
after feature matching/relative-pose estimation has completed for a case.
"""

from __future__ import annotations

import argparse
import csv
import gc
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
from PIL import Image

from panorai.estimators import RelativePoseOptions, SphericalRelativePoseEstimator
from panorai.features import SphericalFeaturePipeline
from panorai.image_processing import native_filter_available
from panorai.stereo import SphericalStereoOptions, estimate_spherical_range

SCHEMA = "panorai-industrial-spherical-dense-stereo-study/v1"
SOURCE_FROM_PANORAI = np.asarray(
    ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    dtype=np.float64,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _slug(identifier: str) -> str:
    return identifier.removeprefix("industrial+").replace("+", "__")


def _rotation_error_deg(estimate: np.ndarray, reference: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(estimate @ reference.T) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _direction_error_deg(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate = estimate / np.linalg.norm(estimate)
    reference = reference / np.linalg.norm(reference)
    cosine = float(np.clip(estimate @ reference, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _select_pairs(path: Path, count: int) -> list[dict[str, Any]]:
    candidates = [
        item
        for item in _read_jsonl(path)
        if item["left_family_id"] == item["right_family_id"]
        and 0.8 <= float(item["center_distance_m"]) <= 3.5
    ]
    candidates.sort(
        key=lambda item: (
            -float(item["min_fraction_within_0p25_m"]),
            float(item["center_distance_m"]),
            item["left_panorama_id"],
            item["right_panorama_id"],
        )
    )
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for item in candidates:
        endpoints = {item["left_panorama_id"], item["right_panorama_id"]}
        if endpoints & used:
            continue
        selected.append(item)
        used.update(endpoints)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"could select only {len(selected)} disjoint industrial pairs")
    return selected


def _load_eq_adapter(adapter_root: Path):
    sys.path.insert(0, str(adapter_root.resolve()))
    try:
        from eq_rgb_adapter import (  # type: ignore[import-not-found]
            canonical_to_native_samples,
            rasterize_centered_partial_erp,
        )
    finally:
        sys.path.pop(0)
    return canonical_to_native_samples, rasterize_centered_partial_erp


def _adapt_rgb(
    panorama_id: str,
    dataset_root: Path,
    cache_dir: Path,
    shape_hw: tuple[int, int],
    rasterize_centered_partial_erp: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    stem = _slug(panorama_id)
    rgb_path = cache_dir / f"{stem}-rgb.npy"
    mask_path = cache_dir / f"{stem}-observed.npy"
    provenance_path = cache_dir / f"{stem}-rgb.json"
    if rgb_path.is_file() and mask_path.is_file() and provenance_path.is_file():
        return (
            np.load(rgb_path),
            np.load(mask_path).astype(bool, copy=False),
            json.loads(provenance_path.read_text()),
        )
    source_path = dataset_root / "images" / f"{panorama_id}_rgb.png"
    with Image.open(source_path) as image:
        source = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = shape_hw
    rgb, observed, fraction = rasterize_centered_partial_erp(
        source, target_width=width, target_height=height
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(rgb_path, rgb)
    np.save(mask_path, observed)
    valid_rows = np.flatnonzero(observed[:, 0])
    positive_fraction = fraction[fraction > 0]
    provenance = {
        "source_path": str(source_path),
        "source_shape_hwc": list(source.shape),
        "output_shape_hwc": list(rgb.shape),
        "observed_fraction": float(observed.mean()),
        "observed_row_range_inclusive": (
            [int(valid_rows[0]), int(valid_rows[-1])] if len(valid_rows) else None
        ),
        "minimum_nonzero_angular_coverage": (
            float(np.min(positive_fraction)) if positive_fraction.size else 0.0
        ),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    return rgb, observed, provenance


def _source_pose(dataset_root: Path, panorama_id: str) -> np.ndarray:
    with np.load(dataset_root / "npzs" / f"{panorama_id}.npz") as archive:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(archive["rotation_matrix"], dtype=np.float64)
        transform[:3, 3] = np.asarray(archive["translation"], dtype=np.float64).reshape(
            3
        )
    return transform


def _reference_pose(
    dataset_root: Path, from_id: str, to_id: str
) -> tuple[np.ndarray, np.ndarray]:
    source_from_a = _source_pose(dataset_root, from_id)
    source_from_b = _source_pose(dataset_root, to_id)
    b_from_a_source = np.linalg.inv(source_from_b) @ source_from_a
    adapter = SOURCE_FROM_PANORAI
    return (
        adapter.T @ b_from_a_source[:3, :3] @ adapter,
        adapter.T @ b_from_a_source[:3, 3],
    )


def _reference_range(
    panorama_id: str,
    dataset_root: Path,
    cache_dir: Path,
    shape_hw: tuple[int, int],
    canonical_to_native_samples: Any,
) -> np.ndarray:
    output = cache_dir / f"{_slug(panorama_id)}-range-m.npy"
    if output.is_file():
        return np.load(output)
    height, width = shape_hw
    with np.load(dataset_root / "npzs" / f"{panorama_id}.npz") as archive:
        xyz = archive["xyz_image"]
        source_height, source_width = xyz.shape[:2]
        y, x = np.indices(shape_hw, dtype=np.float64)
        samples = canonical_to_native_samples(
            np.column_stack((x.ravel(), y.ravel())),
            source_width=source_width,
            source_height=source_height,
            target_width=width,
            target_height=height,
        )
        source_x = np.rint(samples[:, 0]).astype(np.int64) % source_width
        source_y = np.rint(samples[:, 1]).astype(np.int64)
        supported = (source_y >= 0) & (source_y < source_height)
        points = np.full((height * width, 3), np.nan, dtype=np.float64)
        points[supported] = xyz[source_y[supported], source_x[supported]]
    ranges = np.linalg.norm(points, axis=1).reshape(shape_hw).astype(np.float32)
    ranges[(~np.isfinite(ranges)) | (ranges <= 1e-3)] = np.nan
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(output, ranges)
    return ranges


def _depth_metrics(
    estimate: np.ndarray,
    validity: np.ndarray,
    reference: np.ndarray,
    options: SphericalStereoOptions,
) -> dict[str, float | int]:
    support = (
        np.isfinite(reference)
        & (reference >= options.min_range)
        & (reference <= options.max_range)
    )
    pole_margin = int(math.ceil(reference.shape[0] * options.pole_margin_fraction))
    if pole_margin:
        support[:pole_margin] = False
        support[-pole_margin:] = False
    evaluated = support & validity & np.isfinite(estimate) & (estimate > 0)
    support_count = int(support.sum())
    count = int(evaluated.sum())
    image_pixels = int(reference.size)
    if not count:
        return {
            "image_pixels": image_pixels,
            "reference_support_pixels": support_count,
            "reference_support_fraction": support_count / image_pixels,
            "evaluated_pixels": 0,
            "coverage": 0.0,
            "coverage_of_reference_support": 0.0,
            "valid_fraction_of_full_erp": 0.0,
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
        "image_pixels": image_pixels,
        "reference_support_pixels": support_count,
        "reference_support_fraction": support_count / image_pixels,
        "evaluated_pixels": count,
        "coverage": count / support_count,
        "coverage_of_reference_support": count / support_count,
        "valid_fraction_of_full_erp": count / image_pixels,
        "abs_rel": float(np.mean(absolute / truth)),
        "rmse_m": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "median_abs_error_m": float(np.median(absolute)),
        "delta_1_25": float(np.mean(ratio < 1.25)),
    }


def _empty_result(
    shape_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.full(shape_hw, np.nan, dtype=np.float32),
        np.zeros(shape_hw, dtype=bool),
        np.zeros(shape_hw, dtype=np.float32),
    )


def _run_dense(
    first: np.ndarray,
    second: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    options: SphericalStereoOptions,
):
    started = time.perf_counter()
    result = estimate_spherical_range(
        first, second, rotation, translation, options=options
    )
    return result, time.perf_counter() - started


def _render_case(
    output: Path,
    title: str,
    first: np.ndarray,
    second: np.ndarray,
    reference: np.ndarray,
    estimated: tuple[np.ndarray, np.ndarray, np.ndarray],
    oracle: tuple[np.ndarray, np.ndarray, np.ndarray],
    conservative: tuple[np.ndarray, np.ndarray, np.ndarray],
    metrics: dict[str, dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite = reference[np.isfinite(reference)]
    low, high = np.percentile(finite, (2.0, 98.0))
    cmap = matplotlib.colormaps["turbo"].copy()
    cmap.set_bad("black")

    def rel_error(value: np.ndarray) -> np.ndarray:
        result = np.full(reference.shape, np.nan, dtype=np.float32)
        valid = np.isfinite(value) & np.isfinite(reference) & (reference > 0)
        result[valid] = np.abs(value[valid] - reference[valid]) / reference[valid]
        return result

    figure, axes = plt.subplots(3, 3, figsize=(18, 10), constrained_layout=True)
    panels = (
        (axes[0, 0], first, "ERP A / referência", None),
        (axes[0, 1], second, "ERP B", None),
        (axes[0, 2], reference, "Scanner: range radial [m]", (low, high)),
        (
            axes[1, 0],
            estimated[0],
            "Pose 5pt + escala ref.\n"
            f"AbsRel={metrics['estimated_pose']['abs_rel']:.3f}  "
            f"cov={metrics['estimated_pose']['coverage']:.3f}",
            (low, high),
        ),
        (
            axes[1, 1],
            oracle[0],
            "Pose ref., ida\n"
            f"AbsRel={metrics['reference_pose_one_way']['abs_rel']:.3f}  "
            f"cov={metrics['reference_pose_one_way']['coverage']:.3f}",
            (low, high),
        ),
        (
            axes[1, 2],
            conservative[0],
            "Pose ref., ida-volta\n"
            f"AbsRel={metrics['reference_pose_bidirectional']['abs_rel']:.3f}  "
            f"cov={metrics['reference_pose_bidirectional']['coverage']:.3f}",
            (low, high),
        ),
        (axes[2, 0], rel_error(estimated[0]), "Erro relativo: pose 5pt", (0.0, 1.0)),
        (axes[2, 1], rel_error(oracle[0]), "Erro relativo: pose ref.", (0.0, 1.0)),
        (
            axes[2, 2],
            np.where(conservative[1], conservative[2], np.nan),
            "Confiança após consistência",
            (0.0, 0.25),
        ),
    )
    for axis, image, label, limits in panels:
        if limits is None:
            axis.imshow(image)
        else:
            axis.imshow(
                np.ma.masked_invalid(image),
                cmap=cmap,
                vmin=limits[0],
                vmax=limits[1],
            )
        axis.set_title(label)
        axis.axis("off")
    figure.suptitle(title, fontsize=12)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _render_preview(
    output: Path,
    title: str,
    first: np.ndarray,
    second: np.ndarray,
    reference: np.ndarray,
    estimate: np.ndarray,
    validity: np.ndarray,
    confidence: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    """Render a compact first-result panel for a native-resolution smoke run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite = reference[np.isfinite(reference)]
    low, high = np.percentile(finite, (2.0, 98.0))
    cmap = matplotlib.colormaps["turbo"].copy()
    cmap.set_bad("black")
    relative_error = np.full(reference.shape, np.nan, dtype=np.float32)
    supported = validity & np.isfinite(reference) & (reference > 0)
    relative_error[supported] = (
        np.abs(estimate[supported] - reference[supported]) / reference[supported]
    )

    figure, axes = plt.subplots(2, 3, figsize=(18, 7), constrained_layout=True)
    panels = (
        (axes[0, 0], first, "ERP A (8192x4096)", None, None),
        (axes[0, 1], second, "ERP B (8192x4096)", None, None),
        (axes[0, 2], reference, "Scanner: range radial [m]", (low, high), cmap),
        (
            axes[1, 0],
            estimate,
            "Dense esferico C++: range [m]\n"
            f"AbsRel={metrics['abs_rel']:.3f}  cov={metrics['coverage']:.3f}",
            (low, high),
            cmap,
        ),
        (
            axes[1, 1],
            relative_error,
            "Erro relativo |estimado-ref|/ref",
            (0.0, 1.0),
            cmap,
        ),
        (
            axes[1, 2],
            np.where(validity, confidence, np.nan),
            "Confianca do matching",
            (0.0, 0.25),
            cmap,
        ),
    )
    for axis, image, label, limits, panel_cmap in panels:
        if limits is None:
            axis.imshow(image)
        else:
            axis.imshow(
                np.ma.masked_invalid(image),
                cmap=panel_cmap,
                vmin=limits[0],
                vmax=limits[1],
            )
        axis.set_title(label)
        axis.axis("off")
    figure.suptitle(title, fontsize=13)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _render_overview(panel_paths: list[Path], output: Path) -> None:
    columns = 2
    thumbnail_size = (1350, 750)
    gutter = 12
    rows = math.ceil(len(panel_paths) / columns)
    canvas = Image.new(
        "RGB",
        (
            columns * thumbnail_size[0] + (columns + 1) * gutter,
            rows * thumbnail_size[1] + (rows + 1) * gutter,
        ),
        color=(17, 21, 26),
    )
    for index, panel_path in enumerate(panel_paths):
        with Image.open(panel_path) as source:
            panel = source.convert("RGB")
            panel.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
        column = index % columns
        row = index // columns
        x = gutter + column * (thumbnail_size[0] + gutter)
        y = gutter + row * (thumbnail_size[1] + gutter)
        canvas.paste(panel, (x, y))
    canvas.save(output)


def _aggregate(records: list[dict[str, Any]], mode: str) -> dict[str, float]:
    metrics = [item["depth_metrics"][mode] for item in records]
    weights = np.asarray([item["evaluated_pixels"] for item in metrics], dtype=float)
    result: dict[str, float] = {}
    for name in (
        "coverage",
        "valid_fraction_of_full_erp",
        "abs_rel",
        "rmse_m",
        "median_abs_error_m",
        "delta_1_25",
    ):
        values = np.asarray([item[name] for item in metrics], dtype=float)
        finite = np.isfinite(values)
        result[f"median_case_{name}"] = (
            float(np.median(values[finite])) if np.any(finite) else math.nan
        )
        if name != "coverage" and np.any(finite & (weights > 0)):
            result[f"pixel_weighted_{name}"] = float(
                np.average(values[finite], weights=weights[finite])
            )
    return result


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    height, width = summary["resolution_hw"]
    pair_count = len(summary["cases"])
    lines = [
        "# Industrial spherical dense-stereo study",
        "",
        (
            f"{pair_count} disjoint, high-overlap industrial pairs were selected without reading "
            "dense-stereo outputs. Native centered partial panoramas were "
            f"angular-area resampled to a canonical {width}x{height} ERP."
        ),
        "",
        (
            "The visual-pose (RGB-only) run uses PanorAi five-point/RANSAC "
            "rotation and translation direction, but the scanner reference "
            "baseline magnitude. "
            "It therefore does not evaluate metric scale recovery. The two "
            "reference-pose runs isolate the dense matcher."
        ),
        "",
        (
            "`coverage` means accepted depth pixels divided by scanner-reference "
            "pixels inside the configured range and pole margin. "
            "`valid_fraction_of_full_erp` uses every ERP pixel as denominator."
        ),
        "",
        "## Visual overview",
        "",
        f"![{pair_count} industrial dense-stereo cases](overview.png)",
        "",
        "| # | Pair | overlap | baseline | R err | t err | est AbsRel | ref AbsRel | ref bi AbsRel |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, item in enumerate(summary["cases"], start=1):
        pose = item["pose_estimation"]
        depth = item["depth_metrics"]
        lines.append(
            f"| {index} | {item['from_id'].split('+')[-1]} -> "
            f"{item['to_id'].split('+')[-1]} | {item['selection']['overlap']:.3f} | "
            f"{item['selection']['baseline_m']:.2f} m | "
            f"{pose['rotation_error_deg']:.2f} deg | "
            f"{pose['translation_direction_error_deg']:.2f} deg | "
            f"{depth['estimated_pose']['abs_rel']:.3f} | "
            f"{depth['reference_pose_one_way']['abs_rel']:.3f} | "
            f"{depth['reference_pose_bidirectional']['abs_rel']:.3f} |"
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
            "Full-resolution arrays and one visual panel per case are in `cases/`.",
        )
    )
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    if args.count < 1:
        raise ValueError("count must be positive")
    if args.width != 2 * args.height:
        raise ValueError("output must be a canonical 2:1 ERP")
    if args.filter_backend == "native" and not native_filter_available():
        raise RuntimeError(
            "--filter-backend=native requires the compiled spherical-filter kernel"
        )
    shape_hw = (args.height, args.width)
    canonical_to_native_samples, rasterize = _load_eq_adapter(args.adapter_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".matplotlib-cache"))
    cache_dir = args.output_dir / "cache"
    case_dir = args.output_dir / "cases"
    case_dir.mkdir(exist_ok=True)
    selected = _select_pairs(args.overlap_pairs, args.count)
    if args.preview:
        _run_preview(
            args,
            selected[0],
            shape_hw,
            canonical_to_native_samples,
            rasterize,
            cache_dir,
        )
        return
    selection = {
        "selection_is_independent_of_dense_outputs": True,
        "policy": {
            "same_family": True,
            "baseline_interval_m": [0.8, 3.5],
            "order": "descending minimum bidirectional overlap within 0.25 m",
            "disjoint_endpoints": True,
            "count": args.count,
        },
        "pairs": selected,
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n"
    )

    pipeline = SphericalFeaturePipeline.from_preset(
        "sift-flann",
        face_sampler="icosahedron",
        face_fov_deg=80.0,
        face_shape_hw=(args.face_size, args.face_size),
        face_overlap_deg=15.0,
        edge_margin_px=16,
        ratio_test=0.72,
        max_features=4096,
        angular_dedup_threshold_deg=0.15,
    )
    pose_estimator = SphericalRelativePoseEstimator(
        RelativePoseOptions(
            max_angular_error_deg=0.5,
            min_inlier_ratio=0.05,
            min_num_trials=32,
            max_num_trials=500,
            min_inliers=8,
            local_optimization_steps=3,
            random_seed=7,
            min_median_parallax_deg=0.25,
            stability_trials=0,
            model_competition_trials=32,
        )
    )
    one_way = SphericalStereoOptions(
        min_range=args.min_range,
        max_range=args.max_range,
        num_hypotheses=args.hypotheses,
        window_size=7,
        bidirectional_consistency=False,
        filter_backend=args.filter_backend,
        pyramid_levels=args.pyramid_levels,
        refinement_hypotheses=args.refinement_hypotheses,
        refinement_radius_steps=args.refinement_radius_steps,
    )
    bidirectional = SphericalStereoOptions(
        min_range=args.min_range,
        max_range=args.max_range,
        num_hypotheses=args.hypotheses,
        window_size=7,
        bidirectional_consistency=True,
        filter_backend=args.filter_backend,
        pyramid_levels=args.pyramid_levels,
        refinement_hypotheses=args.refinement_hypotheses,
        refinement_radius_steps=args.refinement_radius_steps,
    )
    records: list[dict[str, Any]] = []

    for index, item in enumerate(selected, start=1):
        from_id, to_id = item["left_panorama_id"], item["right_panorama_id"]
        first, first_mask, first_provenance = _adapt_rgb(
            from_id, args.dataset_root, cache_dir, shape_hw, rasterize
        )
        second, second_mask, second_provenance = _adapt_rgb(
            to_id, args.dataset_root, cache_dir, shape_hw, rasterize
        )

        # The reference transform is intentionally not loaded until after the
        # RGB-only estimate has been serialized in local variables.
        feature_a = pipeline.extract(
            first, panorama_id=from_id, validity_mask=first_mask
        )
        feature_b = pipeline.extract(
            second, panorama_id=to_id, validity_mask=second_mask
        )
        matches = pipeline.match(feature_a, feature_b)
        pose_started = time.perf_counter()
        pose = pose_estimator.estimate(matches.to_bearing_correspondences())
        pose_seconds = time.perf_counter() - pose_started

        reference_rotation, reference_translation = _reference_pose(
            args.dataset_root, from_id, to_id
        )
        baseline = float(np.linalg.norm(reference_translation))
        reference = _reference_range(
            from_id,
            args.dataset_root,
            cache_dir,
            shape_hw,
            canonical_to_native_samples,
        )

        if pose is None:
            estimated_arrays = _empty_result(shape_hw)
            estimated_seconds = 0.0
            pose_record = {
                "returned": False,
                "num_matches": len(matches),
                "num_inliers": 0,
                "rotation_error_deg": math.nan,
                "translation_direction_error_deg": math.nan,
                "translation_axis_error_deg": math.nan,
                "strict_pose_success": False,
                "elapsed_seconds": pose_seconds,
            }
        else:
            estimated, estimated_seconds = _run_dense(
                first,
                second,
                pose.R,
                baseline * pose.t,
                one_way,
            )
            estimated_arrays = (
                estimated.range,
                estimated.validity_mask,
                estimated.confidence,
            )
            rotation_error = _rotation_error_deg(pose.R, reference_rotation)
            direction_error = _direction_error_deg(pose.t, reference_translation)
            pose_record = {
                "returned": True,
                "quality_accepted": pose.quality_report.accepted,
                "quality_rejection_reasons": list(
                    pose.quality_report.rejection_reasons
                ),
                "degenerate": pose.degenerate,
                "degeneracy_reasons": list(pose.degeneracy_reasons),
                "num_matches": len(matches),
                "num_inliers": pose.num_inliers,
                "rotation_error_deg": rotation_error,
                "translation_direction_error_deg": direction_error,
                "translation_axis_error_deg": min(
                    direction_error, 180.0 - direction_error
                ),
                "strict_pose_success": (
                    rotation_error <= 5.0 and direction_error <= 10.0
                ),
                "elapsed_seconds": pose_seconds,
            }

        oracle, oracle_seconds = _run_dense(
            first,
            second,
            reference_rotation,
            reference_translation,
            one_way,
        )
        conservative, conservative_seconds = _run_dense(
            first,
            second,
            reference_rotation,
            reference_translation,
            bidirectional,
        )
        oracle_arrays = (oracle.range, oracle.validity_mask, oracle.confidence)
        conservative_arrays = (
            conservative.range,
            conservative.validity_mask,
            conservative.confidence,
        )
        depth_metrics = {
            "estimated_pose": {
                **_depth_metrics(
                    estimated_arrays[0], estimated_arrays[1], reference, one_way
                ),
                "elapsed_seconds": estimated_seconds,
            },
            "reference_pose_one_way": {
                **_depth_metrics(
                    oracle.range, oracle.validity_mask, reference, one_way
                ),
                "elapsed_seconds": oracle_seconds,
            },
            "reference_pose_bidirectional": {
                **_depth_metrics(
                    conservative.range,
                    conservative.validity_mask,
                    reference,
                    bidirectional,
                ),
                "elapsed_seconds": conservative_seconds,
            },
        }
        stem = f"{index:02d}-{_slug(from_id)}-to-{_slug(to_id)}"
        arrays_path = case_dir / f"{stem}.npz"
        np.savez_compressed(
            arrays_path,
            reference_range_m=reference,
            estimated_range_m=estimated_arrays[0],
            estimated_validity=estimated_arrays[1],
            estimated_confidence=estimated_arrays[2],
            reference_pose_one_way_range_m=oracle.range,
            reference_pose_one_way_validity=oracle.validity_mask,
            reference_pose_one_way_confidence=oracle.confidence,
            reference_pose_bidirectional_range_m=conservative.range,
            reference_pose_bidirectional_validity=conservative.validity_mask,
            reference_pose_bidirectional_confidence=conservative.confidence,
        )
        panel_path = case_dir / f"{stem}.png"
        _render_case(
            panel_path,
            f"Industrial case {index:02d}: {from_id} -> {to_id}",
            first,
            second,
            reference,
            estimated_arrays,
            oracle_arrays,
            conservative_arrays,
            depth_metrics,
        )
        record = {
            "from_id": from_id,
            "to_id": to_id,
            "family_id": item["left_family_id"],
            "selection": {
                "overlap": float(item["min_fraction_within_0p25_m"]),
                "baseline_m": float(item["center_distance_m"]),
            },
            "rgb_provenance": [first_provenance, second_provenance],
            "pose_estimation": pose_record,
            "depth_metrics": depth_metrics,
            "panel": str(panel_path.resolve()),
            "arrays": str(arrays_path.resolve()),
        }
        records.append(record)
        (args.output_dir / "partial-summary.json").write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n"
        )
        print(
            f"[{index:02d}/{args.count}] {from_id.split('+')[-1]} -> "
            f"{to_id.split('+')[-1]} | pose R="
            f"{pose_record['rotation_error_deg']:.2f} deg, t="
            f"{pose_record['translation_direction_error_deg']:.2f} deg | "
            f"reference-pose AbsRel="
            f"{depth_metrics['reference_pose_one_way']['abs_rel']:.3f}",
            flush=True,
        )
        del first, second, reference, feature_a, feature_b, matches
        gc.collect()

    modes = (
        "estimated_pose",
        "reference_pose_one_way",
        "reference_pose_bidirectional",
    )
    summary = {
        "schema": SCHEMA,
        "dataset": "private industrial scanner-derived corpus",
        "resolution_hw": list(shape_hw),
        "resolution_regime": f"erp-{args.width}x{args.height}",
        "reference_units": "m_empirically_supported_provenance_pending",
        "selection": selection["policy"],
        "pose_scale_limitation": (
            "visual-pose depth uses the scanner reference baseline magnitude"
        ),
        "dense_options_one_way": one_way.to_dict(),
        "dense_options_bidirectional": bidirectional.to_dict(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "native_spherical_filter": native_filter_available(),
        },
        "aggregate": {mode: _aggregate(records, mode) for mode in modes},
        "strict_pose_success_count": sum(
            item["pose_estimation"]["strict_pose_success"] for item in records
        ),
        "cases": records,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
        fields = [
            "case",
            "from_id",
            "to_id",
            "mode",
            "coverage",
            "coverage_of_reference_support",
            "valid_fraction_of_full_erp",
            "reference_support_fraction",
            "abs_rel",
            "rmse_m",
            "median_abs_error_m",
            "delta_1_25",
            "evaluated_pixels",
            "elapsed_seconds",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(records, start=1):
            for mode in modes:
                row = {
                    "case": index,
                    "from_id": record["from_id"],
                    "to_id": record["to_id"],
                    "mode": mode,
                    **record["depth_metrics"][mode],
                }
                writer.writerow({field: row[field] for field in fields})
    _render_overview(
        [Path(record["panel"]) for record in records],
        args.output_dir / "overview.png",
    )
    _write_report(args.output_dir / "REPORT.md", summary)
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


def _run_preview(
    args: argparse.Namespace,
    item: dict[str, Any],
    shape_hw: tuple[int, int],
    canonical_to_native_samples: Any,
    rasterize: Any,
    cache_dir: Path,
) -> None:
    """Run one reference-pose, one-way case for an early high-res visual."""
    from_id, to_id = item["left_panorama_id"], item["right_panorama_id"]
    first, first_mask, first_provenance = _adapt_rgb(
        from_id, args.dataset_root, cache_dir, shape_hw, rasterize
    )
    second, second_mask, second_provenance = _adapt_rgb(
        to_id, args.dataset_root, cache_dir, shape_hw, rasterize
    )
    reference_rotation, reference_translation = _reference_pose(
        args.dataset_root, from_id, to_id
    )
    reference = _reference_range(
        from_id,
        args.dataset_root,
        cache_dir,
        shape_hw,
        canonical_to_native_samples,
    )
    options = SphericalStereoOptions(
        min_range=args.min_range,
        max_range=args.max_range,
        num_hypotheses=args.hypotheses,
        window_size=7,
        bidirectional_consistency=False,
        filter_backend=args.filter_backend,
        pyramid_levels=args.pyramid_levels,
        refinement_hypotheses=args.refinement_hypotheses,
        refinement_radius_steps=args.refinement_radius_steps,
    )
    result, elapsed_seconds = _run_dense(
        first,
        second,
        reference_rotation,
        reference_translation,
        options,
    )
    metrics = {
        **_depth_metrics(result.range, result.validity_mask, reference, options),
        "elapsed_seconds": elapsed_seconds,
    }
    stem = f"preview-{_slug(from_id)}-to-{_slug(to_id)}"
    arrays_path = args.output_dir / f"{stem}.npz"
    np.savez_compressed(
        arrays_path,
        reference_range_m=reference,
        estimated_range_m=result.range,
        estimated_validity=result.validity_mask,
        estimated_confidence=result.confidence,
    )
    panel_path = args.output_dir / f"{stem}.png"
    _render_preview(
        panel_path,
        (
            f"Industrial preview: {from_id} -> {to_id} | "
            f"D={args.hypotheses}, L={args.pyramid_levels}, "
            f"Dr={args.refinement_hypotheses}, backend={args.filter_backend}, "
            f"{elapsed_seconds:.1f} s"
        ),
        first,
        second,
        reference,
        result.range,
        result.validity_mask,
        result.confidence,
        metrics,
    )
    summary = {
        "schema": f"{SCHEMA}-preview",
        "preliminary": True,
        "from_id": from_id,
        "to_id": to_id,
        "resolution_hw": list(shape_hw),
        "selection": {
            "overlap": float(item["min_fraction_within_0p25_m"]),
            "baseline_m": float(item["center_distance_m"]),
        },
        "rgb_provenance": [first_provenance, second_provenance],
        "observed_fraction": [float(first_mask.mean()), float(second_mask.mean())],
        "options": options.to_dict(),
        "native_spherical_filter": native_filter_available(),
        "metrics": metrics,
        "panel": str(panel_path.resolve()),
        "arrays": str(arrays_path.resolve()),
    }
    (args.output_dir / "preview-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlap-pairs", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--adapter-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--height", type=int, default=4096)
    parser.add_argument("--width", type=int, default=8192)
    parser.add_argument("--face-size", type=int, default=2048)
    parser.add_argument("--min-range", type=float, default=0.30)
    parser.add_argument("--max-range", type=float, default=30.0)
    parser.add_argument("--hypotheses", type=int, default=96)
    parser.add_argument("--pyramid-levels", type=int, default=3)
    parser.add_argument("--refinement-hypotheses", type=int, default=9)
    parser.add_argument("--refinement-radius-steps", type=float, default=4.0)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="run only the first pair with reference pose and one-way matching",
    )
    parser.add_argument(
        "--filter-backend", choices=("auto", "numpy", "native"), default="auto"
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
