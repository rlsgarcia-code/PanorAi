#!/usr/bin/env python3
"""Run native P74 bidirectional spherical cost-volume depth matching."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.spherical_monocular_depth.protocol import (  # noqa: E402
    sha256,
    write_binary_ply,
)
from benchmarks.spherical_multiview_depth.bidirectional import (  # noqa: E402
    BIDIRECTIONAL_INTERFACE,
    BidirectionalCostVolumeOptions,
    BidirectionalSphericalCostVolume,
)
from benchmarks.spherical_multiview_depth.refinement import (  # noqa: E402
    DepthPrior,
    reprojection_score,
)
from benchmarks.spherical_multiview_depth.run_p74_experiment import (  # noqa: E402
    HELDOUT_ID,
    OPTIMIZATION_IDS,
    TARGET_ID,
    _evaluate,
    _json_dump,
    _load_source,
    _paths,
    _verify_binary_ply,
    _write_native_depth_png,
    _write_panel,
)
from benchmarks.spherical_multiview_depth.p74 import (  # noqa: E402
    load_native_angular_rgb,
)


SCHEMA = "panorai-p74-bidirectional-spherical-depth-cost-volume/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search radial range around the frozen W121 seed through B->A and "
            "A->B spherical cost volumes with fixed RGB and registered R,t."
        )
    )
    parser.add_argument("--p74-root", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--evaluation-validity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hypotheses", type=int, default=17)
    parser.add_argument("--range-factor", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--minimum-confidence", type=float, default=0.02)
    parser.add_argument("--maximum-cycle-px", type=float, default=2.0)
    parser.add_argument("--maximum-cycle-log-range", type=float, default=0.08)
    parser.add_argument("--row-batch", type=int, default=16)
    parser.add_argument("--threads", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--normal-stride", type=int, default=8)
    parser.add_argument("--ply-max-points", type=int, default=1_500_000)
    parser.add_argument("--preview-stride", type=int, default=8)
    return parser.parse_args()


def _save_variant(
    output: Path,
    name: str,
    prediction: np.ndarray,
) -> tuple[Path, str]:
    path = output / f"{TARGET_ID}-{name}-radial-m.npy"
    np.save(path, prediction)
    _write_native_depth_png(
        output / f"{TARGET_ID}-{name}-depth-over-15m.png", prediction
    )
    return path, sha256(path)


def write_diagnostic_panel(
    path: Path,
    *,
    seed: np.ndarray,
    reciprocal: np.ndarray,
    confidence: np.ndarray,
    accepted_count: np.ndarray,
    truth: np.ndarray,
    validity: np.ndarray,
    stride: int,
) -> None:
    """Visualize exactly where reciprocity changed the frozen depth seed."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if stride < 1:
        raise ValueError("diagnostic panel stride must be positive")
    selection = np.asarray(validity, dtype=bool)
    log_delta = np.full(seed.shape, np.nan, dtype=np.float32)
    relative_improvement = np.full(seed.shape, np.nan, dtype=np.float32)
    log_delta[selection] = np.log(reciprocal[selection] / seed[selection])
    seed_error = np.abs(seed[selection] - truth[selection]) / truth[selection]
    reciprocal_error = (
        np.abs(reciprocal[selection] - truth[selection]) / truth[selection]
    )
    relative_improvement[selection] = seed_error - reciprocal_error
    entries = (
        ("log(reciprocal / seed)", log_delta, "coolwarm", -0.25, 0.25),
        ("reciprocal confidence", confidence, "viridis", 0.0, 1.0),
        ("accepted source count", accepted_count, "viridis", 0.0, 2.0),
        (
            "GT diagnostic: relative-error improvement",
            relative_improvement,
            "coolwarm",
            -0.25,
            0.25,
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(18, 9), constrained_layout=True)
    for axis, (title, values, cmap, vmin, vmax) in zip(
        axes.ravel(), entries, strict=True
    ):
        image = axis.imshow(values[::stride, ::stride], cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, fraction=0.025, pad=0.01)
    figure.suptitle(
        f"Bidirectional spherical depth diagnostics; every {stride}th native sample"
    )
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    if not np.isfinite(args.range_factor) or args.range_factor <= 1.0:
        raise ValueError("range-factor must exceed 1")
    args.output.mkdir(parents=True, exist_ok=True)

    import torch

    torch.set_num_threads(args.threads)
    prior_memmap = np.load(args.prior, mmap_mode="r")
    if prior_memmap.ndim != 2:
        raise ValueError("prior must be a native HxW radial-range NPY")
    shape_hw = tuple(int(value) for value in prior_memmap.shape)
    seed = np.clip(np.asarray(prior_memmap), 0.3, 15.0).astype(np.float32)
    target_rgb_path, target_npz = _paths(args.p74_root, TARGET_ID)
    target_rgb, target_support = load_native_angular_rgb(target_rgb_path, shape_hw)
    optimization_validity = (
        target_support & np.isfinite(seed) & (seed >= 0.3) & (seed <= 15.0)
    )
    # Only W119/W124 are opened before predictions and hashes freeze.
    sources = tuple(
        _load_source(args.p74_root, target_npz, panorama_id, shape_hw)
        for panorama_id in OPTIMIZATION_IDS
    )
    options = BidirectionalCostVolumeOptions(
        hypotheses=args.hypotheses,
        inverse_log_radius=float(np.log(args.range_factor)),
        temperature=args.temperature,
        minimum_confidence=args.minimum_confidence,
        maximum_cycle_px=args.maximum_cycle_px,
        maximum_cycle_log_range=args.maximum_cycle_log_range,
        row_batch=args.row_batch,
    )
    print(
        json.dumps(
            {
                "event": "bidirectional-start",
                "shape_hw": shape_hw,
                "options": options.to_dict(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    start = perf_counter()
    result = BidirectionalSphericalCostVolume(options).infer(
        DepthPrior(
            seed,
            optimization_validity,
            provenance=f"{args.prior} sha256={sha256(args.prior)}",
        ),
        target_rgb,
        sources,
        device=args.device,
        progress=lambda row: print(
            json.dumps({"event": "source-complete", **row}, sort_keys=True),
            flush=True,
        ),
    )
    runtime_seconds = perf_counter() - start
    forward = result.forward_range_m.cpu().numpy().astype(np.float32, copy=True)
    reciprocal = result.reciprocal_range_m.cpu().numpy().astype(np.float32, copy=True)
    forward_confidence = (
        result.forward_confidence.cpu().numpy().astype(np.float32, copy=True)
    )
    reciprocal_confidence = (
        result.reciprocal_confidence.cpu().numpy().astype(np.float32, copy=True)
    )
    accepted_count = result.reciprocal_accepted_count.cpu().numpy().copy()
    source_summaries = list(result.source_summaries)
    del result
    gc.collect()

    variants = {
        "cnn-seed": seed,
        "forward-cost-volume": forward,
        "bidirectional-reciprocal": reciprocal,
    }
    variant_records: dict[str, dict[str, Any]] = {}
    for name, prediction in variants.items():
        prediction_path, prediction_sha = _save_variant(args.output, name, prediction)
        variant_records[name] = {
            "prediction": str(prediction_path),
            "prediction_sha256": prediction_sha,
        }
    diagnostics = {
        "forward-confidence": forward_confidence,
        "reciprocal-confidence": reciprocal_confidence,
        "reciprocal-accepted-count": accepted_count,
    }
    diagnostic_hashes = {}
    for name, values in diagnostics.items():
        path = args.output / f"{TARGET_ID}-{name}.npy"
        np.save(path, values)
        diagnostic_hashes[name] = {"path": str(path), "sha256": sha256(path)}
    validity_path = args.output / f"{TARGET_ID}-optimization-validity.npy"
    np.save(validity_path, optimization_validity)
    freeze_manifest = {
        "schema": SCHEMA,
        "event": "predictions-frozen-before-heldout-open",
        "interface": BIDIRECTIONAL_INTERFACE,
        "target_id": TARGET_ID,
        "optimization_ids": list(OPTIMIZATION_IDS),
        "heldout_id": HELDOUT_ID,
        "shape_hw": list(shape_hw),
        "options": options.to_dict(),
        "runtime_seconds": runtime_seconds,
        "source_summaries": source_summaries,
        "variants": variant_records,
        "diagnostics": diagnostic_hashes,
        "optimization_validity_sha256": sha256(validity_path),
    }
    freeze_path = args.output / "FROZEN-BEFORE-W122.json"
    _json_dump(freeze_path, freeze_manifest)
    freeze_sha256 = sha256(freeze_path)
    print(json.dumps({"event": "outputs-frozen", "sha256": freeze_sha256}), flush=True)

    # W122 is first opened here. It never contributes a depth hypothesis.
    heldout = _load_source(args.p74_root, target_npz, HELDOUT_ID, shape_hw)
    for name, prediction in variants.items():
        score = reprojection_score(
            prediction,
            optimization_validity,
            target_rgb,
            heldout,
            row_batch=args.row_batch,
            device=args.device,
        )
        variant_records[name]["heldout_w122_reprojection"] = score
        print(
            json.dumps({"event": "heldout-score", "variant": name, **score}),
            flush=True,
        )
    del heldout
    gc.collect()

    # Evaluation-only ground truth is loaded after prediction freeze and W122 scoring.
    truth = np.load(args.ground_truth, mmap_mode="r")
    evaluation_validity = np.load(args.evaluation_validity, mmap_mode="r")
    if truth.shape != shape_hw or evaluation_validity.shape != shape_hw:
        raise ValueError("ground truth and evaluation validity must match native shape")
    common_validity = (
        np.asarray(evaluation_validity, dtype=bool)
        & np.isfinite(truth)
        & (truth >= 0.3)
        & (truth <= 15.0)
        & optimization_validity
    )
    common_path = args.output / f"{TARGET_ID}-common-evaluation-validity.npy"
    np.save(common_path, common_validity)
    for name, prediction in variants.items():
        variant_records[name]["evaluation"] = _evaluate(
            prediction,
            truth,
            common_validity,
            normal_stride=args.normal_stride,
        )
        ply_path = args.output / f"{TARGET_ID}-{name}-view-15m.ply"
        ply = write_binary_ply(
            ply_path,
            prediction,
            target_rgb,
            common_validity,
            max_points=args.ply_max_points,
        )
        ply["verification"] = _verify_binary_ply(ply_path, maximum_radius_m=15.0)
        ply["sha256"] = sha256(ply_path)
        variant_records[name]["ply"] = ply

    panel_path = args.output / "p74-w121-bidirectional-cost-volume-panel.png"
    _write_panel(
        panel_path,
        target_rgb,
        truth,
        {
            "CNN seed": seed,
            "Forward cost volume": forward,
            "Bidirectional reciprocal": reciprocal,
        },
        stride=args.preview_stride,
    )
    diagnostic_panel_path = (
        args.output / "p74-w121-bidirectional-cost-volume-diagnostics.png"
    )
    write_diagnostic_panel(
        diagnostic_panel_path,
        seed=seed,
        reciprocal=reciprocal,
        confidence=reciprocal_confidence,
        accepted_count=accepted_count,
        truth=truth,
        validity=common_validity,
        stride=args.preview_stride,
    )
    report = {
        "schema": SCHEMA,
        "status": "experimental-pose-oracle-control-not-publication-evidence",
        "interface": BIDIRECTIONAL_INTERFACE,
        "dataset": {
            "name": "P74",
            "external": True,
            "target_id": TARGET_ID,
            "optimization_ids": list(OPTIMIZATION_IDS),
            "heldout_id": HELDOUT_ID,
            "depth_semantics": "radial range in metres",
            "metric_cap_m": 15.0,
        },
        "protocol": {
            "native_shape_hw": list(shape_hw),
            "model_or_prediction_resize": False,
            "pyramid": False,
            "search_coordinate": "local inverse radial range",
            "directions": ["target B to source A", "source A back to target B"],
            "optimized_or_selected_quantity": "radial range only",
            "fixed_quantities": ["RGB", "photometric features", "R", "t", "CNN"],
            "optimization_uses_ground_truth": False,
            "heldout_opened_after_prediction_freeze": True,
            "freeze_manifest": str(freeze_path),
            "freeze_manifest_sha256": freeze_sha256,
            "common_evaluation_pixels": int(common_validity.sum()),
        },
        "inputs": {
            "prior": str(args.prior),
            "prior_sha256": sha256(args.prior),
            "ground_truth": str(args.ground_truth),
            "ground_truth_sha256": sha256(args.ground_truth),
            "evaluation_validity": str(args.evaluation_validity),
            "evaluation_validity_sha256": sha256(args.evaluation_validity),
        },
        "matcher": {
            "options": options.to_dict(),
            "runtime_seconds": runtime_seconds,
            "source_summaries": source_summaries,
            "diagnostics": diagnostic_hashes,
        },
        "variants": variant_records,
        "artifacts": {
            "panel": str(panel_path),
            "panel_sha256": sha256(panel_path),
            "diagnostic_panel": str(diagnostic_panel_path),
            "diagnostic_panel_sha256": sha256(diagnostic_panel_path),
            "common_evaluation_validity": str(common_path),
            "common_evaluation_validity_sha256": sha256(common_path),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "device": args.device,
            "torch_threads": args.threads,
        },
    }
    report_path = args.output / "results.json"
    _json_dump(report_path, report)
    summary = {
        name: {
            "abs_rel": record["evaluation"]["metric_depth"]["abs_rel"],
            "si_3d_rmse": record["evaluation"]["scale_invariant_structure"][
                "scale_aligned_relative_3d_rmse"
            ],
            "normal_mean_deg": record["evaluation"]["surface_normals"][
                "normal_mean_deg"
            ],
            "heldout_cost": record["heldout_w122_reprojection"][
                "weighted_mean_feature_cost"
            ],
        }
        for name, record in variant_records.items()
    }
    print(json.dumps({"event": "complete", "summary": summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
