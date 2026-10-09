#!/usr/bin/env python3
"""Solve a continuous spherical residual from frozen P74 grid proposals."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import scipy

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.spherical_monocular_depth.protocol import (  # noqa: E402
    sha256,
    write_binary_ply,
)
from benchmarks.spherical_multiview_depth.continuous_residual import (  # noqa: E402
    INTERFACE,
    ContinuousResidualOptions,
    solve_continuous_grid_residual,
)
from benchmarks.spherical_multiview_depth.p74 import (  # noqa: E402
    load_native_angular_rgb,
)
from benchmarks.spherical_multiview_depth.run_p74_experiment import (  # noqa: E402
    TARGET_ID,
    _evaluate,
    _json_dump,
    _paths,
    _verify_binary_ply,
    _write_native_depth_png,
    _write_panel,
)


SCHEMA = "panorai-p74-continuous-grid-residual/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse frozen P74 tangent-grid proposals and solve a continuous "
            "edge-aware spherical log-depth residual without resizing observations."
        )
    )
    parser.add_argument("--p74-root", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--grid-proposals", type=Path, required=True)
    parser.add_argument("--bounded-splat", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--evaluation-validity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-weight", type=float, default=64.0)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=12.0)
    parser.add_argument("--color-sigma", type=float, default=0.12)
    parser.add_argument("--log-depth-sigma", type=float, default=0.25)
    parser.add_argument("--minimum-edge-weight", type=float, default=0.02)
    parser.add_argument("--cg-rtol", type=float, default=1e-8)
    parser.add_argument("--cg-maxiter", type=int, default=2_000)
    parser.add_argument("--native-row-chunk", type=int, default=64)
    parser.add_argument("--change-threshold-log", type=float, default=0.005)
    parser.add_argument("--threads", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--normal-stride", type=int, default=8)
    parser.add_argument("--ply-max-points", type=int, default=1_500_000)
    parser.add_argument("--preview-stride", type=int, default=8)
    return parser.parse_args()


def _save_prediction(
    output: Path,
    name: str,
    value: np.ndarray,
) -> dict[str, Any]:
    path = output / f"{TARGET_ID}-{name}-radial-m.npy"
    np.save(path, value)
    png_path = output / f"{TARGET_ID}-{name}-depth-over-15m.png"
    _write_native_depth_png(png_path, value)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "png": str(png_path),
        "png_sha256": sha256(png_path),
    }


def _load_grid_proposals(path: Path) -> dict[str, np.ndarray]:
    required = (
        "rows",
        "columns",
        "prior_range_m",
        "consensus_range_m",
        "consensus_accepted",
        "consensus_confidence",
    )
    with np.load(path) as archive:
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError(f"grid proposal archive is missing: {missing}")
        return {name: np.asarray(archive[name]).copy() for name in required}


def _changed_diagnostic(
    prediction: np.ndarray,
    prior: np.ndarray,
    truth: np.ndarray,
    common: np.ndarray,
    changed: np.ndarray,
) -> dict[str, Any]:
    selected = common & changed
    if not selected.any():
        return {"pixels": 0}
    prior_error = np.abs(prior[selected] - truth[selected])
    prediction_error = np.abs(prediction[selected] - truth[selected])
    return {
        "pixels": int(selected.sum()),
        "prior_abs_rel": float(np.mean(prior_error / truth[selected])),
        "prediction_abs_rel": float(np.mean(prediction_error / truth[selected])),
        "fraction_improved": float(np.mean(prediction_error < prior_error)),
    }


def main() -> int:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    if not math.isfinite(args.change_threshold_log) or args.change_threshold_log <= 0:
        raise ValueError("change-threshold-log must be positive and finite")
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(args.threads)

    prior_memmap = np.load(args.prior, mmap_mode="r")
    if prior_memmap.ndim != 2:
        raise ValueError("prior must be a native HxW radial-range map")
    shape_hw = tuple(int(item) for item in prior_memmap.shape)
    prior = np.clip(np.asarray(prior_memmap), 0.3, 15.0).astype(np.float32)
    bounded_splat = np.asarray(np.load(args.bounded_splat, mmap_mode="r")).astype(
        np.float32
    )
    if bounded_splat.shape != shape_hw:
        raise ValueError("bounded splat must match the native prior shape")
    grid = _load_grid_proposals(args.grid_proposals)
    rows = grid["rows"].astype(np.int64)
    columns = grid["columns"].astype(np.int64)
    if not np.allclose(
        grid["prior_range_m"], prior[rows, columns], atol=1e-6, rtol=1e-6
    ):
        raise ValueError("frozen grid proposals do not match the supplied prior")

    target_path, _ = _paths(args.p74_root, TARGET_ID)
    target_rgb, target_support = load_native_angular_rgb(target_path, shape_hw)
    target_validity = (
        target_support & np.isfinite(prior) & (prior >= 0.3) & (prior <= 15.0)
    )
    options = ContinuousResidualOptions(
        seed_weight=args.seed_weight,
        anchor_weight=args.anchor_weight,
        smoothness_weight=args.smoothness_weight,
        color_sigma=args.color_sigma,
        log_depth_sigma=args.log_depth_sigma,
        minimum_edge_weight=args.minimum_edge_weight,
        cg_rtol=args.cg_rtol,
        cg_maxiter=args.cg_maxiter,
        native_row_chunk=args.native_row_chunk,
    )
    print(
        json.dumps(
            {
                "event": "continuous-residual-start",
                "shape_hw": shape_hw,
                "accepted_consensus_seeds": int(grid["consensus_accepted"].sum()),
                "options": options.to_dict(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    started = perf_counter()
    result = solve_continuous_grid_residual(
        prior,
        target_rgb,
        target_validity,
        rows,
        columns,
        grid["consensus_range_m"],
        grid["consensus_accepted"],
        grid["consensus_confidence"],
        options=options,
    )
    solve_seconds = perf_counter() - started
    print(
        json.dumps(
            {
                "event": "continuous-residual-solved",
                "runtime_seconds": solve_seconds,
                **result.diagnostics,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    prediction_record = _save_prediction(
        args.output, "continuous-consensus", result.radial_range_m
    )
    native_residual_path = args.output / f"{TARGET_ID}-continuous-log-residual.npy"
    grid_residual_path = args.output / f"{TARGET_ID}-continuous-grid-residual.npz"
    np.save(native_residual_path, result.native_log_residual)
    np.savez_compressed(
        grid_residual_path,
        rows=result.grid_rows,
        columns=result.grid_columns,
        log_residual=result.grid_log_residual,
    )
    freeze = {
        "schema": SCHEMA,
        "event": "continuous-prediction-frozen-before-ground-truth-open",
        "interface": INTERFACE,
        "shape_hw": list(shape_hw),
        "image_or_depth_resize": False,
        "correction_field_interpolation": (
            "bilinear evaluation of the solved periodic residual lattice on native "
            "pixel centres"
        ),
        "proposal_mode": "strict-two-source-consensus",
        "accepted_seed_count": result.accepted_seed_count,
        "options": options.to_dict(),
        "solver_diagnostics": result.diagnostics,
        "runtime_seconds": solve_seconds,
        "inputs": {
            "prior": str(args.prior),
            "prior_sha256": sha256(args.prior),
            "grid_proposals": str(args.grid_proposals),
            "grid_proposals_sha256": sha256(args.grid_proposals),
            "bounded_splat": str(args.bounded_splat),
            "bounded_splat_sha256": sha256(args.bounded_splat),
            "target_rgb": str(target_path),
            "target_rgb_sha256": sha256(target_path),
        },
        "prediction": prediction_record,
        "native_log_residual": str(native_residual_path),
        "native_log_residual_sha256": sha256(native_residual_path),
        "grid_log_residual": str(grid_residual_path),
        "grid_log_residual_sha256": sha256(grid_residual_path),
    }
    freeze_path = args.output / "FROZEN-BEFORE-GT.json"
    _json_dump(freeze_path, freeze)
    print(
        json.dumps(
            {"event": "prediction-frozen", "sha256": sha256(freeze_path)},
            sort_keys=True,
        ),
        flush=True,
    )

    # Ground truth is first opened here, after the prediction is immutable.
    truth = np.load(args.ground_truth, mmap_mode="r")
    evaluation_validity = np.load(args.evaluation_validity, mmap_mode="r")
    if truth.shape != shape_hw or evaluation_validity.shape != shape_hw:
        raise ValueError("ground truth and evaluation validity must match the prior")
    common = (
        target_validity
        & np.asarray(evaluation_validity, dtype=bool)
        & np.isfinite(truth)
        & (truth >= 0.3)
        & (truth <= 15.0)
    )
    predictions = {
        "cnn-seed": prior,
        "grid-consensus-splat": bounded_splat,
        "continuous-consensus": result.radial_range_m,
    }
    evaluations: dict[str, Any] = {}
    for name, prediction in predictions.items():
        evaluation = _evaluate(
            prediction,
            truth,
            common,
            normal_stride=args.normal_stride,
        )
        ply_path = args.output / f"{TARGET_ID}-{name}-view-15m.ply"
        ply = write_binary_ply(
            ply_path,
            prediction,
            target_rgb,
            common,
            max_points=args.ply_max_points,
        )
        ply["verification"] = _verify_binary_ply(ply_path, maximum_radius_m=15.0)
        ply["sha256"] = sha256(ply_path)
        evaluation["ply"] = ply
        evaluations[name] = evaluation

    splat_changed = np.abs(np.log(bounded_splat / prior)) >= args.change_threshold_log
    continuous_changed = np.abs(result.native_log_residual) >= args.change_threshold_log
    changed_diagnostics = {
        "grid-consensus-splat": _changed_diagnostic(
            bounded_splat,
            prior,
            truth,
            common,
            splat_changed,
        ),
        "continuous-consensus": _changed_diagnostic(
            result.radial_range_m,
            prior,
            truth,
            common,
            continuous_changed,
        ),
    }
    panel_path = args.output / "p74-w121-continuous-residual-panel.png"
    _write_panel(
        panel_path,
        target_rgb,
        truth,
        {
            "CNN seed": prior,
            "Bounded consensus splat": bounded_splat,
            "Continuous consensus residual": result.radial_range_m,
        },
        stride=args.preview_stride,
    )
    report = {
        "schema": SCHEMA,
        "status": "experimental-frozen-proposal-control",
        "interface": INTERFACE,
        "protocol": {
            "native_shape_hw": list(shape_hw),
            "image_or_depth_resize": False,
            "proposal_source": "frozen VAL-031 strict two-source consensus",
            "ground_truth_used_for_prediction": False,
            "changed_region_threshold_abs_log": args.change_threshold_log,
        },
        "configuration": options.to_dict(),
        "solver": result.diagnostics,
        "timings": {"solve_and_native_evaluation_seconds": solve_seconds},
        "changed_region_diagnostics": changed_diagnostics,
        "evaluations": evaluations,
        "inputs": {
            **freeze["inputs"],
            "ground_truth": str(args.ground_truth),
            "ground_truth_sha256": sha256(args.ground_truth),
            "evaluation_validity": str(args.evaluation_validity),
            "evaluation_validity_sha256": sha256(args.evaluation_validity),
        },
        "artifacts": {
            "freeze": str(freeze_path),
            "freeze_sha256": sha256(freeze_path),
            "prediction": prediction_record,
            "native_log_residual": str(native_residual_path),
            "native_log_residual_sha256": sha256(native_residual_path),
            "grid_log_residual": str(grid_residual_path),
            "grid_log_residual_sha256": sha256(grid_residual_path),
            "panel": str(panel_path),
            "panel_sha256": sha256(panel_path),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "opencv": cv2.__version__,
            "threads": args.threads,
        },
    }
    report_path = args.output / "results.json"
    _json_dump(report_path, report)
    print(
        json.dumps(
            {
                "event": "complete",
                "results": str(report_path),
                "abs_rel": {
                    name: value["metric_depth"]["abs_rel"]
                    for name, value in evaluations.items()
                },
                "normal_mean_deg": {
                    name: value["surface_normals"]["normal_mean_deg"]
                    for name, value in evaluations.items()
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
