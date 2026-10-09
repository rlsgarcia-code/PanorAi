#!/usr/bin/env python3
"""Run native P74 semi-dense spherical grid tangent matching."""

from __future__ import annotations

import argparse
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
from benchmarks.spherical_multiview_depth.grid_tangent import (  # noqa: E402
    INTERFACE,
    GridTangentOptions,
    fuse_grid_tangent_proposals,
    infer_grid_tangent_proposal,
    propagate_fused_grid,
    regular_spherical_grid,
)
from benchmarks.spherical_multiview_depth.p74 import (  # noqa: E402
    load_native_angular_rgb,
    load_registered_pose,
)
from benchmarks.spherical_multiview_depth.run_p74_experiment import (  # noqa: E402
    OPTIMIZATION_IDS,
    TARGET_ID,
    _evaluate,
    _json_dump,
    _paths,
    _verify_binary_ply,
    _write_native_depth_png,
    _write_panel,
)


SCHEMA = "panorai-p74-grid-tangent-depth/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate native semi-dense tangent-patch depth search on a regular "
            "W121 grid with fixed registered W119/W124 poses."
        )
    )
    parser.add_argument("--p74-root", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--evaluation-validity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--patch-samples", type=int, default=7)
    parser.add_argument("--support-radius-deg", type=float, default=1.0)
    parser.add_argument("--hypotheses", type=int, default=17)
    parser.add_argument("--range-factor", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--maximum-zncc-cost", type=float, default=0.40)
    parser.add_argument("--minimum-cost-margin", type=float, default=0.0001)
    parser.add_argument("--maximum-source-log-disagreement", type=float, default=0.10)
    parser.add_argument("--threads", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--normal-stride", type=int, default=8)
    parser.add_argument("--ply-max-points", type=int, default=1_500_000)
    parser.add_argument("--preview-stride", type=int, default=8)
    return parser.parse_args()


def _save_prediction(output: Path, name: str, value: np.ndarray) -> dict[str, str]:
    path = output / f"{TARGET_ID}-{name}-radial-m.npy"
    np.save(path, value)
    _write_native_depth_png(output / f"{TARGET_ID}-{name}-depth-over-15m.png", value)
    return {"path": str(path), "sha256": sha256(path)}


def _grid_gt_diagnostic(
    rows: np.ndarray,
    columns: np.ndarray,
    prior: np.ndarray,
    proposed: np.ndarray,
    accepted: np.ndarray,
    truth: np.ndarray,
    validity: np.ndarray,
) -> dict[str, Any]:
    selection = (
        np.asarray(accepted, dtype=bool)
        & validity[rows, columns]
        & np.isfinite(truth[rows, columns])
        & np.isfinite(proposed)
    )
    if not selection.any():
        return {"count": 0}
    gt = truth[rows[selection], columns[selection]]
    prior_error = np.abs(prior[selection] - gt) / gt
    proposed_error = np.abs(proposed[selection] - gt) / gt
    return {
        "count": int(selection.sum()),
        "prior_abs_rel": float(np.mean(prior_error)),
        "proposed_abs_rel": float(np.mean(proposed_error)),
        "fraction_improved": float(np.mean(proposed_error < prior_error)),
        "median_prior_abs_rel": float(np.median(prior_error)),
        "median_proposed_abs_rel": float(np.median(proposed_error)),
    }


def _save_grid_npz(path: Path, proposals: tuple[Any, ...], fused: Any) -> None:
    values: dict[str, np.ndarray] = {
        "rows": fused.rows,
        "columns": fused.columns,
        "prior_range_m": fused.prior_range_m,
        "union_range_m": fused.union_range_m,
        "consensus_range_m": fused.consensus_range_m,
        "union_accepted": fused.union_accepted,
        "consensus_accepted": fused.consensus_accepted,
        "union_confidence": fused.union_confidence,
        "consensus_confidence": fused.consensus_confidence,
        "accepted_source_count": fused.accepted_source_count,
        "source_log_disagreement": fused.source_log_disagreement,
    }
    for index, proposal in enumerate(proposals):
        for name in (
            "proposed_range_m",
            "accepted",
            "confidence",
            "best_cost",
            "cost_margin",
            "target_patch_std",
            "best_hypothesis_index",
        ):
            values[f"source_{index}_{name}"] = np.asarray(getattr(proposal, name))
    np.savez_compressed(path, **values)


def main() -> int:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(args.threads)
    import torch

    torch.set_num_threads(args.threads)
    prior_memmap = np.load(args.prior, mmap_mode="r")
    if prior_memmap.ndim != 2:
        raise ValueError("prior must be a native HxW radial-range map")
    shape_hw = tuple(int(item) for item in prior_memmap.shape)
    seed = np.clip(np.asarray(prior_memmap), 0.3, 15.0).astype(np.float32)
    target_path, target_npz = _paths(args.p74_root, TARGET_ID)
    target_rgb, target_support = load_native_angular_rgb(target_path, shape_hw)
    target_validity = (
        target_support & np.isfinite(seed) & (seed >= 0.3) & (seed <= 15.0)
    )
    options = GridTangentOptions(
        stride_px=args.stride,
        patch_samples=args.patch_samples,
        support_radius_deg=args.support_radius_deg,
        hypotheses=args.hypotheses,
        range_factor=args.range_factor,
        batch_size=args.batch_size,
        maximum_zncc_cost=args.maximum_zncc_cost,
        minimum_cost_margin=args.minimum_cost_margin,
        maximum_source_log_disagreement=args.maximum_source_log_disagreement,
    )
    rows, columns = regular_spherical_grid(shape_hw, options.stride_px)
    print(
        json.dumps(
            {
                "event": "grid-start",
                "shape_hw": shape_hw,
                "grid_points": int(rows.size),
                "options": options.to_dict(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    proposals = []
    source_records = []
    started_all = perf_counter()
    for source_id in OPTIMIZATION_IDS:
        source_path, source_npz = _paths(args.p74_root, source_id)
        source_rgb, source_support = load_native_angular_rgb(source_path, shape_hw)
        pose = load_registered_pose(target_npz, source_npz)
        started = perf_counter()
        proposal = infer_grid_tangent_proposal(
            seed,
            target_rgb,
            source_rgb,
            target_validity,
            source_support,
            pose.rotation_source_from_target,
            pose.translation_source_from_target_m,
            rows=rows,
            columns=columns,
            options=options,
            source_view_id=source_id,
            device=args.device,
            progress=lambda item: print(
                json.dumps({"event": "grid-progress", **item}, sort_keys=True),
                flush=True,
            ),
        )
        runtime = perf_counter() - started
        proposals.append(proposal)
        source_records.append(
            {
                "source_id": source_id,
                "baseline_m": float(
                    np.linalg.norm(pose.translation_source_from_target_m)
                ),
                "runtime_seconds": runtime,
                "proposal": proposal.describe(),
            }
        )
        print(
            json.dumps(
                {
                    "event": "source-complete",
                    "runtime_seconds": runtime,
                    **proposal.describe(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    proposals_tuple = tuple(proposals)
    fused = fuse_grid_tangent_proposals(proposals_tuple)
    started = perf_counter()
    union = propagate_fused_grid(seed, target_rgb, fused, mode="union")
    union_seconds = perf_counter() - started
    started = perf_counter()
    consensus = propagate_fused_grid(seed, target_rgb, fused, mode="consensus")
    consensus_seconds = perf_counter() - started
    total_seconds = perf_counter() - started_all

    predictions = {
        "cnn-seed": seed,
        "grid-union": union.radial_range_m,
        "grid-consensus": consensus.radial_range_m,
    }
    prediction_records = {
        name: _save_prediction(args.output, name, value)
        for name, value in predictions.items()
    }
    grid_path = args.output / f"{TARGET_ID}-grid-proposals.npz"
    _save_grid_npz(grid_path, proposals_tuple, fused)
    for name, result in (("grid-union", union), ("grid-consensus", consensus)):
        np.save(
            args.output / f"{TARGET_ID}-{name}-log-residual.npy", result.log_residual
        )
        np.save(
            args.output / f"{TARGET_ID}-{name}-weight.npy", result.accumulated_weight
        )
        np.save(args.output / f"{TARGET_ID}-{name}-changed.npy", result.changed_mask)
    freeze = {
        "schema": SCHEMA,
        "event": "predictions-frozen-before-ground-truth-open",
        "interface": INTERFACE,
        "shape_hw": list(shape_hw),
        "grid_points": int(rows.size),
        "source_ids": list(OPTIMIZATION_IDS),
        "options": options.to_dict(),
        "source_records": source_records,
        "union_accepted_grid_points": int(fused.union_accepted.sum()),
        "consensus_accepted_grid_points": int(fused.consensus_accepted.sum()),
        "union_changed_pixels": int(union.changed_mask.sum()),
        "consensus_changed_pixels": int(consensus.changed_mask.sum()),
        "predictions": prediction_records,
        "grid_proposals": str(grid_path),
        "grid_proposals_sha256": sha256(grid_path),
    }
    freeze_path = args.output / "FROZEN-BEFORE-GT.json"
    _json_dump(freeze_path, freeze)

    truth = np.load(args.ground_truth, mmap_mode="r")
    evaluation_validity = np.load(args.evaluation_validity, mmap_mode="r")
    if truth.shape != shape_hw or evaluation_validity.shape != shape_hw:
        raise ValueError("ground truth and validity must match the native prior")
    common = (
        target_validity
        & np.asarray(evaluation_validity, dtype=bool)
        & np.isfinite(truth)
        & (truth >= 0.3)
        & (truth <= 15.0)
    )
    for row, proposal in zip(source_records, proposals_tuple, strict=True):
        row["ground_truth_diagnostic"] = _grid_gt_diagnostic(
            proposal.rows,
            proposal.columns,
            proposal.prior_range_m,
            proposal.proposed_range_m,
            proposal.accepted,
            truth,
            common,
        )
    fused_diagnostics = {
        "union": _grid_gt_diagnostic(
            fused.rows,
            fused.columns,
            fused.prior_range_m,
            fused.union_range_m,
            fused.union_accepted,
            truth,
            common,
        ),
        "consensus": _grid_gt_diagnostic(
            fused.rows,
            fused.columns,
            fused.prior_range_m,
            fused.consensus_range_m,
            fused.consensus_accepted,
            truth,
            common,
        ),
    }
    evaluations: dict[str, Any] = {}
    for name, prediction in predictions.items():
        evaluation = _evaluate(
            prediction, truth, common, normal_stride=args.normal_stride
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
    changed_diagnostics = {}
    for name, result in (("grid-union", union), ("grid-consensus", consensus)):
        selected = common & result.changed_mask
        changed_diagnostics[name] = (
            {
                "pixels": int(selected.sum()),
                "seed_abs_rel": float(
                    np.mean(np.abs(seed[selected] - truth[selected]) / truth[selected])
                ),
                "prediction_abs_rel": float(
                    np.mean(
                        np.abs(predictions[name][selected] - truth[selected])
                        / truth[selected]
                    )
                ),
                "fraction_improved": float(
                    np.mean(
                        np.abs(predictions[name][selected] - truth[selected])
                        < np.abs(seed[selected] - truth[selected])
                    )
                ),
            }
            if selected.any()
            else {"pixels": 0}
        )
    panel_path = args.output / "p74-w121-grid-tangent-panel.png"
    _write_panel(
        panel_path,
        target_rgb,
        truth,
        {
            "CNN seed": seed,
            "Grid union": union.radial_range_m,
            "Grid two-source consensus": consensus.radial_range_m,
        },
        stride=args.preview_stride,
    )
    report = {
        "schema": SCHEMA,
        "status": "experimental-registered-pose-control",
        "interface": INTERFACE,
        "protocol": {
            "native_shape_hw": list(shape_hw),
            "image_or_depth_resize": False,
            "grid_stride_px": options.stride_px,
            "patch": (
                f"{options.patch_samples}x{options.patch_samples} tangent samples "
                f"over +/-{options.support_radius_deg} degrees"
            ),
            "search_coordinate": "local log inverse radial range",
            "pose": "fixed registered metric R,t",
            "sources": list(OPTIMIZATION_IDS),
            "ground_truth_used_for_prediction": False,
        },
        "configuration": options.to_dict(),
        "sources": source_records,
        "fused_grid_diagnostics": fused_diagnostics,
        "changed_region_diagnostics": changed_diagnostics,
        "evaluations": evaluations,
        "timings": {
            "total_seconds": total_seconds,
            "union_propagation_seconds": union_seconds,
            "consensus_propagation_seconds": consensus_seconds,
        },
        "inputs": {
            "prior": str(args.prior),
            "prior_sha256": sha256(args.prior),
            "ground_truth": str(args.ground_truth),
            "ground_truth_sha256": sha256(args.ground_truth),
            "evaluation_validity": str(args.evaluation_validity),
            "evaluation_validity_sha256": sha256(args.evaluation_validity),
        },
        "artifacts": {
            "freeze": str(freeze_path),
            "freeze_sha256": sha256(freeze_path),
            "grid_proposals": str(grid_path),
            "grid_proposals_sha256": sha256(grid_path),
            "panel": str(panel_path),
            "panel_sha256": sha256(panel_path),
            "predictions": prediction_records,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "device": args.device,
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
                "fused_grid": fused_diagnostics,
                "changed_regions": changed_diagnostics,
                "abs_rel": {
                    name: value["metric_depth"]["abs_rel"]
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
