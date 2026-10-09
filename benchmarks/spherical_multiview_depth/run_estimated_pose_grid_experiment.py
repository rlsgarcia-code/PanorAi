#!/usr/bin/env python3
"""Compare dense depth from image-estimated and registered metric poses."""

from __future__ import annotations

import argparse
import gc
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
    ContinuousResidualOptions,
    solve_continuous_grid_residual,
)
from benchmarks.spherical_multiview_depth.grid_tangent import (  # noqa: E402
    GridTangentOptions,
    infer_grid_tangent_proposal,
    regular_spherical_grid,
)
from benchmarks.spherical_multiview_depth.p74 import (  # noqa: E402
    RegisteredPose,
    load_native_angular_rgb,
    load_registered_pose,
)
from benchmarks.spherical_multiview_depth.pose_control import (  # noqa: E402
    INTERFACE as POSE_INTERFACE,
    load_quality_accepted_metric_pose,
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


SCHEMA = "panorai-p74-estimated-pose-depth-control/v1"
DEFAULT_SOURCE_ID = "P-74+MD-04_concluido_408+W_119"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare native W121-W119 tangent-grid depth using a quality-accepted "
            "image-estimated pose against the registered P74 oracle pose."
        )
    )
    parser.add_argument("--p74-root", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--estimated-pose-results", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--evaluation-validity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-id", default=DEFAULT_SOURCE_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--patch-samples", type=int, default=7)
    parser.add_argument("--support-radius-deg", type=float, default=1.0)
    parser.add_argument("--hypotheses", type=int, default=17)
    parser.add_argument("--range-factor", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--maximum-zncc-cost", type=float, default=0.40)
    parser.add_argument("--minimum-cost-margin", type=float, default=0.0001)
    parser.add_argument("--seed-weight", type=float, default=64.0)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=12.0)
    parser.add_argument("--color-sigma", type=float, default=0.12)
    parser.add_argument("--log-depth-sigma", type=float, default=0.25)
    parser.add_argument("--minimum-edge-weight", type=float, default=0.02)
    parser.add_argument("--change-threshold-log", type=float, default=0.005)
    parser.add_argument("--threads", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--normal-stride", type=int, default=8)
    parser.add_argument("--ply-max-points", type=int, default=1_500_000)
    parser.add_argument("--preview-stride", type=int, default=8)
    return parser.parse_args()


def _registered_pose_record(pose: RegisteredPose) -> dict[str, Any]:
    translation = np.asarray(pose.translation_source_from_target_m)
    return {
        "role": "registered-full-metric-oracle-control",
        "convention": pose.convention,
        "rotation_source_from_target": np.asarray(
            pose.rotation_source_from_target
        ).tolist(),
        "translation_source_from_target_m": translation.tolist(),
        "baseline_m": float(np.linalg.norm(translation)),
    }


def _save_prediction(
    output: Path,
    name: str,
    prediction: np.ndarray,
) -> dict[str, str]:
    path = output / f"{TARGET_ID}-{name}-radial-m.npy"
    png_path = output / f"{TARGET_ID}-{name}-depth-over-15m.png"
    np.save(path, prediction)
    _write_native_depth_png(png_path, prediction)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "png": str(png_path),
        "png_sha256": sha256(png_path),
    }


def _save_proposals(
    path: Path, rows: np.ndarray, columns: np.ndarray, variants
) -> None:
    values: dict[str, np.ndarray] = {"rows": rows, "columns": columns}
    for name, proposal in variants.items():
        for member in (
            "prior_range_m",
            "proposed_range_m",
            "accepted",
            "confidence",
            "best_cost",
            "cost_margin",
            "target_patch_std",
            "best_hypothesis_index",
        ):
            values[f"{name}_{member}"] = np.asarray(getattr(proposal, member))
    np.savez_compressed(path, **values)


def _proposal_overlap(first, second) -> dict[str, Any]:
    first_mask = np.asarray(first.accepted, dtype=bool)
    second_mask = np.asarray(second.accepted, dtype=bool)
    intersection = first_mask & second_mask
    union = first_mask | second_mask
    disagreement = np.abs(
        np.log(
            np.asarray(first.proposed_range_m)[intersection]
            / np.asarray(second.proposed_range_m)[intersection]
        )
    )
    return {
        "first_accepted": int(first_mask.sum()),
        "second_accepted": int(second_mask.sum()),
        "intersection": int(intersection.sum()),
        "union": int(union.sum()),
        "jaccard": float(intersection.sum() / union.sum()) if union.any() else 0.0,
        "median_abs_log_range_disagreement": (
            float(np.median(disagreement)) if disagreement.size else None
        ),
        "p90_abs_log_range_disagreement": (
            float(np.quantile(disagreement, 0.9)) if disagreement.size else None
        ),
    }


def _proposal_gt_diagnostic(proposal, truth, common) -> dict[str, Any]:
    selected = (
        np.asarray(proposal.accepted, dtype=bool)
        & common[proposal.rows, proposal.columns]
        & np.isfinite(proposal.proposed_range_m)
    )
    if not selected.any():
        return {"count": 0}
    ground_truth = truth[proposal.rows[selected], proposal.columns[selected]]
    prior_error = np.abs(proposal.prior_range_m[selected] - ground_truth) / ground_truth
    proposed_error = (
        np.abs(proposal.proposed_range_m[selected] - ground_truth) / ground_truth
    )
    return {
        "count": int(selected.sum()),
        "prior_abs_rel": float(np.mean(prior_error)),
        "proposed_abs_rel": float(np.mean(proposed_error)),
        "fraction_improved": float(np.mean(proposed_error < prior_error)),
        "median_prior_abs_rel": float(np.median(prior_error)),
        "median_proposed_abs_rel": float(np.median(proposed_error)),
    }


def _changed_diagnostic(prediction, prior, truth, common, threshold) -> dict[str, Any]:
    log_residual = np.log(np.clip(prediction, 0.3, 15.0) / prior)
    selected = common & (np.abs(log_residual) >= threshold)
    if not selected.any():
        return {"pixels": 0}
    before = np.abs(prior[selected] - truth[selected])
    after = np.abs(prediction[selected] - truth[selected])
    return {
        "pixels": int(selected.sum()),
        "prior_abs_rel": float(np.mean(before / truth[selected])),
        "prediction_abs_rel": float(np.mean(after / truth[selected])),
        "fraction_improved": float(np.mean(after < before)),
    }


def main() -> int:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    if not math.isfinite(args.change_threshold_log) or args.change_threshold_log <= 0:
        raise ValueError("change-threshold-log must be positive and finite")
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(args.threads)
    import torch

    torch.set_num_threads(args.threads)
    prior_memmap = np.load(args.prior, mmap_mode="r")
    if prior_memmap.ndim != 2:
        raise ValueError("prior must be a native HxW radial-range map")
    shape_hw = tuple(int(value) for value in prior_memmap.shape)
    prior = np.clip(np.asarray(prior_memmap), 0.3, 15.0).astype(np.float32)

    target_path, target_npz = _paths(args.p74_root, TARGET_ID)
    source_path, source_npz = _paths(args.p74_root, args.source_id)
    registered = load_registered_pose(target_npz, source_npz)
    estimated = load_quality_accepted_metric_pose(
        args.estimated_pose_results,
        args.source_id,
        registered,
    )
    target_rgb, target_support = load_native_angular_rgb(target_path, shape_hw)
    source_rgb, source_support = load_native_angular_rgb(source_path, shape_hw)
    target_validity = (
        target_support & np.isfinite(prior) & (prior >= 0.3) & (prior <= 15.0)
    )

    grid_options = GridTangentOptions(
        stride_px=args.stride,
        patch_samples=args.patch_samples,
        support_radius_deg=args.support_radius_deg,
        hypotheses=args.hypotheses,
        range_factor=args.range_factor,
        batch_size=args.batch_size,
        maximum_zncc_cost=args.maximum_zncc_cost,
        minimum_cost_margin=args.minimum_cost_margin,
    )
    residual_options = ContinuousResidualOptions(
        seed_weight=args.seed_weight,
        anchor_weight=args.anchor_weight,
        smoothness_weight=args.smoothness_weight,
        color_sigma=args.color_sigma,
        log_depth_sigma=args.log_depth_sigma,
        minimum_edge_weight=args.minimum_edge_weight,
    )
    rows, columns = regular_spherical_grid(shape_hw, grid_options.stride_px)
    poses = {
        "oracle-pose": (
            registered.rotation_source_from_target,
            registered.translation_source_from_target_m,
        ),
        "estimated-pose": (
            estimated.rotation_source_from_target,
            estimated.translation_source_from_target_m,
        ),
    }
    proposals = {}
    proposal_timings = {}
    for name, (rotation, translation) in poses.items():
        started = perf_counter()
        proposal = infer_grid_tangent_proposal(
            prior,
            target_rgb,
            source_rgb,
            target_validity,
            source_support,
            rotation,
            translation,
            rows=rows,
            columns=columns,
            options=grid_options,
            source_view_id=f"{args.source_id}:{name}",
            device=args.device,
            progress=lambda item, variant=name: print(
                json.dumps(
                    {"event": "grid-progress", "variant": variant, **item},
                    sort_keys=True,
                ),
                flush=True,
            ),
        )
        proposal_timings[name] = perf_counter() - started
        proposals[name] = proposal
        print(
            json.dumps(
                {
                    "event": "grid-complete",
                    "variant": name,
                    "runtime_seconds": proposal_timings[name],
                    **proposal.describe(),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    proposal_path = args.output / f"{TARGET_ID}-pose-control-grid-proposals.npz"
    _save_proposals(proposal_path, rows, columns, proposals)
    predictions = {"cnn-seed": prior}
    prediction_records = {
        "cnn-seed": {
            "path": str(args.prior),
            "sha256": sha256(args.prior),
        }
    }
    residual_records = {}
    residual_timings = {}
    for name, proposal in proposals.items():
        started = perf_counter()
        result = solve_continuous_grid_residual(
            prior,
            target_rgb,
            target_validity,
            proposal.rows,
            proposal.columns,
            proposal.proposed_range_m,
            proposal.accepted,
            proposal.confidence,
            options=residual_options,
        )
        residual_timings[name] = perf_counter() - started
        predictions[name] = result.radial_range_m
        prediction_records[name] = _save_prediction(
            args.output, name, result.radial_range_m
        )
        native_residual_path = args.output / f"{TARGET_ID}-{name}-log-residual.npy"
        grid_residual_path = args.output / f"{TARGET_ID}-{name}-grid-residual.npz"
        np.save(native_residual_path, result.native_log_residual)
        np.savez_compressed(
            grid_residual_path,
            rows=result.grid_rows,
            columns=result.grid_columns,
            log_residual=result.grid_log_residual,
        )
        residual_records[name] = {
            "runtime_seconds": residual_timings[name],
            "diagnostics": result.diagnostics,
            "native_log_residual": str(native_residual_path),
            "native_log_residual_sha256": sha256(native_residual_path),
            "grid_log_residual": str(grid_residual_path),
            "grid_log_residual_sha256": sha256(grid_residual_path),
        }
        del result
        gc.collect()

    prediction_difference = np.abs(
        np.log(predictions["estimated-pose"] / predictions["oracle-pose"])
    )
    pose_sensitivity = {
        "mean_abs_log_prediction_difference": float(np.mean(prediction_difference)),
        "median_abs_log_prediction_difference": float(np.median(prediction_difference)),
        "p90_abs_log_prediction_difference": float(
            np.quantile(prediction_difference, 0.9)
        ),
        "maximum_abs_log_prediction_difference": float(np.max(prediction_difference)),
    }
    freeze = {
        "schema": SCHEMA,
        "event": "pose-control-predictions-frozen-before-ground-truth-open",
        "pose_interface": POSE_INTERFACE,
        "shape_hw": list(shape_hw),
        "image_or_depth_resize": False,
        "depth_prior_model": "Metric3D-v1 ConvNeXt-Large/Hourglass spherical native",
        "source_id": args.source_id,
        "estimated_pose": estimated.describe(),
        "oracle_pose": _registered_pose_record(registered),
        "estimated_translation_scale_note": (
            "R and translation direction are image-estimated; only the metric "
            "baseline norm is taken from the registered control"
        ),
        "grid_options": grid_options.to_dict(),
        "residual_options": residual_options.to_dict(),
        "proposal_records": {
            name: {"runtime_seconds": proposal_timings[name], **proposal.describe()}
            for name, proposal in proposals.items()
        },
        "proposal_overlap": _proposal_overlap(
            proposals["oracle-pose"], proposals["estimated-pose"]
        ),
        "residual_records": residual_records,
        "prediction_records": prediction_records,
        "pose_sensitivity": pose_sensitivity,
        "inputs": {
            "prior": str(args.prior),
            "prior_sha256": sha256(args.prior),
            "estimated_pose_results": str(args.estimated_pose_results),
            "estimated_pose_results_sha256": sha256(args.estimated_pose_results),
            "target_rgb": str(target_path),
            "target_rgb_sha256": sha256(target_path),
            "source_rgb": str(source_path),
            "source_rgb_sha256": sha256(source_path),
            "proposals": str(proposal_path),
            "proposals_sha256": sha256(proposal_path),
        },
    }
    freeze_path = args.output / "FROZEN-BEFORE-GT.json"
    _json_dump(freeze_path, freeze)
    print(
        json.dumps(
            {"event": "predictions-frozen", "sha256": sha256(freeze_path)},
            sort_keys=True,
        ),
        flush=True,
    )

    # Ground truth is first opened after both pose-route predictions are frozen.
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
    evaluations = {}
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
    proposal_gt = {
        name: _proposal_gt_diagnostic(proposal, truth, common)
        for name, proposal in proposals.items()
    }
    changed_diagnostics = {
        name: _changed_diagnostic(
            predictions[name],
            prior,
            truth,
            common,
            args.change_threshold_log,
        )
        for name in ("oracle-pose", "estimated-pose")
    }
    panel_path = args.output / "p74-w121-estimated-vs-oracle-pose-panel.png"
    _write_panel(
        panel_path,
        target_rgb,
        truth,
        {
            "ConvNeXt-Large CNN seed": prior,
            "Grid continuous, oracle R,t": predictions["oracle-pose"],
            "Grid continuous, estimated R,t direction": predictions["estimated-pose"],
        },
        stride=args.preview_stride,
    )
    report = {
        "schema": SCHEMA,
        "status": "experimental-estimated-pose-versus-oracle-control",
        "protocol": {
            "native_shape_hw": list(shape_hw),
            "image_or_depth_resize": False,
            "depth_prior_model": freeze["depth_prior_model"],
            "ground_truth_used_for_prediction": False,
            "estimated_pose_scale": "registered baseline norm only",
            "single_source_reason": (
                "W119 passed the frozen pose gate; W124 was rejected and excluded"
            ),
            "changed_region_threshold_abs_log": args.change_threshold_log,
        },
        "pose_control": {
            "estimated": estimated.describe(),
            "oracle": _registered_pose_record(registered),
        },
        "configuration": {
            "grid": grid_options.to_dict(),
            "continuous_residual": residual_options.to_dict(),
        },
        "proposal_records": freeze["proposal_records"],
        "proposal_overlap": freeze["proposal_overlap"],
        "proposal_ground_truth_diagnostics": proposal_gt,
        "pose_sensitivity": pose_sensitivity,
        "changed_region_diagnostics": changed_diagnostics,
        "evaluations": evaluations,
        "timings": {
            "grid_seconds": proposal_timings,
            "continuous_residual_seconds": residual_timings,
        },
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
            "prediction_records": prediction_records,
            "residual_records": residual_records,
            "panel": str(panel_path),
            "panel_sha256": sha256(panel_path),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
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
                "abs_rel": {
                    name: value["metric_depth"]["abs_rel"]
                    for name, value in evaluations.items()
                },
                "normal_mean_deg": {
                    name: value["surface_normals"]["normal_mean_deg"]
                    for name, value in evaluations.items()
                },
                "proposal_gt": proposal_gt,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
