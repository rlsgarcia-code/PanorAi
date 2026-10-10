#!/usr/bin/env python3
"""Refine a native Metric3D ERP prior using fused two-view Gaussian centres."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.two_view_gaussian_splatting.p74 import (  # noqa: E402
    load_native_angular_rgb,
)
from benchmarks.two_view_gaussian_splatting.gaussian_depth_feedback import (  # noqa: E402
    GaussianDepthFeedbackOptions,
    rasterize_gaussian_depth_anchors,
    read_gaussian_centres_ply,
    refine_depth_from_gaussian_anchors,
)

TARGET_ID = "P-74+MD-05_concluido_326+G046"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project a fused G046/G047 Gaussian-centre cloud into G046, solve "
            "an edge-aware log-depth correction, and apply it to the native "
            "Metric3D radial-range prior."
        )
    )
    parser.add_argument("--p74-root", required=True, type=Path)
    parser.add_argument("--prior", required=True, type=Path)
    parser.add_argument("--gaussian-cloud", required=True, type=Path)
    parser.add_argument("--landmarks", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--solve-height", type=int, default=1024)
    parser.add_argument("--solve-width", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--gaussian-sigma-px", type=float, default=0.85)
    parser.add_argument("--gaussian-radius-px", type=int, default=2)
    parser.add_argument("--maximum-anchor-log-residual", type=float, default=0.45)
    parser.add_argument("--maximum-log-correction", type=float, default=0.60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.solve_height < 32 or args.solve_width != 2 * args.solve_height:
        raise SystemExit("solve lattice must be 2:1 and at least 32x64")
    prior = np.load(args.prior, mmap_mode="r")
    if prior.ndim != 2 or prior.shape[1] != 2 * prior.shape[0]:
        raise SystemExit("prior must be a 2:1 HW radial-range map")
    solve_shape = (args.solve_height, args.solve_width)
    target_rgb, target_support = load_native_angular_rgb(
        args.p74_root / "images" / f"{TARGET_ID}_rgb.png",
        solve_shape,
    )
    points, colors, observations, view_mask = read_gaussian_centres_ply(
        args.gaussian_cloud
    )
    landmarks = np.asarray(np.load(args.landmarks), dtype=np.float64)
    options = GaussianDepthFeedbackOptions(
        gaussian_sigma_px=args.gaussian_sigma_px,
        gaussian_radius_px=args.gaussian_radius_px,
        maximum_anchor_log_residual=args.maximum_anchor_log_residual,
        maximum_log_correction=args.maximum_log_correction,
        iterations=args.iterations,
    )
    anchors = rasterize_gaussian_depth_anchors(
        points,
        colors,
        observations,
        view_mask,
        target_rgb,
        options=options,
    )
    anchors.confidence[~target_support] = 0.0
    anchors.radial_m[~target_support] = np.nan
    anchors.view_mask[~target_support] = 0
    anchors.point_count[~target_support] = 0
    feedback = refine_depth_from_gaussian_anchors(
        prior,
        target_rgb,
        anchors,
        landmark_points_target=landmarks,
        options=options,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "refined_radial_m": args.output / "refined-radial-m.npy",
        "refined_confidence": args.output / "refined-confidence.npy",
        "anchor_radial_m": args.output / "gaussian-anchor-radial-m.npy",
        "anchor_confidence": args.output / "gaussian-anchor-confidence.npy",
        "anchor_view_mask": args.output / "gaussian-anchor-view-mask.npy",
        "anchor_point_count": args.output / "gaussian-anchor-point-count.npy",
        "correction_log_solve": args.output / "log-correction-solve.npy",
        "accepted_anchor": args.output / "accepted-anchor.npy",
        "hard_landmark_anchor": args.output / "hard-landmark-anchor.npy",
    }
    np.save(artifacts["refined_radial_m"], feedback.refined_radial_m)
    np.save(artifacts["refined_confidence"], feedback.refined_confidence)
    np.save(artifacts["anchor_radial_m"], feedback.anchors.radial_m)
    np.save(artifacts["anchor_confidence"], feedback.anchors.confidence)
    np.save(artifacts["anchor_view_mask"], feedback.anchors.view_mask)
    np.save(artifacts["anchor_point_count"], feedback.anchors.point_count)
    np.save(artifacts["correction_log_solve"], feedback.correction_log_solve)
    np.save(artifacts["accepted_anchor"], feedback.accepted_anchor)
    np.save(artifacts["hard_landmark_anchor"], feedback.hard_landmark_anchor)

    # Prediction artifacts are frozen before the evaluation-only GT is opened.
    freeze = {
        "schema": "panorai-two-view-gaussian-depth-feedback-freeze/v1",
        "event": "prediction-frozen-before-ground-truth-open",
        "coordinate_frame": TARGET_ID,
        "depth_semantics": "radial range in metres",
        "ground_truth_used_for_prediction": False,
        "native_shape_hw": list(prior.shape),
        "solve_shape_hw": list(solve_shape),
        "options": options.to_dict(),
        "inputs": {
            "prior": _file_record(args.prior),
            "gaussian_cloud": _file_record(args.gaussian_cloud),
            "landmarks": _file_record(args.landmarks),
        },
        "artifacts": {name: _file_record(path) for name, path in artifacts.items()},
        "feedback": feedback.report,
    }
    freeze_path = args.output / "prediction-freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    ground_truth = np.load(args.ground_truth, mmap_mode="r")
    if ground_truth.shape != prior.shape:
        raise SystemExit("ground truth must match the native prior shape")
    evaluation_valid = (
        np.isfinite(prior)
        & (prior >= options.min_range_m)
        & (prior <= options.max_range_m)
        & np.isfinite(ground_truth)
        & (ground_truth >= options.min_range_m)
        & (ground_truth <= options.max_range_m)
    )
    confident = evaluation_valid & (feedback.refined_confidence >= 0.10)
    changed = evaluation_valid & (
        np.abs(np.log(feedback.refined_radial_m / np.asarray(prior))) >= math.log(1.01)
    )
    metrics = {
        "all_evaluation_pixels": {
            "prior": _depth_metrics(prior, ground_truth, evaluation_valid),
            "refined": _depth_metrics(
                feedback.refined_radial_m, ground_truth, evaluation_valid
            ),
        },
        "confidence_at_least_0_10": {
            "prior": _depth_metrics(prior, ground_truth, confident),
            "refined": _depth_metrics(
                feedback.refined_radial_m, ground_truth, confident
            ),
        },
        "changed_at_least_1_percent": {
            "prior": _depth_metrics(prior, ground_truth, changed),
            "refined": _depth_metrics(feedback.refined_radial_m, ground_truth, changed),
            "improved_pixel_fraction": _improved_fraction(
                prior, feedback.refined_radial_m, ground_truth, changed
            ),
        },
        "landmarks": {
            "prior": _landmark_metrics(prior, landmarks),
            "refined": _landmark_metrics(feedback.refined_radial_m, landmarks),
        },
    }
    panel_path = args.output / "gaussian-depth-feedback-panel.png"
    _write_panel(
        panel_path,
        target_rgb,
        prior,
        feedback,
        ground_truth,
        options,
    )
    results = {
        "schema": "panorai-two-view-gaussian-depth-feedback/v1",
        "research_only": True,
        "coordinate_frame": TARGET_ID,
        "depth_semantics": "radial range in metres",
        "ground_truth_used_for_prediction": False,
        "feedback": feedback.report,
        "point_cloud": {
            "points": int(points.shape[0]),
            "view_mask_1": int(np.count_nonzero(view_mask == 1)),
            "view_mask_2": int(np.count_nonzero(view_mask == 2)),
            "view_mask_3": int(np.count_nonzero(view_mask == 3)),
        },
        "metrics": metrics,
        "inputs": {
            "ground_truth": _file_record(args.ground_truth),
            "prediction_freeze": _file_record(freeze_path),
        },
        "artifacts": {
            **{name: _file_record(path) for name, path in artifacts.items()},
            "panel": _file_record(panel_path),
        },
    }
    results_path = args.output / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
    return 0


def _depth_metrics(
    prediction: Any, target: Any, valid: np.ndarray
) -> dict[str, float | int]:
    prediction_array = np.asarray(prediction)
    target_array = np.asarray(target)
    mask = valid & np.isfinite(prediction_array) & (prediction_array > 0.0)
    if not np.any(mask):
        return {
            "valid_pixels": 0,
            "abs_rel": float("nan"),
            "rmse_m": float("nan"),
            "delta_1": float("nan"),
            "median_log_ratio": float("nan"),
        }
    pred = prediction_array[mask].astype(np.float64)
    truth = target_array[mask].astype(np.float64)
    ratio = np.maximum(pred / truth, truth / pred)
    return {
        "valid_pixels": int(mask.sum()),
        "abs_rel": float(np.mean(np.abs(pred - truth) / truth)),
        "rmse_m": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "delta_1": float(np.mean(ratio < 1.25)),
        "median_log_ratio": float(np.median(np.log(pred / truth))),
    }


def _landmark_metrics(
    prediction: Any, landmark_points: np.ndarray
) -> dict[str, float | int]:
    array = np.asarray(prediction)
    ranges = np.linalg.norm(landmark_points, axis=1)
    safe = np.maximum(ranges, 1e-12)
    bearings = landmark_points / safe[:, None]
    longitude = np.arctan2(bearings[:, 0], bearings[:, 2])
    latitude = np.arcsin(np.clip(bearings[:, 1], -1.0, 1.0))
    rows = np.clip(
        np.rint((np.pi / 2.0 - latitude) / np.pi * array.shape[0] - 0.5).astype(
            np.int64
        ),
        0,
        array.shape[0] - 1,
    )
    columns = np.mod(
        np.rint((longitude + np.pi) / (2.0 * np.pi) * array.shape[1] - 0.5).astype(
            np.int64
        ),
        array.shape[1],
    )
    pred = array[rows, columns].astype(np.float64)
    valid = np.isfinite(pred) & (pred > 0.0) & np.isfinite(ranges) & (ranges > 0.0)
    pred = pred[valid]
    truth = ranges[valid]
    ratio = np.maximum(pred / truth, truth / pred)
    return {
        "count": int(valid.sum()),
        "abs_rel": float(np.mean(np.abs(pred - truth) / truth)),
        "delta_1": float(np.mean(ratio < 1.25)),
        "median_abs_log": float(np.median(np.abs(np.log(pred / truth)))),
    }


def _improved_fraction(
    prior: Any, refined: np.ndarray, ground_truth: Any, valid: np.ndarray
) -> float:
    if not np.any(valid):
        return float("nan")
    prior_error = np.abs(np.asarray(prior)[valid] - np.asarray(ground_truth)[valid])
    refined_error = np.abs(refined[valid] - np.asarray(ground_truth)[valid])
    return float(np.mean(refined_error < prior_error))


def _write_panel(
    path: Path,
    target_rgb: np.ndarray,
    prior: Any,
    feedback: Any,
    ground_truth: Any,
    options: GaussianDepthFeedbackOptions,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shape = target_rgb.shape[:2]
    prior_preview = cv2.resize(
        np.asarray(prior), (shape[1], shape[0]), interpolation=cv2.INTER_AREA
    )
    refined_preview = cv2.resize(
        feedback.refined_radial_m,
        (shape[1], shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    truth_preview = cv2.resize(
        np.asarray(ground_truth), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
    )
    correction = feedback.correction_log_solve
    confidence = cv2.resize(
        feedback.refined_confidence,
        (shape[1], shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    before_error = np.abs(prior_preview - truth_preview) / np.maximum(
        truth_preview, 1e-6
    )
    after_error = np.abs(refined_preview - truth_preview) / np.maximum(
        truth_preview, 1e-6
    )
    improvement = np.clip(before_error - after_error, -0.25, 0.25)
    figure, axes = plt.subplots(2, 3, figsize=(18, 8), facecolor="#080a0d")
    panels = (
        (target_rgb, "G046 RGB", None, None),
        (prior_preview, "Metric3D radial depth", "turbo", (0.3, 8.0)),
        (
            feedback.anchors.radial_m,
            "Visible Gaussian anchors",
            "turbo",
            (0.3, 8.0),
        ),
        (refined_preview, "Refined native depth", "turbo", (0.3, 8.0)),
        (
            correction,
            "Solved log-depth correction",
            "coolwarm",
            (-options.maximum_log_correction, options.maximum_log_correction),
        ),
        (improvement, "GT diagnostic: error reduction", "RdYlGn", (-0.25, 0.25)),
    )
    for axis, (image, title, cmap, limits) in zip(axes.flat, panels, strict=True):
        if cmap is None:
            axis.imshow(image)
        else:
            axis.imshow(image, cmap=cmap, vmin=limits[0], vmax=limits[1])
        axis.set_title(title, color="white", fontsize=12)
        axis.set_axis_off()
    figure.text(
        0.5,
        0.015,
        f"Native confidence >=0.10: {float(np.mean(confidence >= 0.10)):.1%}",
        ha="center",
        color="#c8d0d8",
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    figure.savefig(path, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
