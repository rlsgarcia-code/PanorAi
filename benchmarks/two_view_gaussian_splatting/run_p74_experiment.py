#!/usr/bin/env python3
"""Run the frozen P74 two-view Gaussian depth-prior feasibility test."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.two_view_gaussian_splatting.p74 import (
    load_native_angular_rgb,
    load_registered_pose,
)
from benchmarks.two_view_gaussian_splatting.gaussian_depth import (
    GaussianDepthOptions,
    align_depth_scale_from_landmarks,
    optimize_gaussian_depth,
    render_spherical_gaussians,
)

TARGET_ID = "P-74+MD-05_concluido_326+G046"
SOURCE_ID = "P-74+MD-05_concluido_326+G047"
HELDOUT_ID = "P-74+MD-05_concluido_326+G048"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize a visible-surface spherical Gaussian representation from "
            "P74 G046/G047 and evaluate only afterward on G048/registered depth."
        )
    )
    parser.add_argument("--p74-root", required=True, type=Path)
    parser.add_argument("--prior", required=True, type=Path)
    parser.add_argument("--landmarks", required=True, type=Path)
    parser.add_argument("--refined-pose", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument(
        "--gaussian-sigma-px",
        type=float,
        default=0.85,
        help="screen-space Gaussian standard deviation in output pixels",
    )
    parser.add_argument(
        "--gaussian-radius-px",
        type=int,
        default=1,
        help="optimization raster radius",
    )
    parser.add_argument(
        "--render-gaussian-sigma-px",
        type=float,
        default=None,
        help="optional evaluation-only Gaussian sigma; does not change optimization",
    )
    parser.add_argument(
        "--render-gaussian-radius-px",
        type=int,
        default=None,
        help="optional evaluation-only raster radius; does not change optimization",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="compute device; auto selects Apple MPS when it is exposed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.height < 32 or args.width != 2 * args.height:
        raise SystemExit("the feasibility lattice must be 2:1 and at least 32x64")
    device = _resolve_device(args.device)
    shape = (args.height, args.width)
    images = args.p74_root / "images"
    npzs = args.p74_root / "npzs"
    target_rgb, target_support = load_native_angular_rgb(
        images / f"{TARGET_ID}_rgb.png", shape
    )
    source_rgb, source_support = load_native_angular_rgb(
        images / f"{SOURCE_ID}_rgb.png", shape
    )
    registered_source = load_registered_pose(
        npzs / f"{TARGET_ID}.npz", npzs / f"{SOURCE_ID}.npz"
    )
    refined_pose = json.loads(args.refined_pose.read_text(encoding="utf-8"))
    if refined_pose.get("source_commit") != "e01f9e83cf7ca27ef4fbf72e45f58bf49d95d744":
        raise SystemExit("refined pose must identify exact VAL-042 commit e01f9e83")
    rotation_source = np.asarray(
        refined_pose["rotation_source_from_target"], dtype=np.float64
    )
    translation_source = np.asarray(
        refined_pose["translation_source_from_target_m"], dtype=np.float64
    )

    prior_native = np.load(args.prior, mmap_mode="r")
    landmarks = np.load(args.landmarks)
    aligned_native, alignment = align_depth_scale_from_landmarks(
        prior_native, landmarks
    )
    prior = _nearest_resize(prior_native, shape)
    aligned = _nearest_resize(aligned_native, shape)
    target_valid = (
        target_support & np.isfinite(aligned) & (aligned >= 0.3) & (aligned <= 15.0)
    )
    settings = GaussianDepthOptions(
        iterations=args.iterations,
        gaussian_sigma_px=args.gaussian_sigma_px,
        gaussian_radius_px=args.gaussian_radius_px,
    )
    render_settings = replace(
        settings,
        gaussian_sigma_px=(
            args.render_gaussian_sigma_px
            if args.render_gaussian_sigma_px is not None
            else settings.gaussian_sigma_px
        ),
        gaussian_radius_px=(
            args.render_gaussian_radius_px
            if args.render_gaussian_radius_px is not None
            else settings.gaussian_radius_px
        ),
    )
    optimized, optimization = optimize_gaussian_depth(
        target_rgb,
        aligned,
        target_valid,
        source_rgb,
        source_support,
        rotation_source,
        translation_source,
        landmarks,
        options=settings,
        device=device,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    optimized_path = args.output / "optimized-gaussian-radial-m.npy"
    aligned_path = args.output / "landmark-aligned-prior-radial-m.npy"
    np.save(optimized_path, optimized)
    np.save(aligned_path, aligned)
    frozen = {
        "schema": "panorai-two-view-gaussian-depth-freeze/v1",
        "training_views": [TARGET_ID, SOURCE_ID],
        "heldout_view": HELDOUT_ID,
        "pose_source": "post-BA VAL-042 pose reproduced from exact commit e01f9e83",
        "unseen_geometry_created": False,
        "optimized_sha256": _sha256(optimized_path),
        "aligned_sha256": _sha256(aligned_path),
        "alignment": alignment,
        "optimization": optimization,
        "device": str(device),
        "refined_pose_sha256": _sha256(args.refined_pose),
    }
    freeze_path = args.output / "FROZEN-BEFORE-EVALUATION.json"
    freeze_path.write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Held-out image and ground truth are intentionally opened only after freeze.
    heldout_rgb, heldout_support = load_native_angular_rgb(
        images / f"{HELDOUT_ID}_rgb.png", shape
    )
    pose_heldout = load_registered_pose(
        npzs / f"{TARGET_ID}.npz", npzs / f"{HELDOUT_ID}.npz"
    )
    ground_truth_native = np.load(args.ground_truth, mmap_mode="r")
    ground_truth = _nearest_resize(ground_truth_native, shape)
    evaluation_valid = (
        target_support
        & np.isfinite(ground_truth)
        & (ground_truth >= settings.min_range_m)
        & (ground_truth <= settings.max_range_m)
    )
    candidates = {
        "monocular-prior": np.asarray(prior, dtype=np.float32),
        "landmark-aligned": aligned,
        "gaussian-optimized": optimized,
        "oracle-depth-renderer-control": np.asarray(ground_truth, dtype=np.float32),
    }
    results: dict[str, Any] = {
        "schema": "panorai-two-view-gaussian-depth/v1",
        "research_only": True,
        "shape_hw": list(shape),
        "training_views": [TARGET_ID, SOURCE_ID],
        "heldout_view": HELDOUT_ID,
        "pose_note": (
            "R,t are the reproduced post-BA VAL-042 pose from exact commit e01f9e83; "
            "the registered pose is used only for a pose-error diagnostic and G048 evaluation."
        ),
        "refined_pose": {
            "path": str(args.refined_pose),
            "sha256": _sha256(args.refined_pose),
            "rotation_error_deg": _rotation_error_deg(
                rotation_source, registered_source.rotation_source_from_target
            ),
            "translation_direction_error_deg": _angle_deg(
                translation_source, registered_source.translation_source_from_target_m
            ),
        },
        "alignment": alignment,
        "optimization": optimization,
        "device": str(device),
        "render_options": {
            "gaussian_sigma_px": render_settings.gaussian_sigma_px,
            "gaussian_radius_px": render_settings.gaussian_radius_px,
        },
        "freeze_sha256": _sha256(freeze_path),
        "candidates": {},
    }
    renders: dict[str, np.ndarray] = {}
    source_renders: dict[str, np.ndarray] = {}
    source_coverages: dict[str, np.ndarray] = {}
    heldout_coverages: dict[str, np.ndarray] = {}
    for name, depth in candidates.items():
        depth_metrics = _depth_metrics(depth, ground_truth, evaluation_valid)
        landmark_metrics = _landmark_metrics(depth, landmarks)
        source_render, source_coverage = _render(
            target_rgb,
            depth,
            target_valid,
            rotation_source,
            translation_source,
            render_settings,
            landmarks,
            device,
        )
        heldout_render, heldout_coverage = _render(
            target_rgb,
            depth,
            target_valid,
            pose_heldout.rotation_source_from_target,
            pose_heldout.translation_source_from_target_m,
            render_settings,
            landmarks,
            device,
        )
        source_photo = _photo_metrics(
            source_render, source_rgb, source_coverage, source_support
        )
        heldout_photo = _photo_metrics(
            heldout_render, heldout_rgb, heldout_coverage, heldout_support
        )
        results["candidates"][name] = {
            "target_depth": depth_metrics,
            "ba_landmark_agreement": landmark_metrics,
            "source_reprojection": source_photo,
            "heldout_reprojection": heldout_photo,
        }
        renders[name] = heldout_render
        source_renders[name] = source_render
        source_coverages[name] = source_coverage
        heldout_coverages[name] = heldout_coverage

    evaluated_names = ("monocular-prior", "landmark-aligned", "gaussian-optimized")
    results["fixed_mask_reprojection"] = {
        "source": _fixed_mask_photo_metrics(
            source_renders,
            source_coverages,
            source_rgb,
            source_support,
            evaluated_names,
        ),
        "heldout": _fixed_mask_photo_metrics(
            renders,
            heldout_coverages,
            heldout_rgb,
            heldout_support,
            evaluated_names,
        ),
    }

    panel_path = args.output / "heldout-render-comparison.png"
    _write_panel(panel_path, heldout_rgb, renders)
    results["artifacts"] = {
        "optimized": {"path": str(optimized_path), "sha256": _sha256(optimized_path)},
        "aligned": {"path": str(aligned_path), "sha256": _sha256(aligned_path)},
        "freeze": {"path": str(freeze_path), "sha256": _sha256(freeze_path)},
        "panel": {"path": str(panel_path), "sha256": _sha256(panel_path)},
    }
    result_path = args.output / "results.json"
    result_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"results": str(result_path), "candidates": results["candidates"]}, indent=2
        )
    )
    return 0


def _nearest_resize(array: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    return cv2.resize(
        np.asarray(array), (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(np.float32)


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    delta = np.asarray(first) @ np.asarray(second).T
    cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def _render(
    target_rgb: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    options: GaussianDepthOptions,
    landmark_points: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(landmark_points, dtype=np.float64)
    ranges = np.linalg.norm(points, axis=1)
    bearings = points / np.maximum(ranges[:, None], 1e-12)
    height, width = depth.shape
    longitude = np.arctan2(bearings[:, 0], bearings[:, 2])
    latitude = np.arcsin(np.clip(bearings[:, 1], -1.0, 1.0))
    columns = np.mod(
        np.rint((longitude + np.pi) / (2.0 * np.pi) * width - 0.5).astype(np.int64),
        width,
    )
    rows = np.clip(
        np.rint((np.pi / 2.0 - latitude) / np.pi * height - 0.5).astype(np.int64),
        0,
        height - 1,
    )
    landmark_colors = target_rgb[rows, columns].astype(np.float32) / 255.0
    with torch.no_grad():
        rendered, coverage, _ = render_spherical_gaussians(
            torch.as_tensor(target_rgb.astype(np.float32) / 255.0, device=device),
            torch.as_tensor(depth, dtype=torch.float32, device=device),
            torch.as_tensor(valid, dtype=torch.bool, device=device),
            torch.as_tensor(rotation, dtype=torch.float32, device=device),
            torch.as_tensor(translation, dtype=torch.float32, device=device),
            extra_points_target=torch.as_tensor(
                points, dtype=torch.float32, device=device
            ),
            extra_colors=torch.as_tensor(
                landmark_colors, dtype=torch.float32, device=device
            ),
            sigma_px=options.gaussian_sigma_px,
            radius_px=options.gaussian_radius_px,
            opacity=options.opacity,
            occlusion_tau_m=options.occlusion_tau_m,
        )
    return rendered.cpu().numpy(), coverage.cpu().numpy()


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "mps":
        raise SystemExit(
            "MPS was requested but is not exposed to this process; "
            "run outside a GPU-restricted sandbox or use --device cpu"
        )
    return torch.device("cpu")


def _depth_metrics(
    prediction: np.ndarray, target: np.ndarray, valid: np.ndarray
) -> dict[str, float | int]:
    mask = valid & np.isfinite(prediction) & (prediction > 0.0)
    pred = prediction[mask].astype(np.float64)
    truth = target[mask].astype(np.float64)
    ratio = np.maximum(pred / truth, truth / pred)
    return {
        "valid_pixels": int(mask.sum()),
        "abs_rel": float(np.mean(np.abs(pred - truth) / truth)),
        "rmse_m": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "delta_1": float(np.mean(ratio < 1.25)),
        "median_log_ratio": float(np.median(np.log(pred / truth))),
    }


def _landmark_metrics(
    prediction: np.ndarray, landmark_points: np.ndarray
) -> dict[str, float | int]:
    points = np.asarray(landmark_points, dtype=np.float64)
    truth = np.linalg.norm(points, axis=1)
    bearings = points / np.maximum(truth[:, None], 1e-12)
    height, width = prediction.shape
    longitude = np.arctan2(bearings[:, 0], bearings[:, 2])
    latitude = np.arcsin(np.clip(bearings[:, 1], -1.0, 1.0))
    columns = np.mod(
        np.rint((longitude + np.pi) / (2.0 * np.pi) * width - 0.5).astype(np.int64),
        width,
    )
    rows = np.clip(
        np.rint((np.pi / 2.0 - latitude) / np.pi * height - 0.5).astype(np.int64),
        0,
        height - 1,
    )
    pred = prediction[rows, columns].astype(np.float64)
    valid = np.isfinite(pred) & (pred > 0.0) & np.isfinite(truth) & (truth > 0.0)
    pred = pred[valid]
    truth = truth[valid]
    ratio = np.maximum(pred / truth, truth / pred)
    return {
        "count": int(valid.sum()),
        "abs_rel": float(np.mean(np.abs(pred - truth) / truth)),
        "delta_1": float(np.mean(ratio < 1.25)),
        "median_abs_log": float(np.median(np.abs(np.log(pred / truth)))),
    }


def _photo_metrics(
    rendered: np.ndarray,
    target_rgb_u8: np.ndarray,
    coverage: np.ndarray,
    support: np.ndarray,
) -> dict[str, float | int]:
    target = target_rgb_u8.astype(np.float32) / 255.0
    mask = support & (coverage >= 0.08)
    weight = coverage[mask].astype(np.float64)
    error = np.mean(np.abs(rendered[mask] - target[mask]), axis=1).astype(np.float64)
    return {
        "covered_pixels": int(mask.sum()),
        "covered_fraction": float(mask.mean()),
        "weighted_rgb_l1": float(np.sum(error * weight) / max(np.sum(weight), 1e-12)),
        "rgb_l1": float(np.mean(error)) if error.size else float("nan"),
    }


def _fixed_mask_photo_metrics(
    renders: dict[str, np.ndarray],
    coverages: dict[str, np.ndarray],
    target_rgb_u8: np.ndarray,
    support: np.ndarray,
    evaluated_names: tuple[str, ...],
) -> dict[str, Any]:
    target = target_rgb_u8.astype(np.float32) / 255.0
    common = support.copy()
    for name in evaluated_names:
        common &= coverages[name] >= 0.08
    oracle_visible = support & (coverages["oracle-depth-renderer-control"] >= 0.08)
    oracle_count = max(int(oracle_visible.sum()), 1)
    metrics: dict[str, Any] = {
        "common_pixel_count": int(common.sum()),
        "oracle_visible_pixel_count": int(oracle_visible.sum()),
        "candidates": {},
    }
    for name in (*evaluated_names, "oracle-depth-renderer-control"):
        render = renders[name]
        candidate_visible = support & (coverages[name] >= 0.08)
        common_error = np.mean(np.abs(render[common] - target[common]), axis=1)
        oracle_mask_error = np.mean(
            np.abs(render[oracle_visible] - target[oracle_visible]), axis=1
        )
        metrics["candidates"][name] = {
            "common_rgb_l1": float(np.mean(common_error)),
            "oracle_mask_rgb_l1_with_uncovered_penalty": float(
                np.mean(oracle_mask_error)
            ),
            "coverage_recall_against_oracle": float(
                np.count_nonzero(candidate_visible & oracle_visible) / oracle_count
            ),
        }
    return metrics


def _write_panel(
    path: Path, heldout_rgb_u8: np.ndarray, renders: dict[str, np.ndarray]
) -> None:
    tiles = [heldout_rgb_u8]
    labels = ["heldout RGB"]
    for name, render in renders.items():
        tiles.append(np.clip(np.rint(render * 255.0), 0, 255).astype(np.uint8))
        labels.append(name)
    labeled: list[np.ndarray] = []
    for label, tile in zip(labels, tiles, strict=True):
        canvas = tile.copy()
        cv2.rectangle(
            canvas, (0, 0), (min(canvas.shape[1] - 1, 230), 18), (0, 0, 0), -1
        )
        cv2.putText(
            canvas,
            label,
            (4, 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        labeled.append(canvas)
    panel = np.concatenate(labeled, axis=0)
    cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
