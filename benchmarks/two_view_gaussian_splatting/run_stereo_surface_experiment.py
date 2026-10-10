#!/usr/bin/env python3
"""Build a partial Gaussian surface from direct two-view spherical stereo."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
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
    render_spherical_gaussians,
)
from benchmarks.two_view_gaussian_splatting.monocular_surface import (
    PriorConsistencyOptions,
    filter_prior_by_other_view,
    load_and_calibrate_prior,
    merge_prior_with_stereo,
)
from benchmarks.two_view_gaussian_splatting.surface_densification import (
    SurfaceDensificationOptions,
    densify_stereo_surface,
)
from panorai.stereo import SphericalStereoOptions, estimate_spherical_range

TARGET_ID = "P-74+MD-05_concluido_326+G046"
SOURCE_ID = "P-74+MD-05_concluido_326+G047"
HELDOUT_ID = "P-74+MD-05_concluido_326+G048"


@dataclass(slots=True)
class StereoSurface:
    radial: np.ndarray
    valid: np.ndarray
    confidence: np.ndarray
    seed_radial: np.ndarray
    seed_valid: np.ndarray
    densification: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate a partial two-view spherical surface from G046/G047 and "
            "render it as Gaussians into held-out G048."
        )
    )
    parser.add_argument("--p74-root", required=True, type=Path)
    parser.add_argument("--landmarks", required=True, type=Path)
    parser.add_argument("--refined-pose", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-prior", type=Path)
    parser.add_argument("--source-prior", type=Path)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--render-gaussian-sigma-px", type=float, default=0.75)
    parser.add_argument("--render-gaussian-radius-px", type=int, default=3)
    parser.add_argument("--minimum-seed-confidence", type=float, default=0.25)
    parser.add_argument("--maximum-densification-distance-px", type=float, default=12.0)
    parser.add_argument("--candidate-blend", type=float, default=0.0)
    parser.add_argument("--maximum-prior-log-disagreement", type=float, default=0.15)
    parser.add_argument("--maximum-prior-rgb-l1", type=float, default=0.16)
    parser.add_argument(
        "--prior-mode", choices=("consistent", "all-seen"), default="consistent"
    )
    parser.add_argument("--interpolation-alpha", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.height < 32 or args.width != 2 * args.height:
        raise SystemExit("the stereo lattice must be 2:1 and at least 32x64")
    if args.render_gaussian_sigma_px <= 0 or args.render_gaussian_radius_px < 1:
        raise SystemExit("Gaussian sigma/radius must be positive")
    if (args.target_prior is None) != (args.source_prior is None):
        raise SystemExit("target-prior and source-prior must be provided together")
    if not 0.0 <= args.interpolation_alpha <= 1.0:
        raise SystemExit("interpolation-alpha must lie in [0, 1]")
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
    pose = json.loads(args.refined_pose.read_text(encoding="utf-8"))
    if pose.get("source_commit") != "e01f9e83cf7ca27ef4fbf72e45f58bf49d95d744":
        raise SystemExit("refined pose must identify exact VAL-042 commit e01f9e83")
    rotation_source = np.asarray(pose["rotation_source_from_target"], dtype=np.float64)
    translation_source = np.asarray(
        pose["translation_source_from_target_m"], dtype=np.float64
    )

    stereo_options = SphericalStereoOptions(
        min_range=0.3,
        max_range=15.0,
        num_hypotheses=96,
        window_size=7,
        pole_margin_fraction=0.04,
        min_texture_std=0.015,
        min_confidence=0.015,
        max_matching_cost=0.65,
        bidirectional_consistency=True,
        pyramid_levels=3,
        refinement_hypotheses=9,
        refinement_radius_steps=4.0,
    )
    landmarks = np.asarray(np.load(args.landmarks), dtype=np.float64)
    densification_options = SurfaceDensificationOptions(
        minimum_seed_confidence=args.minimum_seed_confidence,
        maximum_distance_px=args.maximum_densification_distance_px,
        iterations=max(1, int(math.ceil(args.maximum_densification_distance_px))),
        candidate_blend=args.candidate_blend,
        min_range_m=stereo_options.min_range,
        max_range_m=stereo_options.max_range,
    )
    target_surface = _build_surface(
        target_rgb,
        source_rgb,
        target_support,
        rotation_source,
        translation_source,
        landmarks,
        stereo_options=stereo_options,
        densification_options=densification_options,
    )
    rotation_target_from_source = rotation_source.T
    translation_target_from_source = -rotation_target_from_source @ translation_source
    landmarks_source = landmarks @ rotation_source.T + translation_source[None]
    source_surface = _build_surface(
        source_rgb,
        target_rgb,
        source_support,
        rotation_target_from_source,
        translation_target_from_source,
        landmarks_source,
        stereo_options=stereo_options,
        densification_options=densification_options,
    )
    prior_fusion: dict[str, Any] | None = None
    if args.target_prior is not None and args.source_prior is not None:
        target_calibrated, target_calibrated_valid, target_calibration = (
            load_and_calibrate_prior(
                args.target_prior,
                shape,
                landmarks,
                min_range_m=stereo_options.min_range,
                max_range_m=stereo_options.max_range,
            )
        )
        source_calibrated, source_calibrated_valid, source_calibration = (
            load_and_calibrate_prior(
                args.source_prior,
                shape,
                landmarks_source,
                min_range_m=stereo_options.min_range,
                max_range_m=stereo_options.max_range,
            )
        )
        consistency_options = PriorConsistencyOptions(
            maximum_log_depth_disagreement=args.maximum_prior_log_disagreement,
            maximum_rgb_l1=args.maximum_prior_rgb_l1,
            min_range_m=stereo_options.min_range,
            max_range_m=stereo_options.max_range,
        )
        if args.prior_mode == "consistent":
            (
                target_prior,
                target_prior_valid,
                target_prior_confidence,
                target_consistency,
            ) = filter_prior_by_other_view(
                target_calibrated,
                target_calibrated_valid,
                source_calibrated,
                source_calibrated_valid,
                target_rgb,
                source_rgb,
                target_support,
                source_support,
                rotation_source,
                translation_source,
                options=consistency_options,
            )
            (
                source_prior,
                source_prior_valid,
                source_prior_confidence,
                source_consistency,
            ) = filter_prior_by_other_view(
                source_calibrated,
                source_calibrated_valid,
                target_calibrated,
                target_calibrated_valid,
                source_rgb,
                target_rgb,
                source_support,
                target_support,
                rotation_target_from_source,
                translation_target_from_source,
                options=consistency_options,
            )
        else:
            target_prior_valid = target_calibrated_valid & target_support
            source_prior_valid = source_calibrated_valid & source_support
            target_prior = np.where(
                target_prior_valid, target_calibrated, np.nan
            ).astype(np.float32)
            source_prior = np.where(
                source_prior_valid, source_calibrated, np.nan
            ).astype(np.float32)
            target_prior_confidence = target_prior_valid.astype(np.float32)
            source_prior_confidence = source_prior_valid.astype(np.float32)
            target_consistency = {
                "mode": "all-seen",
                "accepted_pixels": int(target_prior_valid.sum()),
                "accepted_fraction": float(target_prior_valid.mean()),
            }
            source_consistency = {
                "mode": "all-seen",
                "accepted_pixels": int(source_prior_valid.sum()),
                "accepted_fraction": float(source_prior_valid.mean()),
            }
        target_surface.radial, target_surface.valid, target_surface.confidence = (
            merge_prior_with_stereo(
                target_surface.radial,
                target_surface.valid,
                target_surface.confidence,
                target_prior,
                target_prior_valid,
                target_prior_confidence,
            )
        )
        source_surface.radial, source_surface.valid, source_surface.confidence = (
            merge_prior_with_stereo(
                source_surface.radial,
                source_surface.valid,
                source_surface.confidence,
                source_prior,
                source_prior_valid,
                source_prior_confidence,
            )
        )
        prior_fusion = {
            "mode": args.prior_mode,
            "target_calibration": target_calibration,
            "source_calibration": source_calibration,
            "target_consistency": target_consistency,
            "source_consistency": source_consistency,
            "target_prior_sha256": _sha256(args.target_prior),
            "source_prior_sha256": _sha256(args.source_prior),
        }
    radial = target_surface.radial
    valid = target_surface.valid
    confidence = target_surface.confidence
    selected_seed = target_surface.seed_valid
    selected_seed_radial = target_surface.seed_radial
    densification = target_surface.densification

    args.output.mkdir(parents=True, exist_ok=True)
    seed_radial_path = args.output / "stereo-seed-radial-m.npy"
    seed_valid_path = args.output / "stereo-seed-valid.npy"
    radial_path = args.output / "stereo-surface-radial-m.npy"
    valid_path = args.output / "stereo-surface-valid.npy"
    confidence_path = args.output / "stereo-confidence.npy"
    source_radial_path = args.output / "source-stereo-surface-radial-m.npy"
    source_valid_path = args.output / "source-stereo-surface-valid.npy"
    np.save(seed_radial_path, selected_seed_radial)
    np.save(seed_valid_path, selected_seed)
    np.save(radial_path, radial)
    np.save(valid_path, valid)
    np.save(confidence_path, confidence)
    np.save(source_radial_path, source_surface.radial)
    np.save(source_valid_path, source_surface.valid)
    frozen = {
        "schema": "panorai-two-view-stereo-surface-freeze/v2",
        "training_views": [TARGET_ID, SOURCE_ID],
        "heldout_view": HELDOUT_ID,
        "source_images_used": 2,
        "unseen_geometry_created": False,
        "stereo_options": stereo_options.to_dict(),
        "densification": densification,
        "source_densification": source_surface.densification,
        "prior_fusion": prior_fusion,
        "seed_valid_pixels": int(selected_seed.sum()),
        "seed_valid_fraction": float(selected_seed.mean()),
        "valid_pixels": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "source_valid_pixels": int(source_surface.valid.sum()),
        "source_valid_fraction": float(source_surface.valid.mean()),
        "radial_sha256": _sha256(radial_path),
        "seed_radial_sha256": _sha256(seed_radial_path),
        "seed_valid_sha256": _sha256(seed_valid_path),
        "valid_sha256": _sha256(valid_path),
        "confidence_sha256": _sha256(confidence_path),
        "source_radial_sha256": _sha256(source_radial_path),
        "source_valid_sha256": _sha256(source_valid_path),
        "pose_sha256": _sha256(args.refined_pose),
        "landmarks_sha256": _sha256(args.landmarks),
    }
    freeze_path = args.output / "FROZEN-BEFORE-EVALUATION.json"
    freeze_path.write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Held-out RGB and registered depth are opened only after the two-view
    # representation and its hashes have been frozen.
    heldout_rgb, heldout_support = load_native_angular_rgb(
        images / f"{HELDOUT_ID}_rgb.png", shape
    )
    heldout_pose = load_registered_pose(
        npzs / f"{TARGET_ID}.npz", npzs / f"{HELDOUT_ID}.npz"
    )
    ground_truth_native = np.load(args.ground_truth, mmap_mode="r")
    ground_truth = cv2.resize(
        np.asarray(ground_truth_native),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.float32)

    source_target_layer = _render_surface(
        target_rgb,
        target_surface,
        rotation_source,
        translation_source,
        device,
        sigma_px=args.render_gaussian_sigma_px,
        radius_px=args.render_gaussian_radius_px,
    )
    source_identity_layer = _render_surface(
        source_rgb,
        source_surface,
        np.eye(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        device,
        sigma_px=args.render_gaussian_sigma_px,
        radius_px=args.render_gaussian_radius_px,
    )
    source_render, source_coverage = _composite_layers(
        source_target_layer, source_identity_layer
    )
    heldout_target_layer = _render_surface(
        target_rgb,
        target_surface,
        heldout_pose.rotation_source_from_target,
        heldout_pose.translation_source_from_target_m,
        device,
        sigma_px=args.render_gaussian_sigma_px,
        radius_px=args.render_gaussian_radius_px,
    )
    rotation_heldout_from_source = (
        heldout_pose.rotation_source_from_target @ rotation_target_from_source
    )
    translation_heldout_from_source = (
        heldout_pose.translation_source_from_target_m
        + heldout_pose.rotation_source_from_target @ translation_target_from_source
    )
    heldout_source_layer = _render_surface(
        source_rgb,
        source_surface,
        rotation_heldout_from_source,
        translation_heldout_from_source,
        device,
        sigma_px=args.render_gaussian_sigma_px,
        radius_px=args.render_gaussian_radius_px,
    )
    heldout_render, heldout_coverage = _composite_layers(
        heldout_target_layer, heldout_source_layer
    )
    midpoint_render: np.ndarray | None = None
    midpoint_coverage: np.ndarray | None = None
    if args.target_prior is not None:
        midpoint_rotation, midpoint_translation = _interpolate_pose(
            rotation_source,
            translation_source,
            args.interpolation_alpha,
        )
        midpoint_target_layer = _render_surface(
            target_rgb,
            target_surface,
            midpoint_rotation,
            midpoint_translation,
            device,
            sigma_px=args.render_gaussian_sigma_px,
            radius_px=args.render_gaussian_radius_px,
        )
        midpoint_rotation_from_source = midpoint_rotation @ rotation_target_from_source
        midpoint_translation_from_source = (
            midpoint_translation + midpoint_rotation @ translation_target_from_source
        )
        midpoint_source_layer = _render_surface(
            source_rgb,
            source_surface,
            midpoint_rotation_from_source,
            midpoint_translation_from_source,
            device,
            sigma_px=args.render_gaussian_sigma_px,
            radius_px=args.render_gaussian_radius_px,
        )
        if args.interpolation_alpha <= 0.5:
            midpoint_render, midpoint_coverage = _composite_primary_with_fallback(
                midpoint_target_layer, midpoint_source_layer
            )
        else:
            midpoint_render, midpoint_coverage = _composite_primary_with_fallback(
                midpoint_source_layer, midpoint_target_layer
            )
    source_render_path = args.output / "source-g047-render.png"
    heldout_render_path = args.output / "heldout-g048-render.png"
    _write_rgb(source_render_path, source_render)
    _write_rgb(heldout_render_path, heldout_render)
    midpoint_render_path = args.output / "interpolated-midpoint-render.png"
    if midpoint_render is not None:
        _write_rgb(midpoint_render_path, midpoint_render)
    panel_path = args.output / "two-view-stereo-surface-panel.png"
    _write_panel(
        panel_path,
        target_rgb,
        source_rgb,
        source_render,
        heldout_rgb,
        heldout_render,
    )

    evaluation_valid = (
        valid
        & np.isfinite(ground_truth)
        & (ground_truth >= stereo_options.min_range)
        & (ground_truth <= stereo_options.max_range)
    )
    results: dict[str, Any] = {
        "schema": "panorai-two-view-stereo-surface/v2",
        "research_only": True,
        "training_views": [TARGET_ID, SOURCE_ID],
        "heldout_view": HELDOUT_ID,
        "shape_hw": list(shape),
        "device": str(device),
        "source_images_used": 2,
        "stereo_options": stereo_options.to_dict(),
        "densification": densification,
        "source_densification": source_surface.densification,
        "prior_fusion": prior_fusion,
        "render_options": {
            "gaussian_sigma_px": args.render_gaussian_sigma_px,
            "gaussian_radius_px": args.render_gaussian_radius_px,
            "surface_aligned_ewa": True,
            "interpolation_alpha": args.interpolation_alpha,
        },
        "surface": {
            "seed_valid_pixels": int(selected_seed.sum()),
            "seed_valid_fraction": float(selected_seed.mean()),
            "seed_target_depth": _depth_metrics(
                selected_seed_radial,
                ground_truth,
                selected_seed
                & np.isfinite(ground_truth)
                & (ground_truth >= stereo_options.min_range)
                & (ground_truth <= stereo_options.max_range),
            ),
            "valid_pixels": int(valid.sum()),
            "valid_fraction": float(valid.mean()),
            "target_depth": _depth_metrics(radial, ground_truth, evaluation_valid),
            "source_valid_pixels": int(source_surface.valid.sum()),
            "source_valid_fraction": float(source_surface.valid.mean()),
        },
        "source_reprojection": _photo_metrics(
            source_render, source_rgb, source_coverage, source_support
        ),
        "heldout_reprojection": _photo_metrics(
            heldout_render, heldout_rgb, heldout_coverage, heldout_support
        ),
        "interpolated_view": (
            {
                "covered_pixels": int((midpoint_coverage >= 0.08).sum()),
                "covered_fraction": float((midpoint_coverage >= 0.08).mean()),
            }
            if midpoint_coverage is not None
            else None
        ),
        "artifacts": {
            "freeze": {"path": str(freeze_path), "sha256": _sha256(freeze_path)},
            "radial": {"path": str(radial_path), "sha256": _sha256(radial_path)},
            "seed_radial": {
                "path": str(seed_radial_path),
                "sha256": _sha256(seed_radial_path),
            },
            "source_radial": {
                "path": str(source_radial_path),
                "sha256": _sha256(source_radial_path),
            },
            "source_render": {
                "path": str(source_render_path),
                "sha256": _sha256(source_render_path),
            },
            "heldout_render": {
                "path": str(heldout_render_path),
                "sha256": _sha256(heldout_render_path),
            },
            "interpolated_render": (
                {
                    "path": str(midpoint_render_path),
                    "sha256": _sha256(midpoint_render_path),
                }
                if midpoint_render is not None
                else None
            ),
            "panel": {"path": str(panel_path), "sha256": _sha256(panel_path)},
        },
    }
    results_path = args.output / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
    return 0


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "mps":
        raise SystemExit("MPS was requested but is not exposed to this process")
    return torch.device("cpu")


def _landmark_pixels(
    points: np.ndarray, shape_hw: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    ranges = np.linalg.norm(points, axis=1)
    bearings = points / np.maximum(ranges[:, None], 1e-12)
    height, width = shape_hw
    longitude = np.arctan2(bearings[:, 0], bearings[:, 2])
    latitude = np.arcsin(np.clip(bearings[:, 1], -1.0, 1.0))
    x = np.mod(
        np.rint((longitude + np.pi) / (2.0 * np.pi) * width - 0.5).astype(np.int64),
        width,
    )
    y = np.clip(
        np.rint((np.pi / 2.0 - latitude) / np.pi * height - 0.5).astype(np.int64),
        0,
        height - 1,
    )
    return np.stack((x, y), axis=1), ranges


def _build_surface(
    reference_rgb: np.ndarray,
    other_rgb: np.ndarray,
    reference_support: np.ndarray,
    rotation_other_from_reference: np.ndarray,
    translation_other_from_reference: np.ndarray,
    landmarks_reference: np.ndarray,
    *,
    stereo_options: SphericalStereoOptions,
    densification_options: SurfaceDensificationOptions,
) -> StereoSurface:
    stereo = estimate_spherical_range(
        reference_rgb,
        other_rgb,
        rotation_other_from_reference,
        translation_other_from_reference,
        options=stereo_options,
    )
    radial = np.asarray(stereo.range, dtype=np.float32).copy()
    valid = np.asarray(stereo.validity_mask, dtype=bool).copy() & reference_support
    confidence = np.asarray(stereo.confidence, dtype=np.float32).copy()
    radial[~valid] = np.nan
    confidence[~valid] = 0.0
    landmark_pixels, landmark_ranges = _landmark_pixels(
        landmarks_reference, radial.shape
    )
    landmark_x = landmark_pixels[:, 0]
    landmark_y = landmark_pixels[:, 1]
    hard_seed = np.zeros(radial.shape, dtype=bool)
    hard_seed[landmark_y, landmark_x] = True
    radial[landmark_y, landmark_x] = landmark_ranges.astype(np.float32)
    confidence[landmark_y, landmark_x] = 1.0
    valid[landmark_y, landmark_x] = True
    selected_seed = valid & (
        (confidence >= densification_options.minimum_seed_confidence) | hard_seed
    )
    selected_seed_radial = np.where(selected_seed, radial, np.nan).astype(np.float32)
    dense_radial, dense_valid, dense_confidence, report = densify_stereo_surface(
        radial,
        confidence,
        reference_rgb,
        reference_support,
        hard_seed_hw=hard_seed,
        options=densification_options,
    )
    return StereoSurface(
        radial=dense_radial,
        valid=dense_valid,
        confidence=dense_confidence,
        seed_radial=selected_seed_radial,
        seed_valid=selected_seed,
        densification=report,
    )


def _render_surface(
    rgb: np.ndarray,
    surface: StereoSurface,
    rotation: np.ndarray,
    translation: np.ndarray,
    device: torch.device,
    *,
    sigma_px: float,
    radius_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    safe_radial = np.nan_to_num(surface.radial, nan=1.0, posinf=1.0, neginf=1.0)
    with torch.no_grad():
        rendered, coverage, depth = render_spherical_gaussians(
            torch.as_tensor(rgb.astype(np.float32) / 255.0, device=device),
            torch.as_tensor(safe_radial, dtype=torch.float32, device=device),
            torch.as_tensor(surface.valid, dtype=torch.bool, device=device),
            torch.as_tensor(rotation, dtype=torch.float32, device=device),
            torch.as_tensor(translation, dtype=torch.float32, device=device),
            sigma_px=sigma_px,
            radius_px=radius_px,
            opacity=0.95,
            occlusion_tau_m=0.05,
            surface_aligned=True,
        )
    return (
        rendered.cpu().numpy(),
        coverage.cpu().numpy(),
        depth.cpu().numpy(),
    )


def _composite_layers(
    *layers: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    colors = np.stack([layer[0] for layer in layers], axis=0)
    coverages = np.stack([layer[1] for layer in layers], axis=0)
    depths = np.stack([layer[2] for layer in layers], axis=0)
    finite = np.isfinite(depths) & (coverages > 0.0)
    safe_depth = np.where(finite, depths, np.inf)
    nearest = np.min(safe_depth, axis=0)
    relative = np.where(finite, np.maximum(depths - nearest[None], 0.0), np.inf)
    weights = coverages * np.exp(-relative / 0.05)
    weight_sum = np.sum(weights, axis=0)
    rendered = np.sum(weights[..., None] * colors, axis=0) / np.maximum(
        weight_sum[..., None], 1e-8
    )
    rendered[weight_sum <= 0.0] = 0.0
    coverage = 1.0 - np.prod(1.0 - np.clip(coverages, 0.0, 1.0), axis=0)
    return rendered.astype(np.float32), coverage.astype(np.float32)


def _composite_primary_with_fallback(
    primary: tuple[np.ndarray, np.ndarray, np.ndarray],
    fallback: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    primary_saturation: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve the nearer training view and fill only its disocclusions."""

    primary_color, primary_coverage, _ = primary
    fallback_color, fallback_coverage, _ = fallback
    primary_weight = np.clip(primary_coverage / primary_saturation, 0.0, 1.0)
    fallback_weight = np.clip(fallback_coverage, 0.0, 1.0) * (1.0 - primary_weight)
    total = primary_weight + fallback_weight
    rendered = (
        primary_weight[..., None] * primary_color
        + fallback_weight[..., None] * fallback_color
    ) / np.maximum(total[..., None], 1e-8)
    rendered[total <= 0.0] = 0.0
    return rendered.astype(np.float32), np.clip(total, 0.0, 1.0).astype(np.float32)


def _interpolate_pose(
    rotation_source_from_target: np.ndarray,
    translation_source_from_target: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate target-to-source camera pose along its shortest rotation."""

    rotation_vector = cv2.Rodrigues(rotation_source_from_target)[0].reshape(3)
    rotation = cv2.Rodrigues(alpha * rotation_vector)[0]
    source_center_target = (
        -rotation_source_from_target.T @ translation_source_from_target
    )
    center_target = alpha * source_center_target
    translation = -rotation @ center_target
    return rotation.astype(np.float64), translation.astype(np.float64)


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
        "median_abs_rel": float(np.median(np.abs(pred - truth) / truth)),
    }


def _photo_metrics(
    rendered: np.ndarray,
    target_rgb_u8: np.ndarray,
    coverage: np.ndarray,
    support: np.ndarray,
) -> dict[str, float | int]:
    target = target_rgb_u8.astype(np.float32) / 255.0
    mask = support & (coverage >= 0.08)
    error = np.mean(np.abs(rendered[mask] - target[mask]), axis=1)
    return {
        "covered_pixels": int(mask.sum()),
        "covered_fraction": float(mask.mean()),
        "rgb_l1": float(np.mean(error)) if error.size else float("nan"),
    }


def _write_rgb(path: Path, image: np.ndarray) -> None:
    rgb = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _write_panel(
    path: Path,
    target_rgb: np.ndarray,
    source_rgb: np.ndarray,
    source_render: np.ndarray,
    heldout_rgb: np.ndarray,
    heldout_render: np.ndarray,
) -> None:
    images = [
        target_rgb,
        source_rgb,
        np.clip(np.rint(source_render * 255.0), 0, 255).astype(np.uint8),
        heldout_rgb,
        np.clip(np.rint(heldout_render * 255.0), 0, 255).astype(np.uint8),
    ]
    labels = ["G046 target", "G047 real", "G047 render", "G048 heldout", "G048 render"]
    labeled: list[np.ndarray] = []
    for label, image in zip(labels, images, strict=True):
        canvas = image.copy()
        cv2.rectangle(canvas, (0, 0), (190, 20), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            label,
            (5, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
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
