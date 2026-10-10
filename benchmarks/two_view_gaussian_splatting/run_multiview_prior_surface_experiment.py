#!/usr/bin/env python3
"""Evaluate an additional monocular-prior Gaussian surface on a held-out view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.two_view_gaussian_splatting.monocular_surface import (  # noqa: E402
    load_and_calibrate_prior,
)
from benchmarks.two_view_gaussian_splatting.p74 import (  # noqa: E402
    load_native_angular_rgb,
    load_registered_pose,
)
from benchmarks.two_view_gaussian_splatting.run_multiview_registered_surface_experiment import (  # noqa: E402
    P74_PREFIX,
    _composite_ordered_fallback,
    _sha256,
    _write_labeled_panel,
)
from benchmarks.two_view_gaussian_splatting.run_stereo_surface_experiment import (  # noqa: E402
    StereoSurface,
    _photo_metrics,
    _render_surface,
    _resolve_device,
    _write_rgb,
)

BASE_TARGET_ID = "G046"
BASE_SOURCE_ID = "G047"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append one landmark-calibrated monocular surface to a frozen "
            "G046/G047 Gaussian reconstruction and evaluate on a held-out view."
        )
    )
    parser.add_argument("--p74-root", required=True, type=Path)
    parser.add_argument("--frozen-two-view-run", required=True, type=Path)
    parser.add_argument("--landmarks", required=True, type=Path)
    parser.add_argument("--refined-pose", required=True, type=Path)
    parser.add_argument("--additional-id", default="G049")
    parser.add_argument("--additional-prior", required=True, type=Path)
    parser.add_argument("--target-id", default="G048")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--antialias-samples", type=int, default=1)
    parser.add_argument("--gaussian-sigma-px", type=float, default=1.3)
    parser.add_argument("--gaussian-radius-px", type=int, default=4)
    parser.add_argument("--occlusion-tau-m", type=float, default=0.08)
    parser.add_argument("--point-chunk-size", type=int, default=65536)
    parser.add_argument(
        "--compositing-mode",
        choices=("normalized", "alpha"),
        default="alpha",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shape = (args.height, args.width)
    if args.height < 32 or args.width != 2 * args.height:
        raise SystemExit("render lattice must be 2:1 and at least 32x64")
    if args.target_id in (BASE_TARGET_ID, BASE_SOURCE_ID, args.additional_id):
        raise SystemExit("target-id must be held out from every training view")
    if len({BASE_TARGET_ID, BASE_SOURCE_ID, args.additional_id}) != 3:
        raise SystemExit("additional-id must differ from G046 and G047")
    if (
        min(
            args.antialias_samples,
            args.gaussian_sigma_px,
            args.gaussian_radius_px,
            args.occlusion_tau_m,
            args.point_chunk_size,
        )
        <= 0
    ):
        raise SystemExit("render parameters must be positive")

    frozen_results = json.loads(
        (args.frozen_two_view_run / "results.json").read_text(encoding="utf-8")
    )
    expected_training = [
        f"{P74_PREFIX}{BASE_TARGET_ID}",
        f"{P74_PREFIX}{BASE_SOURCE_ID}",
    ]
    if frozen_results.get("training_views") != expected_training:
        raise SystemExit("frozen run must contain the G046/G047 training pair")
    if tuple(frozen_results.get("shape_hw", ())) != shape:
        raise SystemExit("frozen run shape must match requested render shape")

    images = args.p74_root / "images"
    npzs = args.p74_root / "npzs"
    paths = {
        view_id: npzs / f"{P74_PREFIX}{view_id}.npz"
        for view_id in (
            BASE_TARGET_ID,
            BASE_SOURCE_ID,
            args.additional_id,
            args.target_id,
        )
    }
    rgb: dict[str, np.ndarray] = {}
    support: dict[str, np.ndarray] = {}
    for view_id in paths:
        rgb[view_id], support[view_id] = load_native_angular_rgb(
            images / f"{P74_PREFIX}{view_id}_rgb.png",
            shape,
            antialias_samples=args.antialias_samples,
        )

    base_target_surface = _load_frozen_surface(
        args.frozen_two_view_run / "stereo-surface-radial-m.npy",
        args.frozen_two_view_run / "stereo-surface-valid.npy",
        args.frozen_two_view_run / "stereo-confidence.npy",
        expected_shape=shape,
    )
    base_source_surface = _load_frozen_surface(
        args.frozen_two_view_run / "source-stereo-surface-radial-m.npy",
        args.frozen_two_view_run / "source-stereo-surface-valid.npy",
        None,
        expected_shape=shape,
    )

    base_landmarks = np.asarray(np.load(args.landmarks), dtype=np.float64)
    pose_additional_from_base = load_registered_pose(
        paths[BASE_TARGET_ID], paths[args.additional_id]
    )
    additional_landmarks = (
        base_landmarks @ pose_additional_from_base.rotation_source_from_target.T
        + pose_additional_from_base.translation_source_from_target_m[None]
    )
    additional_radial, additional_valid, calibration = load_and_calibrate_prior(
        args.additional_prior,
        shape,
        additional_landmarks,
        min_range_m=0.3,
        max_range_m=15.0,
    )
    additional_valid &= support[args.additional_id]
    additional_surface = StereoSurface(
        radial=np.where(additional_valid, additional_radial, np.nan).astype(np.float32),
        valid=additional_valid,
        confidence=additional_valid.astype(np.float32),
        seed_radial=additional_radial,
        seed_valid=additional_valid,
        densification={"monocular_prior_surface": True},
    )

    refined = json.loads(args.refined_pose.read_text(encoding="utf-8"))
    rotation_g047_from_g046 = np.asarray(
        refined["rotation_source_from_target"], dtype=np.float64
    )
    translation_g047_from_g046 = np.asarray(
        refined["translation_source_from_target_m"], dtype=np.float64
    )
    rotation_g046_from_g047 = rotation_g047_from_g046.T
    translation_g046_from_g047 = -rotation_g046_from_g047 @ translation_g047_from_g046
    pose_target_from_g046 = load_registered_pose(
        paths[BASE_TARGET_ID], paths[args.target_id]
    )
    rotations = {
        BASE_TARGET_ID: pose_target_from_g046.rotation_source_from_target,
        BASE_SOURCE_ID: (
            pose_target_from_g046.rotation_source_from_target @ rotation_g046_from_g047
        ),
        args.additional_id: load_registered_pose(
            paths[args.additional_id], paths[args.target_id]
        ).rotation_source_from_target,
    }
    translations = {
        BASE_TARGET_ID: pose_target_from_g046.translation_source_from_target_m,
        BASE_SOURCE_ID: (
            pose_target_from_g046.translation_source_from_target_m
            + pose_target_from_g046.rotation_source_from_target
            @ translation_g046_from_g047
        ),
        args.additional_id: load_registered_pose(
            paths[args.additional_id], paths[args.target_id]
        ).translation_source_from_target_m,
    }
    surfaces = {
        BASE_TARGET_ID: base_target_surface,
        BASE_SOURCE_ID: base_source_surface,
        args.additional_id: additional_surface,
    }
    device = _resolve_device(args.device)
    layers = {
        view_id: _render_surface(
            rgb[view_id],
            surfaces[view_id],
            rotations[view_id],
            translations[view_id],
            device,
            sigma_px=args.gaussian_sigma_px,
            radius_px=args.gaussian_radius_px,
            occlusion_tau_m=args.occlusion_tau_m,
            point_chunk_size=args.point_chunk_size,
            projected_covariance_mode="legacy",
            compositing_mode=args.compositing_mode,
        )
        for view_id in surfaces
    }
    distances = {
        view_id: float(np.linalg.norm(translations[view_id])) for view_id in surfaces
    }
    base_order = sorted((BASE_TARGET_ID, BASE_SOURCE_ID), key=distances.__getitem__)
    nearest_order = sorted(surfaces, key=distances.__getitem__)
    variants = {
        "two-view": base_order,
        "additional-only": [args.additional_id],
        "three-view-nearest": nearest_order,
        "two-view-plus-additional-fill": [*base_order, args.additional_id],
    }

    args.output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema": "panorai-multiview-prior-surface/v1",
        "research_only": True,
        "training_views": list(surfaces),
        "target_view": args.target_id,
        "target_excluded_from_training": True,
        "shape_hw": list(shape),
        "compositing_mode": args.compositing_mode,
        "device": str(device),
        "additional_prior_calibration": calibration,
        "additional_prior_sha256": _sha256(args.additional_prior),
        "source_distances_to_target_m": distances,
        "variants": {},
        "artifacts": {},
    }
    panel = [(f"{args.target_id} real", rgb[args.target_id])]
    for name, order in variants.items():
        rendered, coverage = _composite_ordered_fallback(
            tuple(layers[view_id] for view_id in order)
        )
        metrics = _photo_metrics(
            rendered, rgb[args.target_id], coverage, support[args.target_id]
        )
        render_path = args.output / f"{name}-render.png"
        _write_rgb(render_path, rendered)
        result["variants"][name] = {"source_ids": order, "metrics": metrics}
        result["artifacts"][name] = {
            "path": str(render_path),
            "sha256": _sha256(render_path),
        }
        panel.append((name, rendered))

    baseline = result["variants"]["two-view"]["metrics"]
    nearest = result["variants"]["three-view-nearest"]["metrics"]
    fill = result["variants"]["two-view-plus-additional-fill"]["metrics"]
    result["gates"] = {
        "nearest_order": _gate(nearest, baseline),
        "fallback_only": _gate(fill, baseline),
    }
    panel_path = args.output / "multiview-prior-comparison.png"
    _write_labeled_panel(panel_path, panel)
    result["artifacts"]["panel"] = {
        "path": str(panel_path),
        "sha256": _sha256(panel_path),
    }
    result_path = args.output / "results.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _load_frozen_surface(
    radial_path: Path,
    valid_path: Path,
    confidence_path: Path | None,
    *,
    expected_shape: tuple[int, int],
) -> StereoSurface:
    radial = np.asarray(np.load(radial_path), dtype=np.float32)
    valid = np.asarray(np.load(valid_path), dtype=bool)
    if radial.shape != expected_shape or valid.shape != expected_shape:
        raise ValueError("frozen surface shape does not match render shape")
    confidence = (
        np.asarray(np.load(confidence_path), dtype=np.float32)
        if confidence_path is not None
        else valid.astype(np.float32)
    )
    if confidence.shape != expected_shape:
        raise ValueError("frozen confidence shape does not match render shape")
    return StereoSurface(
        radial=radial,
        valid=valid,
        confidence=confidence,
        seed_radial=radial,
        seed_valid=valid,
        densification={"loaded_from_frozen_run": True},
    )


def _gate(
    candidate: dict[str, float | int], baseline: dict[str, float | int]
) -> dict[str, bool]:
    checks = {
        "coverage_non_decreasing": (
            float(candidate["covered_fraction"]) >= float(baseline["covered_fraction"])
        ),
        "rgb_l1_improved": float(candidate["rgb_l1"]) < float(baseline["rgb_l1"]),
    }
    return {**checks, "accepted": all(checks.values())}


if __name__ == "__main__":
    raise SystemExit(main())
