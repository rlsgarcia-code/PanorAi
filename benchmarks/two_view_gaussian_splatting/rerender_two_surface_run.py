#!/usr/bin/env python3
"""Rerender a frozen two-surface P74 run with view-aware composition."""

from __future__ import annotations

import argparse
import cv2
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.two_view_gaussian_splatting.p74 import (
    load_native_angular_rgb,
    load_registered_pose,
)
from benchmarks.two_view_gaussian_splatting.run_stereo_surface_experiment import (
    HELDOUT_ID,
    SOURCE_ID,
    TARGET_ID,
    StereoSurface,
    _composite_primary_with_fallback,
    _composite_view_weighted,
    _interpolate_pose,
    _photo_metrics,
    _render_surface,
    _resolve_device,
    _write_panel,
    _write_rgb,
)
from benchmarks.two_view_gaussian_splatting.surface_quality import (
    SurfaceQualityOptions,
    assess_surface_quality,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerender frozen G046/G047 visible surfaces without recomputing "
            "stereo or allowing the farther surface to overwrite the nearer one."
        )
    )
    parser.add_argument("--p74-root", required=True, type=Path)
    parser.add_argument("--surface-run", required=True, type=Path)
    parser.add_argument("--refined-pose", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--interpolation-alpha", type=float, default=0.10)
    parser.add_argument("--gaussian-sigma-px", type=float, default=0.85)
    parser.add_argument("--gaussian-radius-px", type=int, default=3)
    parser.add_argument("--occlusion-tau-m", type=float, default=0.20)
    parser.add_argument("--point-chunk-size", type=int, default=65536)
    parser.add_argument("--antialias-samples", type=int, default=2)
    parser.add_argument("--alpha-depth-bins", type=int, default=8)
    parser.add_argument(
        "--composition-mode",
        choices=("primary", "continuous"),
        default="primary",
        help="continuous is an interpolation ablation; primary is accepted",
    )
    parser.add_argument(
        "--compositing-mode",
        choices=("normalized", "alpha"),
        default="normalized",
        help="alpha is physically ordered; normalized is the legacy ablation",
    )
    parser.add_argument(
        "--projected-covariance-mode",
        choices=("legacy", "jacobian"),
        default="legacy",
        help="jacobian enables the centered adaptive projected footprint",
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=None,
        help="optional accepted run used as a non-regression gate",
    )
    parser.add_argument(
        "--surface-cleaning",
        action="store_true",
        help="remove isolated floaters and derive evidence-weighted opacity",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.interpolation_alpha <= 1.0:
        raise SystemExit("interpolation-alpha must lie in [0, 1]")
    if (
        min(
            args.gaussian_sigma_px,
            args.gaussian_radius_px,
            args.occlusion_tau_m,
            args.point_chunk_size,
            args.antialias_samples,
            args.alpha_depth_bins,
        )
        <= 0
    ):
        raise SystemExit("render parameters must be positive")
    target_surface = _load_surface(args.surface_run, source=False)
    source_surface = _load_surface(args.surface_run, source=True)
    if target_surface.radial.shape != source_surface.radial.shape:
        raise SystemExit("target and source frozen surfaces must share shape")
    shape = target_surface.radial.shape
    if shape[1] != 2 * shape[0]:
        raise SystemExit("frozen surfaces must use a 2:1 ERP lattice")

    pose = json.loads(args.refined_pose.read_text(encoding="utf-8"))
    if pose.get("source_commit") != "e01f9e83cf7ca27ef4fbf72e45f58bf49d95d744":
        raise SystemExit("refined pose must identify exact VAL-042 commit e01f9e83")
    rotation_source = np.asarray(pose["rotation_source_from_target"], dtype=np.float64)
    translation_source = np.asarray(
        pose["translation_source_from_target_m"], dtype=np.float64
    )
    rotation_target_from_source = rotation_source.T
    translation_target_from_source = -rotation_target_from_source @ translation_source

    images = args.p74_root / "images"
    npzs = args.p74_root / "npzs"
    target_rgb, target_support = load_native_angular_rgb(
        images / f"{TARGET_ID}_rgb.png",
        shape,
        antialias_samples=args.antialias_samples,
    )
    source_rgb, source_support = load_native_angular_rgb(
        images / f"{SOURCE_ID}_rgb.png",
        shape,
        antialias_samples=args.antialias_samples,
    )
    heldout_rgb, heldout_support = load_native_angular_rgb(
        images / f"{HELDOUT_ID}_rgb.png",
        shape,
        antialias_samples=args.antialias_samples,
    )
    heldout_pose = load_registered_pose(
        npzs / f"{TARGET_ID}.npz", npzs / f"{HELDOUT_ID}.npz"
    )
    quality_options = SurfaceQualityOptions()
    if not args.surface_cleaning:
        target_quality = {"disabled": True}
        source_quality = {"disabled": True}
        target_surface.confidence = target_surface.valid.astype(np.float32)
        source_surface.confidence = source_surface.valid.astype(np.float32)
    else:
        target_cleaned, target_opacity, target_quality = assess_surface_quality(
            target_surface.radial,
            target_surface.valid,
            target_rgb,
            base_confidence_hw=target_surface.confidence,
            options=quality_options,
        )
        source_cleaned, source_opacity, source_quality = assess_surface_quality(
            source_surface.radial,
            source_surface.valid,
            source_rgb,
            base_confidence_hw=source_surface.confidence,
            options=quality_options,
        )
        target_surface.valid = target_cleaned
        target_surface.confidence = target_opacity
        source_surface.valid = source_cleaned
        source_surface.confidence = source_opacity
    device = _resolve_device(args.device)
    render_kwargs = {
        "sigma_px": args.gaussian_sigma_px,
        "radius_px": args.gaussian_radius_px,
        "occlusion_tau_m": args.occlusion_tau_m,
        "point_chunk_size": args.point_chunk_size,
        "use_surface_confidence": True,
        "compositing_mode": args.compositing_mode,
        "alpha_depth_bins": args.alpha_depth_bins,
        "projected_covariance_mode": args.projected_covariance_mode,
    }

    source_identity_layer = _render_surface(
        source_rgb,
        source_surface,
        np.eye(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        device,
        **render_kwargs,
    )
    source_target_layer = _render_surface(
        target_rgb,
        target_surface,
        rotation_source,
        translation_source,
        device,
        **render_kwargs,
    )
    source_render, source_coverage = _composite_primary_with_fallback(
        source_identity_layer, source_target_layer
    )

    midpoint_rotation, midpoint_translation = _interpolate_pose(
        rotation_source, translation_source, args.interpolation_alpha
    )
    midpoint_target_layer = _render_surface(
        target_rgb,
        target_surface,
        midpoint_rotation,
        midpoint_translation,
        device,
        **render_kwargs,
    )
    midpoint_source_layer = _render_surface(
        source_rgb,
        source_surface,
        midpoint_rotation @ rotation_target_from_source,
        midpoint_translation + midpoint_rotation @ translation_target_from_source,
        device,
        **render_kwargs,
    )
    if args.composition_mode == "continuous":
        midpoint_render, midpoint_coverage = _composite_view_weighted(
            (midpoint_target_layer, midpoint_source_layer),
            (1.0 - args.interpolation_alpha, args.interpolation_alpha),
        )
    elif args.interpolation_alpha <= 0.5:
        midpoint_render, midpoint_coverage = _composite_primary_with_fallback(
            midpoint_target_layer, midpoint_source_layer
        )
    else:
        midpoint_render, midpoint_coverage = _composite_primary_with_fallback(
            midpoint_source_layer, midpoint_target_layer
        )

    heldout_target_layer = _render_surface(
        target_rgb,
        target_surface,
        heldout_pose.rotation_source_from_target,
        heldout_pose.translation_source_from_target_m,
        device,
        **render_kwargs,
    )
    heldout_source_layer = _render_surface(
        source_rgb,
        source_surface,
        heldout_pose.rotation_source_from_target @ rotation_target_from_source,
        heldout_pose.translation_source_from_target_m
        + heldout_pose.rotation_source_from_target @ translation_target_from_source,
        device,
        **render_kwargs,
    )
    target_center = np.zeros(3, dtype=np.float64)
    source_center = -rotation_source.T @ translation_source
    heldout_center = (
        -heldout_pose.rotation_source_from_target.T
        @ heldout_pose.translation_source_from_target_m
    )
    target_distance = max(float(np.linalg.norm(heldout_center - target_center)), 1e-6)
    source_distance = max(float(np.linalg.norm(heldout_center - source_center)), 1e-6)
    if args.composition_mode == "continuous":
        heldout_render, heldout_coverage = _composite_view_weighted(
            (heldout_target_layer, heldout_source_layer),
            (1.0 / target_distance**2, 1.0 / source_distance**2),
        )
    elif target_distance <= source_distance:
        heldout_render, heldout_coverage = _composite_primary_with_fallback(
            heldout_target_layer, heldout_source_layer
        )
    else:
        heldout_render, heldout_coverage = _composite_primary_with_fallback(
            heldout_source_layer, heldout_target_layer
        )

    args.output.mkdir(parents=True, exist_ok=True)
    source_path = args.output / "source-view-aware-render.png"
    midpoint_path = args.output / "interpolated-view-aware-render.png"
    heldout_path = args.output / "heldout-view-aware-render.png"
    panel_path = args.output / "view-aware-two-surface-panel.png"
    _write_rgb(source_path, source_render)
    _write_rgb(midpoint_path, midpoint_render)
    _write_rgb(heldout_path, heldout_render)
    _write_panel(
        panel_path,
        target_rgb,
        source_rgb,
        source_render,
        heldout_rgb,
        heldout_render,
    )
    source_metrics = _photo_metrics(
        source_render, source_rgb, source_coverage, source_support
    )
    heldout_metrics = _photo_metrics(
        heldout_render, heldout_rgb, heldout_coverage, heldout_support
    )
    midpoint_coverage_fraction = float((midpoint_coverage >= 0.08).mean())
    results: dict[str, Any] = {
        "schema": "panorai-two-surface-view-aware-render/v2",
        "research_only": True,
        "surface_run": str(args.surface_run),
        "shape_hw": list(shape),
        "device": str(device),
        "composition": args.composition_mode,
        "surface_quality": {
            "target": target_quality,
            "source": source_quality,
        },
        "render_options": {
            "gaussian_sigma_px": args.gaussian_sigma_px,
            "gaussian_radius_px": args.gaussian_radius_px,
            "occlusion_tau_m": args.occlusion_tau_m,
            "point_chunk_size": args.point_chunk_size,
            "antialias_samples_per_axis": args.antialias_samples,
            "compositing_mode": args.compositing_mode,
            "alpha_depth_bins": args.alpha_depth_bins,
            "projected_covariance_mode": args.projected_covariance_mode,
        },
        "source_reprojection": source_metrics,
        "midpoint": {
            "alpha": args.interpolation_alpha,
            "covered_fraction": midpoint_coverage_fraction,
        },
        "heldout_reprojection": heldout_metrics,
        "artifacts": {},
    }
    if args.baseline_results is not None:
        baseline = json.loads(args.baseline_results.read_text(encoding="utf-8"))
        baseline_source = baseline["source_reprojection"]
        baseline_midpoint = baseline["midpoint"]
        baseline_midpoint_path = Path(baseline["artifacts"]["midpoint_render"]["path"])
        baseline_midpoint_bgr = cv2.imread(
            str(baseline_midpoint_path), cv2.IMREAD_COLOR
        )
        if baseline_midpoint_bgr is None:
            raise SystemExit(
                f"cannot read baseline midpoint render: {baseline_midpoint_path}"
            )
        baseline_midpoint_rgb = (
            cv2.cvtColor(
                baseline_midpoint_bgr,
                cv2.COLOR_BGR2RGB,
            ).astype(np.float32)
            / 255.0
        )
        midpoint_rgb_l1_vs_baseline = float(
            np.mean(np.abs(midpoint_render - baseline_midpoint_rgb))
        )
        checks = _non_regression_checks(
            source_metrics,
            midpoint_coverage_fraction,
            baseline_source,
            baseline_midpoint["covered_fraction"],
            midpoint_rgb_l1_vs_baseline,
        )
        results["acceptance_gate"] = {
            "baseline_results": str(args.baseline_results),
            "checks": checks,
            "accepted": bool(all(checks.values())),
            "heldout_is_diagnostic_only": True,
            "midpoint_rgb_l1_vs_baseline": midpoint_rgb_l1_vs_baseline,
        }
    for name, path in (
        ("source_render", source_path),
        ("midpoint_render", midpoint_path),
        ("heldout_render", heldout_path),
        ("panel", panel_path),
    ):
        results["artifacts"][name] = {
            "path": str(path),
            "sha256": _sha256(path),
        }
    results_path = args.output / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def _non_regression_checks(
    source_metrics: dict[str, float | int],
    midpoint_coverage: float,
    baseline_source_metrics: dict[str, float | int],
    baseline_midpoint_coverage: float,
    midpoint_rgb_l1_vs_baseline: float = 0.0,
) -> dict[str, bool]:
    """Return training-view gates; held-out evidence remains diagnostic only."""

    return {
        "source_rgb_l1_not_worse_by_more_than_0_001": (
            float(source_metrics["rgb_l1"])
            <= float(baseline_source_metrics["rgb_l1"]) + 0.001
        ),
        "source_coverage_not_lower_by_more_than_0_005": (
            float(source_metrics["covered_fraction"])
            >= float(baseline_source_metrics["covered_fraction"]) - 0.005
        ),
        "midpoint_coverage_not_lower_by_more_than_0_005": (
            float(midpoint_coverage) >= float(baseline_midpoint_coverage) - 0.005
        ),
        "midpoint_rgb_l1_vs_baseline_not_above_0_02": (
            float(midpoint_rgb_l1_vs_baseline) <= 0.02
        ),
    }


def _load_surface(run: Path, *, source: bool) -> StereoSurface:
    prefix = "source-" if source else ""
    radial = np.asarray(
        np.load(run / f"{prefix}stereo-surface-radial-m.npy"), dtype=np.float32
    )
    valid = np.asarray(np.load(run / f"{prefix}stereo-surface-valid.npy"), dtype=bool)
    if radial.shape != valid.shape:
        raise ValueError("frozen radial and validity arrays must share shape")
    radial = np.where(valid, radial, np.nan).astype(np.float32)
    return StereoSurface(
        radial=radial,
        valid=valid,
        confidence=valid.astype(np.float32),
        seed_radial=np.full(radial.shape, np.nan, dtype=np.float32),
        seed_valid=np.zeros(radial.shape, dtype=bool),
        densification={"reused_surface_run": str(run)},
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
