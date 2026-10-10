#!/usr/bin/env python3
"""Evaluate ordered N-view Gaussian surfaces with registered P74 geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.two_view_gaussian_splatting.p74 import (
    load_native_angular_rgb,
    load_native_registered_radial,
    load_registered_pose,
)
from benchmarks.two_view_gaussian_splatting.run_stereo_surface_experiment import (
    StereoSurface,
    _photo_metrics,
    _render_surface,
    _resolve_device,
    _write_rgb,
)

P74_PREFIX = "P-74+MD-05_concluido_326+"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leave-one-view-out oracle for deciding whether additional observed "
            "Gaussian surfaces can improve P74 novel-view rendering."
        )
    )
    parser.add_argument("--p74-root", required=True, type=Path)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--source-ids", required=True, nargs="+")
    parser.add_argument("--baseline-source-ids", nargs="+", default=("G046", "G047"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--antialias-samples", type=int, default=1)
    parser.add_argument("--gaussian-sigma-px", type=float, default=0.85)
    parser.add_argument("--gaussian-radius-px", type=int, default=3)
    parser.add_argument("--occlusion-tau-m", type=float, default=0.20)
    parser.add_argument("--point-chunk-size", type=int, default=65536)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shape = (args.height, args.width)
    if args.height < 32 or args.width != 2 * args.height:
        raise SystemExit("render lattice must be 2:1 and at least 32x64")
    if args.target_id in args.source_ids:
        raise SystemExit("target-id must be held out from source-ids")
    if len(set(args.source_ids)) != len(args.source_ids):
        raise SystemExit("source-ids must not contain duplicates")
    if not set(args.baseline_source_ids).issubset(args.source_ids):
        raise SystemExit("baseline-source-ids must be included in source-ids")
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

    images = args.p74_root / "images"
    npzs = args.p74_root / "npzs"
    target_name = f"{P74_PREFIX}{args.target_id}"
    target_npz = npzs / f"{target_name}.npz"
    target_rgb, target_support = load_native_angular_rgb(
        images / f"{target_name}_rgb.png",
        shape,
        antialias_samples=args.antialias_samples,
    )
    device = _resolve_device(args.device)

    layers: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    sources: list[dict[str, Any]] = []
    for source_id in args.source_ids:
        source_name = f"{P74_PREFIX}{source_id}"
        source_npz = npzs / f"{source_name}.npz"
        source_rgb, source_rgb_support = load_native_angular_rgb(
            images / f"{source_name}_rgb.png",
            shape,
            antialias_samples=args.antialias_samples,
        )
        source_radial, source_depth_valid = load_native_registered_radial(
            source_npz,
            shape,
            antialias_samples=args.antialias_samples,
        )
        source_valid = source_rgb_support & source_depth_valid
        surface = StereoSurface(
            radial=source_radial,
            valid=source_valid,
            confidence=source_valid.astype(np.float32),
            seed_radial=source_radial,
            seed_valid=source_valid,
            densification={"registered_surface_oracle": True},
        )
        pose_target_from_source = load_registered_pose(source_npz, target_npz)
        pose_source_from_target = load_registered_pose(target_npz, source_npz)
        source_center_target = (
            -pose_source_from_target.rotation_source_from_target.T
            @ pose_source_from_target.translation_source_from_target_m
        )
        distance = float(np.linalg.norm(source_center_target))
        layer = _render_surface(
            source_rgb,
            surface,
            pose_target_from_source.rotation_source_from_target,
            pose_target_from_source.translation_source_from_target_m,
            device,
            sigma_px=args.gaussian_sigma_px,
            radius_px=args.gaussian_radius_px,
            occlusion_tau_m=args.occlusion_tau_m,
            point_chunk_size=args.point_chunk_size,
            projected_covariance_mode="legacy",
        )
        layers[source_id] = layer
        sources.append(
            {
                "id": source_id,
                "distance_to_target_m": distance,
                "valid_surface_pixels": int(source_valid.sum()),
                "camera_center_target_m": source_center_target.tolist(),
            }
        )

    ordered_ids = [
        entry["id"]
        for entry in sorted(sources, key=lambda entry: entry["distance_to_target_m"])
    ]
    variants: dict[str, list[str]] = {
        "baseline": list(args.baseline_source_ids),
    }
    for count in range(1, len(ordered_ids) + 1):
        variants[f"nearest-{count}"] = ordered_ids[:count]
    variants["all"] = ordered_ids

    args.output.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "schema": "panorai-registered-multiview-gaussian-oracle/v1",
        "research_only": True,
        "oracle_geometry": True,
        "target_id": args.target_id,
        "target_excluded_from_sources": True,
        "shape_hw": list(shape),
        "device": str(device),
        "sources_by_distance": sorted(
            sources, key=lambda entry: entry["distance_to_target_m"]
        ),
        "variants": {},
        "artifacts": {},
    }
    panel_images: list[tuple[str, np.ndarray]] = [
        (f"{args.target_id} real", target_rgb)
    ]
    for variant_name, source_ids in variants.items():
        rendered, coverage = _composite_ordered_fallback(
            tuple(layers[source_id] for source_id in source_ids)
        )
        metrics = _photo_metrics(rendered, target_rgb, coverage, target_support)
        render_path = args.output / f"{variant_name}-render.png"
        _write_rgb(render_path, rendered)
        results["variants"][variant_name] = {
            "source_ids": source_ids,
            "metrics": metrics,
        }
        results["artifacts"][variant_name] = {
            "path": str(render_path),
            "sha256": _sha256(render_path),
        }
        if variant_name in ("baseline", "nearest-1", "nearest-2", "nearest-3", "all"):
            panel_images.append((variant_name, rendered))

    baseline = results["variants"]["baseline"]["metrics"]
    all_metrics = results["variants"]["all"]["metrics"]
    results["multiview_gate"] = {
        "coverage_non_decreasing": (
            all_metrics["covered_fraction"] >= baseline["covered_fraction"]
        ),
        "rgb_l1_improved": all_metrics["rgb_l1"] < baseline["rgb_l1"],
    }
    results["multiview_gate"]["accepted"] = bool(
        all(results["multiview_gate"].values())
    )
    panel_path = args.output / "multiview-comparison.png"
    _write_labeled_panel(panel_path, panel_images)
    results["artifacts"]["panel"] = {
        "path": str(panel_path),
        "sha256": _sha256(panel_path),
    }
    results_path = args.output / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def _composite_ordered_fallback(
    layers: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...],
    *,
    presence_saturation: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Cascade nearest-to-farthest observed layers only into unsupported pixels."""

    if not layers:
        raise ValueError("at least one layer is required")
    if presence_saturation <= 0.0:
        raise ValueError("presence_saturation must be positive")
    shape = layers[0][1].shape
    numerator = np.zeros((*shape, 3), dtype=np.float32)
    total = np.zeros(shape, dtype=np.float32)
    for color, coverage, _ in layers:
        if color.shape != (*shape, 3) or coverage.shape != shape:
            raise ValueError("all layers must share image shape")
        presence = np.clip(coverage / presence_saturation, 0.0, 1.0)
        contribution = presence * (1.0 - total)
        numerator += contribution[..., None] * color
        total = np.clip(total + contribution, 0.0, 1.0)
    rendered = numerator / np.maximum(total[..., None], 1e-8)
    rendered[total <= 0.0] = 0.0
    return rendered.astype(np.float32), total.astype(np.float32)


def _write_labeled_panel(path: Path, entries: list[tuple[str, np.ndarray]]) -> None:
    labeled: list[np.ndarray] = []
    for label, image in entries:
        if image.dtype == np.uint8:
            canvas = image.copy()
        else:
            canvas = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
        cv2.rectangle(canvas, (0, 0), (240, 24), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            label,
            (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
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
