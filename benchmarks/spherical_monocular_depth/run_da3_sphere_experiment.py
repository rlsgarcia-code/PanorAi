#!/usr/bin/env python3
"""Evaluate deterministic spherical feature sharing for DA3 on P74 W121."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.spherical_monocular_depth.da3_sphere import (  # noqa: E402
    INTERFACE as CONSENSUS_INTERFACE,
    infer_shared_feature_erp,
)
from benchmarks.spherical_monocular_depth.da3_spherical_attention import (  # noqa: E402
    INTERFACE as ATTENTION_INTERFACE,
    infer_sparse_attention_erp,
)
from benchmarks.spherical_monocular_depth.da3_tangent import (  # noqa: E402
    load_da3metric_large,
)
from benchmarks.spherical_monocular_depth.protocol import (  # noqa: E402
    P74_ADAPTER,
    band_metrics,
    depth_metrics,
    json_ready,
    mmap_npy_member,
    native_angular_erp_shape,
    normal_structure_metrics,
    scale_invariant_structure_metrics,
    seam_score,
    sha256,
    write_binary_ply,
)
from benchmarks.spherical_monocular_depth.vit_tangent import (  # noqa: E402
    make_native_tangent_plan,
)
from benchmarks.spherical_multiview_depth.make_point_cloud_viewer import (  # noqa: E402
    build_static_preview,
    build_viewer,
)
from benchmarks.spherical_multiview_depth.p74 import (  # noqa: E402
    load_native_angular_rgb,
)


CONSENSUS_SCHEMA = "panorai-p74-da3-spherical-overlap-features/v1"
ATTENTION_SCHEMA = "panorai-p74-da3-sparse-spherical-attention/v1"
TARGET_ID = "P-74+MD-04_concluido_408+W_121"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Share the four DA3 ViT feature levels across overlapping native-density "
            "tangent views before the unchanged DPT decoder."
        )
    )
    parser.add_argument("--p74-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cnn-prior", type=Path, required=True)
    parser.add_argument("--vit-prior", type=Path, required=True)
    parser.add_argument("--vit-validity", type=Path, required=True)
    parser.add_argument("--da3-prior", type=Path, required=True)
    parser.add_argument("--da3-validity", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--evaluation-validity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--adapter",
        choices=("consensus", "sparse-attention"),
        default="consensus",
        help="Spherical communication rule; consensus preserves the VAL-036 path.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--view-height", type=int, default=812)
    parser.add_argument("--view-width", type=int, default=1400)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--feature-sharing-alpha", type=float, default=1.0)
    parser.add_argument("--maximum-sources-per-target", type=int, default=6)
    parser.add_argument(
        "--position-transport",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace sampled source 2D position with the target position for sparse attention.",
    )
    parser.add_argument("--max-depth-m", type=float, default=15.0)
    parser.add_argument("--p74-row-chunk", type=int, default=32)
    parser.add_argument("--normal-step-deg", type=float, default=0.35)
    parser.add_argument("--ply-max-points", type=int, default=1_500_000)
    parser.add_argument("--keep-feature-stores", action="store_true")
    return parser.parse_args()


def _evaluate(
    prediction: np.ndarray,
    truth: np.ndarray,
    validity: np.ndarray,
    *,
    normal_stride: int,
) -> dict[str, Any]:
    return {
        "metric_depth": depth_metrics(prediction, truth, validity),
        "scale_invariant_structure": scale_invariant_structure_metrics(
            prediction, truth, validity
        ),
        "surface_normals": normal_structure_metrics(
            prediction, truth, validity, stride_px=normal_stride
        ),
        "bands": band_metrics(prediction, truth, validity),
        "seam": seam_score(prediction, validity),
    }


def _write_depth_png(path: Path, radial_m: np.ndarray, max_depth_m: float) -> None:
    encoded = np.zeros(radial_m.shape, dtype=np.uint16)
    finite = np.isfinite(radial_m) & (radial_m > 0.0)
    encoded[finite] = np.rint(
        np.clip(radial_m[finite] / max_depth_m, 0.0, 1.0) * np.iinfo(np.uint16).max
    ).astype(np.uint16)
    if not cv2.imwrite(str(path), encoded):
        raise RuntimeError(f"failed to write {path}")


def _write_panel(
    path: Path,
    rgb: np.ndarray,
    truth: np.ndarray,
    da3: np.ndarray,
    shared: np.ndarray,
    *,
    max_depth_m: float,
    candidate_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stride = max(1, rgb.shape[0] // 768)
    figure, axes = plt.subplots(4, 1, figsize=(16, 12))
    axes[0].imshow(rgb[::stride, ::stride], aspect="auto")
    axes[0].set_title("P74 RGB — native angular lattice (display decimation only)")
    for axis, name, value in zip(
        axes[1:],
        ("P74 GT", "DA3 Metric Large", candidate_label),
        (truth, da3, shared),
        strict=True,
    ):
        image = axis.imshow(
            value[::stride, ::stride],
            cmap="turbo",
            vmin=0.0,
            vmax=max_depth_m,
            aspect="auto",
        )
        axis.set_title(name)
    for axis in axes:
        axis.axis("off")
    figure.subplots_adjust(left=0.04, right=0.91, top=0.96, bottom=0.03, hspace=0.28)
    colorbar_axis = figure.add_axes((0.93, 0.05, 0.015, 0.67))
    figure.colorbar(image, cax=colorbar_axis, label="radial range (m)")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _native_array(path: Path, shape: tuple[int, int], name: str) -> np.ndarray:
    value = np.load(path, mmap_mode="r")
    if value.shape != shape:
        raise ValueError(f"{name} shape {value.shape} != native ERP {shape}")
    return value


def main() -> int:
    args = parse_args()
    if not np.isfinite(args.max_depth_m) or args.max_depth_m <= 0.0:
        raise ValueError("max-depth-m must be finite and positive")
    args.output.mkdir(parents=True, exist_ok=True)
    npz = args.p74_root / "npzs" / f"{TARGET_ID}.npz"
    rgb_path = args.p74_root / "images" / f"{TARGET_ID}_rgb.png"
    source_shape = tuple(mmap_npy_member(npz, "xyz_image.npy").shape[:2])
    output_shape = native_angular_erp_shape(source_shape)
    rgb, source_support = load_native_angular_rgb(
        rgb_path, output_shape, row_chunk=args.p74_row_chunk
    )
    plan = make_native_tangent_plan(
        output_shape,
        view_shape_hw=(args.view_height, args.view_width),
        overlap_fraction=args.overlap,
        minimum_latitude_deg=-60.0,
        maximum_latitude_deg=90.0,
    )

    import torch

    torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))
    torch.manual_seed(0)
    model, model_report = load_da3metric_large(
        args.source, args.checkpoint, device=args.device
    )
    model_report.update(
        {
            "source_root": str(args.source),
            "source_archive": str(args.source_archive),
            "source_archive_sha256": sha256(args.source_archive),
            "source_license_sha256": sha256(args.source / "LICENSE"),
        }
    )
    start = perf_counter()
    if args.adapter == "consensus":
        infer = infer_shared_feature_erp
        schema = CONSENSUS_SCHEMA
        interface = CONSENSUS_INTERFACE
        candidate_slug = "da3-shared"
        candidate_key = "da3-shared-features-common-support"
        candidate_label = "DA3 shared spherical ViT features"
        route = "native-density tangent ViT with spherical overlap feature sharing"
    else:
        infer = infer_sparse_attention_erp
        schema = ATTENTION_SCHEMA
        interface = ATTENTION_INTERFACE
        candidate_slug = "da3-sparse-attention"
        candidate_key = "da3-sparse-attention-common-support"
        candidate_label = "DA3 sparse spherical attention"
        route = "native-density tangent ViT with sparse panorama-ray attention"
    inference_kwargs = {
        "scratch_dir": args.output / "feature-store",
        "device": args.device,
        "alpha": args.feature_sharing_alpha,
        "maximum_sources_per_target": args.maximum_sources_per_target,
        "keep_feature_stores": args.keep_feature_stores,
    }
    if args.adapter == "sparse-attention":
        inference_kwargs["position_transport"] = args.position_transport
    prediction, prediction_validity, inference_report = infer(
        model, rgb, source_support, plan, **inference_kwargs
    )
    runtime_seconds = perf_counter() - start

    raw_path = args.output / f"{TARGET_ID}-{candidate_slug}-raw-radial-m.npy"
    np.save(raw_path, prediction)
    prediction = np.minimum(prediction, args.max_depth_m)
    prediction_path = args.output / f"{TARGET_ID}-{candidate_slug}-radial-m.npy"
    validity_path = args.output / f"{TARGET_ID}-{candidate_slug}-validity.npy"
    np.save(prediction_path, prediction)
    np.save(validity_path, prediction_validity)
    png_path = args.output / f"{TARGET_ID}-{candidate_slug}-depth-over-15m.png"
    _write_depth_png(png_path, prediction, args.max_depth_m)
    frozen = {
        "schema": f"{schema}/frozen-before-ground-truth",
        "prediction": {"path": str(prediction_path), "sha256": sha256(prediction_path)},
        "raw_prediction": {"path": str(raw_path), "sha256": sha256(raw_path)},
        "prediction_validity": {
            "path": str(validity_path),
            "sha256": sha256(validity_path),
        },
        "visualization": {"path": str(png_path), "sha256": sha256(png_path)},
        "model": model_report,
        "plan": plan.to_dict(),
        "inference": inference_report,
        "runtime_seconds": runtime_seconds,
        "ground_truth_opened": False,
    }
    frozen_path = args.output / "FROZEN-BEFORE-GT.json"
    frozen_path.write_text(
        json.dumps(json_ready(frozen), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    truth = _native_array(args.ground_truth, output_shape, "truth")
    evaluation_validity = _native_array(
        args.evaluation_validity, output_shape, "evaluation validity"
    )
    cnn = _native_array(args.cnn_prior, output_shape, "CNN prior")
    vit = _native_array(args.vit_prior, output_shape, "ViT prior")
    vit_validity = _native_array(args.vit_validity, output_shape, "ViT validity")
    da3 = _native_array(args.da3_prior, output_shape, "DA3 prior")
    da3_validity = _native_array(args.da3_validity, output_shape, "DA3 validity")
    common = (
        np.asarray(evaluation_validity, dtype=bool)
        & prediction_validity
        & np.asarray(vit_validity, dtype=bool)
        & np.asarray(da3_validity, dtype=bool)
        & np.isfinite(cnn)
        & (cnn > 0.0)
        & np.isfinite(vit)
        & (vit > 0.0)
        & np.isfinite(da3)
        & (da3 > 0.0)
    )
    controls = {
        "convnext-large-common-support": np.minimum(np.asarray(cnn), args.max_depth_m),
        "metric3dv2-vit-large-common-support": np.minimum(
            np.asarray(vit), args.max_depth_m
        ),
        "da3metric-large-common-support": np.minimum(np.asarray(da3), args.max_depth_m),
        candidate_key: prediction,
    }
    normal_stride = max(1, round(output_shape[0] * args.normal_step_deg / 180.0))
    evaluations = {
        name: _evaluate(value, truth, common, normal_stride=normal_stride)
        for name, value in controls.items()
    }

    clouds: list[tuple[str, Path]] = []
    for label, slug, value in (
        ("Ground truth", "gt", truth),
        (
            "DA3 Metric Large",
            "da3metric-large",
            controls["da3metric-large-common-support"],
        ),
        (
            candidate_label,
            candidate_slug,
            controls[candidate_key],
        ),
    ):
        report = write_binary_ply(
            args.output / f"{TARGET_ID}-{slug}-view-15m.ply",
            value,
            rgb,
            common,
            max_points=args.ply_max_points,
        )
        clouds.append((label, Path(report["path"])))
    panel_name = (
        "p74-w121-da3-vs-shared-panel.png"
        if args.adapter == "consensus"
        else f"p74-w121-da3-vs-{candidate_slug}-panel.png"
    )
    panel_path = args.output / panel_name
    _write_panel(
        panel_path,
        rgb,
        truth,
        controls["da3metric-large-common-support"],
        prediction,
        max_depth_m=args.max_depth_m,
        candidate_label=candidate_label,
    )
    viewer_path = args.output / "viewer.html"
    viewer = build_viewer(clouds, viewer_path, max_points=80_000)
    preview_path = args.output / "point-cloud-preview.png"
    build_static_preview(clouds, preview_path, max_points=25_000)

    result = {
        "schema": schema,
        "status": "development-only-not-publication-evidence",
        "interface": interface,
        "dataset": {
            "name": "P74",
            "panorama_id": TARGET_ID,
            "adapter": P74_ADAPTER,
            "source_shape_hw": list(source_shape),
            "evaluation_shape_hw": list(output_shape),
            "depth_semantics": "radial range in metres",
            "maximum_evaluated_depth_m": args.max_depth_m,
        },
        "model": model_report,
        "protocol": {
            "route": route,
            "adapter": args.adapter,
            "source_erp_resize": False,
            "model_input_resize": False,
            "prediction_resize": False,
            "training": False,
            "feature_sharing_alpha": args.feature_sharing_alpha,
            "maximum_sources_per_target": args.maximum_sources_per_target,
            "position_transport": (
                args.position_transport if args.adapter == "sparse-attention" else None
            ),
            "shared_levels": [4, 11, 17, 23],
            "dpt": "unchanged official planar DPT per tangent view",
            "dpt_spherical": False,
            "random_seed": 0,
            "plan": plan.to_dict(),
            "fusion": inference_report,
            "evaluation_support": "identical CNN/Metric3Dv2/DA3/shared intersection",
            "ground_truth_opened_after_prediction_freeze": True,
        },
        "runtime_seconds": runtime_seconds,
        "evaluations": evaluations,
        "artifacts": {
            "prediction": str(prediction_path),
            "raw_prediction": str(raw_path),
            "prediction_validity": str(validity_path),
            "frozen_before_gt": str(frozen_path),
            "panel": str(panel_path),
            "preview": str(preview_path),
            "viewer": viewer,
            "point_clouds": [str(path) for _, path in clouds],
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "peak_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "limitations": (
            [
                "The four ViT levels communicate over panorama-frame rays, but the DPT remains planar inside each tangent view.",
                "This isolates stage A; it is not yet the sparse spherical DPT required by stage C.",
                "One previously studied development panorama is evaluated.",
            ]
            if args.adapter == "consensus"
            else [
                "One frozen ViT block receives sparse panorama-ray attention, but the other blocks retain local learned 2D positions.",
                "The DPT remains planar inside each tangent view.",
                "This exploratory follow-up was designed after observing VAL-036 and is not an outcome-blind publication result.",
                "One previously studied development panorama is evaluated.",
            ]
        ),
    }
    results_path = args.output / "results.json"
    results_path.write_text(
        json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            json_ready({"results": str(results_path), "evaluations": evaluations}),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
