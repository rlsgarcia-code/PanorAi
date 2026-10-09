#!/usr/bin/env python3
"""Run Depth Anything 3 Metric Large on the frozen native P74 tangent atlas."""

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

from benchmarks.spherical_monocular_depth.da3_tangent import (  # noqa: E402
    CANONICAL_FOCAL_PX,
    INTERFACE,
    infer_native_tangent_erp,
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


SCHEMA = "panorai-p74-depth-anything-3-metric-tangent/v1"
TARGET_ID = "P-74+MD-04_concluido_408+W_121"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DA3 Metric Large on the same native-density spherical "
            "tangent atlas used by the Metric3Dv2 control."
        )
    )
    parser.add_argument("--p74-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cnn-prior", type=Path, required=True)
    parser.add_argument("--vit-prior", type=Path, required=True)
    parser.add_argument("--vit-validity", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--evaluation-validity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--view-height", type=int, default=812)
    parser.add_argument("--view-width", type=int, default=1400)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--max-depth-m", type=float, default=15.0)
    parser.add_argument("--p74-row-chunk", type=int, default=32)
    parser.add_argument("--normal-step-deg", type=float, default=0.35)
    parser.add_argument("--ply-max-points", type=int, default=1_500_000)
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
    cnn: np.ndarray,
    vit: np.ndarray,
    da3: np.ndarray,
    *,
    max_depth_m: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stride = max(1, rgb.shape[0] // 768)
    figure, axes = plt.subplots(5, 1, figsize=(16, 15))
    axes[0].imshow(rgb[::stride, ::stride], aspect="auto")
    axes[0].set_title("P74 RGB — native angular lattice (display decimation only)")
    names = (
        "P74 GT",
        "Metric3D-v1 ConvNeXt-Large",
        "Metric3Dv2 ViT-Large",
        "Depth Anything 3 Metric Large",
    )
    for axis, name, value in zip(axes[1:], names, (truth, cnn, vit, da3), strict=True):
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
    figure.subplots_adjust(left=0.04, right=0.91, top=0.97, bottom=0.02, hspace=0.28)
    colorbar_axis = figure.add_axes((0.93, 0.04, 0.015, 0.73))
    figure.colorbar(image, cax=colorbar_axis, label="radial range (m)")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _require_native_array(path: Path, shape: tuple[int, int], name: str) -> np.ndarray:
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
    model, load_report = load_da3metric_large(
        args.source, args.checkpoint, device=args.device
    )
    load_report["source_root"] = str(args.source)
    load_report["source_archive"] = str(args.source_archive)
    load_report["source_archive_sha256"] = sha256(args.source_archive)
    load_report["source_license_sha256"] = sha256(args.source / "LICENSE")
    start = perf_counter()
    prediction, prediction_validity, inference_report = infer_native_tangent_erp(
        model,
        rgb,
        source_support,
        plan,
        device=args.device,
    )
    runtime_seconds = perf_counter() - start

    raw_prediction_path = args.output / f"{TARGET_ID}-da3metric-large-raw-radial-m.npy"
    np.save(raw_prediction_path, prediction)
    prediction = np.minimum(prediction, args.max_depth_m)
    prediction_path = args.output / f"{TARGET_ID}-da3metric-large-radial-m.npy"
    validity_path = args.output / f"{TARGET_ID}-da3metric-large-validity.npy"
    np.save(prediction_path, prediction)
    np.save(validity_path, prediction_validity)
    png_path = args.output / f"{TARGET_ID}-da3metric-large-depth-over-15m.png"
    _write_depth_png(png_path, prediction, args.max_depth_m)
    frozen = {
        "schema": f"{SCHEMA}/frozen-before-ground-truth",
        "prediction": {"path": str(prediction_path), "sha256": sha256(prediction_path)},
        "raw_prediction": {
            "path": str(raw_prediction_path),
            "sha256": sha256(raw_prediction_path),
        },
        "prediction_validity": {
            "path": str(validity_path),
            "sha256": sha256(validity_path),
        },
        "visualization": {"path": str(png_path), "sha256": sha256(png_path)},
        "model": load_report,
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

    # No evaluation/control map is opened before the DA3 output is frozen.
    truth = _require_native_array(args.ground_truth, output_shape, "truth")
    evaluation_validity = _require_native_array(
        args.evaluation_validity, output_shape, "evaluation validity"
    )
    cnn = _require_native_array(args.cnn_prior, output_shape, "CNN prior")
    vit = _require_native_array(args.vit_prior, output_shape, "ViT prior")
    vit_validity = _require_native_array(
        args.vit_validity, output_shape, "ViT validity"
    )
    common_validity = (
        np.asarray(evaluation_validity, dtype=bool)
        & prediction_validity
        & np.asarray(vit_validity, dtype=bool)
        & np.isfinite(cnn)
        & (cnn > 0.0)
        & np.isfinite(vit)
        & (vit > 0.0)
    )
    cnn_capped = np.minimum(np.asarray(cnn), args.max_depth_m)
    vit_capped = np.minimum(np.asarray(vit), args.max_depth_m)
    normal_stride = max(1, round(output_shape[0] * args.normal_step_deg / 180.0))
    evaluations = {
        "convnext-large-common-support": _evaluate(
            cnn_capped, truth, common_validity, normal_stride=normal_stride
        ),
        "metric3dv2-vit-large-common-support": _evaluate(
            vit_capped, truth, common_validity, normal_stride=normal_stride
        ),
        "da3metric-large-common-support": _evaluate(
            prediction, truth, common_validity, normal_stride=normal_stride
        ),
    }

    cloud_inputs = []
    for label, slug, depth in (
        ("Ground truth", "gt", truth),
        ("ConvNeXt-Large", "cnn-large", cnn_capped),
        ("Metric3Dv2 ViT-Large", "metric3dv2-vit-large", vit_capped),
        ("DA3 Metric Large", "da3metric-large", prediction),
    ):
        report = write_binary_ply(
            args.output / f"{TARGET_ID}-{slug}-view-15m.ply",
            depth,
            rgb,
            common_validity,
            max_points=args.ply_max_points,
        )
        cloud_inputs.append((label, Path(report["path"])))
    panel_path = args.output / "p74-w121-cnn-vit-da3-panel.png"
    _write_panel(
        panel_path,
        rgb,
        truth,
        cnn_capped,
        vit_capped,
        prediction,
        max_depth_m=args.max_depth_m,
    )
    viewer_path = args.output / "viewer.html"
    viewer = build_viewer(cloud_inputs, viewer_path, max_points=80_000)
    preview_path = args.output / "point-cloud-preview.png"
    build_static_preview(cloud_inputs, preview_path, max_points=25_000)

    result = {
        "schema": SCHEMA,
        "status": "development-only-not-publication-evidence",
        "interface": INTERFACE,
        "dataset": {
            "name": "P74",
            "panorama_id": TARGET_ID,
            "adapter": P74_ADAPTER,
            "source_shape_hw": list(source_shape),
            "evaluation_shape_hw": list(output_shape),
            "depth_semantics": "radial range in metres",
            "maximum_evaluated_depth_m": args.max_depth_m,
        },
        "model": load_report,
        "protocol": {
            "route": "native-density spherical tangent windows",
            "source_erp_resize": False,
            "high_level_da3_input_resize": False,
            "image_pyramid": False,
            "prefilter": False,
            "prediction_resize": False,
            "view_projection": "canonical gnomonic bilinear sampling",
            "native_angular_focal_px": plan.focal_px,
            "canonical_model_focal_px": CANONICAL_FOCAL_PX,
            "official_axial_to_metric_scale": plan.focal_px / CANONICAL_FOCAL_PX,
            "network_depth_semantics": "pinhole axial z-depth",
            "evaluation_depth_semantics": "radial range",
            "plan": plan.to_dict(),
            "fusion": inference_report,
            "metric_weighting": "ERP pixel-cell solid angle",
            "evaluation_support": "identical DA3/Metric3Dv2/CNN intersection",
            "ground_truth_opened_after_prediction_freeze": True,
        },
        "runtime_seconds": runtime_seconds,
        "evaluations": evaluations,
        "artifacts": {
            "prediction": str(prediction_path),
            "raw_prediction": str(raw_prediction_path),
            "prediction_validity": str(validity_path),
            "frozen_before_gt": str(frozen_path),
            "panel": str(panel_path),
            "preview": str(preview_path),
            "viewer": viewer,
            "point_clouds": [str(path) for _, path in cloud_inputs],
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "peak_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "limitations": [
            "This is a perspective/tangent DA3 control, not spherical-token attention.",
            "One previously studied development panorama is evaluated.",
            "The checkpoint and upstream source remain external to PanorAi artifacts.",
            "The 56.09x34.34-degree atlas is held fixed for model-to-model comparison; it is not DA3's default resized 504-pixel protocol.",
        ],
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
