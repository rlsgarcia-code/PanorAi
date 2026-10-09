#!/usr/bin/env python3
"""Run frozen Metric3D-v1 CNN via spherical and six-face P74 routes."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from protocol import (
    FROZEN_SAMPLE,
    MODEL_DEPTH_RANGE_M,
    P74_ADAPTER,
    SCHEMA,
    axial_to_radial,
    band_metrics,
    depth_metrics,
    json_ready,
    load_p74_frame,
    mmap_npy_member,
    native_angular_cube_face_size,
    native_angular_erp_shape,
    normal_structure_metrics,
    patch_metric3d_device_assumptions,
    port_metric3d_spatial_layers,
    scale_invariant_structure_metrics,
    seam_score,
    sha256,
    write_binary_ply,
)


MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
METRIC3D_CONVNEXT_INPUT_HW = (544, 1216)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p74-root", type=Path, required=True)
    parser.add_argument("--metric3d-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument(
        "--native-angular",
        action="store_true",
        help="infer each sample on a 2:1 ERP matched to native P74 angular density",
    )
    parser.add_argument(
        "--face-size",
        type=int,
        default=0,
        help="cube face size; zero matches the current ERP angular density",
    )
    parser.add_argument("--p74-row-chunk", type=int, default=64)
    parser.add_argument("--spherical-chunk-elements", type=int, default=100_000_000)
    parser.add_argument(
        "--normal-stride",
        type=int,
        default=0,
        help="normal finite-difference stride; zero derives it from --normal-step-deg",
    )
    parser.add_argument("--normal-step-deg", type=float, default=0.35)
    parser.add_argument("--ply-max-points", type=int, default=2_000_000)
    parser.add_argument("--only", choices=FROZEN_SAMPLE)
    return parser.parse_args()


def load_model(
    source: Path,
    checkpoint: Path,
    *,
    spherical: bool,
    max_sampled_elements: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    import torch

    model = torch.hub.load(
        str(source), "metric3d_convnext_tiny", source="local", pretrain=False
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
    patch_metric3d_device_assumptions(model)
    port_report: dict[str, Any] = {
        "ported_layer_count": 0,
        "remaining_planar_learned_spatial_layers": "all (cubemap control)",
    }
    if spherical:
        port_report = port_metric3d_spatial_layers(
            model, max_sampled_elements=max_sampled_elements
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "port": port_report,
    }


def tensor_input(rgb: np.ndarray) -> Any:
    import torch

    values = (rgb.astype(np.float32) - MEAN) / STD
    return torch.from_numpy(np.moveaxis(values, -1, 0))[None]


def infer_spherical(model: Any, rgb: np.ndarray) -> np.ndarray:
    import torch

    with torch.inference_mode():
        prediction = model.inference({"input": tensor_input(rgb)})[0]
    canonical = prediction[0, 0].detach().cpu().numpy()
    effective_focal = rgb.shape[1] / (2.0 * math.pi)
    metric = canonical * (effective_focal / 1000.0)
    return np.clip(metric, *MODEL_DEPTH_RANGE_M).astype(np.float32)


def infer_cubemap(model: Any, rgb: np.ndarray, face_size: int) -> np.ndarray:
    import torch
    from panorai.geometry import CUBE_FACE_ORDER, cubemap_to_equirectangular
    from panorai.geometry import equirectangular_to_cubemap

    faces = equirectangular_to_cubemap(
        rgb.astype(np.float32), face_size, interpolation="bilinear"
    )
    radial_faces: dict[str, np.ndarray] = {}
    if face_size < 1 or face_size % 32:
        raise ValueError(
            "resize-free cubemap face_size must be a positive multiple of 32"
        )
    reference_height, reference_width = METRIC3D_CONVNEXT_INPUT_HW
    input_height = face_size
    input_width = int(
        math.ceil(face_size * reference_width / reference_height / 32.0) * 32
    )
    for face in CUBE_FACE_ORDER:
        face_rgb = np.asarray(faces[face].data)
        pad_total = input_width - input_height
        pad_left = pad_total // 2
        padded = cv2.copyMakeBorder(
            face_rgb,
            0,
            0,
            pad_left,
            pad_total - pad_left,
            cv2.BORDER_CONSTANT,
            value=tuple(float(item) for item in MEAN),
        )
        with torch.inference_mode():
            pred = model.inference({"input": tensor_input(padded)})[0][0, 0]
        axial = pred[:, pad_left : pad_left + input_height].detach().cpu().numpy()
        face_focal = input_height / 2.0
        radial_faces[face] = axial_to_radial(axial * (face_focal / 1000.0))
    result = cubemap_to_equirectangular(
        radial_faces,
        rgb.shape[:2],
        interpolation="bilinear",
        invalid_policy="propagate",
    )
    return np.clip(
        np.asarray(result.data, dtype=np.float32), *MODEL_DEPTH_RANGE_M
    ).astype(np.float32)


def render_panel(
    path: Path,
    rgb: np.ndarray,
    target: np.ndarray,
    spherical: np.ndarray,
    cube: np.ndarray,
    validity: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    if rgb.shape[1] > 1600:
        panel_width = 1600
        panel_height = round(rgb.shape[0] * panel_width / rgb.shape[1])
        rgb = cv2.resize(rgb, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
        target = cv2.resize(
            target, (panel_width, panel_height), interpolation=cv2.INTER_NEAREST
        )
        spherical = cv2.resize(
            spherical, (panel_width, panel_height), interpolation=cv2.INTER_AREA
        )
        cube = cv2.resize(
            cube, (panel_width, panel_height), interpolation=cv2.INTER_AREA
        )
        validity = cv2.resize(
            validity.astype(np.uint8),
            (panel_width, panel_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    valid_values = target[validity]
    vmax = float(np.quantile(valid_values, 0.98))
    maps = [target, spherical, cube, np.abs(spherical - target), np.abs(cube - target)]
    titles = [
        "P74 radial GT",
        "ported spherical CNN",
        "six-face cubemap",
        "spherical |error|",
        "cubemap |error|",
    ]
    figure, axes = plt.subplots(3, 2, figsize=(14, 9), constrained_layout=True)
    axes = axes.ravel()
    axes[0].imshow(rgb)
    axes[0].set_title("P74 RGB (canonical ERP)")
    for axis, values, title in zip(axes[1:], maps, titles, strict=True):
        shown = values.copy()
        shown[~validity] = np.nan
        limit = vmax if "error" not in title else max(1.0, vmax / 2.0)
        image = axis.imshow(shown, cmap="magma", vmin=0.0, vmax=limit)
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.025)
    for axis in axes:
        axis.set_axis_off()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if not args.native_angular and args.width != 2 * args.height:
        raise ValueError("spherical route requires a 2:1 canonical ERP")
    args.output.mkdir(parents=True, exist_ok=True)
    ids = (args.only,) if args.only else FROZEN_SAMPLE
    import torch

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    normal_model, normal_load = load_model(
        args.metric3d_source, args.checkpoint, spherical=False
    )
    spherical_model, spherical_load = load_model(
        args.metric3d_source,
        args.checkpoint,
        spherical=True,
        max_sampled_elements=args.spherical_chunk_elements,
    )
    rows: list[dict[str, Any]] = []
    for panorama_id in ids:
        npz = args.p74_root / "npzs" / f"{panorama_id}.npz"
        rgb_path = args.p74_root / "images" / f"{panorama_id}_rgb.png"
        source_shape = tuple(mmap_npy_member(npz, "xyz_image.npy").shape[:2])
        output_shape = (
            native_angular_erp_shape(source_shape)
            if args.native_angular
            else (args.height, args.width)
        )
        face_size = (
            args.face_size
            if args.face_size > 0
            else native_angular_cube_face_size(output_shape)
        )
        normal_stride = (
            args.normal_stride
            if args.normal_stride > 0
            else max(1, round(output_shape[0] * args.normal_step_deg / 180.0))
        )
        frame = load_p74_frame(
            npz, rgb_path, output_shape, row_chunk=args.p74_row_chunk
        )
        evaluation_mask = (
            frame.validity
            & (frame.radial_range >= MODEL_DEPTH_RANGE_M[0])
            & (frame.radial_range <= MODEL_DEPTH_RANGE_M[1])
        )
        start = perf_counter()
        spherical = infer_spherical(spherical_model, frame.rgb)
        spherical_seconds = perf_counter() - start
        start = perf_counter()
        cube = infer_cubemap(normal_model, frame.rgb, face_size)
        cube_seconds = perf_counter() - start
        predictions = {"spherical": spherical, "cubemap": cube}
        methods: dict[str, Any] = {}
        for name, prediction in predictions.items():
            methods[name] = {
                "metrics": depth_metrics(
                    prediction, frame.radial_range, evaluation_mask
                ),
                "scale_invariant_structure": scale_invariant_structure_metrics(
                    prediction, frame.radial_range, evaluation_mask
                ),
                "surface_normals": normal_structure_metrics(
                    prediction,
                    frame.radial_range,
                    evaluation_mask,
                    stride_px=normal_stride,
                ),
                "bands": band_metrics(prediction, frame.radial_range, evaluation_mask),
                "seam": seam_score(prediction, evaluation_mask),
                "runtime_seconds": spherical_seconds
                if name == "spherical"
                else cube_seconds,
            }
            np.save(args.output / f"{panorama_id}-{name}-radial.npy", prediction)
            methods[name]["ply"] = write_binary_ply(
                args.output / f"{panorama_id}-{name}-view.ply",
                prediction,
                frame.rgb,
                evaluation_mask,
                max_points=args.ply_max_points,
            )
        np.save(args.output / f"{panorama_id}-gt-radial.npy", frame.radial_range)
        np.save(
            args.output / f"{panorama_id}-evaluation-validity.npy",
            evaluation_mask,
        )
        gt_ply = write_binary_ply(
            args.output / f"{panorama_id}-gt-view.ply",
            frame.radial_range,
            frame.rgb,
            evaluation_mask,
            max_points=args.ply_max_points,
        )
        render_panel(
            args.output / f"{panorama_id}-panel.png",
            frame.rgb,
            frame.radial_range,
            spherical,
            cube,
            evaluation_mask,
        )
        row = {
            "panorama_id": frame.panorama_id,
            "family": frame.family,
            "source_shape_hw": frame.source_shape_hw,
            "evaluation_shape_hw": list(output_shape),
            "valid_pixels": int(frame.validity.sum()),
            "evaluation_pixels": int(evaluation_mask.sum()),
            "cubemap_face_size": face_size,
            "gt_ply": gt_ply,
            "methods": methods,
        }
        rows.append(row)
        print(json.dumps(json_ready(row), sort_keys=True), flush=True)

    aggregate: dict[str, Any] = {}
    for method in ("spherical", "cubemap"):
        keys = (
            "abs_rel",
            "sq_rel",
            "rmse_m",
            "rmse_log",
            "silog",
            "delta_1",
            "delta_2",
            "delta_3",
            "prediction_at_0_3m_fraction",
        )
        aggregate[method] = {
            key: float(
                np.mean([row["methods"][method]["metrics"][key] for row in rows])
            )
            for key in keys
        }
        aggregate[method]["runtime_seconds_mean"] = float(
            np.mean([row["methods"][method]["runtime_seconds"] for row in rows])
        )
        for key in (
            "scale_aligned_relative_3d_rmse",
            "scale_invariant_log_rmse",
            "log_depth_correlation",
        ):
            aggregate[method][key] = float(
                np.mean(
                    [
                        row["methods"][method]["scale_invariant_structure"][key]
                        for row in rows
                    ]
                )
            )
        for key in ("normal_mean_deg", "normal_median_deg", "normal_p90_deg"):
            aggregate[method][key] = float(
                np.mean(
                    [row["methods"][method]["surface_normals"][key] for row in rows]
                )
            )
    source_commit = subprocess.run(
        ["git", "-C", str(args.metric3d_source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_dirty = bool(
        subprocess.run(
            ["git", "-C", str(args.metric3d_source), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    result = {
        "schema": SCHEMA,
        "status": "preliminary-portability-evidence-not-publication-evidence",
        "dataset": {
            "name": "P74",
            "adapter": P74_ADAPTER,
            "depth_semantics": "radial range from organized scan-local XYZ",
            "validity": "explicit finite XYZ and radial > 1e-8 intersect geometric support",
            "frozen_sample": list(ids),
        },
        "model": {
            "name": "Metric3D-v1 ConvNeXt-Tiny Hourglass",
            "source_commit": source_commit,
            "source_dirty": source_dirty,
            "checkpoint_sha256": sha256(args.checkpoint),
            "checkpoint_external": True,
            "learned_parameters_frozen": True,
            "normal_load": normal_load,
            "spherical_load": spherical_load,
        },
        "protocol": {
            "evaluation_shape_hw": "native-angular-per-source"
            if args.native_angular
            else [args.height, args.width],
            "native_angular_shape_rule": "infer full-sphere height from P74 0--150 degree vertical sampling, snap to 32, width=2*height",
            "cubemap_face_size": "frame_width/pi rounded upward to 32"
            if args.face_size == 0
            else args.face_size,
            "cubemap_model_input_hw": "face height; width padded to preserve the Metric3D 544:1216 canvas ratio, both multiples of 32",
            "spherical_effective_focal_px": "frame_width/(2*pi)",
            "cubemap_focal_px": "face_size/2",
            "metric_weighting": "ERP pixel-cell solid angle (cos latitude)",
            "structural_metric": "optimal single-scale aligned 3D RMSE normalized by target RMS range",
            "normal_metric": "oriented local surface-normal angle; target-continuous stencil; multiplicative-scale invariant",
            "evaluation_depth_range_m": list(MODEL_DEPTH_RANGE_M),
            "prediction_clamp_m": list(MODEL_DEPTH_RANGE_M),
            "spherical_chunk_elements": args.spherical_chunk_elements,
            "p74_row_chunk": args.p74_row_chunk,
            "normal_stride_px": "per-frame from normal_step_deg"
            if args.normal_stride == 0
            else args.normal_stride,
            "normal_step_deg": args.normal_step_deg,
            "ply_max_points": args.ply_max_points,
            "ply_policy": "deterministic ERP-grid decimation of dense radial predictions for interactive inspection",
            "inference_resize": False,
            "resampling_boundary": "only coordinate-defined P74-to-ERP and ERP/cubemap projection sampling; panel scaling and PLY decimation are evaluation-external",
            "tangent_view_oracle": False,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "peak_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "frames": rows,
        "aggregate_macro_mean": aggregate,
    }
    (args.output / "results.json").write_text(
        json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"aggregate_macro_mean": aggregate}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
