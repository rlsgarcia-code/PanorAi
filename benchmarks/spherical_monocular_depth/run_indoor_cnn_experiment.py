#!/usr/bin/env python3
"""Evaluate the indoor-trained CNNDepth ResNet-101 on frozen P74 scenes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np

from indoor_cnn import (
    CHECKPOINT_SHA256,
    MODEL_DEPTH_RANGE_M,
    SOURCE_COMMIT,
    infer_cubemap,
    infer_spherical,
    load_indoor_cnn,
)
from protocol import (
    FROZEN_SAMPLE,
    P74_ADAPTER,
    band_metrics,
    depth_metrics,
    json_ready,
    load_p74_frame,
    mmap_npy_member,
    native_angular_cube_face_size,
    native_angular_erp_shape,
    normal_structure_metrics,
    scale_invariant_structure_metrics,
    seam_score,
    sha256,
    write_binary_ply,
)


SCHEMA = "panorai-p74-indoor-cnn-depth/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p74-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", choices=("cubemap", "spherical"), required=True)
    parser.add_argument("--only", choices=FROZEN_SAMPLE)
    parser.add_argument("--p74-row-chunk", type=int, default=32)
    parser.add_argument("--spherical-chunk-elements", type=int, default=50_000_000)
    parser.add_argument("--normal-step-deg", type=float, default=0.35)
    parser.add_argument("--ply-max-points", type=int, default=1_500_000)
    return parser.parse_args()


def evaluate(
    prediction: np.ndarray,
    truth: np.ndarray,
    validity: np.ndarray,
    *,
    normal_stride: int,
    seconds: float,
) -> dict[str, Any]:
    return {
        "metrics": depth_metrics(prediction, truth, validity),
        "scale_invariant_structure": scale_invariant_structure_metrics(
            prediction, truth, validity
        ),
        "surface_normals": normal_structure_metrics(
            prediction, truth, validity, stride_px=normal_stride
        ),
        "bands": band_metrics(prediction, truth, validity),
        "seam": seam_score(prediction, validity),
        "runtime_seconds": seconds,
    }


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import torch

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    model, load_report = load_indoor_cnn(
        args.source,
        args.config,
        args.checkpoint,
        spherical=args.route == "spherical",
        max_sampled_elements=args.spherical_chunk_elements,
    )
    ids = (args.only,) if args.only else FROZEN_SAMPLE
    frames: list[dict[str, Any]] = []
    for panorama_id in ids:
        npz = args.p74_root / "npzs" / f"{panorama_id}.npz"
        rgb_path = args.p74_root / "images" / f"{panorama_id}_rgb.png"
        source_shape = tuple(mmap_npy_member(npz, "xyz_image.npy").shape[:2])
        output_shape = native_angular_erp_shape(source_shape)
        face_size = native_angular_cube_face_size(output_shape)
        normal_stride = max(
            1, round(output_shape[0] * args.normal_step_deg / 180.0)
        )
        frame = load_p74_frame(
            npz, rgb_path, output_shape, row_chunk=args.p74_row_chunk
        )
        validity = (
            frame.validity
            & (frame.radial_range >= MODEL_DEPTH_RANGE_M[0])
            & (frame.radial_range <= MODEL_DEPTH_RANGE_M[1])
        )
        start = perf_counter()
        if args.route == "spherical":
            prediction = infer_spherical(model, frame.rgb)
        else:
            prediction = infer_cubemap(model, frame.rgb, face_size)
        seconds = perf_counter() - start
        method = evaluate(
            prediction,
            frame.radial_range,
            validity,
            normal_stride=normal_stride,
            seconds=seconds,
        )
        prediction_path = args.output / f"{panorama_id}-{args.route}-radial.npy"
        np.save(prediction_path, prediction)
        np.save(args.output / f"{panorama_id}-evaluation-validity.npy", validity)
        method["prediction_sha256"] = sha256(prediction_path)
        method["ply"] = write_binary_ply(
            args.output / f"{panorama_id}-{args.route}-view.ply",
            prediction,
            frame.rgb,
            validity,
            max_points=args.ply_max_points,
        )
        row = {
            "panorama_id": panorama_id,
            "family": frame.family,
            "source_shape_hw": list(source_shape),
            "evaluation_shape_hw": list(output_shape),
            "cubemap_face_size": face_size,
            "evaluation_pixels": int(validity.sum()),
            "method": method,
        }
        frames.append(row)
        print(json.dumps(json_ready(row), sort_keys=True), flush=True)

    source_head = subprocess.run(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_diff = subprocess.run(
        ["git", "-C", str(args.source), "diff", "--", "dac/models/cnn_depth.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    result = {
        "schema": SCHEMA,
        "status": "development-only-not-publication-evidence",
        "dataset": {
            "name": "P74",
            "adapter": P74_ADAPTER,
            "frozen_sample": list(ids),
            "depth_semantics": "radial range in metres",
        },
        "model": {
            "name": "Depth Any Camera CNNDepth ResNet-101 indoor",
            "training_domain": "HM3D + Taskonomy + Hypersim, 670k indoor images",
            "source_commit_expected": SOURCE_COMMIT,
            "source_commit_observed": source_head,
            "source_local_lazy_import_patch": source_diff,
            "checkpoint_sha256_expected": CHECKPOINT_SHA256,
            "checkpoint_sha256_observed": sha256(args.checkpoint),
            "checkpoint_external": True,
            "load": load_report,
        },
        "protocol": {
            "route": args.route,
            "native_angular": True,
            "inference_input_resize": False,
            "external_model_internal_upsampling": (
                "nearest and bilinear interpolation retained exactly from CNNDepth"
            ),
            "prediction_resize_outside_model": False,
            "cubemap_focal_px": "face_size/2",
            "spherical_effective_focal_px": "ERP_width/(2*pi)",
            "canonical_focal_px": 519,
            "prediction_clamp_m": list(MODEL_DEPTH_RANGE_M),
            "metric_weighting": "ERP pixel-cell solid angle",
            "normal_step_deg": args.normal_step_deg,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "peak_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "frames": frames,
    }
    result_path = args.output / f"results-{args.route}.json"
    result_path.write_text(
        json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"results": str(result_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
