#!/usr/bin/env python3
"""Evaluate the Matterport3D-trained UniFuse CNN on native-angular P74."""

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

import numpy as np

from panoramic_cnn import (
    CHECKPOINT_SHA256,
    MODEL_DEPTH_RANGE_M,
    SOURCE_COMMIT,
    infer_unifuse,
    load_unifuse,
)
from protocol import (
    FROZEN_SAMPLE,
    P74_ADAPTER,
    band_metrics,
    depth_metrics,
    json_ready,
    load_p74_frame,
    mmap_npy_member,
    native_angular_erp_shape,
    normal_structure_metrics,
    scale_invariant_structure_metrics,
    seam_score,
    sha256,
    write_binary_ply,
)


SCHEMA = "panorai-p74-unifuse-matterport3d/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p74-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", choices=FROZEN_SAMPLE, required=True)
    parser.add_argument("--p74-row-chunk", type=int, default=32)
    parser.add_argument("--normal-step-deg", type=float, default=0.35)
    parser.add_argument("--ply-max-points", type=int, default=1_500_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import torch

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    npz = args.p74_root / "npzs" / f"{args.only}.npz"
    rgb_path = args.p74_root / "images" / f"{args.only}_rgb.png"
    source_shape = tuple(mmap_npy_member(npz, "xyz_image.npy").shape[:2])
    base_shape = native_angular_erp_shape(source_shape)
    # UniFuse's five encoder reductions and cube-face branch require a
    # 64-pixel input lattice.  Round upward from the already non-minifying
    # angular target; do not resize an intermediate ERP.
    output_height = math.ceil(base_shape[0] / 64) * 64
    output_shape = (output_height, 2 * output_height)
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
    model, load_report = load_unifuse(
        args.source,
        args.checkpoint,
        height=output_shape[0],
        width=output_shape[1],
    )
    start = perf_counter()
    prediction = infer_unifuse(model, frame.rgb)
    seconds = perf_counter() - start
    method = {
        "metrics": depth_metrics(prediction, frame.radial_range, validity),
        "scale_invariant_structure": scale_invariant_structure_metrics(
            prediction, frame.radial_range, validity
        ),
        "surface_normals": normal_structure_metrics(
            prediction, frame.radial_range, validity, stride_px=normal_stride
        ),
        "bands": band_metrics(prediction, frame.radial_range, validity),
        "seam": seam_score(prediction, validity),
        "runtime_seconds": seconds,
    }
    prediction_path = args.output / f"{args.only}-unifuse-radial.npy"
    np.save(prediction_path, prediction)
    np.save(args.output / f"{args.only}-evaluation-validity.npy", validity)
    method["prediction_sha256"] = sha256(prediction_path)
    method["ply"] = write_binary_ply(
        args.output / f"{args.only}-unifuse-view.ply",
        prediction,
        frame.rgb,
        validity,
        max_points=args.ply_max_points,
    )
    source_head = subprocess.run(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = {
        "schema": SCHEMA,
        "status": "development-only-not-publication-evidence",
        "dataset": {
            "name": "P74",
            "adapter": P74_ADAPTER,
            "panorama_id": args.only,
            "depth_semantics": "radial range in metres",
        },
        "model": {
            "name": "UniFuse ResNet-18 Matterport3D",
            "source_commit_expected": SOURCE_COMMIT,
            "source_commit_observed": source_head,
            "checkpoint_sha256_expected": CHECKPOINT_SHA256,
            "checkpoint_sha256_observed": sha256(args.checkpoint),
            "checkpoint_external": True,
            "load": load_report,
        },
        "protocol": {
            "native_angular": True,
            "evaluation_shape_hw": list(output_shape),
            "pre_alignment_native_shape_hw": list(base_shape),
            "model_lattice_multiple": 64,
            "inference_input_resize": False,
            "prediction_resize_outside_model": False,
            "internal_upsampling": "nearest, unchanged from UniFuse",
            "erp_to_cube": "PanorAi bilinear projection at face_size=ERP_height/2",
            "prediction_domain_m": list(MODEL_DEPTH_RANGE_M),
            "metric_weighting": "ERP pixel-cell solid angle",
            "normal_step_deg": args.normal_step_deg,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "peak_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "frame": {
            "panorama_id": args.only,
            "family": frame.family,
            "source_shape_hw": list(source_shape),
            "evaluation_pixels": int(validity.sum()),
            "method": method,
        },
    }
    path = args.output / f"results-{args.only}-unifuse.json"
    path.write_text(
        json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(result["frame"]), sort_keys=True), flush=True)
    print(json.dumps({"results": str(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
