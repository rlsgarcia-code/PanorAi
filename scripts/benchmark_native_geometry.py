#!/usr/bin/env python3
"""Compare the installed native cubemap kernel with the PERF-007 fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any

import numpy as np


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _measure(projector, faces, output_shape, repetitions: int) -> list[float]:
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        projector.back_project(faces, output_shape)
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--face-height", type=int, default=512)
    parser.add_argument("--face-width", type=int, default=512)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=15)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing evidence: {args.output}")
    if args.repetitions < 3 or args.warmups < 1:
        parser.error("use at least one warmup and three repetitions")
    if (
        min(
            args.height,
            args.width,
            args.face_height,
            args.face_width,
            args.channels,
        )
        <= 0
    ):
        parser.error("all image dimensions and channels must be positive")

    import panorai
    from panorai.geometry import CUBE_FACE_ORDER, CubemapProjector, CubemapSpec
    from panorai.geometry import _native

    if not _native.native_geometry_available():
        raise SystemExit("native geometry extension is unavailable")
    dtype = np.dtype(args.dtype)
    face_shape = (args.face_height, args.face_width)
    output_shape = (args.height, args.width)
    faces = {
        face: np.random.default_rng(800 + index)
        .random((*face_shape, args.channels))
        .astype(dtype)
        for index, face in enumerate(CUBE_FACE_ORDER)
    }
    projector = CubemapProjector(CubemapSpec(face_shape))
    extension = _native._native_geometry

    _native._native_geometry = None
    try:
        fallback_reference = projector.back_project(faces, output_shape).data.copy()
        for _ in range(args.warmups - 1):
            projector.back_project(faces, output_shape)
        fallback_samples = _measure(projector, faces, output_shape, args.repetitions)
    finally:
        _native._native_geometry = extension

    native_reference = projector.back_project(faces, output_shape).data.copy()
    for _ in range(args.warmups - 1):
        projector.back_project(faces, output_shape)
    native_samples = _measure(projector, faces, output_shape, args.repetitions)
    np.testing.assert_array_equal(native_reference, fallback_reference)

    def summary(samples: list[float]) -> dict[str, Any]:
        return {
            "median_ms": statistics.median(samples),
            "p95_ms": _percentile(samples, 0.95),
            "raw_ms": samples,
        }

    fallback = summary(fallback_samples)
    native = summary(native_samples)
    evidence = {
        "schema": "panorai-native-geometry-benchmark-v1",
        "package": {
            "version": panorai.__version__,
            "origin": str(Path(panorai.__file__).resolve()),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "numpy": np.__version__,
        },
        "case": {
            "operation": "cubemap_to_equirectangular",
            "output_shape_hw": output_shape,
            "face_shape_hw": face_shape,
            "channels": args.channels,
            "dtype": args.dtype,
            "interpolation": "bilinear",
            "invalid_policy": "propagate",
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "measurement_order": ["fallback", "native"],
        },
        "fallback": fallback,
        "native": native,
        "speedup": fallback["median_ms"] / native["median_ms"],
        "parity": "bit-exact",
        "output_sha256": hashlib.sha256(native_reference.tobytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
