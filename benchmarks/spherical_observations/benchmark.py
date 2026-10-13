#!/usr/bin/env python3
"""Measure the Stable spherical-observation workflow on 96 gnomonic faces."""

from __future__ import annotations

import argparse
import inspect
from importlib.metadata import version
import json
import platform
from pathlib import Path
import statistics
import subprocess
import time
import tracemalloc

import numpy as np
import panorai

from panorai.data import (
    EquirectangularImage,
    FacePointBatch,
    FaceRegionBatch,
    FaceViewOutput,
)


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--faces", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _measure(operation, *, warmup: int, repeats: int) -> dict[str, object]:
    for _ in range(warmup):
        operation()
    elapsed_ms = []
    peaks = []
    for _ in range(repeats):
        tracemalloc.start()
        started = time.perf_counter_ns()
        result = operation()
        elapsed_ms.append((time.perf_counter_ns() - started) / 1e6)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)
        if result is None:
            raise RuntimeError("benchmark operation returned no result")
    return {
        "samples_ms": elapsed_ms,
        "median_ms": statistics.median(elapsed_ms),
        "p95_ms": float(np.percentile(elapsed_ms, 95)),
        "peak_memory_bytes": max(peaks),
    }


def main() -> None:
    args = _arguments()
    if args.faces <= 0 or args.repeats <= 0 or args.warmup < 0:
        raise ValueError("faces/repeats must be positive and warmup non-negative")
    image = np.zeros((256, 512, 3), dtype=np.uint8)
    views = EquirectangularImage(image).views(
        "fibonacci", count=args.faces, size=64, fov=80
    )
    point_xy = np.column_stack(
        (
            np.linspace(8, 55, 32, dtype=np.float64),
            np.linspace(55, 8, 32, dtype=np.float64),
        )
    )
    boxes = np.asarray(
        [[8, 8, 28, 28], [24, 8, 48, 30], [8, 30, 30, 54], [30, 30, 55, 55]],
        dtype=np.float64,
    )
    point_batch = FacePointBatch(point_xy, scores=np.linspace(0.1, 1.0, 32))
    region_batch = FaceRegionBatch.from_boxes(
        boxes,
        scores=np.asarray([0.7, 0.8, 0.9, 1.0]),
        class_ids=np.asarray([0, 1, 2, 3]),
    )
    points = views.map_points(lambda face: point_batch, source_id="benchmark")
    regions = views.map_regions(lambda face: region_batch, source_id="benchmark")
    view_output = FaceViewOutput(
        scores=np.linspace(0.1, 1.0, 16),
        class_ids=np.arange(16),
        embedding=np.linspace(-1.0, 1.0, 128),
    )
    view_evidence = views.map_views(
        lambda face: view_output, source_id="benchmark"
    )
    results = {
        "contract": "panorai-spherical-observations/v1",
        "panorai_version": panorai.__version__,
        "distribution_version": version("panorai"),
        "package_origin": str(Path(inspect.getfile(panorai)).resolve()),
        "source_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "configuration": {
            "faces": args.faces,
            "face_shape_hw": [64, 64],
            "face_fov_deg": 80,
            "erp_shape_hw": [256, 512],
            "points_per_face": 32,
            "regions_per_face": 4,
            "view_score_count": 16,
            "view_embedding_dimensions": 128,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "map_points": _measure(
            lambda: views.map_points(lambda face: point_batch, source_id="benchmark"),
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        "point_angular_nms": _measure(
            lambda: points.angular_nms(0.1, class_aware=False),
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        "map_regions": _measure(
            lambda: views.map_regions(lambda face: region_batch, source_id="benchmark"),
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        "region_nms": _measure(
            lambda: regions.nms(
                0.8, class_aware=True, evaluation_shape_hw=(128, 256)
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        "map_views": _measure(
            lambda: views.map_views(
                lambda face: view_output, source_id="benchmark"
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        "view_coverage_mean": _measure(
            lambda: view_evidence.aggregate(
                "coverage_mean", evaluation_shape_hw=(128, 256)
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        ),
    }
    payload = json.dumps(results, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
