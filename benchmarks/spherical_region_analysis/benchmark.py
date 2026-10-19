#!/usr/bin/env python3
"""Reproducible performance probe for Stable spherical region analysis."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Callable

import numpy as np


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _fixture(shape_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape_hw
    rows, columns = np.indices(shape_hw)
    latitude_band = (rows >= height // 4) & (rows < 3 * height // 4)
    seam_band = (columns < width // 10) | (columns >= 9 * width // 10)
    wave = rows < (
        height // 2
        + np.rint(height / 16.0 * np.sin(2.0 * np.pi * columns / width)).astype(int)
    )
    mask = latitude_band & seam_band
    mask |= wave & (columns >= width // 3) & (columns < 2 * width // 3)
    labels = np.zeros(shape_hw, dtype=np.int32)
    labels[mask] = 1
    labels[(~wave) & (columns >= width // 3) & (columns < 2 * width // 3)] = 2
    return mask, labels


def _timings(
    operation: Callable[[], object],
    *,
    warmup: int,
    repetitions: int,
) -> dict[str, object]:
    for _ in range(warmup):
        operation()
    raw = []
    for _ in range(repetitions):
        started = time.perf_counter()
        operation()
        raw.append(time.perf_counter() - started)
    return {
        "raw_seconds": raw,
        "median_seconds": float(np.median(raw)),
        "p95_seconds": float(np.percentile(raw, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--max-median-seconds", type=float, default=10.0)
    parser.add_argument("--max-peak-rss-bytes", type=int, default=2_147_483_648)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--require-installed", action="store_true")
    args = parser.parse_args()

    if not args.require_installed:
        source_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(source_root))

    import panorai
    from panorai.ops import spherical as S

    origin = Path(panorai.__file__).resolve()
    if args.require_installed:
        if args.source_root is None:
            parser.error("--require-installed requires --source-root")
        try:
            origin.relative_to(args.source_root.resolve())
        except ValueError:
            pass
        else:
            raise RuntimeError(f"panorai resolved inside source checkout: {origin}")

    results: dict[str, object] = {}
    for shape in ((256, 512), (512, 1024)):
        mask, labels = _fixture(shape)
        cases = {
            "connected_components": lambda: S.connected_components(mask),
            "measure_regions": lambda: S.measure_regions(labels),
            "trace_contours": lambda: S.trace_contours(labels),
            "skeletonize": lambda: S.skeletonize(mask),
        }
        shape_results = {
            name: _timings(
                operation,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
            for name, operation in cases.items()
        }
        results[f"{shape[0]}x{shape[1]}"] = shape_results

    peak_rss = _peak_rss_bytes()
    payload = {
        "benchmark": "panorai-spherical-region-analysis-performance/v1",
        "artifact": {
            "package_version": version("panorai"),
            "origin": str(origin),
            "source_commit": _commit(),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "method": {
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "timer": "time.perf_counter",
        },
        "results": results,
        "peak_rss_bytes": peak_rss,
        "limits": {
            "max_median_seconds": args.max_median_seconds,
            "max_peak_rss_bytes": args.max_peak_rss_bytes,
        },
    }
    medians = [
        operation["median_seconds"]
        for shape_result in results.values()
        for operation in shape_result.values()
    ]
    payload["passed"] = bool(
        max(medians) <= args.max_median_seconds and peak_rss <= args.max_peak_rss_bytes
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
