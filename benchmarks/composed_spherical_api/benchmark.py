#!/usr/bin/env python3
"""Benchmark cold, warm, and cached paths of the composed spherical API."""

from __future__ import annotations

import argparse
import hashlib
import inspect
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import tempfile
import time
from typing import Any, Callable

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _summarize(
    samples_ms: list[float], *, rss_before: int, rss_after: int
) -> dict[str, Any]:
    return {
        "samples_ms": samples_ms,
        "median_ms": statistics.median(samples_ms),
        "p95_ms": float(np.percentile(samples_ms, 95)),
        "peak_rss_bytes_before": rss_before,
        "peak_rss_bytes_after": rss_after,
        "peak_rss_growth_bytes": max(0, rss_after - rss_before),
    }


def _measure(
    operation: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        if operation() is None:
            raise RuntimeError("benchmark operation returned None during warm-up")
    samples_ms: list[float] = []
    rss_before = _peak_rss_bytes()
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = operation()
        samples_ms.append((time.perf_counter_ns() - started) / 1e6)
        if result is None:
            raise RuntimeError("benchmark operation returned None")
    return _summarize(
        samples_ms,
        rss_before=rss_before,
        rss_after=_peak_rss_bytes(),
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.repeats <= 0 or args.warmup < 0:
        raise ValueError("repeats must be positive and warmup non-negative")
    wheel = args.wheel.resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise ValueError("--wheel must identify the exact installed wheel")

    import panorai
    from panorai.ops.spherical import GaussianBlur
    from panorai.pipeline import OnSphere, Output, Pipeline

    origin = Path(inspect.getfile(panorai)).resolve()
    try:
        origin.relative_to(args.source_root.resolve())
    except ValueError:
        pass
    else:
        raise RuntimeError(f"panorai resolved inside the source checkout: {origin}")

    distribution_version = version("panorai")
    if distribution_version != panorai.__version__:
        raise RuntimeError("package version and distribution metadata disagree")

    values = np.linspace(0, 1, 64 * 128 * 3, dtype=np.float32).reshape(64, 128, 3)
    panorama = panorai.EquirectangularImage(values)

    def build(memory: Path | None = None) -> Pipeline:
        return Pipeline(
            OnSphere(Output(GaussianBlur(5, 1.1), kind="features"), output="features"),
            memory=memory,
        )

    warm_pipeline = build()
    with tempfile.TemporaryDirectory(prefix="panorai-composed-benchmark-") as directory:
        cache_root = Path(directory)
        cache_counter = 0

        def cold_construct_and_run():
            return build()(panorama)

        def cache_miss():
            nonlocal cache_counter
            cache_counter += 1
            return build(cache_root / f"miss-{cache_counter}").run(panorama)

        cached_pipeline = build(cache_root / "hit")
        seeded = cached_pipeline.run(panorama)
        if seeded.manifest()["cache"]["hit"]:
            raise RuntimeError("cache seed unexpectedly reported a hit")

        def cache_hit():
            run = cached_pipeline.run(panorama)
            if not run.manifest()["cache"]["hit"]:
                raise RuntimeError("cached benchmark path did not report a hit")
            return run

        measurements = {
            "cold_construct_and_run": _measure(
                cold_construct_and_run,
                warmup=0,
                repeats=args.repeats,
            ),
            "warm_pipeline_run": _measure(
                lambda: warm_pipeline(panorama),
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "cache_miss": _measure(
                cache_miss,
                warmup=0,
                repeats=args.repeats,
            ),
            "cache_hit": _measure(
                cache_hit,
                warmup=args.warmup,
                repeats=args.repeats,
            ),
        }

    payload = {
        "status": "ok",
        "contract": "panorai-composed-spherical-benchmark/v1",
        "source_commit": args.source_commit,
        "wheel": wheel.name,
        "wheel_sha256": _sha256(wheel),
        "wheel_size_bytes": wheel.stat().st_size,
        "panorai_version": panorai.__version__,
        "distribution_version": distribution_version,
        "package_origin": str(origin),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
        "configuration": {
            "erp_shape_hwc": list(values.shape),
            "dtype": str(values.dtype),
            "operation": "GaussianBlur(kernel_size=5, sigma=1.1)",
            "warmup": args.warmup,
            "repeats": args.repeats,
            "accelerator_synchronization": "not-applicable-numpy-cpu",
        },
        "measurements": measurements,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(json.dumps(payload, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
