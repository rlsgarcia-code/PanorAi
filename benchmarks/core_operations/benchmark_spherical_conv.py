#!/usr/bin/env python3
"""Compare the exact NumPy and fused native spherical-convolution backends."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import inspect
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc

import numpy as np
import panorai

from panorai.core import SphericalExecutionSpec
from panorai.core.numpy import SphericalConv2d


CASES = {
    "rgb-stem": ((1, 3, 128, 256), (32, 3, 3, 3), 1),
    "channel-mixing": ((1, 32, 64, 128), (64, 32, 3, 3), 1),
    "depthwise": ((1, 32, 128, 256), (32, 1, 3, 3), 32),
}


def _p95(samples: list[float]) -> float:
    return sorted(samples)[max(0, math.ceil(0.95 * len(samples)) - 1)]


def _maximum_rss() -> tuple[int | None, str | None]:
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return None, None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value, "bytes" if sys.platform == "darwin" else "kibibytes"


def _operation(case: str, backend: str) -> tuple[SphericalConv2d, np.ndarray]:
    shape, weight_shape, groups = CASES[case]
    rng = np.random.default_rng(1307)
    values = rng.standard_normal(size=shape, dtype=np.float32)
    weights = rng.standard_normal(size=weight_shape, dtype=np.float32)
    bias = rng.standard_normal(size=(weight_shape[0],), dtype=np.float32)
    layer = SphericalConv2d(
        weights,
        bias,
        padding=1,
        groups=groups,
        execution=SphericalExecutionSpec(backend=backend),
    )
    return layer, values


def _worker(case: str, backend: str, warmup: int, repeats: int) -> dict[str, object]:
    layer, values = _operation(case, backend)

    rss_before, rss_unit = _maximum_rss()
    first = layer(values)
    rss_after, _ = _maximum_rss()
    checksum = float(np.sum(first, dtype=np.float64))

    for _ in range(warmup):
        layer(values)
    samples_ms: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        layer(values)
        samples_ms.append((time.perf_counter_ns() - started) / 1e6)

    tracemalloc.start()
    try:
        memory_result = layer(values)
        _, traced_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    if memory_result.shape != first.shape:
        raise RuntimeError("benchmark result shape changed")
    return {
        "backend": backend,
        "case": case,
        "input_shape": list(CASES[case][0]),
        "weight_shape": list(CASES[case][1]),
        "groups": CASES[case][2],
        "samples_ms": samples_ms,
        "median_ms": statistics.median(samples_ms),
        "p95_ms": _p95(samples_ms),
        "python_traced_peak_bytes": traced_peak,
        "maximum_rss_delta": (
            None if rss_before is None or rss_after is None else rss_after - rss_before
        ),
        "maximum_rss_unit": rss_unit,
        "checksum": checksum,
    }


def _run_child(
    script: Path, case: str, backend: str, warmup: int, repeats: int
) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--worker",
            "--case",
            case,
            "--backend",
            backend,
            "--warmup",
            str(warmup),
            "--repeats",
            str(repeats),
        ],
        cwd=Path(os.environ.get("TMPDIR", tempfile.gettempdir())),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=tuple(CASES), help=argparse.SUPPRESS)
    parser.add_argument(
        "--backend", choices=("numpy", "native"), help=argparse.SUPPRESS
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.repeats <= 0 or args.warmup < 0:
        raise ValueError("repeats must be positive and warmup non-negative")
    if args.worker:
        if args.case is None or args.backend is None:
            raise ValueError("worker mode requires case and backend")
        print(json.dumps(_worker(args.case, args.backend, args.warmup, args.repeats)))
        return

    script = Path(__file__).resolve()
    measurements = [
        _run_child(script, case, backend, args.warmup, args.repeats)
        for case in CASES
        for backend in ("numpy", "native")
    ]
    by_case = {
        case: {
            "speedup_native_over_numpy": next(
                item["median_ms"]
                for item in measurements
                if item["case"] == case and item["backend"] == "numpy"
            )
            / next(
                item["median_ms"]
                for item in measurements
                if item["case"] == case and item["backend"] == "native"
            )
        }
        for case in CASES
    }
    payload = {
        "contract": "panorai-core-operations/v1",
        "operation": "SphericalConv2d",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "panorai_version": panorai.__version__,
        "distribution_version": version("panorai"),
        "package_origin": str(Path(inspect.getfile(panorai)).resolve()),
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "measurements": measurements,
        "comparisons": by_case,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
