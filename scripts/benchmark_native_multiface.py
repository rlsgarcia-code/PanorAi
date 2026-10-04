#!/usr/bin/env python3
"""Benchmark native arbitrary-face generation and Gaussian reconstruction.

Every route and operation runs in a fresh process. The report retains first-call
latency, raw reused samples, median, P95, peak RSS, package versions, hardware,
and compact numerical fingerprints for the native/Python parity check.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = ("generation", "gaussian", "end-to-end")
ROUTES = ("python", "native")


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "raw_seconds": values,
        "median_seconds": statistics.median(values),
        "p95_seconds": _p95(values),
    }


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _fingerprint(value) -> dict[str, Any]:
    import numpy as np

    if isinstance(value, np.ndarray):
        arrays = (value,)
        shape = list(value.shape)
        dtype = str(value.dtype)
    else:
        arrays = tuple(face.image for face in value)
        shape = [len(arrays), *arrays[0].shape]
        dtype = str(arrays[0].dtype)
    finite_count = 0
    value_sum = 0.0
    squared_sum = 0.0
    for array in arrays:
        finite = np.isfinite(array)
        finite_values = array[finite].astype(np.float64, copy=False)
        finite_count += int(np.count_nonzero(finite))
        value_sum += float(np.sum(finite_values, dtype=np.float64))
        squared_sum += float(np.sum(finite_values * finite_values, dtype=np.float64))
    return {
        "shape": shape,
        "dtype": dtype,
        "finite_count": finite_count,
        "sum": value_sum,
        "squared_sum": squared_sum,
    }


def _layout_arguments(layout: str, count: int) -> dict[str, int]:
    return {"count": count} if layout in {"fibonacci", "spiral"} else {}


def _worker(args) -> dict[str, Any]:
    import numpy as np
    import panorai as pa
    from panorai.geometry import _native

    origin = Path(pa.__file__).resolve()
    if args.require_installed and origin.is_relative_to(REPOSITORY_ROOT):
        raise RuntimeError(f"panorai resolved inside the checkout: {origin}")
    native_available = _native.native_geometry_available()
    if args.route == "native" and not native_available:
        raise RuntimeError("the native geometry extension is unavailable")
    if args.route == "python":
        _native._native_geometry = None

    source = np.random.default_rng(908).random(
        (args.height, args.width, args.channels), dtype=np.float32
    )
    panorama = pa.EquirectangularImage(source)
    view_kwargs = {
        "layout": args.layout,
        "size": (args.face_height, args.face_width),
        "fov": (args.hfov, args.vfov),
        **_layout_arguments(args.layout, args.count),
    }
    prepared_views = None
    if args.operation == "gaussian":
        prepared_views = panorama.views(**view_kwargs)

    baseline_rss = _rss_bytes()

    def operation():
        if args.operation == "generation":
            return panorama.views(**view_kwargs)
        if args.operation == "gaussian":
            return prepared_views.reconstruct(blend="gaussian").image
        views = panorama.views(**view_kwargs)
        return views.reconstruct(blend="gaussian").image

    started = time.perf_counter()
    result = operation()
    first = time.perf_counter() - started
    fingerprint = _fingerprint(result)
    del result
    gc.collect()
    samples = []
    for _ in range(args.repetitions):
        started = time.perf_counter()
        result = operation()
        samples.append(time.perf_counter() - started)
        del result
        gc.collect()
    return {
        "operation": args.operation,
        "route": args.route,
        "sample_semantics": {
            "generation": "fresh face set; warm process after first sample",
            "gaussian": "same face set; cached back plan after first sample",
            "end-to-end": "fresh face set and back plan; warm process after first sample",
        }[args.operation],
        "layout": args.layout,
        "view_count": (
            len(prepared_views)
            if prepared_views is not None
            else (
                args.count
                if args.layout in {"fibonacci", "spiral"}
                else {"cube": 6, "icosahedron": 12}[args.layout]
            )
        ),
        "erp_shape_hwc": [args.height, args.width, args.channels],
        "face_shape_hw": [args.face_height, args.face_width],
        "first_call_seconds": first,
        "reused": _summary(samples),
        "peak_rss_bytes": _rss_bytes(),
        "peak_rss_delta_bytes": max(0, _rss_bytes() - baseline_rss),
        "fingerprint": fingerprint,
        "native_extension_available_at_import": native_available,
        "panorai_origin": str(origin),
        "panorai_version": metadata.version("panorai"),
        "numpy_version": metadata.version("numpy"),
    }


def _run_fresh(args, route: str, operation: str) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--route",
        route,
        "--operation",
        operation,
        "--layout",
        args.layout,
        "--count",
        str(args.count),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--channels",
        str(args.channels),
        "--face-height",
        str(args.face_height),
        "--face-width",
        str(args.face_width),
        "--hfov",
        str(args.hfov),
        "--vfov",
        str(args.vfov),
        "--repetitions",
        str(args.repetitions),
    ]
    if args.require_installed:
        command.append("--require-installed")
    environment = os.environ.copy()
    if not args.require_installed:
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(REPOSITORY_ROOT)
            if not existing
            else f"{REPOSITORY_ROOT}{os.pathsep}{existing}"
        )
    with tempfile.TemporaryDirectory(prefix="panorai-native-multiface-") as cwd:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    return json.loads(completed.stdout)


def _fingerprints_match(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if any(first[key] != second[key] for key in ("shape", "dtype", "finite_count")):
        return False
    scale = max(1.0, abs(first["sum"]), abs(second["sum"]))
    square_scale = max(1.0, abs(first["squared_sum"]), abs(second["squared_sum"]))
    return bool(
        abs(first["sum"] - second["sum"]) <= 2e-7 * scale
        and abs(first["squared_sum"] - second["squared_sum"]) <= 2e-7 * square_scale
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--route", choices=ROUTES, default="native")
    parser.add_argument("--operation", choices=OPERATIONS, default="end-to-end")
    parser.add_argument(
        "--layout",
        choices=("cube", "icosahedron", "fibonacci", "spiral"),
        default="fibonacci",
    )
    parser.add_argument("--count", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--face-height", type=int, default=256)
    parser.add_argument("--face-width", type=int, default=256)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--vfov", type=float, default=90.0)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--require-installed", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if (
        args.repetitions <= 0
        or args.count <= 0
        or min(
            args.height,
            args.width,
            args.channels,
            args.face_height,
            args.face_width,
        )
        <= 0
    ):
        raise SystemExit("shapes, channels, count, and repetitions must be positive")
    if args.worker:
        print(json.dumps(_worker(args), sort_keys=True))
        return 0

    results = {
        operation: {route: _run_fresh(args, route, operation) for route in ROUTES}
        for operation in OPERATIONS
    }
    comparisons = {}
    for operation, routes in results.items():
        python_result = routes["python"]
        native_result = routes["native"]
        fingerprints_match = _fingerprints_match(
            python_result["fingerprint"], native_result["fingerprint"]
        )
        if not fingerprints_match:
            raise RuntimeError(f"native/Python parity failed for {operation}")
        comparisons[operation] = {
            "fingerprints_match": fingerprints_match,
            "speedup_first": python_result["first_call_seconds"]
            / native_result["first_call_seconds"],
            "speedup_reused_median": python_result["reused"]["median_seconds"]
            / native_result["reused"]["median_seconds"],
            "peak_rss_delta_reduction": (
                None
                if python_result["peak_rss_delta_bytes"] == 0
                else 1.0
                - native_result["peak_rss_delta_bytes"]
                / python_result["peak_rss_delta_bytes"]
            ),
        }
    report = {
        "schema": "panorai-native-multiface-benchmark/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "configuration": {
            "layout": args.layout,
            "count": args.count,
            "erp_shape_hwc": [args.height, args.width, args.channels],
            "face_shape_hw": [args.face_height, args.face_width],
            "fov_degrees": [args.hfov, args.vfov],
            "repetitions": args.repetitions,
        },
        "results": results,
        "comparisons": comparisons,
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
