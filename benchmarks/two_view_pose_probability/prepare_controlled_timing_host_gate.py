#!/usr/bin/env python3
"""Build a signed-off host gate from an exact-route validation result."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

try:
    from run_controlled_timing_benchmark import (  # type: ignore[import-not-found]
        EXPECTED_VERSION,
        EXPECTED_WHEEL_SHA256,
        HOST_GATE_SCHEMA,
        validate_route_result,
    )
    from run_resumable_population_replay import (  # type: ignore[import-not-found]
        _atomic_json,
        _canonical_sha256,
        _sha256,
    )
except ImportError:
    from benchmarks.two_view_pose_probability.run_controlled_timing_benchmark import (
        EXPECTED_VERSION,
        EXPECTED_WHEEL_SHA256,
        HOST_GATE_SCHEMA,
        validate_route_result,
    )
    from benchmarks.two_view_pose_probability.run_resumable_population_replay import (
        _atomic_json,
        _canonical_sha256,
        _sha256,
    )


THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",
)


def build_gate(args: argparse.Namespace) -> dict[str, Any]:
    attestations = {
        "stable_power": args.attest_stable_power,
        "thermal_throttling": not args.attest_no_thermal_throttling,
        "unrelated_intensive_processes": (
            not args.attest_no_unrelated_intensive_processes
        ),
        "output_directory_exclusive": args.attest_output_exclusive,
    }
    if not all(
        (
            attestations["stable_power"],
            not attestations["thermal_throttling"],
            not attestations["unrelated_intensive_processes"],
            attestations["output_directory_exclusive"],
        )
    ):
        raise ValueError("all four host-control attestations are required")
    if args.required_free_memory_gib <= 0:
        raise ValueError("required free memory must be positive")
    if args.free_memory_gib < args.required_free_memory_gib:
        raise ValueError("observed free memory is below the declared requirement")
    if _sha256(args.wheel) != EXPECTED_WHEEL_SHA256:
        raise ValueError("wheel hash differs from frozen PanorAi 3.5.0 artifact")
    result = json.loads(args.route_result.read_text(encoding="utf-8"))
    validate_route_result(
        result,
        expected_source_commit=args.expected_source_commit,
        forbidden_checkout=args.forbidden_checkout,
    )
    system = result["system"]
    return {
        "schema": HOST_GATE_SCHEMA,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "approved": True,
        **attestations,
        "free_memory_gib": args.free_memory_gib,
        "required_free_memory_gib": args.required_free_memory_gib,
        "route_validation": {
            "validated": True,
            "package_version": EXPECTED_VERSION,
            "source_commit": args.expected_source_commit,
            "wheel_sha256": EXPECTED_WHEEL_SHA256,
            "convolution_backend": "native",
            "detector_method": "detect_batch",
            "detector_batch_size": 2,
            "patch_provider_max_workers": 4,
            "numpy_fallback_permitted": False,
            "explicit_validity_masks": True,
            "route_sha256": _canonical_sha256(result["route"]),
            "resolution_hw": result["resolution_hw"],
            "opencv_threads": system["opencv_threads"],
            "patch_workers": system["patch_workers"],
            "result_path": str(args.route_result.resolve()),
            "result_sha256": _sha256(args.route_result),
            "import_path": result["package"]["import_path"],
        },
        "system": {
            "cpu_model": args.cpu_model,
            "physical_cpu_count": args.physical_cpu_count,
            "logical_cpu_count": system["logical_cpu_count"],
            "ram_gib": args.ram_gib,
            "os": system["platform"],
            "python": system["python"],
            "numpy": args.numpy_version,
            "opencv": system["opencv"],
            "power_source": args.power_source,
            "power_mode": args.power_mode,
            "thermal_state": args.thermal_state,
            "thread_environment": {
                name: os.environ.get(name) for name in THREAD_ENVIRONMENT_VARIABLES
            },
        },
        "generator": {
            "python_distribution_numpy": importlib.metadata.version("numpy"),
            "note": (
                "numpy_version is the exact timing-environment value supplied "
                "by the operator; generator distribution is recorded separately"
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-result", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--forbidden-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-model", required=True)
    parser.add_argument("--physical-cpu-count", type=int, required=True)
    parser.add_argument("--ram-gib", type=float, required=True)
    parser.add_argument("--numpy-version", required=True)
    parser.add_argument("--power-source", required=True)
    parser.add_argument("--power-mode", required=True)
    parser.add_argument("--thermal-state", required=True)
    parser.add_argument("--free-memory-gib", type=float, required=True)
    parser.add_argument("--required-free-memory-gib", type=float, required=True)
    parser.add_argument("--attest-stable-power", action="store_true")
    parser.add_argument("--attest-no-thermal-throttling", action="store_true")
    parser.add_argument("--attest-no-unrelated-intensive-processes", action="store_true")
    parser.add_argument("--attest-output-exclusive", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    gate = build_gate(args)
    _atomic_json(args.output, gate)
    print(json.dumps(gate, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
