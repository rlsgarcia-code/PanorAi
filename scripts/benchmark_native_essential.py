#!/usr/bin/env python3
"""Benchmark NumPy and native PanorAi essential-estimation kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import cv2
import numpy as np
import scipy

import panorai
from panorai.estimators import (
    RelativePoseAcceptancePolicy,
    RelativePoseOptions,
    estimate_relative_pose,
    native_kernels_available,
    solve_five_point_essential,
    spherical_tangent_sampson_error,
)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "raw": values,
        "count": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "p95_nearest_rank": _percentile(values, 0.95),
        "maximum": max(values),
    }


def _measure(
    operation: Callable[[], Any], *, warmup: int, repetitions: int
) -> tuple[dict[str, Any], Any]:
    result = None
    for _ in range(warmup):
        result = operation()
    values = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        result = operation()
        values.append((time.perf_counter_ns() - started) / 1_000_000_000)
    return _summary(values), result


def _fixtures(residual_count: int):
    rng = np.random.default_rng(20261003)
    five_points = rng.uniform((-2.0, -1.0, 4.0), (2.0, 1.0, 8.0), size=(5, 3))
    angle = np.deg2rad(4.0)
    rotation = np.asarray(
        (
            (np.cos(angle), 0.0, np.sin(angle)),
            (0.0, 1.0, 0.0),
            (-np.sin(angle), 0.0, np.cos(angle)),
        )
    )
    translation = np.asarray((0.8, 0.1, 0.05))
    five_a = five_points / np.linalg.norm(five_points, axis=1, keepdims=True)
    transformed = five_points @ rotation.T + translation
    five_b = transformed / np.linalg.norm(transformed, axis=1, keepdims=True)

    residual_a = rng.normal(size=(residual_count, 3))
    residual_b = rng.normal(size=(residual_count, 3))
    residual_e = rng.normal(size=(3, 3))

    points = rng.uniform((-2.0, -1.5, 4.0), (2.0, 1.5, 9.0), size=(80, 3))
    estimate_a = points / np.linalg.norm(points, axis=1, keepdims=True)
    estimate_transformed = points @ rotation.T + translation
    estimate_b = estimate_transformed / np.linalg.norm(
        estimate_transformed, axis=1, keepdims=True
    )
    estimate_b[-12:] = estimate_b[-12:][::-1]
    return five_a, five_b, residual_a, residual_b, residual_e, estimate_a, estimate_b


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _source_hash() -> str:
    digest = hashlib.sha256()
    paths = sorted((REPOSITORY_ROOT / "panorai" / "estimators").glob("*.py"))
    paths.extend(
        [
            REPOSITORY_ROOT / "panorai/_native/essential_kernels.cpp",
            REPOSITORY_ROOT / "setup.py",
            Path(__file__).resolve(),
        ]
    )
    for path in paths:
        relative = path.relative_to(REPOSITORY_ROOT).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git(command: list[str]) -> str | None:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return completed.stdout.strip() or None


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if not native_kernels_available():
        raise RuntimeError("native kernels must be built before benchmarking")
    fixtures = _fixtures(args.residual_count)
    five_a, five_b, residual_a, residual_b, residual_e, estimate_a, estimate_b = (
        fixtures
    )
    options = RelativePoseOptions(
        max_angular_error_deg=0.2,
        min_num_trials=16,
        max_num_trials=40,
        min_inliers=12,
        local_optimization_steps=1,
        stability_trials=0,
        model_competition_trials=16,
        random_seed=11,
        compute_backend="numpy",
    )
    policy = RelativePoseAcceptancePolicy(
        min_inliers=12,
        min_inlier_ratio=0.2,
        min_occupied_cells=1,
        min_coverage_entropy=0.0,
        max_median_residual_deg=1.0,
        min_median_parallax_deg=0.0,
        min_cheirality_ratio=0.0,
        min_translation_orientation_margin=0.0,
        require_stability=False,
        require_essential_preferred=False,
    )
    results: dict[str, Any] = {}
    outputs: dict[tuple[str, str], Any] = {}
    for operation_name, repetitions in (
        ("five_point_solver", args.solver_repetitions),
        ("sampson_residuals", args.residual_repetitions),
        ("relative_pose", args.estimator_repetitions),
    ):
        results[operation_name] = {}
        for backend in ("numpy", "native"):
            if operation_name == "five_point_solver":

                def operation(backend: str = backend) -> Any:
                    return solve_five_point_essential(five_a, five_b, backend=backend)
            elif operation_name == "sampson_residuals":

                def operation(backend: str = backend) -> Any:
                    return spherical_tangent_sampson_error(
                        residual_a, residual_b, residual_e, backend=backend
                    )
            else:

                def operation(backend: str = backend) -> Any:
                    return estimate_relative_pose(
                        estimate_a,
                        estimate_b,
                        options=RelativePoseOptions(
                            **{**options.to_dict(), "compute_backend": backend}
                        ),
                        quality_policy=policy,
                    )

            summary, output = _measure(
                operation, warmup=args.warmup, repetitions=repetitions
            )
            results[operation_name][backend] = summary
            outputs[(operation_name, backend)] = output
        numpy_median = results[operation_name]["numpy"]["median"]
        native_median = results[operation_name]["native"]["median"]
        results[operation_name]["speedup_median"] = numpy_median / native_median

    numpy_solutions = outputs[("five_point_solver", "numpy")]
    native_solutions = outputs[("five_point_solver", "native")]
    projective = []
    for candidate in numpy_solutions:
        projective.append(
            min(
                min(
                    np.linalg.norm(candidate - other),
                    np.linalg.norm(candidate + other),
                )
                for other in native_solutions
            )
        )
    residual_difference = float(
        np.max(
            np.abs(
                outputs[("sampson_residuals", "numpy")]
                - outputs[("sampson_residuals", "native")]
            )
        )
    )
    numpy_pose = outputs[("relative_pose", "numpy")]
    native_pose = outputs[("relative_pose", "native")]
    correctness = {
        "essential_solution_count_numpy": len(numpy_solutions),
        "essential_solution_count_native": len(native_solutions),
        "essential_max_projective_distance": max(projective, default=None),
        "residual_max_absolute_difference": residual_difference,
        "relative_pose_both_succeeded": numpy_pose is not None
        and native_pose is not None,
        "relative_pose_inlier_masks_equal": bool(
            numpy_pose is not None
            and native_pose is not None
            and np.array_equal(numpy_pose.inlier_mask, native_pose.inlier_mask)
        ),
        "relative_pose_rotation_difference_frobenius": (
            None
            if numpy_pose is None or native_pose is None
            else float(np.linalg.norm(numpy_pose.R - native_pose.R))
        ),
        "relative_pose_translation_abs_dot": (
            None
            if numpy_pose is None or native_pose is None
            else abs(float(numpy_pose.t @ native_pose.t))
        ),
    }
    correctness["passed"] = bool(
        correctness["essential_solution_count_numpy"]
        == correctness["essential_solution_count_native"]
        and correctness["essential_max_projective_distance"] < 1e-8
        and correctness["residual_max_absolute_difference"] < 1e-12
        and correctness["relative_pose_inlier_masks_equal"]
        and correctness["relative_pose_rotation_difference_frobenius"] < 1e-8
        and correctness["relative_pose_translation_abs_dot"] > 1.0 - 1e-8
    )
    return {
        "schema": "panorai-native-essential-benchmark/v1",
        "source": {
            "git_commit": _git(["git", "rev-parse", "HEAD"]),
            "git_describe": _git(["git", "describe", "--tags", "--always", "--dirty"]),
            "native_estimator_source_sha256": _source_hash(),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "panorai_version": panorai.__version__,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "opencv_version": cv2.__version__,
            "peak_rss_bytes": _rss_bytes(),
        },
        "protocol": {
            "clock": "time.perf_counter_ns",
            "warmup_per_backend_operation": args.warmup,
            "solver_repetitions": args.solver_repetitions,
            "residual_repetitions": args.residual_repetitions,
            "estimator_repetitions": args.estimator_repetitions,
            "residual_count": args.residual_count,
            "percentile": "nearest-rank",
            "sample_retention": "all",
        },
        "results": results,
        "correctness": correctness,
    }


def _positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=_positive, default=3)
    parser.add_argument("--solver-repetitions", type=_positive, default=100)
    parser.add_argument("--residual-repetitions", type=_positive, default=30)
    parser.add_argument("--estimator-repetitions", type=_positive, default=10)
    parser.add_argument("--residual-count", type=_positive, default=100_000)
    args = parser.parse_args()
    report = _run(args)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "results": report["results"],
                "correctness": report["correctness"],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["correctness"]["passed"] else 2)


if __name__ == "__main__":
    main()
