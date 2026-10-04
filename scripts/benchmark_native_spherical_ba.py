#!/usr/bin/env python3
"""Benchmark NumPy finite-difference and native analytic spherical BA."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panorai import __version__ as panorai_version  # noqa: E402
from panorai.reconstruction import SphericalGlobalMapperOptions  # noqa: E402
from panorai.reconstruction._mapper import (  # noqa: E402
    _Observation,
    _Track,
    _bundle_adjust,
)
from panorai.reconstruction._math import rotation_exp  # noqa: E402
from panorai.reconstruction._native import (  # noqa: E402
    native_bundle_kernels_available,
)


def _scene(camera_count: int, point_count: int):
    rng = np.random.default_rng(20261004)
    ids = tuple(f"camera-{index:02d}" for index in range(camera_count))
    rotations = {ids[0]: np.eye(3)}
    centers = {ids[0]: np.zeros(3)}
    for index, panorama_id in enumerate(ids[1:], start=1):
        angle = 2.0 * math.pi * (index - 1) / max(2, camera_count - 1)
        centers[panorama_id] = np.asarray(
            (1.1 * math.cos(angle), 0.8 * math.sin(angle), 0.08 * index)
        )
        rotations[panorama_id] = rotation_exp(
            np.asarray((0.012 * index, -0.018 * index, 0.009 * index))
        )
    points = rng.uniform((-2.5, -1.8, 3.5), (2.5, 1.8, 8.0), size=(point_count, 3))
    tracks = []
    for point_index, point in enumerate(points):
        observations = []
        for panorama_id in ids:
            vector = rotations[panorama_id] @ (point - centers[panorama_id])
            bearing = vector / np.linalg.norm(vector)
            bearing += rng.normal(scale=2e-4, size=3)
            bearing /= np.linalg.norm(bearing)
            observations.append(_Observation(panorama_id, point_index, bearing))
        tracks.append(_Track(observations))
    initial_rotations = {
        item: rotation_exp(rng.normal(scale=0.008, size=3)) @ rotations[item]
        for item in ids
    }
    initial_rotations[ids[0]] = rotations[ids[0]].copy()
    initial_centers = {
        item: centers[item] + rng.normal(scale=0.025, size=3) for item in ids
    }
    initial_centers[ids[0]] = np.zeros(3)
    anchor_axis = int(np.argmax(np.abs(centers[ids[1]])))
    initial_centers[ids[1]][anchor_axis] = centers[ids[1]][anchor_axis]
    initial_points = points + rng.normal(scale=0.04, size=points.shape)
    active = [np.ones(camera_count, dtype=bool) for _ in tracks]
    return (
        ids,
        tracks,
        active,
        initial_rotations,
        initial_centers,
        initial_points,
        f"{ids[1]}:{anchor_axis}",
        np.stack([rotations[item] for item in ids]),
        np.stack([centers[item] for item in ids]),
        points,
    )


def _run(scene, backend: str, max_nfev: int):
    ids, tracks, active, rotations, centers, points, anchor, _, _, _ = scene
    options = SphericalGlobalMapperOptions(
        bundle_compute_backend=backend,
        bundle_max_nfev=max_nfev,
        bundle_loss="cauchy",
    )
    start = time.perf_counter_ns()
    rotations, centers, points, fixed_cost = _bundle_adjust(
        tracks,
        active,
        rotations,
        centers,
        points,
        ids[0],
        anchor,
        options,
        joint=False,
    )
    rotations, centers, points, joint_cost = _bundle_adjust(
        tracks,
        active,
        rotations,
        centers,
        points,
        ids[0],
        anchor,
        options,
        joint=True,
    )
    elapsed = (time.perf_counter_ns() - start) / 1_000_000_000
    return {
        "seconds": elapsed,
        "fixed_cost": list(fixed_cost),
        "joint_cost": list(joint_cost),
        "rotations": np.stack([rotations[item] for item in ids]),
        "centers": np.stack([centers[item] for item in ids]),
        "points": points,
    }


def _summary(values):
    ordered = sorted(float(value) for value in values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "median": statistics.median(ordered),
        "p95_nearest_rank": ordered[p95_index],
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def _accuracy(result, true_rotations, true_centers, true_points):
    relative = np.einsum("nij,nkj->nik", result["rotations"], true_rotations)
    traces = np.trace(relative, axis1=1, axis2=2)
    rotation_errors = np.degrees(np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0)))
    return {
        "rotation_median_deg": float(np.median(rotation_errors)),
        "rotation_max_deg": float(np.max(rotation_errors)),
        "center_rmse": float(
            np.sqrt(np.mean(np.square(result["centers"] - true_centers)))
        ),
        "point_rmse": float(
            np.sqrt(np.mean(np.square(result["points"] - true_points)))
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-count", type=int, default=5)
    parser.add_argument("--point-count", type=int, default=120)
    parser.add_argument("--max-nfev", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=8)
    args = parser.parse_args()
    if not native_bundle_kernels_available():
        raise RuntimeError("native spherical-BA kernel is unavailable")
    scene = _scene(args.camera_count, args.point_count)
    for _ in range(args.warmup):
        _run(scene, "numpy", args.max_nfev)
        _run(scene, "native", args.max_nfev)
    samples = {"numpy": [], "native": []}
    final = {}
    for _ in range(args.repetitions):
        for backend in ("numpy", "native"):
            result = _run(scene, backend, args.max_nfev)
            samples[backend].append(result["seconds"])
            final[backend] = result
    numpy_result = final["numpy"]
    native_result = final["native"]
    true_rotations, true_centers, true_points = scene[-3:]
    payload = {
        "schema": "panorai-native-spherical-ba-benchmark/v1",
        "panorai_version": panorai_version,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "configuration": {
            "camera_count": args.camera_count,
            "point_count": args.point_count,
            "observation_count": args.camera_count * args.point_count,
            "max_nfev_per_phase": args.max_nfev,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
        },
        "raw_seconds": samples,
        "summary_seconds": {
            backend: _summary(values) for backend, values in samples.items()
        },
        "median_speedup": (
            statistics.median(samples["numpy"]) / statistics.median(samples["native"])
        ),
        "correctness": {
            "numpy_fixed_cost": numpy_result["fixed_cost"],
            "native_fixed_cost": native_result["fixed_cost"],
            "numpy_joint_cost": numpy_result["joint_cost"],
            "native_joint_cost": native_result["joint_cost"],
            "max_rotation_frobenius_delta": float(
                np.max(
                    np.linalg.norm(
                        numpy_result["rotations"] - native_result["rotations"],
                        axis=(1, 2),
                    )
                )
            ),
            "max_center_delta": float(
                np.max(
                    np.linalg.norm(
                        numpy_result["centers"] - native_result["centers"], axis=1
                    )
                )
            ),
            "max_point_delta": float(
                np.max(
                    np.linalg.norm(
                        numpy_result["points"] - native_result["points"], axis=1
                    )
                )
            ),
            "numpy_accuracy": _accuracy(
                numpy_result, true_rotations, true_centers, true_points
            ),
            "native_accuracy": _accuracy(
                native_result, true_rotations, true_centers, true_points
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary_seconds"], indent=2, sort_keys=True))
    print(f"median speedup: {payload['median_speedup']:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
