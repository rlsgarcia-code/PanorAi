#!/usr/bin/env python3
"""Reproducible benchmark for Experimental incremental central-ERP SLAM.

The parent process orchestrates two explicitly different measurements:

* ``warm``: inputs are decoded once, one or more complete SLAM sessions are
  discarded as warm-up, then complete fresh sessions are measured in the same
  process;
* ``cold-process``: every sample uses a fresh Python process and includes input
  verification/decoding and session construction in ``measured_seconds``.

Stage timings are inclusive.  In particular, global correction can call code
that is also represented by another stage, so stage totals must not be added.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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
from typing import Any, Iterator

# Executing ``python scripts/...`` otherwise places only ``scripts/`` on
# sys.path.  This benchmark deliberately measures the identified source tree.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import cv2
import numpy as np
import scipy

import panorai
from panorai.estimators import RelativePoseOptions, SphericalRelativePoseEstimator
from panorai.features import SphericalFeaturePipeline
from panorai.slam import SphericalIncrementalSLAM, SphericalIncrementalSLAMOptions


SCHEMA = "panorai-incremental-erp-slam-benchmark/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_selected_jsonl(path: Path, set_id: str) -> dict[str, Any]:
    selected = None
    with path.open() as stream:
        for line in stream:
            item = json.loads(line)
            if item.get("set_id") == set_id:
                if selected is not None:
                    raise RuntimeError(f"duplicate set_id in {path}: {set_id}")
                selected = item
    if selected is None:
        raise KeyError(f"set_id not found: {set_id}")
    return selected


def _read_images(item: dict[str, Any]) -> tuple[tuple[str, np.ndarray], ...]:
    members = sorted(item["members"], key=lambda member: int(member["order"]))
    if len(members) != int(item["member_count"]):
        raise RuntimeError("member_count does not match members")
    images = []
    for member in members:
        path = Path(member["rgb_path"]).resolve()
        if _sha256(path) != member["rgb_sha256"]:
            raise RuntimeError(f"RGB checksum mismatch: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read RGB: {path}")
        expected = tuple(int(value) for value in member["rgb_shape"])
        if image.shape != expected:
            raise RuntimeError(f"RGB shape mismatch: {image.shape} != {expected}")
        images.append((member["view_id"], cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    return tuple(images)


class _StageTimings:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = (time.perf_counter_ns() - started) / 1_000_000_000
            self.values.setdefault(name, []).append(elapsed)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: {
                "call_count": len(values),
                "total_seconds": float(sum(values)),
                "calls_seconds": values,
            }
            for name, values in sorted(self.values.items())
        }


class _TimedPipeline(SphericalFeaturePipeline):
    def __init__(self, delegate: SphericalFeaturePipeline, timings: _StageTimings):
        super().__init__(delegate.config, backend=delegate.backend)
        self._timings = timings

    def extract(self, *args: Any, **kwargs: Any):
        with self._timings.measure("feature_extraction"):
            return super().extract(*args, **kwargs)

    def match(self, *args: Any, **kwargs: Any):
        with self._timings.measure("descriptor_matching"):
            return super().match(*args, **kwargs)


class _TimedEstimator:
    def __init__(self, delegate: Any, timings: _StageTimings):
        self._delegate = delegate
        self._timings = timings

    def estimate(self, *args: Any, **kwargs: Any):
        with self._timings.measure("relative_pose"):
            return self._delegate.estimate(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class _TimedLocalBA:
    def __init__(self, delegate: Any, timings: _StageTimings):
        self._delegate = delegate
        self._timings = timings

    def optimize(self, *args: Any, **kwargs: Any):
        with self._timings.measure("local_bundle_adjustment"):
            return self._delegate.optimize(*args, **kwargs)


class _TimedGlobalMapper:
    def __init__(self, delegate: Any, timings: _StageTimings):
        self._delegate = delegate
        self._timings = timings

    def reconstruct(self, *args: Any, **kwargs: Any):
        with self._timings.measure("global_correction"):
            return self._delegate.reconstruct(*args, **kwargs)


def _build_session(args: argparse.Namespace, timings: _StageTimings):
    pipeline = SphericalFeaturePipeline.from_preset(
        "sift-flann",
        face_sampler="icosahedron",
        face_fov_deg=80.0,
        face_shape_hw=(args.face_size, args.face_size),
        edge_margin_px=16,
        max_features=args.max_features,
        ratio_test=0.75,
    )
    estimator = SphericalRelativePoseEstimator(
        RelativePoseOptions(
            max_angular_error_deg=0.5,
            min_num_trials=32,
            max_num_trials=args.max_num_trials,
            min_inliers=8,
            local_optimization_steps=3,
            random_seed=7,
        )
    )
    timed_estimator = _TimedEstimator(estimator, timings)
    session = SphericalIncrementalSLAM(
        feature_pipeline=_TimedPipeline(pipeline, timings),
        relative_pose_estimator=timed_estimator,
        options=SphericalIncrementalSLAMOptions(
            min_frame_features=64,
            min_pair_matches=20,
            min_map_correspondences=8,
            keyframe_min_interval_s=0.0,
            keyframe_max_interval_s=1.0,
            local_ba_max_points=512,
            local_ba_max_nfev=50,
            edge_admission="accepted",
        ),
    )
    session.local_ba = _TimedLocalBA(session.local_ba, timings)
    session.global_mapper = _TimedGlobalMapper(session.global_mapper, timings)
    return session


def _rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def _run_session(
    args: argparse.Namespace,
    images: tuple[tuple[str, np.ndarray], ...],
) -> tuple[dict[str, Any], Any]:
    timings = _StageTimings()
    with timings.measure("session_construction"):
        session = _build_session(args, timings)
    frame_seconds = []
    started = time.perf_counter_ns()
    for index, (frame_id, image) in enumerate(images):
        frame_started = time.perf_counter_ns()
        session.add_frame(image, timestamp_s=float(index), frame_id=frame_id)
        frame_seconds.append(
            (time.perf_counter_ns() - frame_started) / 1_000_000_000
        )
    finish_started = time.perf_counter_ns()
    result = session.finish()
    finish_seconds = (time.perf_counter_ns() - finish_started) / 1_000_000_000
    slam_seconds = (time.perf_counter_ns() - started) / 1_000_000_000
    sample = {
        "slam_seconds": slam_seconds,
        "frame_seconds": frame_seconds,
        "finish_seconds": finish_seconds,
        "stage_timings_inclusive": timings.to_dict(),
        "result": {
            "success": result.success,
            "failure_reasons": list(result.failure_reasons),
            "registered_frame_count": len(result.trajectory),
            "keyframe_count": len(result.keyframes),
            "map_point_count": len(result.map_points),
            "active_map_point_count": result.diagnostics.active_map_point_count,
            "feature_counts": [frame.feature_count for frame in result.frames],
            "match_counts": [frame.match_count for frame in result.frames],
            "inlier_counts": [frame.inlier_count for frame in result.frames],
        },
    }
    return sample, result


def _rotation_error_deg(actual: np.ndarray, expected: np.ndarray) -> float:
    value = float(np.clip((np.trace(actual @ expected.T) - 1.0) / 2.0, -1, 1))
    return math.degrees(math.acos(value))


def _direction_error_deg(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = actual / np.linalg.norm(actual)
    expected = expected / np.linalg.norm(expected)
    return math.degrees(math.acos(float(np.clip(actual @ expected, -1, 1))))


def _evaluate_result(result: Any, reference: dict[str, Any]) -> dict[str, Any]:
    poses = {pose.frame_id: pose for pose in result.trajectory}
    dataset_id = reference["dataset_id"]
    if dataset_id == "matterport360":
        basis = np.diag((1.0, 1.0, -1.0))
    elif dataset_id == "stanford2d3d":
        basis = np.diag((1.0, -1.0, 1.0))
    else:
        raise ValueError(f"unknown dataset coordinate frame: {dataset_id}")
    rotation_errors = []
    translation_errors = []
    for edge in reference["pairwise_edges"]:
        frame_a = edge["from_view_id"]
        frame_b = edge["to_view_id"]
        if frame_a not in poses or frame_b not in poses:
            continue
        pose_a = poses[frame_a]
        pose_b = poses[frame_b]
        rotation = pose_b.R @ pose_a.R.T
        translation = pose_b.R @ (pose_a.center - pose_b.center)
        translation /= np.linalg.norm(translation)
        rotation = basis @ rotation @ basis.T
        translation = basis @ translation
        target = edge["relative_transform"]
        rotation_errors.append(
            _rotation_error_deg(
                rotation, np.asarray(target["rotation_matrix"], dtype=np.float64)
            )
        )
        translation_errors.append(
            _direction_error_deg(
                translation,
                np.asarray(target["translation_direction"], dtype=np.float64),
            )
        )
    expected_frames = int(reference["member_count"])
    complete = len(poses) == expected_frames and len(rotation_errors) == len(
        reference["pairwise_edges"]
    )
    median_rotation = (
        float(np.median(rotation_errors)) if rotation_errors else None
    )
    median_translation = (
        float(np.median(translation_errors)) if translation_errors else None
    )
    return {
        "reference_opened_after_estimation": True,
        "complete_trajectory": complete,
        "evaluated_edge_count": len(rotation_errors),
        "median_rotation_error_deg": median_rotation,
        "max_rotation_error_deg": max(rotation_errors, default=None),
        "median_translation_direction_error_deg": median_translation,
        "max_translation_direction_error_deg": max(translation_errors, default=None),
        "strict_success": bool(
            result.success
            and complete
            and median_rotation is not None
            and median_translation is not None
            and median_rotation <= 5.0
            and median_translation <= 10.0
        ),
    }


def _worker(args: argparse.Namespace) -> dict[str, Any]:
    inputs = Path(args.inputs).resolve()
    evaluation_path = Path(args.evaluation).resolve()
    item = _load_selected_jsonl(inputs, args.set_id)
    if item.get("allowed_access") != "method_input_only":
        raise RuntimeError("input record does not declare method_input_only access")
    prepare_started = time.perf_counter_ns()
    images = _read_images(item)
    prepare_seconds = (time.perf_counter_ns() - prepare_started) / 1_000_000_000
    if args.mode == "warm":
        for _ in range(args.warmup):
            _run_session(args, images)
    samples = []
    for sample_index in range(args.repetitions):
        measured_started = time.perf_counter_ns()
        sample, result = _run_session(args, images)
        measured_seconds = (time.perf_counter_ns() - measured_started) / 1_000_000_000
        # Reference data is deliberately opened only after the estimate is complete.
        reference = _load_selected_jsonl(evaluation_path, args.set_id)
        sample.update(
            {
                "sample_index": sample_index,
                "mode": args.mode,
                "input_prepare_seconds": prepare_seconds,
                "measured_seconds": (
                    measured_seconds
                    if args.mode == "warm"
                    else measured_seconds + prepare_seconds
                ),
                "peak_rss_bytes_process_high_watermark": _rss_bytes(),
                "correctness": _evaluate_result(result, reference),
            }
        )
        samples.append(sample)
        del result
    return {
        "mode": args.mode,
        "warmup": args.warmup if args.mode == "warm" else 0,
        "samples": samples,
    }


def nearest_rank_percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    if not 0.0 < probability <= 1.0:
        raise ValueError("probability must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("values cannot be empty")
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "raw": numeric,
        "minimum": min(numeric),
        "median": statistics.median(numeric),
        "p95_nearest_rank": nearest_rank_percentile(numeric, 0.95),
        "maximum": max(numeric),
    }


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("samples cannot be empty")
    frame_count = samples[0]["result"]["registered_frame_count"]
    stage_names = sorted(
        {
            name
            for sample in samples
            for name in sample["stage_timings_inclusive"]
        }
    )
    return {
        "sample_count": len(samples),
        "all_strictly_correct": all(
            sample["correctness"]["strict_success"] for sample in samples
        ),
        "measured_seconds": summarize_values(
            [sample["measured_seconds"] for sample in samples]
        ),
        "slam_seconds": summarize_values(
            [sample["slam_seconds"] for sample in samples]
        ),
        "seconds_per_registered_frame": summarize_values(
            [sample["slam_seconds"] / frame_count for sample in samples]
        ),
        "registered_frames_per_second": summarize_values(
            [frame_count / sample["slam_seconds"] for sample in samples]
        ),
        "peak_rss_bytes_process_high_watermark": summarize_values(
            [sample["peak_rss_bytes_process_high_watermark"] for sample in samples]
        ),
        "stage_seconds_inclusive": {
            name: summarize_values(
                [
                    sample["stage_timings_inclusive"].get(name, {}).get(
                        "total_seconds", 0.0
                    )
                    for sample in samples
                ]
            )
            for name in stage_names
        },
        "accuracy": {
            "median_rotation_error_deg": summarize_values(
                [
                    sample["correctness"]["median_rotation_error_deg"]
                    for sample in samples
                ]
            ),
            "median_translation_direction_error_deg": summarize_values(
                [
                    sample["correctness"][
                        "median_translation_direction_error_deg"
                    ]
                    for sample in samples
                ]
            ),
        },
    }


def _command_value(command: list[str]) -> str | None:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    value = completed.stdout.strip()
    return value or None


def _source_fingerprint() -> dict[str, Any]:
    digest = hashlib.sha256()
    paths = sorted((REPOSITORY_ROOT / "panorai").rglob("*.py"))
    paths.extend(
        [
            REPOSITORY_ROOT / "pyproject.toml",
            Path(__file__).resolve(),
        ]
    )
    for path in paths:
        relative = path.relative_to(REPOSITORY_ROOT).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return {
        "git_commit": _command_value(["git", "rev-parse", "HEAD"]),
        "git_describe": _command_value(
            ["git", "describe", "--tags", "--always", "--dirty"]
        ),
        "git_status_porcelain": (_command_value(["git", "status", "--porcelain"]) or "").splitlines(),
        "panorai_python_and_benchmark_sha256": digest.hexdigest(),
    }


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_brand": _command_value(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "memory_bytes": _command_value(["sysctl", "-n", "hw.memsize"]),
        "panorai_version": panorai.__version__,
        "panorai_module": str(Path(panorai.__file__).resolve()),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "opencv_version": cv2.__version__,
    }


def _run_worker(arguments: list[str]) -> tuple[dict[str, Any], float]:
    started = time.perf_counter_ns()
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "worker", *arguments],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    wall = (time.perf_counter_ns() - started) / 1_000_000_000
    return json.loads(completed.stdout), wall


def _benchmark(args: argparse.Namespace) -> int:
    common = [
        "--inputs",
        str(Path(args.inputs).resolve()),
        "--evaluation",
        str(Path(args.evaluation).resolve()),
        "--set-id",
        args.set_id,
        "--face-size",
        str(args.face_size),
        "--max-features",
        str(args.max_features),
        "--max-num-trials",
        str(args.max_num_trials),
    ]
    batches = []
    warm, wall = _run_worker(
        common
        + [
            "--mode",
            "warm",
            "--warmup",
            str(args.warmup),
            "--repetitions",
            str(args.repetitions),
        ]
    )
    warm["worker_wall_seconds"] = wall
    batches.append(warm)
    for _ in range(args.cold_repetitions):
        cold, wall = _run_worker(
            common
            + ["--mode", "cold-process", "--warmup", "0", "--repetitions", "1"]
        )
        cold["worker_wall_seconds"] = wall
        batches.append(cold)
    warm_samples = batches[0]["samples"]
    cold_samples = [batch["samples"][0] for batch in batches[1:]]
    report = {
        "schema": SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": {
            "clock": "time.perf_counter_ns",
            "percentile": "nearest-rank",
            "warm": "inputs decoded once; complete sessions discarded as warm-up; each measured repetition creates a fresh SLAM session in the same process",
            "cold_process": "fresh process per sample; measured_seconds includes input hashing/decoding, session construction, add_frame calls, and finish",
            "peak_memory": "resource.getrusage process high-watermark; warm values include imports and warm-up and are monotonic within the worker",
            "stage_timing": "inclusive; stage totals overlap and must not be summed",
            "reference_policy": "evaluation JSONL is opened only after each estimate completes",
            "sample_retention": "no outlier rejection",
        },
        "source": _source_fingerprint(),
        "environment": _environment(),
        "inputs": {
            "method_inputs": str(Path(args.inputs).resolve()),
            "method_inputs_sha256": _sha256(Path(args.inputs).resolve()),
            "evaluation": str(Path(args.evaluation).resolve()),
            "evaluation_sha256": _sha256(Path(args.evaluation).resolve()),
            "set_id": args.set_id,
        },
        "configuration": {
            "preset": "sift-flann",
            "face_sampler": "icosahedron",
            "face_fov_deg": 80.0,
            "face_size": args.face_size,
            "edge_margin_px": 16,
            "max_features": args.max_features,
            "ratio_test": 0.75,
            "max_angular_error_deg": 0.5,
            "max_num_trials": args.max_num_trials,
            "local_ba_max_points": 512,
            "local_ba_max_nfev": 50,
        },
        "warmup_sessions": args.warmup,
        "warm_repetitions": args.repetitions,
        "cold_process_repetitions": args.cold_repetitions,
        "raw_batches": batches,
        "summary": {
            "warm": summarize_samples(warm_samples),
            "cold_process": summarize_samples(cold_samples),
        },
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "summary": report["summary"]}, indent=2))
    return 0 if all(
        section["all_strictly_correct"] for section in report["summary"].values()
    ) else 2


def _positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    benchmark = commands.add_parser("run")
    benchmark.add_argument("--inputs", required=True)
    benchmark.add_argument("--evaluation", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--set-id", default="multiview-neighbor-0379")
    benchmark.add_argument("--warmup", type=_positive, default=1)
    benchmark.add_argument("--repetitions", type=_positive, default=5)
    benchmark.add_argument("--cold-repetitions", type=_positive, default=3)
    benchmark.add_argument("--face-size", type=_positive, default=512)
    benchmark.add_argument("--max-features", type=_positive, default=4096)
    benchmark.add_argument("--max-num-trials", type=_positive, default=500)
    benchmark.set_defaults(func=_benchmark)

    worker = commands.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--inputs", required=True)
    worker.add_argument("--evaluation", required=True)
    worker.add_argument("--set-id", required=True)
    worker.add_argument("--mode", choices=("warm", "cold-process"), required=True)
    worker.add_argument("--warmup", type=int, required=True)
    worker.add_argument("--repetitions", type=_positive, required=True)
    worker.add_argument("--face-size", type=_positive, required=True)
    worker.add_argument("--max-features", type=_positive, required=True)
    worker.add_argument("--max-num-trials", type=_positive, required=True)
    worker.set_defaults(func=_worker)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = args.func(args)
    if args.command == "worker":
        print(json.dumps(result, sort_keys=True))
    else:
        raise SystemExit(result)


if __name__ == "__main__":
    main()
