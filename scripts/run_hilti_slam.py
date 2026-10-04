#!/usr/bin/env python3
"""Metadata-blind Hilti cam0 replay and post-freeze trajectory evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterator

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o444)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def iter_cam0_frames(
    bag_directory: Path,
    *,
    sample_hz: float,
    start_offset_s: float,
    max_frames: int | None,
) -> Iterator[tuple[str, float, np.ndarray]]:
    """Yield decoded cam0 frames without opening evaluation metadata."""

    try:
        import cv2
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise ImportError(
            "Hilti replay requires `pip install 'panorai[slam]'`."
        ) from exc
    if sample_hz <= 0.0:
        raise ValueError("sample_hz must be positive")
    interval_ns = int(round(1e9 / sample_hz))
    topic = "/cam0/image_raw/compressed"
    first_timestamp: int | None = None
    next_timestamp: int | None = None
    emitted = 0
    with AnyReader([bag_directory]) as reader:
        connections = [item for item in reader.connections if item.topic == topic]
        if len(connections) != 1:
            raise RuntimeError(f"expected exactly one {topic!r} connection")
        for connection, bag_timestamp, rawdata in reader.messages(
            connections=connections
        ):
            if first_timestamp is None:
                first_timestamp = bag_timestamp
                next_timestamp = bag_timestamp + int(round(start_offset_s * 1e9))
            assert next_timestamp is not None
            if bag_timestamp < next_timestamp:
                continue
            message = reader.deserialize(rawdata, connection.msgtype)
            encoded = np.frombuffer(message.data, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"OpenCV failed to decode cam0 at {bag_timestamp}")
            timestamp_s = bag_timestamp / 1e9
            yield f"cam0-{bag_timestamp}", timestamp_s, image
            emitted += 1
            if max_frames is not None and emitted >= max_frames:
                return
            while next_timestamp <= bag_timestamp:
                next_timestamp += interval_ns


def estimate(args: argparse.Namespace) -> int:
    """Run SLAM and freeze its output without opening ground truth."""

    from panorai.estimators import RelativePoseOptions, SphericalRelativePoseEstimator
    from panorai.features import FeatureExtractorConfig, FeatureMatcherConfig
    from panorai.reconstruction import SphericalGlobalMapperOptions
    from panorai.slam import (
        EquidistantFisheyeCamera,
        SphericalSLAMOptions,
        SphericalVisualSLAM,
    )

    bag_directory = args.bag_directory.resolve()
    bag_path = bag_directory / "rosbag.db3"
    calibration_path = args.calibration.resolve()
    if not bag_path.is_file() or not calibration_path.is_file():
        raise FileNotFoundError("bag or calibration path does not exist")
    camera = EquidistantFisheyeCamera.from_kalibr_yaml(
        calibration_path,
        camera_id="cam0",
        max_theta_deg=args.max_theta_deg,
    )
    relative = SphericalRelativePoseEstimator(
        RelativePoseOptions(
            max_angular_error_deg=args.max_angular_error_deg,
            max_num_trials=args.max_num_trials,
            min_num_trials=min(32, args.max_num_trials),
            stability_trials=args.stability_trials,
            stability_ransac_trials=args.stability_ransac_trials,
            model_competition_trials=args.model_competition_trials,
            random_seed=args.random_seed,
        )
    )
    options = SphericalSLAMOptions(
        temporal_window=args.temporal_window,
        min_features_per_frame=args.min_features,
        min_matches_per_pair=args.min_matches,
        edge_margin_px=args.edge_margin_px,
        extractor=FeatureExtractorConfig(
            method="sift",
            max_features=args.max_features,
            edge_margin_px=0,
        ),
        matcher=FeatureMatcherConfig(
            method="flann",
            ratio_test=args.ratio_test,
            deduplicate_matches=False,
        ),
        mapper=SphericalGlobalMapperOptions(edge_admission=args.edge_admission),
    )
    slam = SphericalVisualSLAM(
        camera,
        options=options,
        relative_pose_estimator=relative,
    )
    for frame_id, timestamp_s, image in iter_cam0_frames(
        bag_directory,
        sample_hz=args.sample_hz,
        start_offset_s=args.start_offset_s,
        max_frames=args.max_frames,
    ):
        summary = slam.add_frame(
            image,
            timestamp_s=timestamp_s,
            frame_id=frame_id,
        )
        count = 0 if summary is None else summary.feature_count
        print(f"{frame_id} t={timestamp_s:.6f} features={count}", flush=True)
    result = slam.finish()
    payload = {
        "interface": "panorai-hilti-slam-replay/v1",
        "evaluation_state": "prediction-frozen-before-reference-access",
        "dataset": {
            "repository": "Hilti-Research/hilti-trimble-slam-challenge-2026",
            "sequence": "floor_2_2025-10-28_run_2",
            "license": "CC-BY-NC-SA-3.0",
            "bag_path": str(bag_path),
            "bag_size_bytes": bag_path.stat().st_size,
            "bag_sha256": _sha256(bag_path),
            "calibration_path": str(calibration_path),
            "calibration_sha256": _sha256(calibration_path),
            "camera": "cam0",
        },
        "sampling": {
            "sample_hz": args.sample_hz,
            "start_offset_s": args.start_offset_s,
            "max_frames": args.max_frames,
        },
        "result": result.describe(),
        "poses": [
            {
                "frame_id": pose.frame_id,
                "timestamp_s": pose.timestamp_s,
                "rotation_world_to_camera": pose.R.tolist(),
                "center_world": pose.center.tolist(),
            }
            for pose in result.poses
        ],
        "keyframes": [
            {
                "frame_id": frame.frame_id,
                "timestamp_s": frame.timestamp_s,
                "feature_count": frame.feature_count,
                "checksum_sha256": frame.checksum_sha256,
            }
            for frame in result.keyframes
        ],
        "reconstruction": (
            None if result.reconstruction is None else result.reconstruction.describe()
        ),
        "pairwise_edges": (
            []
            if result.reconstruction is None
            else [
                {
                    "pair": list(edge.pair),
                    "num_inliers": edge.pose.num_inliers,
                    "accepted": edge.accepted,
                    "quality": edge.pose.quality_report.to_dict(),
                }
                for edge in result.reconstruction.pairwise_edges
            ]
        ),
    }
    _atomic_json(args.output.resolve(), payload)
    print(
        f"frozen={args.output.resolve()} success={result.success} poses={len(result.poses)}"
    )
    return 0 if result.success else 2


def _load_tum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path, comments="#", dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError("ground truth must use TUM timestamp tx ty tz qx qy qz qw")
    return values[:, 0], values[:, 1:4]


def _load_tum_poses(path: Path) -> np.ndarray:
    values = np.loadtxt(path, comments="#", dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError("ground truth must use TUM timestamp tx ty tz qx qy qz qw")
    return values


def _reference_motion(timestamps: np.ndarray, groundtruth: np.ndarray) -> dict:
    if len(timestamps) < 2:
        return {"step_count": 0}
    reference_time = groundtruth[:, 0]
    indices = np.searchsorted(reference_time, timestamps)
    indices = np.clip(indices, 1, len(reference_time) - 1)
    left = indices - 1
    choose_right = np.abs(reference_time[indices] - timestamps) < np.abs(
        reference_time[left] - timestamps
    )
    nearest = np.where(choose_right, indices, left)
    positions = groundtruth[nearest, 1:4]
    quaternions = groundtruth[nearest, 4:8]
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    baselines = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    dots = np.abs(np.sum(quaternions[1:] * quaternions[:-1], axis=1))
    rotations = np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))
    intervals = np.diff(timestamps)

    def stats(values: np.ndarray) -> dict[str, float]:
        return {
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "max": float(values.max()),
        }

    return {
        "step_count": len(baselines),
        "interval_s": stats(intervals),
        "baseline_m": stats(baselines),
        "rotation_deg": stats(rotations),
    }


def _umeyama(
    source: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must have equal shape (N, 3)")
    if len(source) < 3:
        raise ValueError("at least three poses are required for Sim(3) alignment")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if variance <= np.finfo(np.float64).eps:
        raise ValueError("predicted trajectory has zero spatial variance")
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def evaluate(args: argparse.Namespace) -> int:
    """Open reference data only after a read-only prediction exists."""

    prediction_path = args.prediction.resolve()
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    if payload.get("evaluation_state") != "prediction-frozen-before-reference-access":
        raise ValueError("prediction does not declare the metadata-blind freeze state")
    poses = payload.get("poses", [])
    groundtruth = _load_tum_poses(args.groundtruth.resolve())
    gt_time, gt_position = groundtruth[:, 0], groundtruth[:, 1:4]
    capture_items = payload.get("keyframes") or poses
    capture_times = np.asarray(
        [item["timestamp_s"] for item in capture_items], dtype=np.float64
    )
    capture_motion = _reference_motion(capture_times, groundtruth)
    if len(poses) < 3:
        report = {
            "interface": "panorai-hilti-slam-evaluation/v1",
            "prediction_path": str(prediction_path),
            "prediction_sha256": _sha256(prediction_path),
            "groundtruth_path": str(args.groundtruth.resolve()),
            "groundtruth_sha256": _sha256(args.groundtruth.resolve()),
            "matched_pose_count": len(poses),
            "capture_reference": capture_motion,
            "sim3": None,
            "ate_m": None,
        }
        _atomic_json(args.output.resolve(), report)
        print(json.dumps({"ate_m": None, "capture_reference": capture_motion}))
        return 0
    timestamps = np.asarray([item["timestamp_s"] for item in poses], dtype=np.float64)
    predicted = np.asarray([item["center_world"] for item in poses], dtype=np.float64)
    inside = (timestamps >= gt_time[0]) & (timestamps <= gt_time[-1])
    timestamps = timestamps[inside]
    predicted = predicted[inside]
    if len(predicted) < 3:
        raise ValueError("fewer than three predicted poses overlap ground truth")
    reference = np.column_stack(
        [np.interp(timestamps, gt_time, gt_position[:, axis]) for axis in range(3)]
    )
    scale, rotation, translation = _umeyama(predicted, reference)
    aligned = (scale * (rotation @ predicted.T)).T + translation
    errors = np.linalg.norm(aligned - reference, axis=1)
    report = {
        "interface": "panorai-hilti-slam-evaluation/v1",
        "prediction_path": str(prediction_path),
        "prediction_sha256": _sha256(prediction_path),
        "groundtruth_path": str(args.groundtruth.resolve()),
        "groundtruth_sha256": _sha256(args.groundtruth.resolve()),
        "matched_pose_count": len(errors),
        "capture_reference": capture_motion,
        "sim3": {
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
        },
        "ate_m": {
            "rmse": float(np.sqrt(np.mean(errors * errors))),
            "median": float(np.median(errors)),
            "p90": float(np.percentile(errors, 90)),
            "max": float(errors.max()),
        },
    }
    _atomic_json(args.output.resolve(), report)
    print(json.dumps(report["ate_m"], sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    estimation = subparsers.add_parser("estimate")
    estimation.add_argument("--bag-directory", type=Path, required=True)
    estimation.add_argument("--calibration", type=Path, required=True)
    estimation.add_argument("--output", type=Path, required=True)
    estimation.add_argument("--sample-hz", type=float, default=1.0)
    estimation.add_argument("--start-offset-s", type=float, default=5.0)
    estimation.add_argument("--max-frames", type=int, default=30)
    estimation.add_argument("--max-theta-deg", type=float, default=98.0)
    estimation.add_argument("--temporal-window", type=int, default=2)
    estimation.add_argument(
        "--edge-admission", choices=("accepted", "successful"), default="accepted"
    )
    estimation.add_argument("--min-features", type=int, default=128)
    estimation.add_argument("--min-matches", type=int, default=20)
    estimation.add_argument("--max-features", type=int, default=2048)
    estimation.add_argument("--edge-margin-px", type=int, default=16)
    estimation.add_argument("--ratio-test", type=float, default=0.75)
    estimation.add_argument("--max-angular-error-deg", type=float, default=1.0)
    estimation.add_argument("--max-num-trials", type=int, default=512)
    estimation.add_argument("--stability-trials", type=int, default=3)
    estimation.add_argument("--stability-ransac-trials", type=int, default=24)
    estimation.add_argument("--model-competition-trials", type=int, default=64)
    estimation.add_argument("--random-seed", type=int, default=20261003)
    estimation.set_defaults(func=estimate)
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--prediction", type=Path, required=True)
    evaluation.add_argument("--groundtruth", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.set_defaults(func=evaluate)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
