#!/usr/bin/env python3
"""Metadata-blind real-ERP replay for the incremental spherical SLAM API.

Run ``estimate`` first.  It reads only the method-input JSONL and referenced
RGB files, then atomically writes a read-only prediction.  Run ``evaluate`` in
a separate command after that freeze; only this second command accepts the
evaluation JSONL containing reference geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import stat
import sys
import time
from typing import Any

import cv2
import numpy as np

import panorai
from panorai.estimators import RelativePoseOptions, SphericalRelativePoseEstimator
from panorai.features import SphericalFeaturePipeline
from panorai.slam import SphericalIncrementalSLAM, SphericalIncrementalSLAMOptions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


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


def _read_rgb(member: dict[str, Any]) -> np.ndarray:
    path = Path(member["rgb_path"]).resolve()
    if sha256(path) != member["rgb_sha256"]:
        raise RuntimeError(f"RGB checksum mismatch: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read RGB: {path}")
    expected = tuple(int(value) for value in member["rgb_shape"])
    if image.shape != expected:
        raise RuntimeError(
            f"RGB shape mismatch for {path}: {image.shape} != {expected}"
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _pose_dict(pose: Any | None) -> dict[str, Any] | None:
    if pose is None:
        return None
    return {
        "R_world_to_panorama": pose.rotation_world_to_camera,
        "center_world": pose.center_world,
    }


def _frame_dict(frame: Any) -> dict[str, Any]:
    return {
        "frame_id": frame.frame_id,
        "timestamp_s": frame.timestamp_s,
        "state": frame.state,
        "pose": _pose_dict(frame.pose),
        "reference_frame_id": frame.reference_frame_id,
        "is_keyframe": frame.is_keyframe,
        "feature_count": frame.feature_count,
        "match_count": frame.match_count,
        "inlier_count": frame.inlier_count,
        "map_correspondence_count": frame.map_correspondence_count,
        "median_parallax_deg": frame.median_parallax_deg,
        "local_map_point_count": frame.local_map_point_count,
        "reasons": frame.reasons,
    }


def _build_session(args: argparse.Namespace) -> SphericalIncrementalSLAM:
    pipeline = SphericalFeaturePipeline.from_preset(
        args.preset,
        face_sampler=args.face_sampler,
        face_fov_deg=args.face_fov_deg,
        face_shape_hw=(args.face_size, args.face_size),
        edge_margin_px=args.edge_margin_px,
        max_features=args.max_features,
        ratio_test=args.ratio_test,
    )
    estimator = SphericalRelativePoseEstimator(
        RelativePoseOptions(
            max_angular_error_deg=args.max_angular_error_deg,
            min_num_trials=args.min_num_trials,
            max_num_trials=args.max_num_trials,
            min_inliers=args.min_inliers,
            local_optimization_steps=args.local_optimization_steps,
            random_seed=args.random_seed,
        )
    )
    options = SphericalIncrementalSLAMOptions(
        min_frame_features=args.min_frame_features,
        min_pair_matches=args.min_pair_matches,
        min_map_correspondences=args.min_map_correspondences,
        keyframe_min_interval_s=0.0,
        keyframe_max_interval_s=1.0,
        local_ba_max_points=args.local_ba_max_points,
        local_ba_max_nfev=args.local_ba_max_nfev,
        edge_admission="accepted",
    )
    return SphericalIncrementalSLAM(
        feature_pipeline=pipeline,
        relative_pose_estimator=estimator,
        options=options,
    )


def estimate(args: argparse.Namespace) -> None:
    inputs = Path(args.inputs).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prediction: {output}")
    item = _load_selected_jsonl(inputs, args.set_id)
    if item.get("allowed_access") != "method_input_only":
        raise RuntimeError("input record does not declare method_input_only access")
    members = sorted(item["members"], key=lambda member: int(member["order"]))
    if len(members) != int(item["member_count"]):
        raise RuntimeError("member_count does not match members")

    session = _build_session(args)
    online = []
    started = time.perf_counter()
    for index, member in enumerate(members):
        image = _read_rgb(member)
        state = session.add_frame(
            image,
            timestamp_s=float(index),
            frame_id=member["view_id"],
        )
        online.append(_frame_dict(state))
    result = session.finish()
    elapsed = time.perf_counter() - started

    prediction = {
        "schema": "panorai-incremental-erp-slam-real-replay/v1",
        "metadata_access": "none; method-input JSONL and referenced RGB only",
        "frozen_before_reference_access": True,
        "panorai_version": panorai.__version__,
        "panorai_module": str(Path(panorai.__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "opencv_version": cv2.__version__,
        "input_path": str(inputs),
        "input_sha256": sha256(inputs),
        "set_id": item["set_id"],
        "dataset_id": item["dataset_id"],
        "spatial_group_id": item["spatial_group_id"],
        "member_ids": [member["view_id"] for member in members],
        "member_rgb_sha256": {
            member["view_id"]: member["rgb_sha256"] for member in members
        },
        "elapsed_seconds": elapsed,
        "method": session.describe(),
        "online_frames": online,
        "final": {
            "success": result.success,
            "failure_reasons": result.failure_reasons,
            "poses": {
                pose.frame_id: _pose_dict(pose) for pose in result.trajectory
            },
            "keyframe_ids": [keyframe.frame_id for keyframe in result.keyframes],
            "map_point_count": len(result.map_points),
            "active_map_point_count": result.diagnostics.active_map_point_count,
            "diagnostics": result.diagnostics.to_dict(),
            "describe": result.describe(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(prediction, indent=2, sort_keys=True, default=_json_default)
            + "\n"
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    output.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(
        json.dumps(
            {
                "prediction": str(output),
                "prediction_sha256": sha256(output),
                "success": result.success,
                "tracked_frames": len(result.trajectory),
                "input_frames": len(members),
                "active_map_points": result.diagnostics.active_map_point_count,
                "elapsed_seconds": elapsed,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _rotation_error_deg(actual: np.ndarray, expected: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(actual @ expected.T) - 1.0) / 2.0, -1, 1))
    return math.degrees(math.acos(cosine))


def _direction_error_deg(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = actual / np.linalg.norm(actual)
    expected = expected / np.linalg.norm(expected)
    return math.degrees(math.acos(float(np.clip(actual @ expected, -1, 1))))


def _change_of_basis(dataset_id: str) -> np.ndarray:
    if dataset_id == "matterport360":
        return np.diag((1.0, 1.0, -1.0))
    if dataset_id == "stanford2d3d":
        return np.diag((1.0, -1.0, 1.0))
    raise ValueError(f"unknown dataset coordinate frame: {dataset_id}")


def _relative_from_global(
    poses: dict[str, dict[str, Any]], panorama_a: str, panorama_b: str
) -> tuple[np.ndarray, np.ndarray]:
    pose_a = poses[panorama_a]
    pose_b = poses[panorama_b]
    rotation_a = np.asarray(pose_a["R_world_to_panorama"], dtype=np.float64)
    rotation_b = np.asarray(pose_b["R_world_to_panorama"], dtype=np.float64)
    center_a = np.asarray(pose_a["center_world"], dtype=np.float64)
    center_b = np.asarray(pose_b["center_world"], dtype=np.float64)
    rotation = rotation_b @ rotation_a.T
    translation = rotation_b @ (center_a - center_b)
    norm = float(np.linalg.norm(translation))
    if norm <= 1e-12:
        raise ValueError("coincident reconstructed centers")
    return rotation, translation / norm


def evaluate(args: argparse.Namespace) -> None:
    prediction_path = Path(args.prediction).resolve()
    if prediction_path.stat().st_mode & 0o222:
        raise RuntimeError("prediction must be read-only before evaluation")
    prediction_hash = sha256(prediction_path)
    if args.prediction_sha256 and prediction_hash != args.prediction_sha256:
        raise RuntimeError("frozen prediction checksum mismatch")
    prediction = json.loads(prediction_path.read_text())
    if not prediction.get("frozen_before_reference_access"):
        raise RuntimeError("prediction lacks the pre-reference freeze declaration")

    reference = _load_selected_jsonl(
        Path(args.evaluation).resolve(), prediction["set_id"]
    )
    poses = prediction["final"]["poses"]
    basis = _change_of_basis(reference["dataset_id"])
    edges = []
    for edge in reference["pairwise_edges"]:
        frame_a = edge["from_view_id"]
        frame_b = edge["to_view_id"]
        if frame_a not in poses or frame_b not in poses:
            continue
        rotation, translation = _relative_from_global(poses, frame_a, frame_b)
        rotation = basis @ rotation @ basis.T
        translation = basis @ translation
        target = edge["relative_transform"]
        edges.append(
            {
                "edge_id": edge["edge_id"],
                "rotation_error_deg": _rotation_error_deg(
                    rotation, np.asarray(target["rotation_matrix"], dtype=np.float64)
                ),
                "translation_direction_error_deg": _direction_error_deg(
                    translation,
                    np.asarray(target["translation_direction"], dtype=np.float64),
                ),
                "baseline_m": edge["baseline_m"],
                "relation": edge["relation"],
            }
        )
    rotation_errors = [item["rotation_error_deg"] for item in edges]
    translation_errors = [item["translation_direction_error_deg"] for item in edges]
    complete = len(poses) == len(prediction["member_ids"]) and len(edges) == len(
        reference["pairwise_edges"]
    )
    report = {
        "schema": "panorai-incremental-erp-slam-real-evaluation/v1",
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_hash,
        "evaluation_path": str(Path(args.evaluation).resolve()),
        "set_id": prediction["set_id"],
        "method_success": prediction["final"]["success"],
        "complete_trajectory": complete,
        "registered_frame_count": len(poses),
        "expected_frame_count": len(prediction["member_ids"]),
        "evaluated_edge_count": len(edges),
        "median_rotation_error_deg": (
            float(np.median(rotation_errors)) if rotation_errors else None
        ),
        "max_rotation_error_deg": max(rotation_errors, default=None),
        "median_translation_direction_error_deg": (
            float(np.median(translation_errors)) if translation_errors else None
        ),
        "max_translation_direction_error_deg": max(translation_errors, default=None),
        "primary_success": bool(
            complete
            and rotation_errors
            and np.median(rotation_errors) <= 15.0
            and np.median(translation_errors) <= 30.0
        ),
        "strict_success": bool(
            complete
            and rotation_errors
            and np.median(rotation_errors) <= 5.0
            and np.median(translation_errors) <= 10.0
        ),
        "pairwise_edges": edges,
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inference = commands.add_parser("estimate")
    inference.add_argument("--inputs", required=True)
    inference.add_argument("--set-id", default="multiview-neighbor-0379")
    inference.add_argument("--output", required=True)
    inference.add_argument("--preset", default="sift-flann")
    inference.add_argument("--face-sampler", default="icosahedron")
    inference.add_argument("--face-fov-deg", type=float, default=80.0)
    inference.add_argument("--face-size", type=int, default=512)
    inference.add_argument("--edge-margin-px", type=int, default=16)
    inference.add_argument("--max-features", type=int, default=4096)
    inference.add_argument("--ratio-test", type=float, default=0.75)
    inference.add_argument("--max-angular-error-deg", type=float, default=0.5)
    inference.add_argument("--min-num-trials", type=int, default=32)
    inference.add_argument("--max-num-trials", type=int, default=500)
    inference.add_argument("--min-inliers", type=int, default=8)
    inference.add_argument("--local-optimization-steps", type=int, default=3)
    inference.add_argument("--random-seed", type=int, default=7)
    inference.add_argument("--min-frame-features", type=int, default=64)
    inference.add_argument("--min-pair-matches", type=int, default=20)
    inference.add_argument("--min-map-correspondences", type=int, default=8)
    inference.add_argument("--local-ba-max-points", type=int, default=512)
    inference.add_argument("--local-ba-max-nfev", type=int, default=50)
    inference.set_defaults(func=estimate)

    scoring = commands.add_parser("evaluate")
    scoring.add_argument("--prediction", required=True)
    scoring.add_argument("--prediction-sha256")
    scoring.add_argument("--evaluation", required=True)
    scoring.add_argument("--output", required=True)
    scoring.set_defaults(func=evaluate)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
