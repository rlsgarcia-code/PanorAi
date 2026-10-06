#!/usr/bin/env python3
"""Metadata-separated real-pair benchmark for spherical relative pose."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable

import numpy as np
import scipy
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panorai.estimators import (  # noqa: E402
    RelativePoseOptions,
    SphericalRelativePoseEstimator,
)
from panorai.features import (  # noqa: E402
    FeatureExtractorConfig,
    FeatureMatcherConfig,
    OpenCVFeatureBackend,
)
from panorai.slam import EquidistantFisheyeCamera  # noqa: E402
from scripts.run_hilti_slam import iter_cam0_frames  # noqa: E402


PREDICTION_SCHEMA = "panorai-real-pair-relative-pose-prediction/v1"
EVALUATION_SCHEMA = "panorai-real-pair-relative-pose-evaluation/v1"
DATASET_NAME = "Hilti-Trimble-Oxford floor_2_2025-10-28_run_2"
BASELINE = "count-first"
_PANORAI_FROM_OPENCV = np.diag((1.0, -1.0, 1.0))


@dataclass(frozen=True)
class Variant:
    name: str
    ranking: str = "count-first"
    refit_steps: int = 0
    decoupled: bool = False


VARIANTS = (
    Variant(BASELINE),
    Variant("count-first-refit", refit_steps=100),
    Variant("msac-first", ranking="msac-first"),
    Variant("msac-first-refit", ranking="msac-first", refit_steps=100),
    Variant("decoupled-guarded", decoupled=True),
)


@dataclass(slots=True)
class FrameFeatures:
    frame_id: str
    timestamp_s: float
    checksum_sha256: str
    pixels_xy: np.ndarray
    bearings_xyz: np.ndarray
    responses: np.ndarray
    descriptors: np.ndarray
    descriptor_type: str
    descriptor_metric: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _atomic_json(path: Path, payload: dict[str, Any], *, readonly: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if readonly:
            path.chmod(0o444)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _parse_offsets(value: str) -> tuple[float, ...]:
    try:
        offsets = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "offsets must be comma-separated numbers"
        ) from exc
    if not offsets or any(not math.isfinite(item) or item < 0.0 for item in offsets):
        raise argparse.ArgumentTypeError("offsets must be finite and non-negative")
    if tuple(sorted(offsets)) != offsets or len(set(offsets)) != len(offsets):
        raise argparse.ArgumentTypeError("offsets must be unique and increasing")
    return offsets


def _selected_indices(
    offsets_s: tuple[float, ...], sample_hz: float, frames_per_window: int
) -> tuple[dict[int, tuple[int, int]], int, float]:
    if sample_hz <= 0.0:
        raise ValueError("sample_hz must be positive")
    if frames_per_window < 2:
        raise ValueError("frames_per_window must be at least two")
    base = offsets_s[0]
    selected: dict[int, tuple[int, int]] = {}
    for window, offset in enumerate(offsets_s):
        start = int(round((offset - base) * sample_hz))
        reconstructed = base + start / sample_hz
        if not math.isclose(reconstructed, offset, abs_tol=1e-9):
            raise ValueError("each offset must lie on the requested sample grid")
        for within in range(frames_per_window):
            index = start + within
            if index in selected:
                raise ValueError("sampling windows must not overlap")
            selected[index] = (window, within)
    return selected, max(selected) + 1, base


def _opencv_gray(image: np.ndarray) -> np.ndarray:
    import cv2

    values = np.asarray(image)
    if values.dtype != np.uint8 or values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("Hilti benchmark expects uint8 BGR images")
    return np.ascontiguousarray(cv2.cvtColor(values, cv2.COLOR_BGR2GRAY))


def _extract(
    image: np.ndarray,
    frame_id: str,
    timestamp_s: float,
    *,
    camera: EquidistantFisheyeCamera,
    backend: OpenCVFeatureBackend,
    extractor: FeatureExtractorConfig,
    support_mask: np.ndarray,
) -> FrameFeatures:
    detected = backend.detect_and_describe(
        _opencv_gray(image), support_mask.astype(np.uint8) * 255, extractor
    )
    pixels = np.asarray(detected["pixels_xy"], dtype=np.float64)
    projection = camera.pixels_to_rays(pixels)
    ix = np.floor(pixels[:, 0] + 0.5).astype(np.int64)
    iy = np.floor(pixels[:, 1] + 0.5).astype(np.int64)
    inside = (
        (ix >= 0)
        & (ix < support_mask.shape[1])
        & (iy >= 0)
        & (iy < support_mask.shape[0])
    )
    supported = np.zeros(len(pixels), dtype=bool)
    supported[inside] = support_mask[iy[inside], ix[inside]]
    keep = np.flatnonzero(projection.valid & supported)
    metadata = detected["metadata"]
    return FrameFeatures(
        frame_id=frame_id,
        timestamp_s=float(timestamp_s),
        checksum_sha256=hashlib.sha256(
            np.ascontiguousarray(image).view(np.uint8)
        ).hexdigest(),
        pixels_xy=pixels[keep],
        bearings_xyz=np.asarray(projection.rays_xyz[keep], dtype=np.float64),
        responses=np.asarray(detected["responses"][keep], dtype=np.float32),
        descriptors=np.asarray(detected["descriptors"][keep]),
        descriptor_type=str(metadata["type"]),
        descriptor_metric=str(metadata["metric"]),
    )


def _match(
    left: FrameFeatures,
    right: FrameFeatures,
    *,
    backend: OpenCVFeatureBackend,
    matcher: FeatureMatcherConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if left.descriptor_type != right.descriptor_type:
        raise RuntimeError("descriptor type changed between real frames")
    raw, _ = backend.match(
        left.descriptors, right.descriptors, left.descriptor_metric, matcher
    )
    chosen: list[dict[str, Any]] = []
    used_query: set[int] = set()
    used_train: set[int] = set()
    for item in sorted(
        raw,
        key=lambda value: (
            value["distance"],
            value["query_idx"],
            value["train_idx"],
        ),
    ):
        query = int(item["query_idx"])
        train = int(item["train_idx"])
        if query in used_query or train in used_train:
            continue
        used_query.add(query)
        used_train.add(train)
        chosen.append(item)
    chosen.sort(key=lambda value: (value["query_idx"], value["train_idx"]))
    indices_a = np.asarray([item["query_idx"] for item in chosen], dtype=np.int64)
    indices_b = np.asarray([item["train_idx"] for item in chosen], dtype=np.int64)
    distances = np.asarray([item["distance"] for item in chosen], dtype=np.float32)
    return left.bearings_xyz[indices_a], right.bearings_xyz[indices_b], distances


def _options(
    seed: int, variant: Variant, args: argparse.Namespace
) -> RelativePoseOptions:
    return RelativePoseOptions(
        max_angular_error_deg=args.max_angular_error_deg,
        max_num_trials=args.max_num_trials,
        min_num_trials=min(32, args.max_num_trials),
        stability_trials=args.stability_trials,
        stability_ransac_trials=args.stability_ransac_trials,
        model_competition_trials=args.model_competition_trials,
        random_seed=seed,
        hypothesis_ranking=variant.ranking,
        nonminimal_refit_max_steps=variant.refit_steps,
        pose_refinement_method="decoupled" if variant.decoupled else "joint",
        decoupled_rotation_trials=args.decoupled_rotation_trials,
        decoupled_refit_max_steps=args.decoupled_refit_max_steps,
        decoupled_translation_min_score_margin=args.decoupled_min_margin,
    )


def _predict_one(
    bearings_a: np.ndarray,
    bearings_b: np.ndarray,
    *,
    seed: int,
    variant: Variant,
    args: argparse.Namespace,
) -> dict[str, Any]:
    options = _options(seed, variant, args)
    started = time.perf_counter_ns()
    result = SphericalRelativePoseEstimator(options).estimate(bearings_a, bearings_b)
    runtime_ms = (time.perf_counter_ns() - started) / 1e6
    if result is None:
        return {
            "returned": False,
            "accepted": False,
            "runtime_ms": runtime_ms,
            "options": options.to_dict(),
        }
    report = result.decoupled_pose_report
    return {
        "returned": True,
        "accepted": bool(result.quality_report.accepted),
        "rejection_reasons": list(result.quality_report.rejection_reasons),
        "rotation_2_from_1": result.R.tolist(),
        "translation_2_from_1": result.t.tolist(),
        "essential_matrix": result.essential_matrix.tolist(),
        "num_inliers": result.num_inliers,
        "num_trials": result.num_trials,
        "inlier_ratio": result.quality_report.inlier_ratio,
        "median_residual_deg": result.quality_report.median_residual_deg,
        "median_parallax_deg": result.median_parallax_deg,
        "cheirality_ratio": result.cheirality_ratio,
        "consensus_refit_steps": result.consensus_refit_steps,
        "decoupled": None if report is None else report.to_dict(),
        "runtime_ms": runtime_ms,
        "options": options.to_dict(),
    }


def _resolve_variants(names: Iterable[str]) -> tuple[Variant, ...]:
    lookup = {item.name: item for item in VARIANTS}
    selected = []
    for name in names:
        if name not in lookup:
            raise ValueError(f"unknown variant {name!r}; choose from {sorted(lookup)}")
        selected.append(lookup[name])
    if not selected:
        raise ValueError("at least one variant is required")
    return tuple(selected)


def estimate(args: argparse.Namespace) -> int:
    bag_directory = args.bag_directory.resolve()
    calibration = args.calibration.resolve()
    bag_path = bag_directory / "rosbag.db3"
    if not bag_path.is_file() or not calibration.is_file():
        raise FileNotFoundError("bag or calibration does not exist")
    variants = _resolve_variants(args.variants.split(","))
    selected, max_frames, base_offset = _selected_indices(
        args.offsets, args.sample_hz, args.frames_per_window
    )
    camera = EquidistantFisheyeCamera.from_kalibr_yaml(
        calibration, camera_id="cam0", max_theta_deg=args.max_theta_deg
    )
    backend = OpenCVFeatureBackend()
    backend.require_version("4.9.0")
    extractor = FeatureExtractorConfig(
        method="sift", max_features=args.max_features, edge_margin_px=0
    )
    matcher = FeatureMatcherConfig(
        method="flann", ratio_test=args.ratio_test, deduplicate_matches=False
    )
    support_mask = camera.valid_pixel_mask(edge_margin_px=args.edge_margin_px)
    frames: dict[tuple[int, int], FrameFeatures] = {}
    extraction_started = time.perf_counter_ns()
    for index, (frame_id, timestamp_s, image) in enumerate(
        iter_cam0_frames(
            bag_directory,
            sample_hz=args.sample_hz,
            start_offset_s=base_offset,
            max_frames=max_frames,
        )
    ):
        location = selected.get(index)
        if location is None:
            continue
        frame = _extract(
            image,
            frame_id,
            timestamp_s,
            camera=camera,
            backend=backend,
            extractor=extractor,
            support_mask=support_mask,
        )
        frames[location] = frame
        print(
            f"extract window={location[0]} frame={location[1]} "
            f"features={len(frame.bearings_xyz)}",
            flush=True,
        )
    extraction_ms = (time.perf_counter_ns() - extraction_started) / 1e6
    if len(frames) != len(selected):
        raise RuntimeError(
            f"decoded {len(frames)} selected frames; expected {len(selected)}"
        )

    pairs: list[dict[str, Any]] = []
    for window in range(len(args.offsets)):
        for within in range(args.frames_per_window - 1):
            left = frames[(window, within)]
            right = frames[(window, within + 1)]
            bearings_a, bearings_b, distances = _match(
                left, right, backend=backend, matcher=matcher
            )
            pair_index = len(pairs)
            predictions = {}
            for variant_index, variant in enumerate(variants):
                seed = args.random_seed + pair_index
                predictions[variant.name] = _predict_one(
                    bearings_a,
                    bearings_b,
                    seed=seed,
                    variant=variant,
                    args=args,
                )
                print(
                    f"pair={pair_index:02d} matches={len(bearings_a)} "
                    f"variant={variant.name} "
                    f"returned={predictions[variant.name]['returned']}",
                    flush=True,
                )
            pairs.append(
                {
                    "pair_index": pair_index,
                    "window_index": window,
                    "frame_index_in_window": within,
                    "frame_id_1": left.frame_id,
                    "frame_id_2": right.frame_id,
                    "timestamp_1_s": left.timestamp_s,
                    "timestamp_2_s": right.timestamp_s,
                    "image_1_sha256": left.checksum_sha256,
                    "image_2_sha256": right.checksum_sha256,
                    "feature_count_1": len(left.bearings_xyz),
                    "feature_count_2": len(right.bearings_xyz),
                    "match_count": len(bearings_a),
                    "descriptor_distance_median": (
                        float(np.median(distances)) if len(distances) else None
                    ),
                    "predictions": predictions,
                }
            )

    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        peak_rss *= 1024
    payload = {
        "schema": PREDICTION_SCHEMA,
        "evaluation_state": "prediction-frozen-before-reference-access",
        "dataset": {
            "name": DATASET_NAME,
            "repository": "Hilti-Research/hilti-trimble-slam-challenge-2026",
            "license": "CC-BY-NC-SA-3.0",
            "camera": "cam0",
            "bag_size_bytes": bag_path.stat().st_size,
            "bag_sha256": _sha256(bag_path),
            "calibration_sha256": _sha256(calibration),
        },
        "source": {
            "commit": _git_value("rev-parse", "HEAD"),
            "describe": _git_value("describe", "--tags", "--always", "--dirty"),
            "dirty": bool(_git_value("status", "--porcelain")),
            "relative_pose_sha256": _sha256(
                ROOT / "panorai/estimators/relative_pose.py"
            ),
            "benchmark_sha256": _sha256(Path(__file__)),
        },
        "configuration": {
            "sample_hz": args.sample_hz,
            "offsets_s": args.offsets,
            "frames_per_window": args.frames_per_window,
            "max_theta_deg": args.max_theta_deg,
            "edge_margin_px": args.edge_margin_px,
            "extractor": extractor.to_dict(),
            "matcher": matcher.to_dict(),
            "variants": [asdict(item) for item in variants],
            "random_seed": args.random_seed,
            "reference_access": "none in estimate subcommand",
            "protocol": (
                None
                if args.protocol is None
                else {
                    "name": args.protocol.name,
                    "sha256": _sha256(args.protocol.resolve()),
                }
            ),
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "opencv": backend.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "measurements": {
            "extraction_ms": extraction_ms,
            "peak_rss_bytes": peak_rss,
            "peak_rss_scope": "complete prediction process",
        },
        "pairs": pairs,
    }
    output = args.output.resolve()
    _atomic_json(output, payload, readonly=True)
    print(f"frozen={output} sha256={_sha256(output)} pairs={len(pairs)}")
    return 0


def _load_groundtruth(path: Path) -> tuple[np.ndarray, np.ndarray, Rotation]:
    values = np.loadtxt(path, comments="#", dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 8 or len(values) < 2:
        raise ValueError("ground truth must have TUM timestamp tx ty tz qx qy qz qw")
    if not np.all(np.diff(values[:, 0]) > 0.0):
        raise ValueError("ground-truth timestamps must increase strictly")
    quaternions = values[:, 4:8]
    norms = np.linalg.norm(quaternions, axis=1)
    if not np.all(np.isfinite(values)) or not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError("ground-truth poses must be finite with unit quaternions")
    return values[:, 0], values[:, 1:4], Rotation.from_quat(quaternions)


def _interpolate_reference(
    query_times: np.ndarray,
    reference_times: np.ndarray,
    positions: np.ndarray,
    rotations_world_from_camera: Rotation,
) -> tuple[np.ndarray, np.ndarray]:
    if np.any(query_times < reference_times[0]) or np.any(
        query_times > reference_times[-1]
    ):
        raise ValueError("prediction timestamps fall outside reference coverage")
    interpolated_positions = np.column_stack(
        [
            np.interp(query_times, reference_times, positions[:, axis])
            for axis in range(3)
        ]
    )
    interpolated_rotations = Slerp(reference_times, rotations_world_from_camera)(
        query_times
    ).as_matrix()
    return interpolated_positions, interpolated_rotations


def _relative_oracle(
    center_1_world: np.ndarray,
    rotation_world_from_camera_1: np.ndarray,
    center_2_world: np.ndarray,
    rotation_world_from_camera_2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_2_from_1_opencv = (
        rotation_world_from_camera_2.T @ rotation_world_from_camera_1
    )
    translation_2_from_1_opencv = rotation_world_from_camera_2.T @ (
        center_1_world - center_2_world
    )
    # Hilti/Kalibr uses the OpenCV optical basis (+X right, +Y down,
    # +Z forward). EquidistantFisheyeCamera deliberately emits PanorAi rays
    # in +X right, +Y up, +Z forward. P is an orthogonal change of basis; the
    # relative rotation is conjugated and the translation is left-multiplied.
    rotation_2_from_1 = (
        _PANORAI_FROM_OPENCV @ rotation_2_from_1_opencv @ _PANORAI_FROM_OPENCV
    )
    translation_2_from_1 = _PANORAI_FROM_OPENCV @ translation_2_from_1_opencv
    norm = float(np.linalg.norm(translation_2_from_1))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("reference pair has zero baseline")
    return rotation_2_from_1, translation_2_from_1 / norm


def _check_relative_oracle() -> None:
    rotation_1 = Rotation.from_euler("xyz", [11.0, -7.0, 4.0], degrees=True).as_matrix()
    rotation_2 = Rotation.from_euler("xyz", [-5.0, 9.0, 18.0], degrees=True).as_matrix()
    center_1 = np.array([0.3, -0.2, 0.5])
    center_2 = np.array([1.2, 0.4, -0.1])
    relative_rotation, relative_translation = _relative_oracle(
        center_1, rotation_1, center_2, rotation_2
    )
    point_world = np.array([3.0, -1.0, 4.0])
    point_1_opencv = rotation_1.T @ (point_world - center_1)
    point_2_opencv = rotation_2.T @ (point_world - center_2)
    point_1 = _PANORAI_FROM_OPENCV @ point_1_opencv
    point_2 = _PANORAI_FROM_OPENCV @ point_2_opencv
    metric_translation = _PANORAI_FROM_OPENCV @ (rotation_2.T @ (center_1 - center_2))
    if not np.allclose(
        point_2, relative_rotation @ point_1 + metric_translation, atol=1e-12
    ):
        raise AssertionError("relative-pose oracle does not map camera 1 to camera 2")
    if not np.allclose(
        relative_translation, metric_translation / np.linalg.norm(metric_translation)
    ):
        raise AssertionError(
            "relative translation direction convention is inconsistent"
        )


def _rotation_error_deg(estimated: np.ndarray, reference: np.ndarray) -> float:
    value = (np.trace(estimated @ reference.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def _direction_error_deg(estimated: np.ndarray, reference: np.ndarray) -> float:
    dot = float(np.dot(estimated, reference))
    return float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(values, quantile, method="higher"))


def _aggregate(rows: list[dict[str, Any]], variants: tuple[str, ...]) -> dict[str, Any]:
    baseline_by_pair = {
        int(row["pair_index"]): row for row in rows if row["variant"] == BASELINE
    }
    output: dict[str, Any] = {}
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        returned = [row for row in selected if row["returned"]]
        accepted = [row for row in returned if row["accepted"]]
        rotation = [float(row["rotation_error_deg"]) for row in returned]
        translation = [float(row["translation_error_deg"]) for row in returned]
        axis = [float(row["translation_axis_error_deg"]) for row in returned]
        accepted_rotation = [float(row["rotation_error_deg"]) for row in accepted]
        accepted_translation = [float(row["translation_error_deg"]) for row in accepted]
        strict = [
            row
            for row in returned
            if row["rotation_error_deg"] < 5.0 and row["translation_error_deg"] < 10.0
        ]
        high_precision = [
            row
            for row in returned
            if row["rotation_error_deg"] < 1.0 and row["translation_error_deg"] < 5.0
        ]
        paired_rotation_wins = paired_translation_wins = 0
        paired_rotation_losses = paired_translation_losses = 0
        if variant != BASELINE and baseline_by_pair:
            for row in returned:
                baseline = baseline_by_pair.get(int(row["pair_index"]))
                if baseline is None or not baseline["returned"]:
                    continue
                delta_r = float(row["rotation_error_deg"]) - float(
                    baseline["rotation_error_deg"]
                )
                delta_t = float(row["translation_error_deg"]) - float(
                    baseline["translation_error_deg"]
                )
                paired_rotation_wins += delta_r < -1e-9
                paired_rotation_losses += delta_r > 1e-9
                paired_translation_wins += delta_t < -1e-9
                paired_translation_losses += delta_t > 1e-9
        output[variant] = {
            "count": len(selected),
            "returned_count": len(returned),
            "accepted_count": len(accepted),
            "strict_count": len(strict),
            "high_precision_count": len(high_precision),
            "accepted_strict_count": sum(
                row["rotation_error_deg"] < 5.0 and row["translation_error_deg"] < 10.0
                for row in accepted
            ),
            "false_accept_count": sum(
                not (
                    row["rotation_error_deg"] < 5.0
                    and row["translation_error_deg"] < 10.0
                )
                for row in accepted
            ),
            "rotation_error_median_deg": float(np.median(rotation))
            if rotation
            else None,
            "rotation_error_p95_deg": _percentile(rotation, 0.95),
            "translation_error_median_deg": float(np.median(translation))
            if translation
            else None,
            "translation_error_p95_deg": _percentile(translation, 0.95),
            "translation_axis_error_median_deg": float(np.median(axis))
            if axis
            else None,
            "translation_axis_error_p95_deg": _percentile(axis, 0.95),
            "accepted_rotation_error_median_deg": (
                float(np.median(accepted_rotation)) if accepted_rotation else None
            ),
            "accepted_translation_error_median_deg": (
                float(np.median(accepted_translation)) if accepted_translation else None
            ),
            "runtime_median_ms": (
                float(np.median([float(row["runtime_ms"]) for row in selected]))
                if selected
                else None
            ),
            "paired_rotation_wins_vs_baseline": paired_rotation_wins,
            "paired_rotation_losses_vs_baseline": paired_rotation_losses,
            "paired_translation_wins_vs_baseline": paired_translation_wins,
            "paired_translation_losses_vs_baseline": paired_translation_losses,
            "decoupled_applied_count": sum(
                bool(row["decoupled_applied"]) for row in selected
            ),
            "decoupled_reasons": {
                reason: sum(row["decoupled_reason"] == reason for row in selected)
                for reason in sorted({str(row["decoupled_reason"]) for row in selected})
            },
        }
    return output


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# Real calibrated-pair relative-pose benchmark",
        "",
        "| Variant | Returned | Accepted | Strict | High precision | R median / P95 (deg) | t median / P95 (deg) | Axis median / P95 (deg) | Runtime median (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in summary["results"].items():
        lines.append(
            f"| {name} | {values['returned_count']}/{values['count']} | "
            f"{values['accepted_count']}/{values['count']} | "
            f"{values['strict_count']}/{values['count']} | "
            f"{values['high_precision_count']}/{values['count']} | "
            f"{values['rotation_error_median_deg']:.3f} / {values['rotation_error_p95_deg']:.3f} | "
            f"{values['translation_error_median_deg']:.3f} / {values['translation_error_p95_deg']:.3f} | "
            f"{values['translation_axis_error_median_deg']:.3f} / {values['translation_axis_error_p95_deg']:.3f} | "
            f"{values['runtime_median_ms']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Strict means `R < 5 deg` and oriented `t < 10 deg`. High precision means",
            "`R < 1 deg` and oriented `t < 5 deg`. Axis error ignores the sign of `t`.",
            "",
            "## Evidence separation",
            "",
            "The estimate command had no ground-truth argument and froze its output",
            "read-only. The evaluate command opened that prediction first and only then",
            "loaded the LiDAR-derived cam0 reference trajectory.",
            "",
            "## Reproducibility",
            "",
            f"- Prediction SHA-256: `{summary['prediction']['sha256']}`",
            f"- Ground-truth SHA-256: `{summary['reference']['groundtruth_sha256']}`",
            f"- Raw CSV SHA-256: `{summary['measurements']['raw_csv_sha256']}`",
            f"- Source commit: `{summary['source']['commit']}` (dirty: `{summary['source']['dirty']}`)",
            f"- Pairs: `{summary['measurements']['pair_count']}`",
            "",
            "## Scope",
            "",
            "This is one real sequence from one calibrated central fisheye lens. Pairs",
            "within a window are temporally correlated. The result is descriptive and",
            "does not justify a default-policy change by itself.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate(args: argparse.Namespace) -> int:
    _check_relative_oracle()
    prediction_path = args.prediction.resolve()
    mode = prediction_path.stat().st_mode
    if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("prediction must be read-only before evaluation")
    prediction_sha = _sha256(prediction_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    if prediction.get("schema") != PREDICTION_SCHEMA:
        raise ValueError("unsupported prediction schema")
    if (
        prediction.get("evaluation_state")
        != "prediction-frozen-before-reference-access"
    ):
        raise ValueError("prediction was not frozen before reference access")
    pairs = prediction.get("pairs", [])
    if not pairs:
        raise ValueError("prediction contains no pairs")

    groundtruth = args.groundtruth.resolve()
    reference_times, positions, rotations = _load_groundtruth(groundtruth)
    query_times = np.asarray(
        [
            value
            for pair in pairs
            for value in (pair["timestamp_1_s"], pair["timestamp_2_s"])
        ],
        dtype=np.float64,
    )
    interpolated_positions, interpolated_rotations = _interpolate_reference(
        query_times, reference_times, positions, rotations
    )

    rows: list[dict[str, Any]] = []
    variant_names = tuple(
        item["name"] for item in prediction["configuration"]["variants"]
    )
    for pair_index, pair in enumerate(pairs):
        center_1, center_2 = interpolated_positions[2 * pair_index : 2 * pair_index + 2]
        rotation_1, rotation_2 = interpolated_rotations[
            2 * pair_index : 2 * pair_index + 2
        ]
        reference_rotation, reference_translation = _relative_oracle(
            center_1, rotation_1, center_2, rotation_2
        )
        baseline_m = float(np.linalg.norm(center_2 - center_1))
        reference_rotation_deg = _rotation_error_deg(reference_rotation, np.eye(3))
        for variant in variant_names:
            result = pair["predictions"][variant]
            base = {
                "pair_index": pair_index,
                "window_index": pair["window_index"],
                "frame_index_in_window": pair["frame_index_in_window"],
                "variant": variant,
                "timestamp_1_s": pair["timestamp_1_s"],
                "timestamp_2_s": pair["timestamp_2_s"],
                "interval_s": pair["timestamp_2_s"] - pair["timestamp_1_s"],
                "reference_baseline_m": baseline_m,
                "reference_rotation_deg": reference_rotation_deg,
                "match_count": pair["match_count"],
                "returned": bool(result["returned"]),
                "accepted": bool(result["accepted"]),
                "num_inliers": int(result.get("num_inliers", 0)),
                "consensus_refit_steps": int(result.get("consensus_refit_steps", 0)),
                "decoupled_applied": bool(
                    result.get("decoupled") and result["decoupled"]["applied"]
                ),
                "decoupled_reason": (
                    "not-requested"
                    if result.get("decoupled") is None
                    else result["decoupled"]["reason"]
                ),
                "runtime_ms": float(result["runtime_ms"]),
            }
            if not result["returned"]:
                rows.append(
                    {
                        **base,
                        "rotation_error_deg": "",
                        "translation_error_deg": "",
                        "translation_axis_error_deg": "",
                    }
                )
                continue
            estimated_rotation = np.asarray(
                result["rotation_2_from_1"], dtype=np.float64
            )
            estimated_translation = np.asarray(
                result["translation_2_from_1"], dtype=np.float64
            )
            rotation_error = _rotation_error_deg(estimated_rotation, reference_rotation)
            translation_error = _direction_error_deg(
                estimated_translation, reference_translation
            )
            rows.append(
                {
                    **base,
                    "rotation_error_deg": rotation_error,
                    "translation_error_deg": translation_error,
                    "translation_axis_error_deg": min(
                        translation_error, 180.0 - translation_error
                    ),
                }
            )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        peak_rss *= 1024
    source = prediction["source"]
    summary = {
        "schema": EVALUATION_SCHEMA,
        "prediction": {
            "schema": prediction["schema"],
            "sha256": prediction_sha,
            "frozen_before_reference_access": True,
        },
        "reference": {
            "dataset": DATASET_NAME,
            "camera_pose_convention": "cam0-to-map (R_world_from_camera_opencv, center_world)",
            "relative_pose_convention": "camera-2-from-camera-1",
            "frame_basis_conversion": "P=diag(1,-1,1): OpenCV +Y down to PanorAi +Y up",
            "interpolation": "linear position plus quaternion SLERP",
            "groundtruth_sha256": _sha256(groundtruth),
            "oracle_self_check": "passed",
        },
        "source": source,
        "configuration": prediction["configuration"],
        "environment": {
            **prediction["environment"],
            "evaluation_python": sys.version.split()[0],
            "evaluation_numpy": np.__version__,
            "evaluation_scipy": scipy.__version__,
        },
        "measurements": {
            "pair_count": len(pairs),
            "row_count": len(rows),
            "raw_csv_sha256": _sha256(raw_path),
            "evaluation_peak_rss_bytes": peak_rss,
            "reference_baseline_median": float(
                np.median([float(row["reference_baseline_m"]) for row in rows])
            ),
            "reference_rotation_median_deg": float(
                np.median([float(row["reference_rotation_deg"]) for row in rows])
            ),
        },
        "thresholds": {
            "strict": "rotation_error_deg < 5 and oriented_translation_error_deg < 10",
            "high_precision": "rotation_error_deg < 1 and oriented_translation_error_deg < 5",
        },
        "results": _aggregate(rows, variant_names),
    }
    _atomic_json(output / "summary.json", summary, readonly=False)
    (output / "BENCHMARK_REPORT.md").write_text(_report(summary), encoding="utf-8")
    print(json.dumps(summary["results"], indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    estimation = commands.add_parser("estimate")
    estimation.add_argument("--bag-directory", type=Path, required=True)
    estimation.add_argument("--calibration", type=Path, required=True)
    estimation.add_argument("--output", type=Path, required=True)
    estimation.add_argument("--sample-hz", type=float, default=5.0)
    estimation.add_argument(
        "--offsets", type=_parse_offsets, default=_parse_offsets("6,20,34,48,62,76")
    )
    estimation.add_argument("--frames-per-window", type=int, default=4)
    estimation.add_argument("--max-theta-deg", type=float, default=98.0)
    estimation.add_argument("--edge-margin-px", type=int, default=16)
    estimation.add_argument("--max-features", type=int, default=2048)
    estimation.add_argument("--ratio-test", type=float, default=0.75)
    estimation.add_argument("--max-angular-error-deg", type=float, default=1.0)
    estimation.add_argument("--max-num-trials", type=int, default=512)
    estimation.add_argument("--stability-trials", type=int, default=3)
    estimation.add_argument("--stability-ransac-trials", type=int, default=24)
    estimation.add_argument("--model-competition-trials", type=int, default=64)
    estimation.add_argument("--decoupled-rotation-trials", type=int, default=512)
    estimation.add_argument("--decoupled-refit-max-steps", type=int, default=100)
    estimation.add_argument("--decoupled-min-margin", type=float, default=0.15)
    estimation.add_argument("--random-seed", type=int, default=20261005)
    estimation.add_argument("--protocol", type=Path)
    estimation.add_argument(
        "--variants", default=",".join(item.name for item in VARIANTS)
    )
    estimation.set_defaults(func=estimate)
    evaluation = commands.add_parser("evaluate")
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
