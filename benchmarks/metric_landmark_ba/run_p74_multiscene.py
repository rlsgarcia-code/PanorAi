#!/usr/bin/env python3
"""Validate metric spherical landmark BA on three predeclared P74 families."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.metric_landmark_ba.p74_protocol import (  # noqa: E402
    RegisteredPose,
    canonicalize_p74_rgb_native,
    load_registered_pose,
    radial_metrics,
    rotation_error_deg,
    sample_erp_nearest,
    sample_registered_radial_range,
    sha256,
    vector_angle_deg,
)
from panorai.estimators import (  # noqa: E402
    RelativePoseOptions,
    SpatiallyWeightedFivePointSampler,
    SphericalRelativePoseEstimator,
)
from panorai.features import SphericalFeaturePipeline  # noqa: E402
from panorai.reconstruction.metric_landmark_ba import (  # noqa: E402
    MetricBearingObservation,
    MetricLandmarkBAOptions,
    MetricRadialRangePrior,
    MetricScaleGauge,
    MetricSphericalCamera,
    MetricSphericalLandmark,
    refine_metric_spherical_landmarks,
)

SCHEMA = "panorai-p74-multiscene-landmark-ba/v1"


@dataclass(frozen=True, slots=True)
class PairSpec:
    family: str
    target_id: str
    source_id: str
    minimum_overlap: float
    prior_filename: str


PAIR_SPECS = (
    PairSpec(
        "W",
        "P-74+MD-04_concluido_408+W_121",
        "P-74+MD-04_concluido_408+W_124",
        0.7237,
        "P-74+MD-04_concluido_408+W_121-spherical-radial.npy",
    ),
    PairSpec(
        "G",
        "P-74+MD-05_concluido_326+G046",
        "P-74+MD-05_concluido_326+G047",
        0.8035,
        "P-74+MD-05_concluido_326+G046-spherical-radial.npy",
    ),
    PairSpec(
        "M",
        "P-74+MD-08_missing_files+M-014",
        "P-74+MD-08_missing_files+M-018",
        0.7885,
        "P-74+MD-08_missing_files+M-014-spherical-radial.npy",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p74-root", required=True, type=Path)
    parser.add_argument("--prior-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=tuple(item.family for item in PAIR_SPECS),
        default=tuple(item.family for item in PAIR_SPECS),
    )
    parser.add_argument("--max-iterations", type=int, default=25)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse an existing per-family metrics.json before running a pair",
    )
    return parser.parse_args()


def _paths(root: Path, panorama_id: str) -> tuple[Path, Path]:
    rgb = root / "images" / f"{panorama_id}_rgb.png"
    npz = root / "npzs" / f"{panorama_id}.npz"
    if not rgb.is_file() or not npz.is_file():
        raise FileNotFoundError(f"missing P74 files for {panorama_id}")
    return rgb, npz


def _unit(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array / np.linalg.norm(array, axis=-1, keepdims=True)


def triangulate_two_view(
    bearing_target: Any,
    bearing_source: Any,
    rotation_source_from_target: Any,
    translation_source_from_target: Any,
) -> np.ndarray | None:
    """Triangulate one point in the target frame by two-ray least squares."""

    first = _unit(np.asarray(bearing_target).reshape(1, 3))[0]
    second = _unit(np.asarray(bearing_source).reshape(1, 3))[0]
    rotation = np.asarray(rotation_source_from_target, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation_source_from_target, dtype=np.float64).reshape(
        3
    )
    parallax = vector_angle_deg(rotation @ first, second)
    if parallax < 0.5:
        return None
    matrix = np.column_stack((rotation @ first, -second))
    scales, _, _, _ = np.linalg.lstsq(matrix, -translation, rcond=None)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        return None
    point = first * scales[0]
    predicted_source = rotation @ point + translation
    error_target = vector_angle_deg(point, first)
    error_source = vector_angle_deg(predicted_source, second)
    if max(error_target, error_source) > 1.5:
        return None
    return point


def _angular_errors(
    rotation: np.ndarray,
    center: np.ndarray,
    points: np.ndarray,
    target_bearings: np.ndarray,
    source_bearings: np.ndarray,
) -> np.ndarray:
    target_predicted = _unit(points)
    source_predicted = _unit((rotation @ (points - center).T).T)
    first = np.degrees(
        np.arccos(
            np.clip(np.sum(target_predicted * target_bearings, axis=1), -1.0, 1.0)
        )
    )
    second = np.degrees(
        np.arccos(
            np.clip(np.sum(source_predicted * source_bearings, axis=1), -1.0, 1.0)
        )
    )
    return np.column_stack((first, second))


def _epipolar_errors_deg(
    rotation: np.ndarray,
    center: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    translation = -rotation @ center
    lines = np.cross(
        np.broadcast_to(translation, second.shape),
        (rotation @ first.T).T,
    )
    denominator = np.linalg.norm(lines, axis=1)
    sine = np.abs(np.sum(second * lines, axis=1)) / np.maximum(denominator, 1e-12)
    return np.degrees(np.arcsin(np.clip(sine, 0.0, 1.0)))


def _problem(
    *,
    target_id: str,
    source_id: str,
    target_bearings: np.ndarray,
    source_bearings: np.ndarray,
    initial_points: np.ndarray,
    prior_ranges: np.ndarray,
    confidences: np.ndarray,
    rotation: np.ndarray,
    center: np.ndarray,
) -> tuple[
    tuple[MetricSphericalCamera, ...],
    tuple[MetricSphericalLandmark, ...],
    tuple[MetricBearingObservation, ...],
    tuple[MetricRadialRangePrior, ...],
]:
    cameras = (
        MetricSphericalCamera(target_id, np.eye(3), np.zeros(3)),
        MetricSphericalCamera(source_id, rotation, center),
    )
    landmarks = []
    observations = []
    priors = []
    for index, (point, first, second, value, confidence) in enumerate(
        zip(
            initial_points,
            target_bearings,
            source_bearings,
            prior_ranges,
            confidences,
        )
    ):
        landmark_id = f"landmark-{index:06d}"
        landmarks.append(MetricSphericalLandmark(landmark_id, point))
        observations.extend(
            (
                MetricBearingObservation(target_id, landmark_id, first, confidence),
                MetricBearingObservation(source_id, landmark_id, second, confidence),
            )
        )
        priors.append(
            MetricRadialRangePrior(
                target_id,
                landmark_id,
                float(value),
                sigma_log_range=0.18,
                confidence=float(confidence),
            )
        )
    return cameras, tuple(landmarks), tuple(observations), tuple(priors)


def _pose_record(rotation: np.ndarray, center: np.ndarray, reference: RegisteredPose):
    translation = -rotation @ center
    return {
        "rotation_error_deg": rotation_error_deg(
            rotation, reference.rotation_source_from_target
        ),
        "translation_direction_error_deg": vector_angle_deg(
            translation, reference.translation_source_from_target_m
        ),
        "center_error_m": float(
            np.linalg.norm(center - reference.source_center_in_target_m)
        ),
        "baseline_m": float(np.linalg.norm(center)),
    }


def _summarize_errors(values: np.ndarray) -> dict[str, float]:
    return {
        "mean_deg": float(np.mean(values)),
        "median_deg": float(np.median(values)),
        "p90_deg": float(np.quantile(values, 0.9)),
    }


def _run_pair(
    spec: PairSpec,
    *,
    p74_root: Path,
    prior_root: Path,
    output: Path,
    max_iterations: int,
) -> dict[str, Any]:
    target_rgb, target_npz = _paths(p74_root, spec.target_id)
    source_rgb, source_npz = _paths(p74_root, spec.source_id)
    prior_path = prior_root / spec.prior_filename
    if not prior_path.is_file():
        raise FileNotFoundError(prior_path)
    prior = np.load(prior_path, mmap_mode="r")
    registered = load_registered_pose(target_npz, source_npz)
    baseline = float(np.linalg.norm(registered.translation_source_from_target_m))

    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    started = perf_counter()
    target_panorama, target_validity = canonicalize_p74_rgb_native(target_rgb)
    source_panorama, source_validity = canonicalize_p74_rgb_native(source_rgb)
    pipeline = SphericalFeaturePipeline.for_relative_pose()
    matches = pipeline.extract_and_match(
        target_panorama,
        source_panorama,
        panorama_id_a=spec.target_id,
        panorama_id_b=spec.source_id,
        validity_mask_a=target_validity,
        validity_mask_b=source_validity,
    )
    pose = SphericalRelativePoseEstimator(
        RelativePoseOptions(
            max_angular_error_deg=1.0,
            max_num_trials=1000,
            stability_trials=6,
            model_competition_trials=128,
            random_seed=7,
            hypothesis_ranking="msac-first",
            nonminimal_refit_max_steps=100,
        ),
        sampler=SpatiallyWeightedFivePointSampler(),
    ).estimate(matches.to_bearing_correspondences())
    frontend_seconds = perf_counter() - started
    if pose is None:
        return {
            "pair": asdict(spec),
            "status": "frontend-abstained",
            "match_count": len(matches),
            "frontend_seconds": frontend_seconds,
        }

    if not pose.quality_report.accepted:
        estimated_rotation = np.asarray(pose.rotation, dtype=np.float64)
        estimated_translation = (
            np.asarray(pose.translation_direction, dtype=np.float64) * baseline
        )
        estimated_center = -estimated_rotation.T @ estimated_translation
        record = {
            "pair": asdict(spec),
            "status": "frontend-quality-rejected",
            "frontend_seconds": frontend_seconds,
            "frontend": {
                "native_target_shape_hw": list(target_panorama.image.shape[:2]),
                "native_source_shape_hw": list(source_panorama.image.shape[:2]),
                "resized_for_frontend": False,
                "configuration": pipeline.describe(),
                "match_count": len(matches),
                "pose_inlier_count": int(pose.num_inliers),
                "quality_accepted": False,
                "median_parallax_deg": float(pose.median_parallax_deg),
                "pose_vs_registered": _pose_record(
                    estimated_rotation,
                    estimated_center,
                    registered,
                ),
            },
            "reason": "relative-pose quality policy rejected the frontend result",
        }
        pair_output = output / spec.family
        pair_output.mkdir(parents=True, exist_ok=True)
        metrics_path = pair_output / "metrics.json"
        metrics_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return record

    selected = np.flatnonzero(np.asarray(matches.valid & pose.inlier_mask, dtype=bool))
    heldout = selected[::5]
    train = np.setdiff1d(selected, heldout, assume_unique=True)
    estimated_rotation = np.asarray(pose.rotation, dtype=np.float64)
    estimated_translation = (
        np.asarray(pose.translation_direction, dtype=np.float64) * baseline
    )
    estimated_center = -estimated_rotation.T @ estimated_translation
    prior_samples = sample_erp_nearest(prior, matches.bearings_a)
    ratio_scores = (
        np.asarray(matches.ratio_scores, dtype=np.float64)
        if matches.ratio_scores is not None
        else np.full(len(matches), 0.72, dtype=np.float64)
    )

    promoted = []
    points = []
    for index in train:
        point = triangulate_two_view(
            matches.bearings_a[index],
            matches.bearings_b[index],
            estimated_rotation,
            estimated_translation,
        )
        prior_value = float(prior_samples[index])
        if (
            point is None
            or not math.isfinite(prior_value)
            or not 0.3 <= prior_value <= 15.0
        ):
            continue
        promoted.append(int(index))
        points.append(point)
    if len(points) < 6:
        return {
            "pair": asdict(spec),
            "status": "insufficient-landmarks",
            "match_count": len(matches),
            "pose_inlier_count": int(selected.size),
            "landmark_count": len(points),
            "frontend_seconds": frontend_seconds,
        }

    promoted_indices = np.asarray(promoted, dtype=np.int64)
    initial_points = np.stack(points)
    first_bearings = np.asarray(matches.bearings_a[promoted_indices], dtype=np.float64)
    second_bearings = np.asarray(matches.bearings_b[promoted_indices], dtype=np.float64)
    promoted_priors = np.asarray(prior_samples[promoted_indices], dtype=np.float64)
    confidences = np.clip(1.0 - ratio_scores[promoted_indices], 0.1, 1.0)
    options = MetricLandmarkBAOptions(
        max_iterations=max_iterations,
        initial_damping=1e-4,
    )
    gauge = MetricScaleGauge(spec.target_id, spec.source_id, baseline, sigma_m=0.01)

    joint_problem = _problem(
        target_id=spec.target_id,
        source_id=spec.source_id,
        target_bearings=first_bearings,
        source_bearings=second_bearings,
        initial_points=initial_points,
        prior_ranges=promoted_priors,
        confidences=confidences,
        rotation=estimated_rotation,
        center=estimated_center,
    )
    joint = refine_metric_spherical_landmarks(
        *joint_problem[:3],
        radial_range_priors=joint_problem[3],
        fixed_camera_ids=(spec.target_id,),
        scale_gauge=gauge,
        options=options,
    )
    fixed_problem = _problem(
        target_id=spec.target_id,
        source_id=spec.source_id,
        target_bearings=first_bearings,
        source_bearings=second_bearings,
        initial_points=initial_points,
        prior_ranges=promoted_priors,
        confidences=confidences,
        rotation=registered.rotation_source_from_target,
        center=registered.source_center_in_target_m,
    )
    fixed = refine_metric_spherical_landmarks(
        *fixed_problem[:3],
        radial_range_priors=fixed_problem[3],
        fixed_camera_ids=(spec.target_id, spec.source_id),
        scale_gauge=gauge,
        options=options,
    )

    joint_points = np.stack([item.position_world_m for item in joint.landmarks])
    fixed_points = np.stack([item.position_world_m for item in fixed.landmarks])
    joint_source = joint.camera_by_id[spec.source_id]
    initial_errors = _angular_errors(
        estimated_rotation,
        estimated_center,
        initial_points,
        first_bearings,
        second_bearings,
    )
    joint_errors = _angular_errors(
        joint_source.rotation_world_to_camera,
        joint_source.center_world_m,
        joint_points,
        first_bearings,
        second_bearings,
    )
    fixed_errors = _angular_errors(
        registered.rotation_source_from_target,
        registered.source_center_in_target_m,
        fixed_points,
        first_bearings,
        second_bearings,
    )
    initial_heldout = _epipolar_errors_deg(
        estimated_rotation,
        estimated_center,
        matches.bearings_a[heldout],
        matches.bearings_b[heldout],
    )
    joint_heldout = _epipolar_errors_deg(
        joint_source.rotation_world_to_camera,
        joint_source.center_world_m,
        matches.bearings_a[heldout],
        matches.bearings_b[heldout],
    )

    pair_output = output / spec.family
    pair_output.mkdir(parents=True, exist_ok=True)
    prediction_path = pair_output / "predictions-before-ground-truth.npz"
    np.savez_compressed(
        prediction_path,
        promoted_indices=promoted_indices,
        target_bearings=first_bearings,
        monocular_ranges_m=promoted_priors,
        initial_ranges_m=np.linalg.norm(initial_points, axis=1),
        joint_ranges_m=np.linalg.norm(joint_points, axis=1),
        fixed_pose_ranges_m=np.linalg.norm(fixed_points, axis=1),
        joint_points_m=joint_points,
        fixed_pose_points_m=fixed_points,
    )
    prediction_hash = sha256(prediction_path)

    # Evaluation-only boundary: organized XYZ is opened only after predictions
    # have been persisted and hashed above.
    truth, truth_valid = sample_registered_radial_range(target_npz, first_bearings)
    metric_sets = {
        "monocular": radial_metrics(promoted_priors, truth, truth_valid),
        "triangulated_initial": radial_metrics(
            np.linalg.norm(initial_points, axis=1), truth, truth_valid
        ),
        "joint_camera_and_landmarks": radial_metrics(
            np.linalg.norm(joint_points, axis=1), truth, truth_valid
        ),
        "registered_pose_landmarks_only": radial_metrics(
            np.linalg.norm(fixed_points, axis=1), truth, truth_valid
        ),
    }
    metrics_path = pair_output / "metrics.json"
    record = {
        "pair": asdict(spec),
        "status": "evaluated",
        "registered_baseline_m": baseline,
        "frontend_seconds": frontend_seconds,
        "frontend": {
            "native_target_shape_hw": list(target_panorama.image.shape[:2]),
            "native_source_shape_hw": list(source_panorama.image.shape[:2]),
            "resized_for_frontend": False,
            "configuration": pipeline.describe(),
            "match_count": len(matches),
            "pose_inlier_count": int(selected.size),
            "quality_accepted": bool(pose.quality_report.accepted),
            "median_parallax_deg": float(pose.median_parallax_deg),
        },
        "landmark_count": int(promoted_indices.size),
        "ground_truth_valid_landmark_count": int(truth_valid.sum()),
        "prediction_freeze": {
            "path": str(prediction_path),
            "sha256": prediction_hash,
            "ground_truth_opened_before_write": False,
        },
        "radial_metrics": metric_sets,
        "angular_residuals": {
            "initial": _summarize_errors(initial_errors.ravel()),
            "joint": _summarize_errors(joint_errors.ravel()),
            "registered_pose_landmarks_only": _summarize_errors(fixed_errors.ravel()),
        },
        "joint_pose": {
            "before": _pose_record(estimated_rotation, estimated_center, registered),
            "after": _pose_record(
                joint_source.rotation_world_to_camera,
                joint_source.center_world_m,
                registered,
            ),
            "heldout_match_count": int(heldout.size),
            "heldout_epipolar_before": _summarize_errors(initial_heldout),
            "heldout_epipolar_after": _summarize_errors(joint_heldout),
        },
        "reports": {
            "joint": joint.report.to_dict(),
            "registered_pose_landmarks_only": fixed.report.to_dict(),
        },
    }
    metrics_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    record["metrics_path"] = str(metrics_path)
    record["metrics_sha256"] = sha256(metrics_path)
    return record


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [item for item in records if item["status"] == "evaluated"]
    if not evaluated:
        return {"evaluated_pair_count": 0}
    methods = (
        "monocular",
        "triangulated_initial",
        "joint_camera_and_landmarks",
        "registered_pose_landmarks_only",
    )
    aggregate: dict[str, Any] = {
        "evaluated_pair_count": len(evaluated),
        "total_landmark_count": sum(item["landmark_count"] for item in evaluated),
        "methods": {},
    }
    for method in methods:
        available = [
            item["radial_metrics"][method]
            for item in evaluated
            if item["radial_metrics"][method].get("count", 0) > 0
        ]
        aggregate["methods"][method] = {
            key: float(np.median([row[key] for row in available]))
            for key in (
                "abs_rel",
                "rmse_m",
                "delta_1",
                "si_log_rmse",
                "scale_aligned_abs_rel",
            )
        }
    return aggregate


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = [item for item in PAIR_SPECS if item.family in args.families]
    records = []
    for spec in selected:
        metrics_path = args.output / spec.family / "metrics.json"
        if args.resume and metrics_path.is_file():
            records.append(json.loads(metrics_path.read_text(encoding="utf-8")))
            print(f"reused {spec.family}: {metrics_path}", flush=True)
            continue
        print(
            f"running {spec.family}: {spec.target_id} -> {spec.source_id}", flush=True
        )
        records.append(
            _run_pair(
                spec,
                p74_root=args.p74_root,
                prior_root=args.prior_root,
                output=args.output,
                max_iterations=args.max_iterations,
            )
        )
    result = {
        "schema": SCHEMA,
        "research_only": True,
        "selection": {
            "basis": "VAL-013 overlap eligibility plus existing VAL-020 priors",
            "families": list(args.families),
            "retuned_per_pair": False,
        },
        "pairs": records,
        "aggregate": _aggregate(records),
    }
    result_path = args.output / "results.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
