#!/usr/bin/env python3
"""Frozen progressive experiments for dense, pose, and sparse matches."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panorai.estimators import (  # noqa: E402
    RelativePoseAcceptancePolicy,
    RelativePoseOptions,
    estimate_relative_pose,
)
from panorai.features import MatchProvenance, SphericalFeatureMatches  # noqa: E402
from panorai.geometry import erp_pixels_to_rays  # noqa: E402
from panorai.stereo import (  # noqa: E402
    DenseMatchFilterOptions,
    SphericalMatchRefinementOptions,
    SphericalStereoOptions,
    estimate_spherical_range,
    filter_matches_by_dense_range,
    refine_matches_on_sphere,
)

SCHEMA = "panorai-dense-pose-match-experiment/v1"
SHAPE_HW = (32, 64)


@dataclass(frozen=True)
class SceneSpec:
    seed: int
    outlier_fraction: float
    baseline: float
    yaw_deg: float
    pitch_deg: float
    sphere_radius: float
    sphere_center: tuple[float, float, float]


def _scene_specs(split: str) -> tuple[SceneSpec, ...]:
    offset = 0 if split == "development" else 10_000
    return tuple(
        SceneSpec(
            seed=offset + index,
            outlier_fraction=outlier,
            baseline=baseline,
            yaw_deg=yaw,
            pitch_deg=pitch,
            sphere_radius=radius,
            sphere_center=center,
        )
        for index, (outlier, baseline, yaw, pitch, radius, center) in enumerate(
            (
                (0.05, 0.34, 3.0, -1.0, 4.2, (0.15, -0.10, 0.05)),
                (0.10, 0.42, -4.0, 1.5, 4.8, (-0.20, 0.15, 0.10)),
                (0.20, 0.50, 5.0, 2.0, 4.5, (0.25, 0.05, -0.15)),
                (0.30, 0.38, -3.5, -2.0, 4.0, (-0.10, -0.20, 0.20)),
                (0.40, 0.55, 4.5, -1.5, 5.0, (0.20, 0.20, 0.00)),
                (0.45, 0.46, -5.0, 2.5, 4.6, (-0.25, 0.10, -0.10)),
            )
        )
    )


def _rotation(spec: SceneSpec) -> np.ndarray:
    yaw = math.radians(spec.yaw_deg)
    pitch = math.radians(spec.pitch_deg)
    rotation_y = np.asarray(
        (
            (math.cos(yaw), 0.0, math.sin(yaw)),
            (0.0, 1.0, 0.0),
            (-math.sin(yaw), 0.0, math.cos(yaw)),
        )
    )
    rotation_x = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, math.cos(pitch), -math.sin(pitch)),
            (0.0, math.sin(pitch), math.cos(pitch)),
        )
    )
    return rotation_x @ rotation_y


def _camera_center(spec: SceneSpec) -> np.ndarray:
    direction = np.asarray((0.91, 0.18, 0.37), dtype=np.float64)
    direction /= np.linalg.norm(direction)
    return direction * spec.baseline


def _ray_lattice(shape_hw: tuple[int, int]) -> np.ndarray:
    y, x = np.indices(shape_hw, dtype=np.float64)
    return erp_pixels_to_rays(np.stack((x, y), axis=-1), shape_hw)


def _sphere_intersections(
    camera_center: np.ndarray,
    rays_world: np.ndarray,
    sphere_center: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    offset = camera_center - sphere_center
    projected = np.sum(rays_world * offset, axis=-1)
    discriminant = projected * projected + radius * radius - float(offset @ offset)
    distance = -projected + np.sqrt(np.maximum(discriminant, 0.0))
    points = camera_center + distance[..., None] * rays_world
    return points, distance


def _render_scene(
    camera_center: np.ndarray,
    rotation_camera_from_a: np.ndarray,
    spec: SceneSpec,
) -> tuple[np.ndarray, np.ndarray]:
    rays_camera = _ray_lattice(SHAPE_HW)
    rays_world = rays_camera @ rotation_camera_from_a
    points, distance = _sphere_intersections(
        camera_center,
        rays_world,
        np.asarray(spec.sphere_center),
        spec.sphere_radius,
    )
    image = np.stack(
        (
            0.5
            + 0.21 * np.sin(5.0 * points[..., 0] + 2.0 * points[..., 2])
            + 0.19 * np.cos(7.0 * points[..., 1]),
            0.5
            + 0.24 * np.sin(4.0 * points[..., 1] - 3.0 * points[..., 2])
            + 0.14 * np.cos(8.0 * points[..., 0]),
            0.5
            + 0.23 * np.cos(6.0 * points[..., 2] + points[..., 0])
            + 0.16 * np.sin(9.0 * points[..., 1]),
        ),
        axis=-1,
    )
    return np.clip(image, 0.0, 1.0).astype(np.float32), distance.astype(np.float32)


def _perturb_bearings(
    bearings: np.ndarray, noise_deg: float, rng: np.random.Generator
) -> np.ndarray:
    noise = rng.normal(size=bearings.shape)
    noise -= np.sum(noise * bearings, axis=1, keepdims=True) * bearings
    noise_norm = np.linalg.norm(noise, axis=1, keepdims=True)
    noise /= np.maximum(noise_norm, np.finfo(np.float64).eps)
    angles = rng.normal(scale=math.radians(noise_deg), size=(len(bearings), 1))
    perturbed = np.cos(angles) * bearings + np.sin(angles) * noise
    return perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)


def _make_matches(
    spec: SceneSpec,
    rotation: np.ndarray,
    translation: np.ndarray,
    reference_range: np.ndarray,
    *,
    angular_noise_deg: float,
) -> tuple[SphericalFeatureMatches, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(spec.seed)
    height, width = SHAPE_HW
    candidates = np.asarray(
        [(x, y) for y in range(4, height - 4) for x in range(width)],
        dtype=np.int64,
    )
    selected = candidates[rng.choice(len(candidates), size=120, replace=False)]
    bearings_a = erp_pixels_to_rays(selected.astype(np.float64), SHAPE_HW)
    ranges = reference_range[selected[:, 1], selected[:, 0]].astype(np.float64)
    points_a = bearings_a * ranges[:, None]
    target = points_a @ rotation.T + translation
    exact_bearings_b = target / np.linalg.norm(target, axis=1, keepdims=True)
    bearings_b = _perturb_bearings(exact_bearings_b, angular_noise_deg, rng)

    inlier = np.ones(len(selected), dtype=bool)
    outlier_count = int(round(spec.outlier_fraction * len(selected)))
    outlier_indices = rng.choice(len(selected), size=outlier_count, replace=False)
    inlier[outlier_indices] = False
    epipolar_count = int(round(0.6 * outlier_count))
    epipolar_indices = outlier_indices[:epipolar_count]
    random_indices = outlier_indices[epipolar_count:]
    if epipolar_count:
        near_factors = rng.uniform(0.40, 0.65, size=epipolar_count)
        far_factors = rng.uniform(1.60, 2.20, size=epipolar_count)
        use_far = rng.random(epipolar_count) >= 0.5
        wrong_ranges = ranges[epipolar_indices] * np.where(
            use_far, far_factors, near_factors
        )
        wrong_points = bearings_a[epipolar_indices] * wrong_ranges[:, None]
        epipolar_targets = wrong_points @ rotation.T + translation
        epipolar_targets /= np.linalg.norm(epipolar_targets, axis=1, keepdims=True)
        bearings_b[epipolar_indices] = _perturb_bearings(epipolar_targets, 0.12, rng)
    if len(random_indices):
        random_targets = rng.normal(size=(len(random_indices), 3))
        random_targets /= np.linalg.norm(random_targets, axis=1, keepdims=True)
        bearings_b[random_indices] = random_targets

    count = len(selected)
    matches = SphericalFeatureMatches(
        panorama_id_a=f"analytic-{spec.seed}-a",
        panorama_id_b=f"analytic-{spec.seed}-b",
        feature_indices_a=np.arange(count),
        feature_indices_b=np.arange(count),
        bearings_a=bearings_a,
        bearings_b=bearings_b,
        descriptor_distances=np.where(inlier, 0.2, 0.8).astype(np.float32),
        ratio_scores=np.where(inlier, 0.6, 0.9).astype(np.float32),
        mutual=np.ones(count, dtype=bool),
        valid=np.ones(count, dtype=bool),
        matcher_name="frozen-analytic",
        matcher_config={"angular_noise_deg": angular_noise_deg},
        backend_name="analytic",
        backend_version="1",
        provenance=MatchProvenance(
            interface="panorai-spherical-features/v1",
            source_checksums=(f"analytic-{spec.seed}-a", f"analytic-{spec.seed}-b"),
            face_pairs=tuple(("erp", "erp") for _ in range(count)),
            face_pair_groups=tuple((("erp", "erp"),) for _ in range(count)),
            deduplicated=False,
        ),
        keypoint_responses=np.ones((count, 2), dtype=np.float32),
        face_ids_a=np.full(count, "erp", dtype=object),
        face_ids_b=np.full(count, "erp", dtype=object),
        stability="experimental",
    )
    return matches, inlier, exact_bearings_b


def _pose_options(seed: int) -> RelativePoseOptions:
    return RelativePoseOptions(
        max_angular_error_deg=1.0,
        min_num_trials=24,
        max_num_trials=240,
        local_optimization_steps=2,
        minimal_solver_starts=8,
        minimal_solver_max_nfev=60,
        refinement_max_nfev=60,
        random_seed=seed,
        stability_trials=0,
        model_competition_trials=24,
        nonminimal_refit_max_steps=30,
    )


def _pose_policy() -> RelativePoseAcceptancePolicy:
    return RelativePoseAcceptancePolicy(
        require_stability=False,
        require_essential_preferred=False,
    )


def _estimate_pose(matches: SphericalFeatureMatches, valid: np.ndarray, seed: int):
    return estimate_relative_pose(
        matches.bearings_a,
        matches.bearings_b,
        valid=valid,
        options=_pose_options(seed),
        quality_policy=_pose_policy(),
    )


def _rotation_error_deg(estimated: np.ndarray, reference: np.ndarray) -> float:
    cosine = (np.trace(estimated @ reference.T) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _direction_error_deg(estimated: np.ndarray, reference: np.ndarray) -> float:
    left = estimated / np.linalg.norm(estimated)
    right = reference / np.linalg.norm(reference)
    return math.degrees(math.acos(float(np.clip(left @ right, -1.0, 1.0))))


def _pose_metrics(result: Any, rotation: np.ndarray, translation: np.ndarray):
    if result is None:
        return {"returned": False, "rotation_error_deg": None, "t_error_deg": None}
    return {
        "returned": True,
        "rotation_error_deg": _rotation_error_deg(result.rotation, rotation),
        "t_error_deg": _direction_error_deg(result.translation_direction, translation),
    }


def _filter_metrics(mask: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    true_positive = int(np.count_nonzero(mask & truth))
    false_positive = int(np.count_nonzero(mask & ~truth))
    false_negative = int(np.count_nonzero(~mask & truth))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "accepted": int(mask.sum()),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "precision": precision,
        "recall": recall,
    }


def _run_scene(spec: SceneSpec, stage: int) -> dict[str, Any]:
    rng = np.random.default_rng(spec.seed + 500_000)
    rotation = _rotation(spec)
    camera_b_in_a = _camera_center(spec)
    translation = -(rotation @ camera_b_in_a)
    reference, reference_range = _render_scene(np.zeros(3), np.eye(3), spec)
    target, _ = _render_scene(camera_b_in_a, rotation, spec)
    target = np.clip(
        target * (1.02 + 0.01 * rng.random())
        + 0.005
        + rng.normal(scale=0.002, size=target.shape),
        0.0,
        1.0,
    ).astype(np.float32)
    angular_noise = 0.12 if stage == 1 else 0.75
    matches, truth, exact_bearings_b = _make_matches(
        spec,
        rotation,
        translation,
        reference_range,
        angular_noise_deg=angular_noise,
    )

    sparse_pose = _estimate_pose(matches, matches.valid, spec.seed)
    sparse_metrics = _pose_metrics(sparse_pose, rotation, translation)
    row: dict[str, Any] = {
        "scene_id": f"analytic-{spec.seed}",
        "outlier_fraction": spec.outlier_fraction,
        "baseline": spec.baseline,
        "sparse_matches": _filter_metrics(matches.valid, truth),
        "sparse_pose": sparse_metrics,
    }
    if sparse_pose is None:
        row["reference_pose_filter"] = None
        row["estimated_pose_filter"] = None
        return row

    stereo_options = SphericalStereoOptions(
        min_range=2.8,
        max_range=6.2,
        num_hypotheses=48,
        window_size=5,
        pole_margin_fraction=0.06,
        min_texture_std=0.002,
        min_confidence=0.0,
        max_matching_cost=0.9,
        bidirectional_consistency=False,
        filter_backend="numpy",
    )
    filter_options = DenseMatchFilterOptions(
        max_angular_error_deg=1.5 if stage == 1 else 2.5,
        min_dense_confidence=0.0,
        min_valid_weight=0.75,
    )
    reference_dense = estimate_spherical_range(
        reference, target, rotation, translation, options=stereo_options
    )
    estimated_translation = sparse_pose.translation_direction * np.linalg.norm(
        translation
    )
    estimated_dense = estimate_spherical_range(
        reference,
        target,
        sparse_pose.rotation,
        estimated_translation,
        options=stereo_options,
    )

    for name, dense, filter_rotation, filter_translation, seed_addition in (
        ("reference_pose_filter", reference_dense, rotation, translation, 1_000),
        (
            "estimated_pose_filter",
            estimated_dense,
            sparse_pose.rotation,
            estimated_translation,
            2_000,
        ),
    ):
        filtered = filter_matches_by_dense_range(
            matches,
            dense,
            filter_rotation,
            filter_translation,
            options=filter_options,
        )
        filtered_pose = _estimate_pose(
            matches, filtered.accepted_mask, spec.seed + seed_addition
        )
        row[name] = {
            **_filter_metrics(filtered.accepted_mask, truth),
            "supported": int(filtered.dense_supported_mask.sum()),
            "unsupported": int(filtered.unsupported_mask.sum()),
            "pose": _pose_metrics(filtered_pose, rotation, translation),
        }
        if name == "estimated_pose_filter" and stage >= 2:
            refinement = refine_matches_on_sphere(
                reference,
                target,
                matches,
                filtered,
                sparse_pose.rotation,
                estimated_translation,
                options=SphericalMatchRefinementOptions(
                    patch_radius_px=2,
                    search_radius_px=0.6,
                    search_step_px=0.05,
                    min_patch_support=0.9,
                    min_patch_std=0.005,
                    min_cost_improvement=0.02,
                    max_angular_shift_deg=1.5,
                ),
            )
            refined_pose = estimate_relative_pose(
                refinement.to_bearing_correspondences(),
                options=_pose_options(spec.seed + 3_000),
                quality_policy=_pose_policy(),
            )
            original_error = _bearing_errors_deg(
                matches.bearings_b[truth], exact_bearings_b[truth]
            )
            refined_error = _bearing_errors_deg(
                refinement.refined_bearings_b[truth], exact_bearings_b[truth]
            )
            applied_gain = refinement.cost_improvement[refinement.applied_mask]
            row["refined_filter"] = {
                **_filter_metrics(refinement.eligible_mask, truth),
                "supported": int(refinement.eligible_mask.sum()),
                "unsupported": int((~refinement.eligible_mask).sum()),
                "evaluated": int(refinement.evaluated_mask.sum()),
                "applied": int(refinement.applied_mask.sum()),
                "median_original_bearing_error_deg": float(np.median(original_error)),
                "median_refined_bearing_error_deg": float(np.median(refined_error)),
                "median_applied_cost_improvement": (
                    float(np.median(applied_gain)) if applied_gain.size else None
                ),
                "pose": _pose_metrics(refined_pose, rotation, translation),
            }
    return row


def _bearing_errors_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.sum(left * right, axis=1), -1.0, 1.0)))


def _aggregate(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    entries = [row[variant] for row in rows if row[variant] is not None]
    true_positive = sum(item["true_positive"] for item in entries)
    false_positive = sum(item["false_positive"] for item in entries)
    truth_count = sum(
        row["sparse_matches"]["true_positive"]
        for row in rows
        if row[variant] is not None
    )
    pose_entries = [item["pose"] for item in entries if item["pose"]["returned"]]
    result = {
        "case_count": len(entries),
        "accepted": sum(item["accepted"] for item in entries),
        "precision": true_positive / max(true_positive + false_positive, 1),
        "recall": true_positive / max(truth_count, 1),
        "mean_supported_fraction": float(
            np.mean([item["supported"] / 120.0 for item in entries])
        ),
        "pose_returned_count": len(pose_entries),
        "median_rotation_error_deg": (
            float(np.median([item["rotation_error_deg"] for item in pose_entries]))
            if pose_entries
            else None
        ),
        "median_t_error_deg": (
            float(np.median([item["t_error_deg"] for item in pose_entries]))
            if pose_entries
            else None
        ),
    }
    if variant == "refined_filter":
        result.update(
            {
                "evaluated": sum(item["evaluated"] for item in entries),
                "applied": sum(item["applied"] for item in entries),
                "median_original_bearing_error_deg": float(
                    np.median(
                        [item["median_original_bearing_error_deg"] for item in entries]
                    )
                ),
                "median_refined_bearing_error_deg": float(
                    np.median(
                        [item["median_refined_bearing_error_deg"] for item in entries]
                    )
                ),
            }
        )
    return result


def _summary(rows: list[dict[str, Any]], split: str, stage: int) -> dict[str, Any]:
    baseline_true = sum(row["sparse_matches"]["true_positive"] for row in rows)
    baseline_false = sum(row["sparse_matches"]["false_positive"] for row in rows)
    sparse_poses = [
        row["sparse_pose"] for row in rows if row["sparse_pose"]["returned"]
    ]
    summary = {
        "schema": SCHEMA,
        "stage": stage,
        "split": split,
        "evidence_target": "source-checkout",
        "case_count": len(rows),
        "sparse": {
            "precision": baseline_true / max(baseline_true + baseline_false, 1),
            "recall": 1.0,
            "pose_returned_count": len(sparse_poses),
            "median_rotation_error_deg": (
                float(np.median([item["rotation_error_deg"] for item in sparse_poses]))
                if sparse_poses
                else None
            ),
            "median_t_error_deg": (
                float(np.median([item["t_error_deg"] for item in sparse_poses]))
                if sparse_poses
                else None
            ),
        },
        "reference_pose_filter": _aggregate(rows, "reference_pose_filter"),
        "estimated_pose_filter": _aggregate(rows, "estimated_pose_filter"),
        "cases": rows,
    }
    candidate = summary["estimated_pose_filter"]
    summary["stage_gate"] = {
        "precision_improved": candidate["precision"] > summary["sparse"]["precision"],
        "recall_at_least_0_60": candidate["recall"] >= 0.60,
        "all_initial_poses_returned": summary["sparse"]["pose_returned_count"]
        == len(rows),
        "all_filtered_poses_returned": candidate["pose_returned_count"] == len(rows),
    }
    if stage >= 2:
        refined = _aggregate(rows, "refined_filter")
        summary["refined_filter"] = refined
        rotation_regressions = [
            row["refined_filter"]["pose"]["rotation_error_deg"]
            - row["estimated_pose_filter"]["pose"]["rotation_error_deg"]
            for row in rows
        ]
        translation_regressions = [
            row["refined_filter"]["pose"]["t_error_deg"]
            - row["estimated_pose_filter"]["pose"]["t_error_deg"]
            for row in rows
        ]
        refined["max_case_rotation_regression_deg"] = float(max(rotation_regressions))
        refined["max_case_t_regression_deg"] = float(max(translation_regressions))
        summary["stage_gate"].update(
            {
                "bearing_error_improved": refined["median_refined_bearing_error_deg"]
                < refined["median_original_bearing_error_deg"],
                "all_refined_poses_returned": refined["pose_returned_count"]
                == len(rows),
                "rotation_not_regressed": refined["median_rotation_error_deg"]
                <= candidate["median_rotation_error_deg"] + 0.10,
                "translation_not_regressed": refined["median_t_error_deg"]
                <= candidate["median_t_error_deg"] + 0.50,
                "per_case_pose_regression_bounded": max(rotation_regressions) <= 0.15
                and max(translation_regressions) <= 0.75,
            }
        )
    summary["stage_gate"]["passed"] = all(summary["stage_gate"].values())
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--split", choices=("development", "heldout"), required=True)
    args = parser.parse_args()
    rows = [_run_scene(spec, args.stage) for spec in _scene_specs(args.split)]
    summary = _summary(rows, args.split, args.stage)
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0 if summary["stage_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
