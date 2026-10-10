#!/usr/bin/env python3
"""Measure when a query-conditioned semantic RANSAC prior helps or hurts.

This is a synthetic causal benchmark: each baseline/guided pair uses the same
bearings, validity mask, estimator configuration, and random seed.  Only the
minimal-set proposal weights differ.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panorai.estimators import RelativePoseOptions, estimate_relative_pose  # noqa: E402
from panorai.features import MatchProvenance, SphericalFeatureMatches  # noqa: E402
from panorai.object_localization import (  # noqa: E402
    SemanticMatchPriorConfig,
    SemanticQuery,
    SemanticRegionObservation,
    build_semantic_match_prior,
)


SCHEMA = "panorai-semantic-match-prior-limits/v1"


@dataclass(frozen=True)
class Scenario:
    name: str
    semantic_source: str
    localized_object: bool
    semantic_count: int
    description: str


SCENARIOS = (
    Scenario(
        "helpful-diffuse",
        "inliers",
        False,
        15,
        "Correct semantic support is angularly distributed.",
    ),
    Scenario(
        "helpful-localized",
        "inliers",
        True,
        15,
        "Correct support lies on one compact object and conflicts with diversity gates.",
    ),
    Scenario(
        "mixed",
        "mixed",
        False,
        20,
        "Semantic support has equal numbers of geometrically correct and false matches.",
    ),
    Scenario(
        "misleading",
        "outliers",
        False,
        15,
        "A wrong semantic prior concentrates proposal mass on outliers.",
    ),
    Scenario(
        "absent",
        "none",
        False,
        0,
        "No shared query support forces exact uniform fallback.",
    ),
)


def _rotation_y(angle: float) -> np.ndarray:
    return np.asarray(
        (
            (math.cos(angle), 0.0, math.sin(angle)),
            (0.0, 1.0, 0.0),
            (-math.sin(angle), 0.0, math.cos(angle)),
        ),
        dtype=np.float64,
    )


def _unit(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _scene(
    scenario: Scenario,
    *,
    seed: int,
    inlier_count: int,
    outlier_count: int,
) -> tuple[SphericalFeatureMatches, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rotation = _rotation_y(0.11)
    translation = np.asarray((0.9, 0.12, 0.25), dtype=np.float64)
    translation /= np.linalg.norm(translation)

    if scenario.localized_object:
        local_count = min(scenario.semantic_count, inlier_count)
        local = np.asarray((0.2, -0.1, 5.0)) + rng.normal(
            scale=(0.16, 0.16, 0.12), size=(local_count, 3)
        )
        global_points = rng.uniform(
            low=(-3.0, -2.0, 3.0),
            high=(3.0, 2.0, 8.0),
            size=(inlier_count - local_count, 3),
        )
        points = np.concatenate((local, global_points), axis=0)
    else:
        points = rng.uniform(
            low=(-3.0, -2.0, 3.0),
            high=(3.0, 2.0, 8.0),
            size=(inlier_count, 3),
        )
    points_b = points @ rotation.T + translation
    inlier_a = _unit(points)
    inlier_b = _unit(points_b)

    outlier_a_points = rng.uniform(
        low=(-3.0, -2.0, 3.0), high=(3.0, 2.0, 8.0), size=(outlier_count, 3)
    )
    outlier_b_points = rng.uniform(
        low=(-3.0, -2.0, 3.0), high=(3.0, 2.0, 8.0), size=(outlier_count, 3)
    )
    bearings_a = np.concatenate((inlier_a, _unit(outlier_a_points)), axis=0)
    bearings_b = np.concatenate((inlier_b, _unit(outlier_b_points)), axis=0)
    count = len(bearings_a)
    matches = SphericalFeatureMatches(
        panorama_id_a="synthetic-a",
        panorama_id_b="synthetic-b",
        feature_indices_a=np.arange(count),
        feature_indices_b=np.arange(count),
        bearings_a=bearings_a,
        bearings_b=bearings_b,
        descriptor_distances=np.zeros(count),
        ratio_scores=np.zeros(count),
        mutual=np.ones(count, dtype=bool),
        valid=np.ones(count, dtype=bool),
        matcher_name="known-correspondence-plus-controlled-outliers",
        matcher_config={},
        backend_name="analytic",
        backend_version="1",
        provenance=MatchProvenance(
            interface=SCHEMA,
            source_checksums=(f"seed-{seed}-a", f"seed-{seed}-b"),
            face_pairs=(("analytic", "analytic"),),
            face_pair_groups=((("analytic", "analytic"),),),
            deduplicated=True,
        ),
        keypoint_responses=np.ones((count, 2)),
        face_ids_a=np.full(count, "analytic", dtype=object),
        face_ids_b=np.full(count, "analytic", dtype=object),
    )
    inlier_mask = np.zeros(count, dtype=bool)
    inlier_mask[:inlier_count] = True
    return matches, inlier_mask, rotation, translation


def _semantic_indices(
    scenario: Scenario, inlier_count: int, outlier_count: int
) -> np.ndarray:
    if scenario.semantic_source == "none":
        return np.empty(0, dtype=np.int64)
    if scenario.semantic_source == "inliers":
        return np.arange(min(scenario.semantic_count, inlier_count), dtype=np.int64)
    if scenario.semantic_source == "outliers":
        return inlier_count + np.arange(
            min(scenario.semantic_count, outlier_count), dtype=np.int64
        )
    half = scenario.semantic_count // 2
    return np.concatenate(
        (
            np.arange(min(half, inlier_count), dtype=np.int64),
            inlier_count + np.arange(min(half, outlier_count), dtype=np.int64),
        )
    )


def _region(view_id: str, indices: np.ndarray) -> SemanticRegionObservation:
    return SemanticRegionObservation(
        region_id=f"query-region-{view_id}",
        view_id=view_id,
        class_id=7,
        class_name="query-object",
        semantic_score=1.0,
        feature_indices=indices,
        membership_weights=np.ones(len(indices)),
    )


def _rotation_error_deg(actual: np.ndarray, expected: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(actual @ expected.T) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _direction_error_deg(actual: np.ndarray, expected: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(actual, expected), -1.0, 1.0))))


def _pose_record(
    result: Any, rotation: np.ndarray, translation: np.ndarray
) -> dict[str, Any]:
    if result is None:
        return {
            "estimated": False,
            "accepted": False,
            "success": False,
            "rotation_error_deg": None,
            "translation_error_deg": None,
            "num_inliers": 0,
            "samples_drawn": 0,
            "uniform_samples": 0,
            "rejection_reasons": ["no-pose-estimated"],
            "degeneracy_reasons": [],
        }
    rotation_error = _rotation_error_deg(result.rotation, rotation)
    translation_error = _direction_error_deg(result.translation_direction, translation)
    return {
        "estimated": True,
        "accepted": bool(result.quality_report.accepted),
        "success": rotation_error <= 5.0 and translation_error <= 10.0,
        "rotation_error_deg": rotation_error,
        "translation_error_deg": translation_error,
        "num_inliers": int(result.num_inliers),
        "samples_drawn": int(result.sampling_diagnostics.samples_drawn),
        "uniform_samples": int(result.sampling_diagnostics.uniform_samples),
        "rejection_reasons": list(result.quality_report.rejection_reasons),
        "degeneracy_reasons": list(result.degeneracy_reasons),
    }


def _run_one(
    scenario: Scenario,
    *,
    seed: int,
    inlier_count: int,
    outlier_count: int,
    max_trials: int,
    uniform_mix: float,
    max_weight_ratio: float,
) -> dict[str, Any]:
    matches, oracle_inliers, rotation, translation = _scene(
        scenario,
        seed=seed,
        inlier_count=inlier_count,
        outlier_count=outlier_count,
    )
    semantic_indices = _semantic_indices(scenario, inlier_count, outlier_count)
    regions_a = (
        () if not len(semantic_indices) else (_region("synthetic-a", semantic_indices),)
    )
    regions_b = (
        () if not len(semantic_indices) else (_region("synthetic-b", semantic_indices),)
    )
    query = SemanticQuery(text="find query object", class_ids=(7,))
    guided_prior = build_semantic_match_prior(
        query,
        regions_a,
        regions_b,
        matches,
        SemanticMatchPriorConfig(
            uniform_mix=uniform_mix,
            max_weight_ratio=max_weight_ratio,
        ),
    )
    baseline_prior = build_semantic_match_prior(
        query,
        regions_a,
        regions_b,
        matches,
        SemanticMatchPriorConfig(uniform_mix=1.0),
    )
    options = RelativePoseOptions(
        max_angular_error_deg=0.75,
        min_inlier_ratio=0.1,
        min_num_trials=max_trials,
        max_num_trials=max_trials,
        min_inliers=8,
        local_optimization_steps=2,
        minimal_solver_starts=8,
        random_seed=seed,
        stability_trials=2,
        stability_ransac_trials=8,
        model_competition_trials=16,
    )
    baseline = estimate_relative_pose(
        baseline_prior.to_bearing_correspondences(matches), options=options
    )
    guided = estimate_relative_pose(
        guided_prior.to_bearing_correspondences(matches), options=options
    )
    if scenario.semantic_source == "none":
        if not np.array_equal(
            baseline_prior.sampling_weights, guided_prior.sampling_weights
        ):
            raise AssertionError("absent semantics must reproduce baseline weights")
    semantic_true = int(oracle_inliers[semantic_indices].sum())
    semantic_precision = (
        semantic_true / len(semantic_indices) if len(semantic_indices) else None
    )
    return {
        "scenario": scenario.name,
        "seed": seed,
        "inlier_count": inlier_count,
        "outlier_count": outlier_count,
        "outlier_ratio": outlier_count / (inlier_count + outlier_count),
        "semantic_count": len(semantic_indices),
        "semantic_true_inliers": semantic_true,
        "semantic_precision": semantic_precision,
        "prior": guided_prior.describe(),
        "baseline": _pose_record(baseline, rotation, translation),
        "guided": _pose_record(guided, rotation, translation),
        "absent_control_exact": (
            _pose_record(baseline, rotation, translation)
            == _pose_record(guided, rotation, translation)
            if scenario.semantic_source == "none"
            else None
        ),
    }


def _aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for scenario in SCENARIOS:
        rows = [row for row in records if row["scenario"] == scenario.name]
        baseline_success = sum(row["baseline"]["success"] for row in rows)
        guided_success = sum(row["guided"]["success"] for row in rows)
        summaries.append(
            {
                "scenario": scenario.name,
                "description": scenario.description,
                "runs": len(rows),
                "baseline_successes": baseline_success,
                "guided_successes": guided_success,
                "success_delta": guided_success - baseline_success,
                "baseline_accepted": sum(row["baseline"]["accepted"] for row in rows),
                "guided_accepted": sum(row["guided"]["accepted"] for row in rows),
                "median_effective_sample_size": float(
                    np.median([row["prior"]["effective_sample_size"] for row in rows])
                ),
                "median_weight_ratio": float(
                    np.median(
                        [row["prior"]["max_to_min_valid_weight_ratio"] for row in rows]
                    )
                ),
            }
        )
    return summaries


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = (
        "scenario",
        "seed",
        "outlier_ratio",
        "semantic_precision",
        "effective_sample_size",
        "weight_ratio",
        "baseline_success",
        "guided_success",
        "baseline_rotation_error_deg",
        "guided_rotation_error_deg",
        "baseline_translation_error_deg",
        "guided_translation_error_deg",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "scenario": row["scenario"],
                    "seed": row["seed"],
                    "outlier_ratio": row["outlier_ratio"],
                    "semantic_precision": row["semantic_precision"],
                    "effective_sample_size": row["prior"]["effective_sample_size"],
                    "weight_ratio": row["prior"]["max_to_min_valid_weight_ratio"],
                    "baseline_success": row["baseline"]["success"],
                    "guided_success": row["guided"]["success"],
                    "baseline_rotation_error_deg": row["baseline"][
                        "rotation_error_deg"
                    ],
                    "guided_rotation_error_deg": row["guided"]["rotation_error_deg"],
                    "baseline_translation_error_deg": row["baseline"][
                        "translation_error_deg"
                    ],
                    "guided_translation_error_deg": row["guided"][
                        "translation_error_deg"
                    ],
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--inliers", type=int, default=30)
    parser.add_argument("--outliers", type=int, default=45)
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--uniform-mix", type=float, default=0.25)
    parser.add_argument("--max-weight-ratio", type=float, default=4.0)
    args = parser.parse_args()
    seed_count = args.seeds or (4 if args.quick else 20)
    max_trials = args.max_trials or (96 if args.quick else 192)
    if seed_count < 1 or args.inliers < 8 or args.outliers < 0 or max_trials < 1:
        parser.error(
            "seeds, inliers, and max-trials must be positive; outliers non-negative"
        )

    records = [
        _run_one(
            scenario,
            seed=20261010 + seed,
            inlier_count=args.inliers,
            outlier_count=args.outliers,
            max_trials=max_trials,
            uniform_mix=args.uniform_mix,
            max_weight_ratio=args.max_weight_ratio,
        )
        for scenario in SCENARIOS
        for seed in range(seed_count)
    ]
    if not all(
        row["absent_control_exact"] for row in records if row["scenario"] == "absent"
    ):
        raise AssertionError("absent semantic control did not reproduce baseline")
    summary = _aggregate(records)
    payload = {
        "schema": SCHEMA,
        "causal_contract": {
            "same_correspondences": True,
            "same_validity": True,
            "same_estimator_options": True,
            "same_random_seed": True,
            "only_difference": "minimal-set proposal weights",
            "all_valid_matches_scored": True,
        },
        "configuration": {
            "seed_count": seed_count,
            "inlier_count": args.inliers,
            "outlier_count": args.outliers,
            "max_trials": max_trials,
            "semantic_prior": {
                "uniform_mix": args.uniform_mix,
                "max_weight_ratio": args.max_weight_ratio,
            },
            "success_thresholds": {
                "rotation_error_deg": 5.0,
                "translation_error_deg": 10.0,
            },
            "scenarios": [asdict(item) for item in SCENARIOS],
        },
        "summary": summary,
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir / "runs.csv", records)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
