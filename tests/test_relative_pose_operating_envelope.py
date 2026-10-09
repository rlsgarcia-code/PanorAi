from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarks.relative_pose_operating_envelope.run_benchmark import (
    Condition,
    aggregate,
    development_conditions,
    evaluate_case,
    generate_scene,
)


def test_development_grid_is_frozen_and_complete() -> None:
    conditions = development_conditions()

    assert len(conditions) == 71
    assert sum(item.family == "parallax-vs-inliers" for item in conditions) == 36
    assert sum(item.family == "count-vs-coverage" for item in conditions) == 15
    assert sum(item.family == "noise-vs-parallax" for item in conditions) == 16
    assert sum(item.family == "negative-control" for item in conditions) == 4
    assert len({(item.family, item.name) for item in conditions}) == len(conditions)


def test_generator_hits_requested_clean_median_parallax() -> None:
    condition = Condition(
        family="test",
        name="known",
        count=80,
        target_parallax_deg=1.0,
        inlier_ratio=1.0,
        noise_deg=0.0,
        coverage="full-sphere",
    )

    scene = generate_scene(condition, 1234)

    assert scene["bearings_a"].shape == (80, 3)
    assert scene["bearings_b"].shape == (80, 3)
    assert np.allclose(np.linalg.norm(scene["bearings_a"], axis=1), 1.0)
    assert np.allclose(np.linalg.norm(scene["bearings_b"], axis=1), 1.0)
    assert scene["true_inliers"].all()
    assert scene["actual_median_parallax_deg"] == pytest.approx(1.0, abs=1e-8)


def test_pure_rotation_control_has_no_translation_parallax() -> None:
    condition = next(
        item for item in development_conditions() if item.name == "pure-rotation"
    )

    scene = generate_scene(condition, 5678)

    assert scene["actual_median_parallax_deg"] == 0.0
    assert scene["true_inliers"].all()


def test_fast_public_estimator_case_serializes_accuracy_and_policy() -> None:
    condition = Condition(
        family="test",
        name="easy",
        count=40,
        target_parallax_deg=2.0,
        inlier_ratio=1.0,
        noise_deg=0.02,
        coverage="full-sphere",
    )

    row = evaluate_case(condition, 9012, fast=True)

    assert row["returned"]
    assert row["strict_correct"]
    assert row["rotation_error_deg"] < 5.0
    assert row["translation_error_deg"] < 10.0
    assert row["options"]["hypothesis_ranking"] == "msac-first"
    assert row["policy"]["require_essential_preferred"] is True
    assert math.isfinite(row["runtime_ms"])


def test_aggregate_distinguishes_precision_from_coverage() -> None:
    rows = [
        {
            "family": "x",
            "name": "y",
            "count": 40,
            "target_parallax_deg": 1.0,
            "inlier_ratio": 0.7,
            "noise_deg": 0.1,
            "coverage": "full-sphere",
            "observable_translation": True,
            "returned": True,
            "accepted": True,
            "strict_correct": True,
            "precise_correct": False,
            "false_accept": False,
            "rotation_error_deg": 1.5,
            "translation_error_deg": 6.0,
            "runtime_ms": 10.0,
            "rejection_reasons": [],
        },
        {
            "family": "x",
            "name": "y",
            "count": 40,
            "target_parallax_deg": 1.0,
            "inlier_ratio": 0.7,
            "noise_deg": 0.1,
            "coverage": "full-sphere",
            "observable_translation": True,
            "returned": True,
            "accepted": False,
            "strict_correct": False,
            "precise_correct": False,
            "false_accept": False,
            "rotation_error_deg": 2.0,
            "translation_error_deg": 20.0,
            "runtime_ms": 20.0,
            "rejection_reasons": ["unstable-translation"],
        },
    ]

    summary = aggregate(rows)[0]

    assert summary["acceptance_rate"] == 0.5
    assert summary["accepted_strict_precision"] == 1.0
    assert summary["strict_rate_all"] == 0.5
    assert summary["rejection_reason_counts"] == {"unstable-translation": 1}
