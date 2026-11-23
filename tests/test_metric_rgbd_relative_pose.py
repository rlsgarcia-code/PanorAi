from __future__ import annotations

import math

import numpy as np
import pytest

from panorai.estimators import (
    MetricRgbdPoseAcceptancePolicy,
    MetricRgbdPoseOptions,
    MetricRgbdPoseResult,
    RelativePoseOptions,
    estimate_metric_rgbd_relative_pose,
)


def _rotation_exp(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    x, y, z = vector
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return (
        np.eye(3)
        + math.sin(angle) / angle * skew
        + (1.0 - math.cos(angle)) / angle**2 * (skew @ skew)
    )


def _scene(
    *, count: int = 96, seed: int = 20261123
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Independent Euclidean oracle with known target-from-source SE(3)."""

    rng = np.random.default_rng(seed)
    rotation = _rotation_exp(np.asarray((0.08, -0.12, 0.04)))
    translation = np.asarray((0.42, -0.11, 0.27))
    points_a = rng.uniform((-3.0, -2.0, 2.0), (3.0, 2.0, 10.0), (count, 3))
    points_b = points_a @ rotation.T + translation
    depth_a = np.linalg.norm(points_a, axis=1)
    depth_b = np.linalg.norm(points_b, axis=1)
    bearings_a = points_a / depth_a[:, None]
    bearings_b = points_b / depth_b[:, None]
    return bearings_a, bearings_b, depth_a, depth_b, rotation, translation


def _rotation_error_deg(estimated: np.ndarray, expected: np.ndarray) -> float:
    cosine = np.clip((np.trace(estimated @ expected.T) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def test_metric_rgbd_pose_recovers_known_se3_without_mutating_inputs() -> None:
    b1, b2, d1, d2, expected_r, expected_t = _scene()
    originals = tuple(value.copy() for value in (b1, b2, d1, d2))

    result = estimate_metric_rgbd_relative_pose(
        b1,
        b2,
        d1,
        d2,
        valid=np.ones(len(b1), dtype=bool),
        options=MetricRgbdPoseOptions(
            max_metric_error=1e-6,
            min_trials=8,
            max_trials=32,
            use_visual_consensus=False,
        ),
    )

    assert isinstance(result, MetricRgbdPoseResult)
    assert result.num_inliers == len(b1)
    assert _rotation_error_deg(result.R, expected_r) < 1e-5
    assert np.linalg.norm(result.t - expected_t) < 1e-10
    assert result.quality_report.accepted
    assert result.describe()["depth_semantics"] == "radial-range-along-unit-bearing"
    assert result.describe()["translation"] == "metric-in-input-radial-depth-units"
    assert not result.R.flags.writeable
    assert not result.t.flags.writeable
    assert not result.inlier_mask.flags.writeable
    for actual, expected in zip((b1, b2, d1, d2), originals, strict=True):
        assert np.array_equal(actual, expected)


def test_metric_rgbd_pose_rejects_outliers_and_preserves_explicit_invalid_rows() -> (
    None
):
    b1, b2, d1, d2, expected_r, expected_t = _scene(count=120, seed=88)
    rng = np.random.default_rng(9)
    outliers = np.arange(0, 24)
    b2[outliers] = rng.normal(size=(len(outliers), 3))
    b2[outliers] /= np.linalg.norm(b2[outliers], axis=1, keepdims=True)
    d2[outliers] = rng.uniform(1.0, 15.0, len(outliers))
    valid = np.ones(len(b1), dtype=bool)
    valid[-3:] = False
    b1[-3:] = np.nan
    d1[-3:] = np.nan

    result = estimate_metric_rgbd_relative_pose(
        b1,
        b2,
        d1,
        d2,
        valid=valid,
        options=MetricRgbdPoseOptions(
            max_metric_error=0.02,
            min_trials=32,
            max_trials=200,
            min_inliers=24,
            use_visual_consensus=False,
        ),
    )

    assert result is not None
    assert _rotation_error_deg(result.R, expected_r) < 1e-4
    assert np.linalg.norm(result.t - expected_t) < 1e-8
    assert not np.any(result.inlier_mask[outliers])
    assert not np.any(result.inlier_mask[-3:])
    assert np.all(np.isinf(result.residuals[-3:]))
    assert result.quality_report.num_valid == len(valid) - 3
    assert result.quality_report.accepted


def test_visual_consensus_is_reused_but_metric_fallback_is_explicit() -> None:
    b1, b2, d1, d2, expected_r, expected_t = _scene(count=64, seed=71)
    result = estimate_metric_rgbd_relative_pose(
        b1,
        b2,
        d1,
        d2,
        valid=np.ones(len(b1), dtype=bool),
        options=MetricRgbdPoseOptions(
            max_metric_error=1e-5,
            min_trials=8,
            max_trials=40,
        ),
        visual_options=RelativePoseOptions(
            max_angular_error_deg=0.1,
            min_num_trials=8,
            max_num_trials=40,
            min_inliers=8,
            local_optimization_steps=1,
            stability_trials=0,
            model_competition_trials=16,
        ),
    )

    assert result is not None
    assert result.visual_pose is not None
    assert result.visual_consensus_used
    assert result.visual_consensus_fallback_reason is None
    assert _rotation_error_deg(result.R, expected_r) < 1e-5
    assert np.linalg.norm(result.t - expected_t) < 1e-9


def test_metric_depth_can_resolve_pure_rotation_without_visual_consensus() -> None:
    rng = np.random.default_rng(123)
    rotation = _rotation_exp(np.asarray((0.0, 0.2, -0.1)))
    points_a = rng.normal(size=(80, 3)) * (1.0, 1.5, 2.0) + (0.0, 0.0, 5.0)
    points_b = points_a @ rotation.T
    d1 = np.linalg.norm(points_a, axis=1)
    d2 = np.linalg.norm(points_b, axis=1)
    b1 = points_a / d1[:, None]
    b2 = points_b / d2[:, None]

    result = estimate_metric_rgbd_relative_pose(
        b1,
        b2,
        d1,
        d2,
        valid=np.ones(len(b1), dtype=bool),
        options=MetricRgbdPoseOptions(
            max_metric_error=1e-6,
            min_trials=8,
            max_trials=32,
            use_visual_consensus=False,
        ),
    )

    assert result is not None
    assert _rotation_error_deg(result.R, rotation) < 1e-5
    assert np.linalg.norm(result.t) < 1e-10


def test_validity_and_radial_depth_contract_fail_closed() -> None:
    b1, b2, d1, d2, _, _ = _scene(count=12)
    options = MetricRgbdPoseOptions(max_metric_error=0.1)
    with pytest.raises(TypeError, match="boolean dtype"):
        estimate_metric_rgbd_relative_pose(
            b1, b2, d1, d2, valid=np.ones(len(b1)), options=options
        )
    invalid_depth = d1.copy()
    invalid_depth[0] = 0.0
    with pytest.raises(ValueError, match="must be positive"):
        estimate_metric_rgbd_relative_pose(
            b1,
            b2,
            invalid_depth,
            d2,
            valid=np.ones(len(b1), dtype=bool),
            options=options,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        MetricRgbdPoseOptions(max_metric_error=0.0)
    with pytest.raises(ValueError, match="must not exceed 1"):
        MetricRgbdPoseAcceptancePolicy(min_inlier_ratio=1.1)


def test_quality_policy_rejection_is_visible_without_hiding_pose() -> None:
    b1, b2, d1, d2, _, _ = _scene(count=32)
    result = estimate_metric_rgbd_relative_pose(
        b1,
        b2,
        d1,
        d2,
        valid=np.ones(len(b1), dtype=bool),
        options=MetricRgbdPoseOptions(
            max_metric_error=1e-5,
            min_trials=8,
            max_trials=24,
            use_visual_consensus=False,
        ),
        acceptance_policy=MetricRgbdPoseAcceptancePolicy(min_inliers=40),
    )

    assert result is not None
    assert not result.quality_report.accepted
    assert result.quality_report.rejection_reasons == ("insufficient-metric-inliers",)
