from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from panorai.estimators import (
    RelativePoseAcceptancePolicy,
    RelativePoseConfidenceCalibrator,
    RelativePoseOptions,
    estimate_relative_pose,
    solve_five_point_essential,
)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def _rotation_exp(vector: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(vector)
    cross = _skew(vector)
    return (
        np.eye(3)
        + math.sin(angle) / angle * cross
        + (1.0 - math.cos(angle)) / angle**2 * (cross @ cross)
    )


def _scene(
    *, count: int, seed: int, radial_min: float = 4.0, radial_max: float = 12.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rotation = _rotation_exp(np.asarray((0.08, -0.13, 0.04)))
    translation = np.asarray((0.8, 0.1, 0.25))
    translation /= np.linalg.norm(translation)
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    points_a = directions * rng.uniform(radial_min, radial_max, size=(count, 1))
    points_b = points_a @ rotation.T + translation
    bearings_a = points_a / np.linalg.norm(points_a, axis=1, keepdims=True)
    bearings_b = points_b / np.linalg.norm(points_b, axis=1, keepdims=True)
    return bearings_a, bearings_b, rotation, translation


def _options(**changes) -> RelativePoseOptions:
    values = dict(
        max_angular_error_deg=0.2,
        min_num_trials=12,
        max_num_trials=80,
        min_inliers=10,
        local_optimization_steps=2,
        random_seed=29,
        stability_trials=5,
        model_competition_trials=80,
    )
    values.update(changes)
    return RelativePoseOptions(**values)


def _rotation_error_deg(estimated: np.ndarray, expected: np.ndarray) -> float:
    cosine = np.clip((np.trace(estimated @ expected.T) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def _direction_error_deg(estimated: np.ndarray, expected: np.ndarray) -> float:
    cosine = np.clip(np.dot(estimated, expected), -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


@pytest.mark.parametrize("seed", [501, 511, 521, 531])
def test_polynomial_five_point_solver_enumerates_ground_truth_root(seed: int) -> None:
    b1, b2, rotation, translation = _scene(count=5, seed=seed)
    expected = _skew(translation) @ rotation
    expected /= np.linalg.norm(expected)

    solutions = solve_five_point_essential(b1, b2)

    assert 1 <= len(solutions) <= 40
    distances = [
        min(np.linalg.norm(item - expected), np.linalg.norm(item + expected))
        for item in solutions
    ]
    assert min(distances) < 1e-8
    for essential in solutions:
        epipolar = np.einsum("ni,ij,nj->n", b2, essential, b1)
        eet = essential @ essential.T
        cubic = 2.0 * eet @ essential - np.trace(eet) * essential
        assert np.max(np.abs(epipolar)) < 1e-8
        assert np.linalg.norm(cubic) < 1e-6


def test_quality_report_accepts_stable_wide_spherical_pose() -> None:
    b1, b2, rotation, translation = _scene(count=80, seed=502)

    result = estimate_relative_pose(b1, b2, options=_options())

    assert result is not None
    assert _rotation_error_deg(result.R, rotation) < 1e-5
    assert _direction_error_deg(result.t, translation) < 1e-5
    quality = result.quality_report
    assert quality.accepted
    assert quality.rejection_reasons == ()
    assert quality.occupied_cells_a >= 12
    assert quality.occupied_cells_b >= 12
    assert quality.stability.successful_trials == 5
    assert quality.stability.rotation_p90_deg < 1e-5
    assert quality.stability.translation_p90_deg < 1e-5
    assert quality.model_competition.preferred_model == "essential"
    assert quality.model_competition.essential_score_margin > 0.5
    assert 0.0 <= quality.raw_quality_score <= 1.0


def test_low_parallax_pose_is_returned_but_quality_gate_rejects_it() -> None:
    b1, b2, _, _ = _scene(count=80, seed=503, radial_min=2000.0, radial_max=4000.0)

    result = estimate_relative_pose(
        b1,
        b2,
        options=_options(max_angular_error_deg=0.02, min_median_parallax_deg=0.5),
    )

    assert result is not None
    quality = result.quality_report
    assert not quality.accepted
    assert "low-parallax" in quality.rejection_reasons
    assert quality.model_competition.rotation_only.num_inliers >= 40


def test_near_pure_rotation_wins_conservative_model_tie() -> None:
    b1, b2, _, _ = _scene(
        count=80,
        seed=508,
        radial_min=10_000.0,
        radial_max=20_000.0,
    )

    result = estimate_relative_pose(
        b1,
        b2,
        options=_options(min_median_parallax_deg=0.0),
    )

    assert result is not None
    quality = result.quality_report
    assert quality.model_competition.rotation_only.num_inliers == len(b1)
    assert quality.model_competition.preferred_model == "rotation-only"
    assert not quality.accepted
    assert "competing-model:rotation-only" in quality.rejection_reasons


def test_stability_can_be_explicitly_disabled_by_policy() -> None:
    b1, b2, _, _ = _scene(count=80, seed=507)
    policy = RelativePoseAcceptancePolicy(require_stability=False)

    result = estimate_relative_pose(
        b1,
        b2,
        options=_options(stability_trials=0),
        quality_policy=policy,
    )

    assert result is not None
    assert result.quality_report.accepted
    assert result.quality_report.stability.successful_trials == 0
    assert not any(
        reason.startswith("unstable-")
        for reason in result.quality_report.rejection_reasons
    )


def test_planar_model_competition_can_be_required_without_hiding_pose() -> None:
    rng = np.random.default_rng(504)
    rotation = _rotation_exp(np.asarray((0.04, -0.1, 0.03)))
    translation = np.asarray((0.7, 0.2, 0.1))
    translation /= np.linalg.norm(translation)
    points_a = np.column_stack(
        (rng.uniform(-4, 4, 80), rng.uniform(-3, 3, 80), np.full(80, 6.0))
    )
    points_b = points_a @ rotation.T + translation
    b1 = points_a / np.linalg.norm(points_a, axis=1, keepdims=True)
    b2 = points_b / np.linalg.norm(points_b, axis=1, keepdims=True)
    policy = RelativePoseAcceptancePolicy(min_essential_score_margin=0.05)

    result = estimate_relative_pose(b1, b2, options=_options(), quality_policy=policy)

    assert result is not None
    assert (
        result.quality_report.model_competition.spherical_homography.num_inliers >= 75
    )
    assert result.quality_report.model_competition.essential_score_margin < 0.05
    assert not result.quality_report.accepted
    assert "weak-essential-margin" in result.quality_report.rejection_reasons


def test_scale_marginal_scoring_and_irls_recover_noisy_outliers() -> None:
    b1, b2, rotation, translation = _scene(count=120, seed=505)
    rng = np.random.default_rng(1505)
    noise = math.radians(0.04)
    b1 += rng.normal(scale=noise, size=b1.shape)
    b2 += rng.normal(scale=noise, size=b2.shape)
    b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
    b2 /= np.linalg.norm(b2, axis=1, keepdims=True)
    outliers = rng.choice(len(b1), size=30, replace=False)
    b2[outliers] = rng.normal(size=(len(outliers), 3))
    b2[outliers] /= np.linalg.norm(b2[outliers], axis=1, keepdims=True)

    result = estimate_relative_pose(
        b1,
        b2,
        options=_options(
            max_angular_error_deg=0.25,
            max_num_trials=180,
            min_inliers=50,
            robust_refinement_steps=3,
        ),
    )

    assert result is not None
    assert result.robust_estimator == "panorai-scale-marginal-lo-ransac-v1"
    assert result.num_inliers >= 85
    assert not result.inlier_mask[outliers].any()
    assert _rotation_error_deg(result.R, rotation) < 0.3
    assert _direction_error_deg(result.t, translation) < 1.0


def test_confidence_calibration_requires_disjoint_evaluation_ids() -> None:
    b1, b2, _, _ = _scene(count=50, seed=506)
    base = estimate_relative_pose(b1, b2, options=_options(max_num_trials=30))
    assert base is not None
    reports = [
        replace(base.quality_report, raw_quality_score=value)
        for value in (0.1, 0.25, 0.5, 0.7, 0.9, 0.95)
    ]
    calibrator = RelativePoseConfidenceCalibrator.fit(
        reports[:4],
        [False, False, True, True],
        sample_ids=["cal-a", "cal-b", "cal-c", "cal-d"],
    )

    probabilities = calibrator.predict_proba(reports[4:])
    assert probabilities.shape == (2,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    evaluation = calibrator.evaluate(
        reports[4:], [True, True], sample_ids=["test-a", "test-b"]
    )
    assert evaluation.count == 2
    assert 0.0 <= evaluation.brier_score <= 1.0
    with pytest.raises(ValueError, match="overlap calibration"):
        calibrator.evaluate(reports[4:], [True, True], sample_ids=["cal-a", "test-b"])
    with pytest.raises(ValueError, match="bins must be positive"):
        calibrator.evaluate(
            reports[4:], [True, True], sample_ids=["test-a", "test-b"], bins=0
        )


def test_acceptance_rejects_ambiguous_translation_orientation() -> None:
    b1, b2, _, _ = _scene(count=80, seed=507)
    base = estimate_relative_pose(b1, b2, options=_options(max_num_trials=40))
    assert base is not None
    ambiguous = replace(
        base.quality_report,
        translation_orientation=replace(
            base.quality_report.translation_orientation,
            alternative_positive_depth_count=base.num_inliers,
            cheirality_margin=0.0,
            ambiguous=True,
        ),
    )
    policy = RelativePoseAcceptancePolicy(
        min_inliers=5,
        min_inlier_ratio=0.0,
        min_occupied_cells=1,
        min_coverage_entropy=0.0,
        max_median_residual_deg=2.0,
        min_median_parallax_deg=0.0,
        min_cheirality_ratio=0.0,
        min_translation_orientation_margin=0.05,
        require_stability=False,
        require_essential_preferred=False,
    )

    decided = ambiguous.with_decision(policy)

    assert not decided.accepted
    assert "ambiguous-translation-orientation" in decided.rejection_reasons
