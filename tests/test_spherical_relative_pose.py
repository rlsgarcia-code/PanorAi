from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from panorai.estimators import (
    FivePointSample,
    RelativePoseOptions,
    SpatiallyWeightedFivePointSampler,
    SphericalRelativePoseEstimator,
    UniformFivePointSampler,
    estimate_relative_pose,
    spherical_tangent_sampson_error,
)
from panorai.features import SphericalBearingCorrespondences
from panorai.estimators.relative_pose import (
    _essential_pose_candidates,
    _pose_from_essential,
    _select_essential_pose_candidate,
)

ROOT = Path(__file__).resolve().parents[1]


def _rotation_exp(vector: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(vector)
    x, y, z = vector
    skew = np.asarray(((0, -z, y), (z, 0, -x), (-y, x, 0)), dtype=np.float64)
    return (
        np.eye(3)
        + math.sin(angle) / angle * skew
        + (1 - math.cos(angle)) / angle**2 * (skew @ skew)
    )


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray(((0, -z, y), (z, 0, -x), (-y, x, 0)), dtype=np.float64)


def _synthetic_bearings(
    *, count: int, seed: int, minimum_depth: float = 3.0, maximum_depth: float = 12.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Independent pinhole-free oracle from known 3D points and SE(3).

    Convention: ``point2 = R @ point1 + t``; bearings are unit vectors in each
    panorama frame.  Translation magnitude is fixed to one only to construct
    the scene and is not an observable expected output.
    """

    rng = np.random.default_rng(seed)
    rotation = _rotation_exp(np.asarray((0.08, -0.13, 0.04)))
    translation = np.asarray((0.8, 0.1, 0.25), dtype=np.float64)
    translation /= np.linalg.norm(translation)
    points1 = rng.uniform(
        (-3.0, -2.0, minimum_depth),
        (3.0, 2.0, maximum_depth),
        size=(count, 3),
    )
    points2 = points1 @ rotation.T + translation
    assert np.all(points1[:, 2] > 0) and np.all(points2[:, 2] > 0)
    bearings1 = points1 / np.linalg.norm(points1, axis=1, keepdims=True)
    bearings2 = points2 / np.linalg.norm(points2, axis=1, keepdims=True)
    return bearings1, bearings2, rotation, translation


def _rotation_error_deg(estimated: np.ndarray, expected: np.ndarray) -> float:
    cosine = np.clip((np.trace(estimated @ expected.T) - 1) / 2, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def _direction_error_deg(estimated: np.ndarray, expected: np.ndarray) -> float:
    return math.degrees(
        math.acos(float(np.clip(np.dot(estimated, expected), -1.0, 1.0)))
    )


def _add_tangent_noise(
    bearings: np.ndarray, rng: np.random.Generator, noise_deg: float
) -> np.ndarray:
    noise = rng.normal(size=bearings.shape)
    noise -= bearings * np.einsum("ni,ni->n", noise, bearings)[:, None]
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    angles = rng.normal(scale=math.radians(noise_deg), size=len(bearings))
    result = bearings * np.cos(angles)[:, None]
    result += noise * np.sin(angles)[:, None]
    return result / np.linalg.norm(result, axis=1, keepdims=True)


def _options(**changes) -> RelativePoseOptions:
    values = dict(
        max_angular_error_deg=0.15,
        confidence=0.999,
        min_inlier_ratio=0.1,
        min_num_trials=8,
        max_num_trials=80,
        min_inliers=10,
        local_optimization_steps=2,
        minimal_solver_starts=16,
        random_seed=17,
    )
    values.update(changes)
    return RelativePoseOptions(**values)


def test_exact_spherical_pose_recovers_known_rotation_and_translation_direction() -> (
    None
):
    bearings1, bearings2, expected_rotation, expected_translation = _synthetic_bearings(
        count=36, seed=20261001
    )
    original1 = bearings1.copy()
    original2 = bearings2.copy()

    result = estimate_relative_pose(bearings1, bearings2, options=_options())

    assert result is not None
    assert result.num_inliers == len(bearings1)
    assert _rotation_error_deg(result.R, expected_rotation) < 1e-5
    assert _direction_error_deg(result.t, expected_translation) < 1e-5
    assert np.allclose(
        result.essential_matrix,
        (_skew(result.t) @ result.R) / math.sqrt(2),
        atol=1e-10,
    )
    assert (
        np.max(
            np.abs(
                np.einsum(
                    "ni,ij,nj->n",
                    bearings2,
                    result.essential_matrix,
                    bearings1,
                )
            )
        )
        < 1e-10
    )
    assert np.array_equal(bearings1, original1)
    assert np.array_equal(bearings2, original2)
    assert not result.rotation.flags.writeable
    assert not result.translation_direction.flags.writeable
    assert result.describe()["translation"] == "unit-direction-only"
    assert result.minimal_solver.startswith("panorai-polynomial-action-matrix-v1")
    orientation = result.quality_report.translation_orientation
    assert orientation.hypothesis_count == 4
    assert orientation.best_positive_depth_count == result.num_inliers
    assert orientation.cheirality_margin > 0.5
    assert not orientation.ambiguous
    assert orientation.selection_method == "parallax-weighted"
    assert orientation.weighted_cheirality_margin == pytest.approx(
        orientation.cheirality_margin
    )
    assert orientation.reliable_correspondence_count >= 5


def test_parallax_weighting_resolves_sign_that_weak_raw_votes_reverse() -> None:
    """Regression oracle from known SE(3), not from the estimator under test.

    The scene contains five nearby points and 395 points at 500--2000 radial
    units. Independent 0.15-degree tangent noise makes the far-point depth
    signs unstable. The historical raw count chooses ``-t`` for the recorded
    seed; bounded parallax evidence keeps the five geometrically informative
    votes and recovers the generating direction.
    """

    rng = np.random.default_rng(10018)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    rotation = _rotation_exp(axis * math.radians(rng.uniform(2.0, 14.0)))
    translation = rng.normal(size=3)
    translation /= np.linalg.norm(translation)
    directions = rng.normal(size=(400, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    depths = np.empty(400)
    depths[:5] = rng.uniform(1.0, 3.0, size=5)
    depths[5:] = rng.uniform(500.0, 2000.0, size=395)
    rng.shuffle(depths)
    points1 = directions * depths[:, None]
    points2 = points1 @ rotation.T + 0.25 * translation
    bearings1 = points1 / np.linalg.norm(points1, axis=1, keepdims=True)
    bearings2 = points2 / np.linalg.norm(points2, axis=1, keepdims=True)
    bearings1 = _add_tangent_noise(bearings1, rng, 0.15)
    bearings2 = _add_tangent_noise(bearings2, rng, 0.15)
    essential = _skew(translation) @ rotation
    essential /= np.linalg.norm(essential)

    baseline = _pose_from_essential(
        essential,
        bearings1,
        bearings2,
        _options(translation_orientation_method="positive-depth-count"),
    )
    weighted = _pose_from_essential(
        essential,
        bearings1,
        bearings2,
        _options(
            translation_orientation_method="parallax-weighted",
            translation_orientation_parallax_scale_deg=1.0,
        ),
    )

    assert baseline is not None and weighted is not None
    assert _direction_error_deg(baseline[1], translation) > 179.0
    assert _direction_error_deg(weighted[1], translation) < 1e-5


def test_parallax_weighting_abstains_without_five_reliable_rays() -> None:
    rng = np.random.default_rng(40001)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    rotation = _rotation_exp(axis * math.radians(rng.uniform(2.0, 14.0)))
    translation = rng.normal(size=3)
    translation /= np.linalg.norm(translation)
    directions = rng.normal(size=(400, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    depths = rng.uniform(500.0, 2000.0, size=400)
    points1 = directions * depths[:, None]
    points2 = points1 @ rotation.T + 0.25 * translation
    bearings1 = points1 / np.linalg.norm(points1, axis=1, keepdims=True)
    bearings2 = points2 / np.linalg.norm(points2, axis=1, keepdims=True)
    bearings1 = _add_tangent_noise(bearings1, rng, 0.15)
    bearings2 = _add_tangent_noise(bearings2, rng, 0.15)
    essential = _skew(translation) @ rotation
    essential /= np.linalg.norm(essential)

    candidates = _essential_pose_candidates(
        essential,
        bearings1,
        bearings2,
        parallax_scale_deg=1.0,
    )
    selected, applied_method = _select_essential_pose_candidate(
        candidates, "parallax-weighted"
    )

    assert selected.reliable_correspondence_count < 5
    assert applied_method == "positive-depth-count-fallback"


def test_lo_ransac_rejects_seeded_outliers_and_is_deterministic() -> None:
    bearings1, bearings2, expected_rotation, expected_translation = _synthetic_bearings(
        count=100, seed=42
    )
    rng = np.random.default_rng(9001)
    noise = math.radians(0.03)
    bearings1 = bearings1 + rng.normal(scale=noise, size=bearings1.shape)
    bearings2 = bearings2 + rng.normal(scale=noise, size=bearings2.shape)
    bearings1 /= np.linalg.norm(bearings1, axis=1, keepdims=True)
    bearings2 /= np.linalg.norm(bearings2, axis=1, keepdims=True)
    outlier_indices = rng.choice(len(bearings1), size=25, replace=False)
    bearings2[outlier_indices] = rng.normal(size=(25, 3))
    bearings2[outlier_indices] /= np.linalg.norm(
        bearings2[outlier_indices], axis=1, keepdims=True
    )
    options = _options(
        max_angular_error_deg=0.2,
        min_num_trials=20,
        max_num_trials=200,
        min_inliers=40,
        random_seed=7,
    )

    first = estimate_relative_pose(bearings1, bearings2, options=options)
    second = estimate_relative_pose(bearings1, bearings2, options=options)

    assert first is not None and second is not None
    assert np.array_equal(first.inlier_mask, second.inlier_mask)
    assert np.array_equal(first.R, second.R)
    assert np.array_equal(first.t, second.t)
    assert first.num_trials == second.num_trials
    assert _rotation_error_deg(first.R, expected_rotation) < 0.3
    assert _direction_error_deg(first.t, expected_translation) < 1.0
    assert first.num_inliers >= 70
    assert not first.inlier_mask[outlier_indices].any()


def test_tangent_sampson_residual_matches_independent_finite_difference() -> None:
    bearings1, bearings2, rotation, translation = _synthetic_bearings(count=2, seed=4)
    bearings2 = bearings2.copy()
    bearings2[0] += np.asarray((0.002, -0.001, 0.0005))
    bearings2[0] /= np.linalg.norm(bearings2[0])
    essential = _skew(translation) @ rotation
    actual = spherical_tangent_sampson_error(bearings1[:1], bearings2[:1], essential)[0]

    def tangent_basis(direction: np.ndarray) -> np.ndarray:
        axis = np.zeros(3)
        axis[np.argmin(np.abs(direction))] = 1
        first = np.cross(direction, axis)
        first /= np.linalg.norm(first)
        return np.stack((first, np.cross(direction, first)), axis=1)

    def constraint(ray1: np.ndarray, ray2: np.ndarray) -> float:
        return float(ray2 @ essential @ ray1)

    epsilon = 1e-7
    derivatives = []
    for which, bearing in enumerate((bearings1[0], bearings2[0])):
        for tangent in tangent_basis(bearing).T:
            plus = bearing + epsilon * tangent
            minus = bearing - epsilon * tangent
            plus /= np.linalg.norm(plus)
            minus /= np.linalg.norm(minus)
            if which == 0:
                derivative = (
                    constraint(plus, bearings2[0]) - constraint(minus, bearings2[0])
                ) / (2 * epsilon)
            else:
                derivative = (
                    constraint(bearings1[0], plus) - constraint(bearings1[0], minus)
                ) / (2 * epsilon)
            derivatives.append(derivative)
    expected = abs(constraint(bearings1[0], bearings2[0])) / np.linalg.norm(derivatives)
    assert actual == pytest.approx(expected, rel=2e-7, abs=1e-12)


def test_explicit_validity_allows_invalid_storage_without_inference() -> None:
    bearings1, bearings2, _, _ = _synthetic_bearings(count=24, seed=81)
    valid = np.ones(24, dtype=bool)
    valid[[2, 9, 17]] = False
    bearings1[[2, 9, 17]] = np.nan
    bearings2[[2, 9, 17]] = 0
    correspondences = SphericalBearingCorrespondences(
        bearings_a=bearings1,
        bearings_b=bearings2,
        weights=valid.astype(np.float32),
        valid=valid,
    )
    estimator = SphericalRelativePoseEstimator(_options(min_inliers=10))

    result = estimator.estimate(correspondences)

    assert result is not None
    assert not result.inlier_mask[~valid].any()
    assert np.isinf(result.residuals_rad[~valid]).all()
    assert result.num_inliers == int(valid.sum())
    assert "random_seed=17" in repr(estimator)
    with pytest.raises(TypeError, match="boolean dtype"):
        estimator.estimate(bearings1, bearings2, valid=valid.astype(np.uint8))
    with pytest.raises(ValueError, match="finite"):
        estimator.estimate(bearings1, bearings2)
    with pytest.raises(ValueError, match="must be omitted"):
        estimator.estimate(correspondences, valid=valid)


def test_low_parallax_is_returned_but_explicitly_marked_degenerate() -> None:
    bearings1, bearings2, _, _ = _synthetic_bearings(
        count=30, seed=19, minimum_depth=1000.0, maximum_depth=2000.0
    )
    result = estimate_relative_pose(
        bearings1,
        bearings2,
        options=_options(
            max_angular_error_deg=0.01,
            min_median_parallax_deg=1.0,
            max_num_trials=100,
        ),
    )

    assert result is not None
    assert result.degenerate
    assert "low-parallax" in result.degeneracy_reasons
    assert result.median_parallax_deg < 1.0


def test_insufficient_or_incompatible_inputs_fail_explicitly() -> None:
    bearings = np.tile((0.0, 0.0, 1.0), (7, 1))
    assert (
        estimate_relative_pose(
            bearings,
            bearings,
            options=_options(min_inliers=8),
        )
        is None
    )
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        estimate_relative_pose(np.zeros((3, 2)), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="same shape"):
        estimate_relative_pose(np.zeros((5, 3)), np.zeros((6, 3)))
    with pytest.raises(ValueError, match="at least 8"):
        RelativePoseOptions(minimal_solver_starts=7)
    with pytest.raises(ValueError, match="must not exceed"):
        RelativePoseOptions(min_num_trials=5, max_num_trials=4)
    with pytest.raises(ValueError, match="max_angular_error_deg"):
        RelativePoseOptions(max_angular_error_deg=0)
    with pytest.raises(ValueError, match="confidence"):
        RelativePoseOptions(confidence=1)


def test_minimum_inlier_ratio_is_enforced() -> None:
    bearings1, bearings2, _, _ = _synthetic_bearings(count=20, seed=123)
    rng = np.random.default_rng(456)
    bearings2[10:] = rng.normal(size=(10, 3))
    bearings2[10:] /= np.linalg.norm(bearings2[10:], axis=1, keepdims=True)

    result = estimate_relative_pose(
        bearings1,
        bearings2,
        options=_options(
            max_num_trials=200,
            min_inliers=5,
            min_inlier_ratio=0.75,
        ),
    )

    assert result is None


def test_estimator_import_is_numpy_scipy_lazy_and_backend_independent() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys; import panorai.estimators; "
            "print(json.dumps(sorted(set(sys.modules) & "
            "{'cv2','pycolmap','torch','scipy'})))",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []


def test_spatial_sampler_is_deterministic_diverse_and_conditioned() -> None:
    bearings1, bearings2, _, _ = _synthetic_bearings(count=80, seed=2027)
    active = np.arange(len(bearings1), dtype=np.int64)
    sampler = SpatiallyWeightedFivePointSampler(
        min_angular_separation_deg=5.0,
        min_unique_cells=3,
        uniform_trial_probability=0.0,
    )
    first_prepared = sampler.prepare(bearings1, bearings2, active)
    second_prepared = sampler.prepare(bearings1, bearings2, active)

    first = first_prepared.draw(np.random.default_rng(91))
    second = second_prepared.draw(np.random.default_rng(91))

    assert first is not None and second is not None
    assert np.array_equal(first.indices, second.indices)
    assert first.strategy == sampler.name
    assert first.relaxation_level == 0
    assert first.min_separation_a_deg >= 5.0 - 1e-10
    assert first.min_separation_b_deg >= 5.0 - 1e-10
    assert first.unique_cells_a >= 3
    assert first.unique_cells_b >= 3
    assert first.design_condition_number <= sampler.max_design_condition_number
    assert not first.indices.flags.writeable


def test_spatial_sampler_relaxes_to_uniform_without_prefiltering() -> None:
    rng = np.random.default_rng(44)
    bearings1 = np.tile((0.0, 0.0, 1.0), (20, 1))
    bearings1 += rng.normal(scale=math.radians(0.2), size=bearings1.shape)
    bearings1 /= np.linalg.norm(bearings1, axis=1, keepdims=True)
    rotation = _rotation_exp(np.asarray((0.01, -0.02, 0.005)))
    bearings2 = bearings1 @ rotation.T
    active = np.arange(20, dtype=np.int64)
    sampler = SpatiallyWeightedFivePointSampler(
        min_angular_separation_deg=30.0,
        min_unique_cells=4,
        uniform_trial_probability=0.0,
        attempts_per_level=4,
    )

    sample = sampler.prepare(bearings1, bearings2, active).draw(
        np.random.default_rng(5)
    )

    assert sample is not None
    assert sample.strategy == "uniform-fallback-five-point-v1"
    assert len(sample.indices) == 5
    assert np.all(np.isin(sample.indices, active))


def test_spatial_sampler_validates_weights_and_keeps_zero_weight_rows_eligible() -> (
    None
):
    bearings1, bearings2, _, _ = _synthetic_bearings(count=12, seed=505)
    active = np.arange(12, dtype=np.int64)
    sampler = SpatiallyWeightedFivePointSampler(uniform_trial_probability=0.0)
    weights = np.ones(12)
    weights[:7] = 0.0

    prepared = sampler.prepare(bearings1, bearings2, active, weights)

    assert np.all(prepared.base_weights > 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        sampler.prepare(bearings1, bearings2, active, -np.ones(12))
    with pytest.raises(ValueError, match=r"shape \(N,\)"):
        sampler.prepare(bearings1, bearings2, active, np.ones(11))


def test_injected_sampler_proposes_five_but_ransac_scores_every_valid_match() -> None:
    bearings1, bearings2, _, _ = _synthetic_bearings(count=30, seed=606)
    weights = np.linspace(0.1, 1.0, len(bearings1))

    class PreparedRecordingSampler:
        name = "recording-five-point-v1"
        supports_uniform_trial_bound = False

        def __init__(self) -> None:
            self.draws = 0

        def draw(self, rng):
            self.draws += 1
            return FivePointSample(
                indices=np.arange(5),
                strategy=self.name,
                relaxation_level=0,
                min_separation_a_deg=1.0,
                min_separation_b_deg=1.0,
                unique_cells_a=5,
                unique_cells_b=5,
                design_condition_number=10.0,
            )

    class RecordingSampler:
        name = "recording-five-point-v1"
        supports_uniform_trial_bound = False

        def __init__(self) -> None:
            self.active = None
            self.weights = None
            self.prepared = PreparedRecordingSampler()

        def prepare(self, b1, b2, active, sampling_weights=None):
            self.active = np.array(active, copy=True)
            self.weights = np.array(sampling_weights, copy=True)
            return self.prepared

        def describe(self):
            return {"name": self.name, "prefilters_correspondences": False}

    sampler = RecordingSampler()
    result = estimate_relative_pose(
        bearings1,
        bearings2,
        sampling_weights=weights,
        sampler=sampler,
        options=_options(
            min_num_trials=1,
            max_num_trials=1,
            min_inliers=10,
        ),
    )

    assert result is not None
    assert np.array_equal(sampler.active, np.arange(30))
    assert np.array_equal(sampler.weights, weights)
    assert sampler.prepared.draws == 1
    assert result.num_inliers == 30
    assert result.sampling_diagnostics.samples_drawn == 1
    assert not result.sampling_diagnostics.adaptive_uniform_trial_bound_enabled


def test_uniform_sampler_preserves_classic_adaptive_trial_semantics() -> None:
    bearings1, bearings2, _, _ = _synthetic_bearings(count=30, seed=707)
    result = estimate_relative_pose(
        bearings1,
        bearings2,
        sampler=UniformFivePointSampler(),
        options=_options(min_num_trials=4, max_num_trials=20),
    )

    assert result is not None
    diagnostics = result.sampling_diagnostics
    assert diagnostics.sampler_name == "uniform-five-point-v1"
    assert diagnostics.uniform_samples == diagnostics.samples_drawn
    assert diagnostics.strict_spatial_samples == 0
    assert diagnostics.adaptive_uniform_trial_bound_enabled
