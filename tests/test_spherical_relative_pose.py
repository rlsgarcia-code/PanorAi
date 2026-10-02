from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from panorai.estimators import (
    RelativePoseOptions,
    SphericalRelativePoseEstimator,
    estimate_relative_pose,
    spherical_tangent_sampson_error,
)
from panorai.features import SphericalBearingCorrespondences


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
    assert result.minimal_solver == "panorai-numerical-five-correspondence-v1"


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
