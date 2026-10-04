from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from panorai.estimators import (
    RelativePoseAcceptancePolicy,
    RelativePoseOptions,
    estimate_relative_pose,
    native_kernels_available,
    solve_five_point_essential,
    spherical_tangent_sampson_error,
)
from panorai.estimators._five_point import _chart_coefficient_matrix
from panorai.estimators._native import (
    native_five_point_coefficients,
    resolve_compute_backend,
)
from panorai._native import _essential as native_extension


pytestmark = pytest.mark.skipif(
    not native_kernels_available(), reason="optional native kernels are not built"
)


def _five_point_fixture():
    rng = np.random.default_rng(20261003)
    points = rng.uniform((-2.0, -1.0, 4.0), (2.0, 1.0, 8.0), size=(5, 3))
    angle = np.deg2rad(4.0)
    rotation = np.asarray(
        (
            (np.cos(angle), 0.0, np.sin(angle)),
            (0.0, 1.0, 0.0),
            (-np.sin(angle), 0.0, np.cos(angle)),
        )
    )
    translation = np.asarray((0.8, 0.1, 0.05))
    first = points / np.linalg.norm(points, axis=1, keepdims=True)
    transformed = points @ rotation.T + translation
    second = transformed / np.linalg.norm(transformed, axis=1, keepdims=True)
    return first, second


def _projective_distance(first: np.ndarray, second: np.ndarray) -> float:
    return min(np.linalg.norm(first - second), np.linalg.norm(first + second))


def test_native_polynomial_coefficients_match_numpy_reference():
    rng = np.random.default_rng(71)
    nullspace = rng.normal(size=(9, 4))
    native = native_five_point_coefficients(nullspace)
    assert native.shape == (4, 10, 20)
    for constant_index in range(4):
        variables = tuple(index for index in range(4) if index != constant_index)
        expected = _chart_coefficient_matrix(nullspace, variables, constant_index)
        np.testing.assert_allclose(
            native[constant_index], expected, atol=1e-14, rtol=1e-14
        )


def test_private_extension_rejects_non_float64_buffers():
    with pytest.raises(ValueError, match="float64"):
        native_extension.five_point_coefficients(np.zeros((9, 4), dtype=np.int64))
    with pytest.raises(ValueError, match="float64"):
        native_extension.sampson_residuals(
            np.zeros((2, 3), dtype=np.int64),
            np.zeros((2, 3), dtype=np.float64),
            np.eye(3, dtype=np.float64),
            False,
        )


def test_native_five_point_solution_set_matches_numpy_reference():
    first, second = _five_point_fixture()
    expected = solve_five_point_essential(first, second, backend="numpy")
    actual = solve_five_point_essential(first, second, backend="native")
    assert len(expected) == len(actual) > 0
    for candidate in expected:
        assert min(_projective_distance(candidate, item) for item in actual) < 1e-8
    for candidate in actual:
        assert min(_projective_distance(candidate, item) for item in expected) < 1e-8


def test_native_spherical_sampson_residual_matches_numpy_and_squared_contract():
    rng = np.random.default_rng(19)
    first = rng.normal(size=(257, 3))
    second = rng.normal(size=(257, 3))
    essential = rng.normal(size=(3, 3))
    expected = spherical_tangent_sampson_error(
        first, second, essential, backend="numpy"
    )
    actual = spherical_tangent_sampson_error(first, second, essential, backend="native")
    squared = spherical_tangent_sampson_error(
        first, second, essential, squared=True, backend="native"
    )
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(squared, expected**2, atol=1e-14, rtol=1e-14)


def _relative_scene():
    rng = np.random.default_rng(37)
    points = rng.uniform((-2.0, -1.5, 4.0), (2.0, 1.5, 9.0), size=(80, 3))
    angle = np.deg2rad(3.0)
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    translation = np.asarray((0.6, 0.08, 0.03))
    first = points / np.linalg.norm(points, axis=1, keepdims=True)
    transformed = points @ rotation.T + translation
    second = transformed / np.linalg.norm(transformed, axis=1, keepdims=True)
    second[-12:] = second[-12:][::-1]
    return first, second


def test_relative_pose_native_and_numpy_backends_preserve_public_decision():
    first, second = _relative_scene()
    base = RelativePoseOptions(
        max_angular_error_deg=0.2,
        min_num_trials=16,
        max_num_trials=40,
        min_inliers=12,
        local_optimization_steps=1,
        stability_trials=0,
        model_competition_trials=16,
        random_seed=11,
        compute_backend="numpy",
    )
    policy = RelativePoseAcceptancePolicy(
        min_inliers=12,
        min_inlier_ratio=0.2,
        min_occupied_cells=1,
        min_coverage_entropy=0.0,
        max_median_residual_deg=1.0,
        min_median_parallax_deg=0.0,
        min_cheirality_ratio=0.0,
        min_translation_orientation_margin=0.0,
        require_stability=False,
        require_essential_preferred=False,
    )
    expected = estimate_relative_pose(
        first, second, options=base, quality_policy=policy
    )
    actual = estimate_relative_pose(
        first,
        second,
        options=replace(base, compute_backend="native"),
        quality_policy=policy,
    )
    assert expected is not None and actual is not None
    assert expected.compute_backend == "numpy"
    assert actual.compute_backend == "native"
    assert expected.quality_report.accepted == actual.quality_report.accepted
    np.testing.assert_array_equal(actual.inlier_mask, expected.inlier_mask)
    np.testing.assert_allclose(actual.R, expected.R, atol=1e-8, rtol=1e-8)
    assert abs(float(actual.t @ expected.t)) > 1.0 - 1e-8
    assert actual.describe()["compute_backend"] == "native"


def test_auto_selects_native_and_explicit_native_fails_when_unavailable(monkeypatch):
    assert resolve_compute_backend("auto") == "native"
    import panorai.estimators._native as dispatch

    monkeypatch.setattr(dispatch, "_native_essential", None)
    monkeypatch.setattr(dispatch, "_native_import_error", ImportError("test missing"))
    assert dispatch.resolve_compute_backend("auto") == "numpy"
    with pytest.raises(RuntimeError, match="test missing"):
        dispatch.resolve_compute_backend("native")
    with pytest.raises(ValueError, match="auto.*numpy.*native"):
        dispatch.resolve_compute_backend("unknown")
