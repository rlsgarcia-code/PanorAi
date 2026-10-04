from __future__ import annotations

import numpy as np
import pytest

from panorai.reconstruction import SphericalGlobalMapper, SphericalGlobalMapperOptions
from panorai.reconstruction._mapper import _spherical_log_residual_batch
from panorai.reconstruction._math import rotation_exp
from panorai.reconstruction._native import (
    native_bundle_kernels_available,
    resolve_bundle_backend,
    spherical_ba_residual_jacobians,
)


pytestmark = pytest.mark.skipif(
    not native_bundle_kernels_available(),
    reason="optional native spherical-BA kernel is not built",
)


def _fixture():
    rng = np.random.default_rng(20261004)
    camera_count = 4
    point_count = 13
    observation_count = 31
    base_rotations = np.stack(
        [rotation_exp(rng.normal(scale=0.2, size=3)) for _ in range(camera_count)]
    )
    rotation_deltas = rng.normal(scale=0.08, size=(camera_count, 3))
    rotations = np.stack(
        [
            rotation_exp(rotation_deltas[index]) @ base_rotations[index]
            for index in range(camera_count)
        ]
    )
    centers = rng.normal(scale=0.5, size=(camera_count, 3))
    points = rng.normal(size=(point_count, 3))
    points[:, 2] += 5.0
    camera_indices = rng.integers(
        0, camera_count, size=observation_count, dtype=np.int64
    )
    point_indices = rng.integers(0, point_count, size=observation_count, dtype=np.int64)
    vectors = np.einsum(
        "nij,nj->ni",
        rotations[camera_indices],
        points[point_indices] - centers[camera_indices],
    )
    predicted = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    measured = predicted + rng.normal(scale=0.03, size=predicted.shape)
    measured /= np.linalg.norm(measured, axis=1, keepdims=True)
    return (
        measured,
        base_rotations,
        rotations,
        centers,
        points,
        camera_indices,
        point_indices,
        rotation_deltas,
        predicted,
    )


def test_native_spherical_ba_residuals_match_numpy_oracle():
    (
        measured,
        _,
        rotations,
        centers,
        points,
        camera_indices,
        point_indices,
        rotation_deltas,
        predicted,
    ) = _fixture()
    residuals, rotation_blocks, center_blocks, point_blocks = (
        spherical_ba_residual_jacobians(
            measured,
            rotations,
            centers,
            points,
            camera_indices,
            point_indices,
            rotation_deltas,
        )
    )
    expected = _spherical_log_residual_batch(measured, predicted)
    np.testing.assert_allclose(residuals, expected, atol=5e-14, rtol=5e-14)
    assert rotation_blocks.shape == (len(measured), 2, 3)
    assert center_blocks.shape == (len(measured), 2, 3)
    assert point_blocks.shape == (len(measured), 2, 3)
    np.testing.assert_allclose(center_blocks, -point_blocks, atol=1e-14, rtol=0)


def test_native_spherical_ba_handles_coincident_antipodal_and_zero_range():
    measured = np.asarray(((0.0, 0.0, 1.0),) * 3)
    rotations = np.eye(3)[None]
    centers = np.zeros((1, 3))
    points = np.asarray(((0.0, 0.0, 2.0), (0.0, 0.0, -2.0), (0.0, 0.0, 0.0)))
    camera_indices = np.zeros(3, dtype=np.int64)
    point_indices = np.arange(3, dtype=np.int64)
    residuals, rotation_blocks, center_blocks, point_blocks = (
        spherical_ba_residual_jacobians(
            measured,
            rotations,
            centers,
            points,
            camera_indices,
            point_indices,
            np.zeros((1, 3)),
        )
    )

    np.testing.assert_allclose(residuals[0], 0.0, atol=0.0)
    np.testing.assert_allclose(residuals[1], (np.pi, 0.0), atol=0.0)
    np.testing.assert_allclose(residuals[2], (np.pi, np.pi), atol=0.0)
    assert np.all(np.isfinite(rotation_blocks))
    assert np.all(np.isfinite(center_blocks))
    assert np.all(np.isfinite(point_blocks))


def test_native_spherical_ba_analytic_blocks_match_central_differences():
    (
        measured,
        base_rotations,
        rotations,
        centers,
        points,
        camera_indices,
        point_indices,
        rotation_deltas,
        _,
    ) = _fixture()
    residuals, rotation_blocks, center_blocks, point_blocks = (
        spherical_ba_residual_jacobians(
            measured,
            rotations,
            centers,
            points,
            camera_indices,
            point_indices,
            rotation_deltas,
        )
    )
    del residuals
    epsilon = 1e-7
    for observation_index in (0, 7, 19):
        camera = int(camera_indices[observation_index])
        point = int(point_indices[observation_index])

        def evaluate(delta_values, center_values, point_values):
            current_rotations = np.stack(
                [
                    rotation_exp(delta_values[index]) @ base_rotations[index]
                    for index in range(len(base_rotations))
                ]
            )
            return spherical_ba_residual_jacobians(
                measured,
                current_rotations,
                center_values,
                point_values,
                camera_indices,
                point_indices,
                delta_values,
            )[0][observation_index]

        for name, expected in (
            ("rotation", rotation_blocks[observation_index]),
            ("center", center_blocks[observation_index]),
            ("point", point_blocks[observation_index]),
        ):
            finite_difference = np.empty((2, 3), dtype=np.float64)
            for axis in range(3):
                lower_delta = rotation_deltas.copy()
                upper_delta = rotation_deltas.copy()
                lower_centers = centers.copy()
                upper_centers = centers.copy()
                lower_points = points.copy()
                upper_points = points.copy()
                if name == "rotation":
                    lower_delta[camera, axis] -= epsilon
                    upper_delta[camera, axis] += epsilon
                elif name == "center":
                    lower_centers[camera, axis] -= epsilon
                    upper_centers[camera, axis] += epsilon
                else:
                    lower_points[point, axis] -= epsilon
                    upper_points[point, axis] += epsilon
                finite_difference[:, axis] = (
                    evaluate(upper_delta, upper_centers, upper_points)
                    - evaluate(lower_delta, lower_centers, lower_points)
                ) / (2.0 * epsilon)
            np.testing.assert_allclose(
                expected, finite_difference, atol=2e-7, rtol=2e-6
            )


def test_native_bundle_dispatch_and_input_guards(monkeypatch):
    assert resolve_bundle_backend("auto") == "native"
    assert SphericalGlobalMapper().bundle_compute_backend == "native"
    with pytest.raises(ValueError, match="auto.*numpy.*native"):
        SphericalGlobalMapperOptions(bundle_compute_backend="other")

    import panorai.reconstruction._native as dispatch

    monkeypatch.setattr(dispatch, "_native_extension", None)
    monkeypatch.setattr(dispatch, "_native_import_error", ImportError("missing test"))
    assert dispatch.resolve_bundle_backend("auto") == "numpy"
    assert SphericalGlobalMapper().bundle_compute_backend == "numpy"
    with pytest.raises(RuntimeError, match="missing test"):
        dispatch.resolve_bundle_backend("native")
    with pytest.raises(RuntimeError, match="missing test"):
        SphericalGlobalMapper(
            options=SphericalGlobalMapperOptions(bundle_compute_backend="native")
        )


def test_native_kernel_rejects_out_of_bounds_indices():
    (
        measured,
        _,
        rotations,
        centers,
        points,
        camera_indices,
        point_indices,
        rotation_deltas,
        _,
    ) = _fixture()
    camera_indices = camera_indices.copy()
    camera_indices[0] = len(rotations)
    with pytest.raises(ValueError, match="out of bounds"):
        spherical_ba_residual_jacobians(
            measured,
            rotations,
            centers,
            points,
            camera_indices,
            point_indices,
            rotation_deltas,
        )
    with pytest.raises(ValueError, match="unit vectors"):
        spherical_ba_residual_jacobians(
            measured * 2.0,
            rotations,
            centers,
            points,
            point_indices % len(rotations),
            point_indices,
            rotation_deltas,
        )
