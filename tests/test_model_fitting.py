from __future__ import annotations

import math

import numpy as np
import pytest

from panorai.model_fitting import (
    HomographyModel,
    ModelFamily,
    RobustFitOptions,
    RotationModel,
    fit_homography,
    fit_rotation,
    robust_fit,
)


def _rotation_y(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    return np.asarray(
        (
            (math.cos(angle), 0.0, math.sin(angle)),
            (0.0, 1.0, 0.0),
            (-math.sin(angle), 0.0, math.cos(angle)),
        )
    )


def test_rotation_fit_recovers_so3_with_outliers_deterministically() -> None:
    random = np.random.default_rng(4)
    source = random.normal(size=(80, 3))
    source /= np.linalg.norm(source, axis=1, keepdims=True)
    expected = _rotation_y(24.0)
    target = source @ expected.T
    target[::9] = random.normal(size=(9, 3))
    target[::9] /= np.linalg.norm(target[::9], axis=1, keepdims=True)
    options = RobustFitOptions(
        residual_threshold=math.radians(0.1),
        min_inliers=60,
        random_seed=12,
    )

    first = fit_rotation(source, target, options=options)
    second = fit_rotation(source, target, options=options)

    np.testing.assert_allclose(first.model, expected, atol=1e-12)
    np.testing.assert_array_equal(first.inlier_mask, second.inlier_mask)
    assert first.num_inliers == 71
    assert first.residual_unit == "radians"
    assert first.method == "msac"
    assert not first.inlier_mask.flags.writeable
    assert isinstance(RotationModel(), ModelFamily)


def test_homography_fit_uses_symmetric_transfer_error_and_rejects_outliers() -> None:
    columns, rows = np.meshgrid(np.linspace(-2, 2, 7), np.linspace(-1, 1, 5))
    source = np.column_stack((columns.ravel(), rows.ravel()))
    expected = np.asarray(((1.2, 0.1, 3.0), (-0.05, 0.9, -2.0), (0.01, 0.02, 1.0)))
    homogeneous = np.column_stack((source, np.ones(len(source)))) @ expected.T
    target = homogeneous[:, :2] / homogeneous[:, 2:]
    target[::8] += 20.0

    result = fit_homography(
        source,
        target,
        options=RobustFitOptions(
            residual_threshold=0.01,
            min_inliers=25,
            random_seed=9,
        ),
    )

    recovered = result.model / result.model[2, 2]
    np.testing.assert_allclose(recovered, expected, atol=1e-10)
    assert result.num_inliers == 30
    assert result.residual_unit == "pixels"
    np.testing.assert_allclose(
        HomographyModel().residuals(result.model, source[1:], target[1:])[
            ~result.inlier_mask[1:]
        ],
        result.residuals[1:][~result.inlier_mask[1:]],
    )


def test_generic_fit_fails_closed_for_incompatible_or_degenerate_data() -> None:
    family = RotationModel()
    with pytest.raises(ValueError, match="at least 3"):
        robust_fit(
            family,
            np.eye(2, 3),
            np.eye(2, 3),
            options=RobustFitOptions(residual_threshold=0.1),
        )
    repeated = np.repeat([[0.0, 0.0, 1.0]], 8, axis=0)
    with pytest.raises(ValueError, match="no consensus"):
        robust_fit(
            family,
            repeated,
            repeated,
            options=RobustFitOptions(
                residual_threshold=0.1, min_trials=8, max_trials=8
            ),
        )
