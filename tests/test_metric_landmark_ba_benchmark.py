from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarks.metric_landmark_ba.p74_protocol import (
    P74_FROM_PANORAI,
    radial_metrics,
    scan_bearings_to_native_pixels,
)
from benchmarks.metric_landmark_ba.run_p74_multiscene import (
    _angular_errors,
    triangulate_two_view,
)


def test_p74_native_mapping_preserves_endpoint_period_and_polar_support():
    canonical = np.asarray(
        (
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
        ),
        dtype=np.float64,
    )
    native = (P74_FROM_PANORAI @ canonical.T).T
    pixels, support = scan_bearings_to_native_pixels(native, (3414, 8248))
    assert support[:3].all()
    assert not support[3]
    assert np.all((pixels[:, 0] >= 0.0) & (pixels[:, 0] < 8247.0))


def test_two_view_triangulation_recovers_metric_point():
    point = np.asarray((0.8, 0.2, 3.0), dtype=np.float64)
    angle = math.radians(4.0)
    rotation = np.asarray(
        (
            (math.cos(angle), 0.0, math.sin(angle)),
            (0.0, 1.0, 0.0),
            (-math.sin(angle), 0.0, math.cos(angle)),
        )
    )
    translation = np.asarray((-1.1, 0.05, 0.0))
    first = point / np.linalg.norm(point)
    second_point = rotation @ point + translation
    second = second_point / np.linalg.norm(second_point)
    recovered = triangulate_two_view(first, second, rotation, translation)
    assert recovered is not None
    np.testing.assert_allclose(recovered, point, atol=1e-10)
    errors = _angular_errors(
        rotation,
        -rotation.T @ translation,
        recovered[None],
        first[None],
        second[None],
    )
    np.testing.assert_allclose(errors, 0.0, atol=1e-6)


def test_two_view_triangulation_rejects_negligible_parallax():
    ray = np.asarray((0.0, 0.0, 1.0))
    assert triangulate_two_view(ray, ray, np.eye(3), np.zeros(3)) is None


def test_sparse_metrics_separate_metric_scale_and_structure():
    truth = np.asarray((1.0, 2.0, 4.0, 8.0))
    scaled = 3.0 * truth
    result = radial_metrics(scaled, truth, np.ones(4, dtype=bool))
    assert result["count"] == 4
    assert result["abs_rel"] == pytest.approx(2.0)
    assert result["median_scale"] == pytest.approx(1.0 / 3.0)
    assert result["scale_aligned_abs_rel"] == pytest.approx(0.0, abs=1e-12)
    assert result["si_log_rmse"] == pytest.approx(0.0, abs=1e-12)


def test_sparse_metrics_use_explicit_validity():
    prediction = np.asarray((2.0, 100.0, np.nan))
    truth = np.asarray((2.0, 1.0, 3.0))
    result = radial_metrics(prediction, truth, np.asarray((True, False, True)))
    assert result["count"] == 1
    assert result["abs_rel"] == 0.0
