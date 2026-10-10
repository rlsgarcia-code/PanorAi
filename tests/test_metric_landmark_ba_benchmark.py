from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarks.metric_landmark_ba.p74_protocol import (
    P74_FROM_PANORAI,
    radial_metrics,
    scan_bearings_to_native_pixels,
)
from benchmarks.metric_landmark_ba.anchor_densification import (
    AnchorDensificationOptions,
    densify_harmonic_log_range,
    densify_log_range_anchors,
)
from benchmarks.metric_landmark_ba.run_p74_multiscene import (
    _angular_errors,
    triangulate_two_view,
)
from benchmarks.metric_landmark_ba.run_anchor_densification import (
    _dense_metrics,
    _float_arrays_bitwise_equal,
    _write_ply,
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


def test_anchor_densification_is_exact_local_and_periodic():
    prior = np.full((64, 128), 2.0, dtype=np.float32)
    longitude = math.pi - 0.01
    bearing = np.asarray(
        ((math.sin(longitude), 0.0, math.cos(longitude)),), dtype=np.float64
    )
    result = densify_log_range_anchors(
        prior,
        bearing,
        np.asarray((4.0,)),
        prior_validity=np.ones(prior.shape, dtype=bool),
        options=AnchorDensificationOptions(
            angular_radius_deg=6.0,
        ),
    )
    assert result.radial_range_m[
        result.anchor_rows[0], result.anchor_columns[0]
    ] == pytest.approx(4.0)
    assert result.support[:, 0].any()
    assert result.support[:, -1].any()
    np.testing.assert_array_equal(
        result.radial_range_m[~result.support], prior[~result.support]
    )


def test_anchor_densification_groups_colliding_anchor_cells():
    prior = np.ones((16, 32), dtype=np.float32)
    bearing = np.asarray(((0.0, 0.0, 1.0), (0.001, 0.0, 1.0)))
    result = densify_log_range_anchors(
        prior,
        bearing,
        np.asarray((2.0, 8.0)),
        prior_validity=np.ones(prior.shape, dtype=bool),
    )
    assert result.input_anchor_count == 2
    assert result.unique_anchor_count == 1
    value = result.radial_range_m[result.anchor_rows[0], result.anchor_columns[0]]
    assert value == pytest.approx(4.0)


def test_harmonic_densification_recovers_smooth_scale_drift():
    prior = np.full((32, 64), 2.0, dtype=np.float32)
    longitudes = np.linspace(-2.5, 2.5, 18)
    latitudes = np.linspace(-0.8, 0.8, 18)
    bearings = np.column_stack(
        (
            np.cos(latitudes) * np.sin(longitudes),
            np.sin(latitudes),
            np.cos(latitudes) * np.cos(longitudes),
        )
    )
    anchor_ranges = 2.0 * np.exp(0.4 + 0.2 * bearings[:, 0])
    result = densify_harmonic_log_range(
        prior,
        bearings,
        anchor_ranges,
        prior_validity=np.ones(prior.shape, dtype=bool),
        maximum_degree=2,
        ridge=1e-6,
    )
    assert result.degree in (1, 2)
    assert result.anchor_mae_log < 1e-5
    assert set(result.cross_validation_mae_log_by_degree) == {0, 1, 2}


def test_dense_region_metrics_do_not_mutate_validity_mask():
    prediction = np.ones((3, 4), dtype=np.float32)
    truth = np.ones_like(prediction)
    validity = np.ones_like(prediction, dtype=bool)
    original = validity.copy()
    region = np.zeros_like(validity)
    region[1, 2] = True

    result = _dense_metrics(prediction, truth, validity, region=region)

    assert result["count"] == 1
    np.testing.assert_array_equal(validity, original)


def test_dense_structure_metric_is_invariant_to_global_scale():
    # Exactly representable proportional fields isolate scale invariance from
    # float32 quantization of a varying input ramp.
    truth = np.ones((4, 8), dtype=np.float32)
    prediction = truth * 3.0

    result = _dense_metrics(
        prediction,
        truth,
        np.ones(truth.shape, dtype=bool),
    )

    assert result["solid_angle_optimal_scale"] == pytest.approx(1.0 / 3.0)
    assert result["solid_angle_scale_aligned_relative_3d_rmse"] == pytest.approx(
        0.0, abs=1e-12
    )


def test_compact_residual_makes_harmonic_landmarks_exact():
    prior = np.full((48, 96), 2.0, dtype=np.float32)
    longitudes = np.asarray((-2.7, -1.8, -0.9, 0.0, 0.9, 1.8, 2.7))
    latitudes = np.asarray((-0.4, 0.2, 0.5, -0.2, 0.3, -0.5, 0.1))
    bearings = np.column_stack(
        (
            np.cos(latitudes) * np.sin(longitudes),
            np.sin(latitudes),
            np.cos(latitudes) * np.cos(longitudes),
        )
    )
    anchor_ranges = 2.0 * np.exp(0.35 + 0.15 * bearings[:, 0])
    validity = np.ones(prior.shape, dtype=bool)
    harmonic = densify_harmonic_log_range(
        prior,
        bearings,
        anchor_ranges,
        prior_validity=validity,
    )
    anchored = densify_log_range_anchors(
        harmonic.radial_range_m,
        bearings,
        anchor_ranges,
        prior_validity=validity,
        options=AnchorDensificationOptions(angular_radius_deg=4.0),
    )

    np.testing.assert_allclose(
        anchored.radial_range_m[anchored.anchor_rows, anchored.anchor_columns],
        anchored.anchor_ranges_m,
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        anchored.radial_range_m[~anchored.support],
        harmonic.radial_range_m[~anchored.support],
    )


def test_harmonic_correction_is_not_clipped_at_log_eight():
    prior = np.ones((24, 48), dtype=np.float32)
    longitudes = np.linspace(-2.8, 2.8, 12)
    latitudes = np.linspace(-0.7, 0.7, 12)
    bearings = np.column_stack(
        (
            np.cos(latitudes) * np.sin(longitudes),
            np.sin(latitudes),
            np.cos(latitudes) * np.cos(longitudes),
        )
    )
    anchor_ranges = np.full(12, 16.0, dtype=np.float64)

    result = densify_harmonic_log_range(
        prior,
        bearings,
        anchor_ranges,
        prior_validity=np.ones(prior.shape, dtype=bool),
        maximum_degree=0,
        ridge=1e-6,
    )

    assert float(np.min(result.log_correction)) > math.log(8.0)
    np.testing.assert_allclose(result.radial_range_m, 16.0, rtol=1e-6, atol=1e-6)


def test_explicit_prior_validity_blocks_finite_unsupported_pixels():
    prior = np.ones((24, 48), dtype=np.float32)
    validity = np.ones(prior.shape, dtype=bool)
    validity[:, :8] = False
    longitudes = np.linspace(-2.8, 2.8, 12)
    latitudes = np.linspace(-0.7, 0.7, 12)
    bearings = np.column_stack(
        (
            np.cos(latitudes) * np.sin(longitudes),
            np.sin(latitudes),
            np.cos(latitudes) * np.cos(longitudes),
        )
    )

    result = densify_harmonic_log_range(
        prior,
        bearings,
        np.full(12, 2.0),
        prior_validity=validity,
        maximum_degree=0,
    )

    np.testing.assert_array_equal(result.radial_range_m[:, :8], prior[:, :8])
    np.testing.assert_allclose(result.radial_range_m[:, 8:], 2.0, rtol=1e-6)


def test_densification_ply_is_xyz_rgb_viewer_compatible(tmp_path):
    radial = np.full((4, 8), 2.0, dtype=np.float32)
    validity = np.ones(radial.shape, dtype=bool)
    path = tmp_path / "surface.ply"

    report = _write_ply(path, radial, validity, stride=2)

    payload = path.read_bytes()
    header, body = payload.split(b"end_header\n", 1)
    assert b"property uchar red" in header
    assert b"property uchar green" in header
    assert b"property uchar blue" in header
    assert len(body) == report["written_points"] * 15


def test_bitwise_identity_accepts_copied_nan_payloads():
    original = np.asarray((1.0, np.nan, -0.0), dtype=np.float32)
    copied = original.copy()

    assert _float_arrays_bitwise_equal(original, copied)
    copied[-1] = 0.0
    assert not _float_arrays_bitwise_equal(original, copied)
