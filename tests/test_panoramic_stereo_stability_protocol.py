from __future__ import annotations

import numpy as np

from benchmarks.panoramic_stereo.stability_protocol import (
    build_evaluation_population,
    same_surface_observability,
    summarize_confidence_selectivity,
    summarize_range_prediction,
)


def test_explicit_sky_is_excluded_without_inspecting_prediction_values():
    truth = np.full((2, 4), 2.0, dtype=np.float32)
    non_sky = np.asarray(
        ((False, False, True, True), (False, True, True, True)), dtype=bool
    )
    support = np.ones_like(non_sky)
    population = build_evaluation_population(
        truth,
        explicit_non_sky=non_sky,
        geometric_support=support,
        observable=support,
        non_sky_policy="analytic-fixture",
    )
    prediction = truth.copy()
    prediction[~non_sky] = 1000.0
    summary = summarize_range_prediction(
        truth,
        prediction,
        population=population.mask,
        prediction_validity=support,
    )

    assert population.audit["excluded_as_sky_pixels"] == 3
    assert population.audit["observable_pixels"] == 5
    assert summary["overall"]["delta_1_25"] == 1.0


def test_missing_prediction_stays_in_fixed_accuracy_denominator():
    truth = np.full((2, 2), 2.0, dtype=np.float32)
    mask = np.ones((2, 2), dtype=bool)
    prediction = truth.copy()
    validity = mask.copy()
    validity[0, 0] = False
    prediction[0, 0] = np.nan

    summary = summarize_range_prediction(
        truth,
        prediction,
        population=mask,
        prediction_validity=validity,
    )["overall"]

    assert summary["population"] == 4
    assert summary["predicted_valid"] == 3
    assert summary["coverage"] == 0.75
    assert summary["delta_1_25"] == 0.75


def test_range_bands_have_deterministic_boundaries_and_empty_band():
    truth = np.asarray(((0.3, 1.999, 2.0, 4.999, 5.0, 12.0),), np.float32)
    mask = np.ones_like(truth, dtype=bool)
    summary = summarize_range_prediction(
        truth,
        truth,
        population=mask,
        prediction_validity=mask,
    )

    bands = summary["range_bands"]
    assert bands["near_0_3_to_2_m"]["population"] == 2
    assert bands["mid_2_to_5_m"]["population"] == 2
    assert bands["far_5_to_12_m"]["population"] == 2

    empty = mask & False
    empty_summary = summarize_range_prediction(
        truth,
        truth,
        population=empty,
        prediction_validity=mask,
    )["overall"]
    assert empty_summary["population"] == 0
    assert empty_summary["coverage"] is None


def test_population_rejects_value_derived_or_non_boolean_masks():
    truth = np.ones((2, 3), dtype=np.float32)
    mask = np.ones_like(truth, dtype=bool)
    with np.testing.assert_raises_regex(ValueError, "explicit_non_sky"):
        build_evaluation_population(
            truth,
            explicit_non_sky=truth,
            geometric_support=mask,
            observable=mask,
            non_sky_policy="invalid-value-derived-fixture",
        )


def test_solid_angle_weights_control_fixed_denominator_metrics():
    truth = np.full((2, 2), 2.0, dtype=np.float32)
    prediction = truth.copy()
    prediction[0, 0] = 4.0
    mask = np.ones_like(truth, dtype=bool)
    weights = np.asarray(((9.0, 1.0), (1.0, 1.0)), dtype=np.float64)

    summary = summarize_range_prediction(
        truth,
        prediction,
        population=mask,
        prediction_validity=mask,
        sample_weights=weights,
    )["overall"]

    assert summary["population"] == 4
    assert summary["population_weight"] == 12.0
    assert summary["coverage"] == 1.0
    assert summary["delta_1_25"] == 0.25
    assert summary["median_abs_error_m_valid"] == 2.0


def test_confidence_selectivity_reports_conditional_and_fixed_denominators():
    truth = np.full((1, 4), 2.0, dtype=np.float32)
    prediction = np.asarray(((2.0, 2.0, 4.0, 4.0),), dtype=np.float32)
    confidence = np.asarray(((0.9, 0.8, 0.2, 0.1),), dtype=np.float32)
    mask = np.ones_like(truth, dtype=bool)

    summary = summarize_confidence_selectivity(
        truth,
        prediction,
        confidence,
        population=mask,
        prediction_validity=mask,
    )

    top = summary["retention"]["0.25"]
    half = summary["retention"]["0.50"]
    full = summary["retention"]["1.00"]
    assert top["conditional_delta_1_25"] == 1.0
    assert top["fixed_population_delta_1_25"] == 0.25
    assert half["conditional_delta_1_25"] == 1.0
    assert full["conditional_delta_1_25"] == 0.5
    assert full["fixed_population_coverage"] == 1.0
    assert summary["monotonic_conditional_delta_1_25"] is True
    assert summary["top_quartile_conditional_delta_gain"] == 0.5


def test_confidence_selectivity_rejects_invalid_fraction_contract():
    truth = np.ones((1, 2), dtype=np.float32)
    mask = np.ones_like(truth, dtype=bool)

    with np.testing.assert_raises_regex(ValueError, "retention_fractions"):
        summarize_confidence_selectivity(
            truth,
            truth,
            truth,
            population=mask,
            prediction_validity=mask,
            retention_fractions=(0.5, 0.5),
        )


def test_same_surface_observability_rejects_occlusion_depth_mismatch():
    truth = np.full((3, 8), 2.0, dtype=np.float32)
    source = truth.copy()
    source[1, 3] = 1.0
    support = np.ones_like(truth, dtype=bool)

    observable = same_surface_observability(
        truth,
        source,
        source_support=support,
        rotation_source_from_reference=np.eye(3),
        translation_source_from_reference=np.zeros(3),
    )

    expected = support.copy()
    expected[1, 3] = False
    np.testing.assert_array_equal(observable, expected)
    assert not observable.flags.writeable


def test_same_surface_observability_wraps_the_erp_seam():
    truth = np.full((5, 12), 3.0, dtype=np.float32)
    source = truth.copy()
    support = np.ones_like(truth, dtype=bool)
    yaw_180 = np.diag((-1.0, 1.0, -1.0))

    observable = same_surface_observability(
        truth,
        source,
        source_support=support,
        rotation_source_from_reference=yaw_180,
        translation_source_from_reference=np.zeros(3),
    )

    assert np.all(observable)
