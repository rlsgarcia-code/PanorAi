from __future__ import annotations

import numpy as np

from benchmarks.panoramic_stereo.run_p74_cost_aggregation_benchmark import (
    COSTS,
    aggregate_cost_volume,
    build_cost_volumes,
    evaluate_cost_volume,
)
from panorai.stereo.classical._operations import _INVALID_COST


def test_every_cost_finds_an_exact_plane_sweep_match_without_aggregation():
    rng = np.random.default_rng(7)
    reference = rng.uniform(0.05, 0.95, (21, 25)).astype(np.float32)
    warped = np.stack(
        (
            np.roll(reference, 3, axis=1),
            np.roll(reference, 4, axis=0),
            reference,
            1.0 - reference,
        )
    )
    valid = np.ones_like(warped, dtype=bool)

    volumes = build_cost_volumes(reference, warped, valid)

    assert tuple(volumes) == COSTS
    core = np.s_[4:-4, 4:-4]
    for volume in volumes.values():
        assert np.mean(np.argmin(volume, axis=2)[core] == 2) > 0.95


def test_sgm_recovers_spatially_consistent_label_from_local_false_minima():
    height, width, labels = 18, 24, 3
    costs = np.full((height, width, labels), 0.8, dtype=np.float32)
    # The locally correct label has a small non-zero unary cost everywhere.
    # Alternating labels 0 and 2 win independently, but their two-label jumps
    # cost P2; the globally coherent label 1 is therefore the energy minimum.
    costs[..., 1] = 0.02
    yy, xx = np.indices((height, width))
    false = (yy + xx) % 2
    costs[..., 0] = np.where(false == 0, 0.0, 0.55)
    costs[..., 2] = np.where(false == 1, 0.0, 0.55)
    guidance = np.full((height, width), 0.5, dtype=np.float32)

    raw_fraction = np.mean(np.argmin(costs, axis=2) == 1)
    aggregated = aggregate_cost_volume(costs, guidance, "sgm4")
    sgm_fraction = np.mean(np.argmin(aggregated, axis=2) == 1)

    assert raw_fraction == 0.0
    assert sgm_fraction > 0.80


def test_box_then_sgm_is_an_explicit_composed_aggregation():
    rng = np.random.default_rng(13)
    costs = rng.uniform(0.0, 1.0, (9, 11, 4)).astype(np.float32)
    guidance = rng.uniform(0.0, 1.0, (9, 11)).astype(np.float32)

    composed = aggregate_cost_volume(costs, guidance, "box-sgm4")
    manual = aggregate_cost_volume(
        aggregate_cost_volume(costs, guidance, "box"), guidance, "sgm4"
    )

    np.testing.assert_allclose(composed, manual)


def test_bilateral_aggregation_does_not_mix_costs_across_guidance_edge():
    guidance = np.zeros((15, 21), dtype=np.float32)
    guidance[:, 11:] = 1.0
    costs = np.zeros((15, 21, 2), dtype=np.float32)
    costs[:, :11, 0] = 0.0
    costs[:, :11, 1] = 1.0
    costs[:, 11:, 0] = 1.0
    costs[:, 11:, 1] = 0.0

    aggregated = aggregate_cost_volume(costs, guidance, "bilateral")

    assert np.all(np.argmin(aggregated[:, :11], axis=2) == 0)
    assert np.all(np.argmin(aggregated[:, 11:], axis=2) == 1)


def test_evaluation_reports_correct_label_rank_and_metric_range():
    inverse_ranges = np.asarray((0.25, 0.5, 1.0), dtype=np.float32)
    costs = np.ones((3, 4, 3), dtype=np.float32)
    costs[..., 1] = 0.0
    truth = np.full((3, 4), 2.0, dtype=np.float32)
    mask = np.ones((3, 4), dtype=bool)

    metrics = evaluate_cost_volume(costs, inverse_ranges, truth, mask)

    assert metrics["evaluated"] == 12
    assert metrics["winner_within_1_label"] == 1.0
    assert metrics["truth_band_top_1"] == 1.0
    assert metrics["delta_1_25"] == 1.0
    assert metrics["median_abs_error_m"] == 0.0


def test_invalid_prediction_remains_in_fixed_evaluation_denominator():
    inverse_ranges = np.asarray((0.25, 0.5, 1.0), dtype=np.float32)
    costs = np.ones((2, 2, 3), dtype=np.float32)
    costs[..., 1] = 0.0
    costs[0, 0, :] = _INVALID_COST
    truth = np.full((2, 2), 2.0, dtype=np.float32)
    mask = np.ones((2, 2), dtype=bool)

    metrics = evaluate_cost_volume(costs, inverse_ranges, truth, mask)

    assert metrics["evaluated"] == 4
    assert metrics["predicted_valid"] == 3
    assert metrics["coverage"] == 0.75
    assert metrics["winner_within_1_label"] == 0.75
    assert metrics["delta_1_25"] == 0.75
