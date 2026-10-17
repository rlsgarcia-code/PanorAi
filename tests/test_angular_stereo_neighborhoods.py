from __future__ import annotations

from dataclasses import replace

import pytest

from benchmarks.panoramic_stereo.run_angular_neighborhood_validation import (
    CELLS,
    ERP_FEATURE_TAP_DEG,
    FACE_COST_TAP_DEG,
    aggregate_records,
    classify_candidate,
    make_angular_candidate,
    verify_common_populations,
)
from benchmarks.panoramic_stereo.run_full_sphere_stability_validation import (
    make_candidate,
)


def _record(
    cell: str,
    dataset: str,
    *,
    coverage: float,
    delta: float,
    population_hash: str = "same",
) -> dict[str, object]:
    return {
        "cell": cell,
        "dataset_id": dataset,
        "pair_id": f"{dataset}-pair",
        "elapsed_seconds": 1.0,
        "evaluation_population_sha256": population_hash,
        "metrics": {
            "overall": {
                "population_weight": 100.0,
                "predicted_valid_weight": 100.0 * coverage,
                "coverage": coverage,
                "delta_1_25": delta,
                "median_abs_error_m_valid": 0.5,
            }
        },
    }


def test_angular_cells_change_only_declared_tap_spacing() -> None:
    assert [cell.name for cell in CELLS] == [
        "legacy-pixel-taps",
        "erp-angular-only",
        "face-angular-only",
        "erp-and-face-angular",
    ]
    assert ERP_FEATURE_TAP_DEG == pytest.approx(0.703125)
    assert 1.0 < FACE_COST_TAP_DEG < 1.1


def test_candidate_builder_preserves_legacy_default_and_sets_angles() -> None:
    legacy = make_candidate(
        face_size=32,
        face_fov_deg=100.0,
        min_range_m=0.3,
        max_range_m=12.0,
        hypotheses=8,
        filter_backend="numpy",
        matching_cost="ad-census",
    )
    angular = make_candidate(
        face_size=32,
        face_fov_deg=100.0,
        min_range_m=0.3,
        max_range_m=12.0,
        hypotheses=8,
        filter_backend="numpy",
        matching_cost="ad-census",
        erp_feature_tap_deg=ERP_FEATURE_TAP_DEG,
        face_cost_tap_deg=FACE_COST_TAP_DEG,
    )

    assert legacy.options.erp_feature_tap_deg is None
    assert legacy.options.face_cost_tap_deg is None
    assert angular.options == replace(
        legacy.options,
        erp_feature_tap_deg=ERP_FEATURE_TAP_DEG,
        face_cost_tap_deg=FACE_COST_TAP_DEG,
    )


def test_controlled_runner_can_change_only_the_regularizer() -> None:
    cell = next(item for item in CELLS if item.name == "erp-and-face-angular")
    sgm4 = make_angular_candidate(cell, filter_backend="numpy")
    tmgm = make_angular_candidate(
        cell,
        filter_backend="numpy",
        regularizer="tmgm",
    )

    assert sgm4.options.regularizer == "sgm4"
    assert tmgm.options == replace(sgm4.options, regularizer="tmgm")


def test_population_drift_is_rejected() -> None:
    records = [
        _record(cell.name, dataset, coverage=0.6, delta=0.3)
        for cell in CELLS
        for dataset in ("stanford2d3d", "matterport360")
    ]
    verify_common_populations(records)
    records[-1]["evaluation_population_sha256"] = "drift"

    with pytest.raises(RuntimeError, match="population drift"):
        verify_common_populations(records)


def test_classification_requires_improvement_in_both_domains() -> None:
    records = []
    for cell in CELLS:
        gain = 0.0 if cell.name == "legacy-pixel-taps" else 0.02
        if cell.name == "erp-and-face-angular":
            gain = 0.04
        for dataset in ("stanford2d3d", "matterport360"):
            records.append(_record(cell.name, dataset, coverage=0.6, delta=0.3 + gain))
    conclusion = classify_candidate(aggregate_records(records))

    assert conclusion["candidate"] == "erp-and-face-angular"
