from __future__ import annotations

import pytest

from benchmarks.panoramic_stereo.run_resolution_lattice_diagnostic import (
    CELLS,
    aggregate_records,
    classify_ceiling,
    verify_common_factorial_populations,
)


def _record(
    cell: str,
    dataset: str,
    pair: str,
    *,
    coverage: float,
    delta: float,
    population_hash: str = "same",
) -> dict[str, object]:
    return {
        "cell": cell,
        "factorial": cell != "historical-anchor",
        "dataset_id": dataset,
        "pair_id": pair,
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


def _complete_records(
    *, spatial_gain: float, range_gain: float, interaction_gain: float
) -> list[dict[str, object]]:
    records = []
    gains = {
        "historical-anchor": 0.0,
        "spatial-low_range-low": 0.0,
        "spatial-low_range-high": range_gain,
        "spatial-high_range-low": spatial_gain,
        "spatial-high_range-high": interaction_gain,
    }
    for cell in CELLS:
        for dataset in ("stanford2d3d", "matterport360"):
            records.append(
                _record(
                    cell.name,
                    dataset,
                    f"{dataset}-pair",
                    coverage=0.6,
                    delta=0.3 + gains[cell.name],
                )
            )
    return records


def test_factorial_cells_share_one_erp_grid_and_form_two_by_two_design() -> None:
    factorial = [cell for cell in CELLS if cell.factorial]

    assert {cell.erp_shape_hw for cell in factorial} == {(512, 1024)}
    assert {(cell.face_size, cell.hypotheses) for cell in factorial} == {
        (128, 48),
        (128, 96),
        (256, 48),
        (256, 96),
    }
    assert CELLS[0].erp_shape_hw == (256, 512)


def test_population_hash_must_not_change_inside_factorial() -> None:
    records = _complete_records(
        spatial_gain=0.01, range_gain=0.02, interaction_gain=0.03
    )
    verify_common_factorial_populations(records)
    records[-1]["evaluation_population_sha256"] = "drift"

    with pytest.raises(RuntimeError, match="population drift"):
        verify_common_factorial_populations(records)


def test_classification_selects_strongest_cross_domain_controlled_effect() -> None:
    records = _complete_records(
        spatial_gain=0.015,
        range_gain=0.025,
        interaction_gain=0.05,
    )
    aggregates = aggregate_records(records)

    conclusion = classify_ceiling(aggregates)

    assert conclusion["classification"] == "interaction"
    assert conclusion["effects"]["interaction"]["coverage_gate_passed"] is True


def test_classification_rejects_domain_specific_or_small_changes() -> None:
    records = _complete_records(
        spatial_gain=0.005,
        range_gain=0.005,
        interaction_gain=0.005,
    )
    aggregates = aggregate_records(records)

    assert classify_ceiling(aggregates)["classification"] == "neither"
