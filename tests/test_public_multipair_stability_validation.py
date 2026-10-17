from __future__ import annotations

import json

import pytest

from benchmarks.panoramic_stereo.run_public_multipair_stability_validation import (
    DEFAULT_MANIFEST,
    aggregate_records,
    load_manifest,
)
from benchmarks.panoramic_stereo.run_real_domain_validation import sha256


def _record(
    dataset_id: str,
    coverage: float,
    delta: float,
    median_error: float,
    population_weight: float = 10.0,
) -> dict:
    selected_weight = coverage * population_weight
    correct_weight = delta * population_weight
    return {
        "dataset_id": dataset_id,
        "metrics": {
            "overall": {
                "coverage": coverage,
                "delta_1_25": delta,
                "median_abs_error_m_valid": median_error,
                "population": 10,
                "population_weight": population_weight,
                "predicted_valid_weight": coverage * population_weight,
            }
        },
        "confidence_selectivity": {
            "population_weight": population_weight,
            "prediction_weight": selected_weight,
            "retention": {
                "1.00": {
                    "selected_weight": selected_weight,
                    "selected_correct_weight": correct_weight,
                    "conditional_delta_1_25": correct_weight / selected_weight,
                }
            },
            "monotonic_conditional_delta_1_25": True,
            "top_quartile_conditional_delta_gain": 0.1,
        },
    }


def test_frozen_public_multipair_manifest_is_outcome_blind_and_balanced():
    manifest = load_manifest(DEFAULT_MANIFEST)

    pairs = manifest["pairs"]
    assert len(pairs) == 11
    assert sum(item["dataset_id"] == "stanford2d3d" for item in pairs) == 5
    assert sum(item["dataset_id"] == "matterport360" for item in pairs) == 6
    assert manifest["selection_policy"]["dense_output_used_for_selection"] is False
    assert sha256(DEFAULT_MANIFEST) == (
        "4910a0285aa6693881b84c1f3b58839154d1032f230f0478da3dfcae93dd0896"
    )


def test_manifest_rejects_reused_view(tmp_path):
    payload = json.loads(DEFAULT_MANIFEST.read_text())
    payload["pairs"][1]["reference"] = payload["pairs"][0]["reference"]
    payload["pairs"][1]["group"] = payload["pairs"][0]["group"]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="reused view identifier"):
        load_manifest(path)


def test_aggregate_records_keeps_micro_and_pair_statistics_distinct():
    records = [
        _record("stanford2d3d", 0.5, 0.4, 0.2, 10.0),
        _record("stanford2d3d", 0.8, 0.6, 0.4, 30.0),
    ]

    aggregate = aggregate_records(records)["stanford2d3d"]

    assert aggregate["pair_count"] == 2
    assert aggregate["micro_coverage"] == pytest.approx(0.725)
    assert aggregate["micro_delta_1_25"] == pytest.approx(0.55)
    assert aggregate["pair_macro"]["median_coverage"] == pytest.approx(0.65)
    assert aggregate["pair_macro"][
        "median_of_pair_median_abs_error_m"
    ] == pytest.approx(0.3)
    assert aggregate["worst_pair"]["minimum_delta_1_25"] == pytest.approx(0.4)
    selectivity = aggregate["confidence_selectivity"]
    assert selectivity["retention"]["1.00"][
        "micro_conditional_delta_1_25"
    ] == pytest.approx(0.55 / 0.725)
    assert selectivity["monotonic_pair_count"] == 2
