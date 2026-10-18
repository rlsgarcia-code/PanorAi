from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.panoramic_stereo.run_heldout_multiscale_validation import (
    EXPECTED_MANIFEST_SHA256,
    aggregate_records,
    load_manifest,
    performance_summary,
    verify_common_populations,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks" / "panoramic_stereo"
MANIFEST = BENCHMARKS / "public_multiscale_cost_manifest.json"
PRIOR_MANIFESTS = tuple(
    BENCHMARKS / name
    for name in (
        "public_multipair_manifest.json",
        "public_multiview_manifest.json",
        "public_three_source_manifest.json",
        "public_confidence_validity_manifest.json",
        "public_cost_default_manifest.json",
        "public_angular_neighborhood_manifest.json",
    )
)


def _identifiers(value: Any) -> tuple[set[str], set[str]]:
    pairs: set[str] = set()
    views: set[str] = set()

    def visit(item: Any, key: str | None = None) -> None:
        if isinstance(item, dict):
            for child_key, child in item.items():
                visit(child, child_key)
        elif isinstance(item, list):
            for child in item:
                visit(child, key)
        elif isinstance(item, str):
            if key == "pair_id":
                pairs.add(item)
            elif key in {"reference", "source"}:
                views.add(item)

    visit(value)
    return pairs, views


def test_manifest_is_hash_pinned_balanced_and_output_blind() -> None:
    payload = load_manifest(MANIFEST)

    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == (
        EXPECTED_MANIFEST_SHA256
    )
    assert payload["frozen_before_dense_evaluation"] is True
    assert Counter(item["dataset_id"] for item in payload["pairs"]) == {
        "stanford2d3d": 6,
        "matterport360": 6,
    }
    policy = payload["selection_policy"]
    assert policy["dense_output_used_for_selection"] is False
    assert policy["rgb_match_used_for_selection"] is False
    assert policy["range_reference_used_for_selection"] is False


def test_manifest_excludes_all_previously_observed_pairs_and_views() -> None:
    selected_pairs, selected_views = _identifiers(json.loads(MANIFEST.read_text()))
    prior_pairs: set[str] = set()
    prior_views: set[str] = set()
    for path in PRIOR_MANIFESTS:
        pairs, views = _identifiers(json.loads(path.read_text()))
        prior_pairs.update(pairs)
        prior_views.update(views)

    assert selected_pairs.isdisjoint(prior_pairs)
    assert selected_views.isdisjoint(prior_views)
    assert json.loads(MANIFEST.read_text())["excluded_manifests"] == [
        {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in PRIOR_MANIFESTS
    ]


def test_manifest_hash_drift_fails_closed(tmp_path: Path) -> None:
    changed = json.loads(MANIFEST.read_text())
    changed["pairs"][0]["baseline_m"] = 1.0
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed))

    with pytest.raises(ValueError, match="manifest hash changed"):
        load_manifest(path)


def _record(
    dataset: str,
    pair_id: str,
    cost: str,
    *,
    coverage: float,
    delta: float,
    gross: float,
    elapsed: float,
    population_hash: str = "same",
) -> dict[str, Any]:
    return {
        "dataset_id": dataset,
        "pair_id": pair_id,
        "configuration": {"matching_cost": cost},
        "elapsed_seconds": elapsed,
        "evaluation_population_sha256": population_hash,
        "metrics": {
            "overall": {
                "population_weight": 100.0,
                "predicted_valid_weight": 100.0 * coverage,
                "delta_1_25": delta,
            }
        },
        "relative_gross_error": {
            "fixed_population_fraction": gross * coverage,
            "conditional_fraction": gross,
        },
    }


def _accepted_records() -> list[dict[str, Any]]:
    records = []
    for dataset in ("stanford2d3d", "matterport360"):
        pair_id = f"{dataset}-pair"
        records.extend(
            (
                _record(
                    dataset,
                    pair_id,
                    "ad-census",
                    coverage=0.50,
                    delta=0.30,
                    gross=0.20,
                    elapsed=10.0,
                ),
                _record(
                    dataset,
                    pair_id,
                    "multiscale-ad-census",
                    coverage=0.49,
                    delta=0.31,
                    gross=0.19,
                    elapsed=12.0,
                ),
            )
        )
    return records


def test_scientific_and_performance_gates_accept_bounded_improvement() -> None:
    records = _accepted_records()

    assert aggregate_records(records)["scientific_acceptance"] is True
    performance = performance_summary(records)
    assert performance["accepted"] is True
    assert performance["median_candidate_over_baseline"] == pytest.approx(1.2)


def test_gate_rejects_pair_regression_and_population_drift() -> None:
    records = _accepted_records()
    candidate = next(
        item
        for item in records
        if item["dataset_id"] == "matterport360"
        and item["configuration"]["matching_cost"] == "multiscale-ad-census"
    )
    candidate["metrics"]["overall"]["delta_1_25"] = 0.27

    assert aggregate_records(records)["scientific_acceptance"] is False
    candidate["evaluation_population_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="changed populations"):
        verify_common_populations(records)
