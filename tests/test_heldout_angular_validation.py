from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.panoramic_stereo.run_heldout_angular_validation import (
    ANGULAR,
    EXPECTED_MANIFEST_SHA256,
    LEGACY,
    aggregate_records,
    load_manifest,
    performance_summary,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "benchmarks"
    / "panoramic_stereo"
    / "public_angular_neighborhood_manifest.json"
)
PRIOR_MANIFESTS = (
    ROOT / "benchmarks" / "panoramic_stereo" / "public_multipair_manifest.json",
    ROOT / "benchmarks" / "panoramic_stereo" / "public_multiview_manifest.json",
    ROOT / "benchmarks" / "panoramic_stereo" / "public_three_source_manifest.json",
    ROOT
    / "benchmarks"
    / "panoramic_stereo"
    / "public_confidence_validity_manifest.json",
    ROOT / "benchmarks" / "panoramic_stereo" / "public_cost_default_manifest.json",
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


def test_manifest_is_frozen_balanced_and_output_blind() -> None:
    payload = load_manifest(MANIFEST)
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == EXPECTED_MANIFEST_SHA256
    assert len(payload["pairs"]) == 12
    assert Counter(item["dataset_id"] for item in payload["pairs"]) == {
        "stanford2d3d": 6,
        "matterport360": 6,
    }
    assert payload["selection_policy"]["dense_output_used_for_selection"] is False
    assert payload["selection_policy"]["rgb_match_used_for_selection"] is False
    assert payload["selection_policy"]["range_reference_used_for_selection"] is False


def test_manifest_excludes_every_prior_pair_and_view() -> None:
    selected_pairs, selected_views = _identifiers(json.loads(MANIFEST.read_text()))
    prior_pairs: set[str] = set()
    prior_views: set[str] = set()
    for path in PRIOR_MANIFESTS:
        pairs, views = _identifiers(json.loads(path.read_text()))
        prior_pairs.update(pairs)
        prior_views.update(views)

    assert selected_pairs.isdisjoint(prior_pairs)
    assert selected_views.isdisjoint(prior_views)


def test_manifest_hash_drift_fails_closed(tmp_path: Path) -> None:
    changed = json.loads(MANIFEST.read_text())
    changed["pairs"][0]["baseline_m"] = 1.0
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed))

    with pytest.raises(ValueError, match="manifest hash changed"):
        load_manifest(path)


def _record(
    dataset: str,
    cell: str,
    *,
    coverage: float,
    delta: float,
    gross: float,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset,
        "cell": cell,
        "elapsed_seconds": elapsed,
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


def test_scientific_and_runtime_gates_accept_bounded_improvement() -> None:
    records = []
    for dataset in ("stanford2d3d", "matterport360"):
        records.extend(
            (
                _record(
                    dataset,
                    LEGACY.name,
                    coverage=0.5,
                    delta=0.3,
                    gross=0.2,
                    elapsed=10.0,
                ),
                _record(
                    dataset,
                    ANGULAR.name,
                    coverage=0.52,
                    delta=0.36,
                    gross=0.21,
                    elapsed=12.0,
                ),
            )
        )

    aggregates = aggregate_records(records)
    performance = performance_summary(records)

    assert all(item["scientific_acceptance"] for item in aggregates.values())
    assert performance["accepted"] is True
    assert performance["candidate_median_runtime_ratio"] == pytest.approx(1.2)


def test_gate_rejects_one_domain_accuracy_regression() -> None:
    records = []
    for dataset in ("stanford2d3d", "matterport360"):
        records.extend(
            (
                _record(
                    dataset,
                    LEGACY.name,
                    coverage=0.5,
                    delta=0.3,
                    gross=0.2,
                    elapsed=10.0,
                ),
                _record(
                    dataset,
                    ANGULAR.name,
                    coverage=0.52,
                    delta=0.29 if dataset == "matterport360" else 0.36,
                    gross=0.2,
                    elapsed=12.0,
                ),
            )
        )

    assert aggregate_records(records)["matterport360"]["scientific_acceptance"] is False
