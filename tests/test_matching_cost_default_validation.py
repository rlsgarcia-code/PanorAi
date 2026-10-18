from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.panoramic_stereo.run_full_sphere_stability_validation import (
    make_candidate,
)
from benchmarks.panoramic_stereo.run_matching_cost_default_validation import (
    aggregate_records,
    candidate_for_cost,
    load_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "benchmarks" / "panoramic_stereo" / "public_cost_default_manifest.json"
)
EXCLUDED = (
    ROOT / "benchmarks" / "panoramic_stereo" / "public_multipair_manifest.json",
    ROOT / "benchmarks" / "panoramic_stereo" / "public_multiview_manifest.json",
    ROOT / "benchmarks" / "panoramic_stereo" / "public_three_source_manifest.json",
    ROOT
    / "benchmarks"
    / "panoramic_stereo"
    / "public_confidence_validity_manifest.json",
)
EXPECTED_MANIFEST_SHA256 = (
    "a9c778e87363be6cb7fbf14e8bade493bccd2347607e3f77b95fac505a89e9cc"
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


def test_cost_manifest_is_frozen_before_dense_inference() -> None:
    manifest = json.loads(MANIFEST.read_text())

    assert manifest["schema"] == "panorai-public-matching-cost-selection/v1"
    assert manifest["protocol_id"] == "VAL-117-heldout-matching-cost-default"
    assert manifest["frozen_before_dense_evaluation"] is True
    assert manifest["selection_policy"]["dense_output_used_for_selection"] is False
    assert manifest["selection_policy"]["rgb_match_used_for_selection"] is False
    assert manifest["selection_policy"]["range_reference_used_for_selection"] is False
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == EXPECTED_MANIFEST_SHA256


def test_cost_manifest_is_balanced_and_has_no_view_reuse() -> None:
    pairs = json.loads(MANIFEST.read_text())["pairs"]

    assert len(pairs) == 12
    assert Counter(pair["dataset_id"] for pair in pairs) == {
        "stanford2d3d": 6,
        "matterport360": 6,
    }
    stanford = [pair for pair in pairs if pair["dataset_id"] == "stanford2d3d"]
    matterport = [pair for pair in pairs if pair["dataset_id"] == "matterport360"]
    assert Counter(pair["group"] for pair in stanford) == {
        "area_2": 2,
        "area_3": 2,
        "area_4": 2,
    }
    assert len({pair["group"] for pair in matterport}) == 6
    views = [view for pair in pairs for view in (pair["reference"], pair["source"])]
    assert len(views) == len(set(views))
    assert all(0.2 <= pair["baseline_m"] <= 5.0 for pair in pairs)
    assert all(pair["minimum_directional_cloud_overlap"] >= 0.35 for pair in pairs)


def test_cost_manifest_excludes_all_prior_pairs_and_views() -> None:
    selected_pairs, selected_views = _identifiers(json.loads(MANIFEST.read_text()))
    prior_pairs: set[str] = set()
    prior_views: set[str] = set()
    for path in EXCLUDED:
        pairs, views = _identifiers(json.loads(path.read_text()))
        prior_pairs.update(pairs)
        prior_views.update(views)

    assert selected_pairs.isdisjoint(prior_pairs)
    assert selected_views.isdisjoint(prior_views)


def test_cost_manifest_records_exact_exclusion_hashes() -> None:
    manifest = json.loads(MANIFEST.read_text())

    assert manifest["excluded_manifests"] == [
        {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in EXCLUDED
    ]


def test_runtime_loader_rejects_any_manifest_hash_drift(tmp_path: Path) -> None:
    assert load_manifest(MANIFEST)["pairs"]
    changed = json.loads(MANIFEST.read_text())
    changed["pairs"][0]["baseline_m"] = 1.0
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed))

    with pytest.raises(ValueError, match="manifest hash changed"):
        load_manifest(path)


def test_candidate_changes_only_matching_cost() -> None:
    base = make_candidate(
        face_size=16,
        face_fov_deg=100.0,
        min_range_m=0.3,
        max_range_m=12.0,
        hypotheses=8,
        filter_backend="numpy",
    )
    candidate = candidate_for_cost(base, "ad-census")

    assert candidate.options == replace(base.options, matching_cost="ad-census")
    with pytest.raises(ValueError, match="unsupported matching cost"):
        candidate_for_cost(base, "learned")


def _record(
    dataset_id: str,
    cost: str,
    *,
    coverage: float,
    delta: float,
    gross: float,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "matching_cost": cost,
        "metrics": {
            "overall": {
                "population_weight": 100.0,
                "predicted_valid_weight": 100.0 * coverage,
                "delta_1_25": delta,
            }
        },
        "label_error": {
            "gross_gt_2_steps_fixed_population_fraction": gross * coverage,
        },
    }


def test_aggregate_records_applies_frozen_scientific_criteria() -> None:
    records = []
    for dataset_id in ("stanford2d3d", "matterport360"):
        records.extend(
            (
                _record(
                    dataset_id,
                    "intensity-gradient",
                    coverage=0.50,
                    delta=0.30,
                    gross=0.25,
                ),
                _record(
                    dataset_id,
                    "ad-census",
                    coverage=0.49,
                    delta=0.32,
                    gross=0.26,
                ),
            )
        )

    aggregate = aggregate_records(records)

    assert aggregate["scientific_acceptance"] is True
    for dataset_id in ("stanford2d3d", "matterport360"):
        assert aggregate[dataset_id]["accepted"] is True
        assert all(aggregate[dataset_id]["acceptance_checks"].values())
