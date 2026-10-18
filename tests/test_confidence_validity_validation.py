from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "benchmarks"
    / "panoramic_stereo"
    / "public_confidence_validity_manifest.json"
)
EXCLUDED = (
    ROOT / "benchmarks" / "panoramic_stereo" / "public_multipair_manifest.json",
    ROOT / "benchmarks" / "panoramic_stereo" / "public_multiview_manifest.json",
    ROOT / "benchmarks" / "panoramic_stereo" / "public_three_source_manifest.json",
)
EXPECTED_MANIFEST_SHA256 = (
    "7ec9d2b5ef2c2619748be559b98bbca600fa0801430b2ad2a50d1f5aa482c8f1"
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


def test_heldout_manifest_is_frozen_before_dense_inference() -> None:
    manifest = json.loads(MANIFEST.read_text())

    assert manifest["schema"] == "panorai-public-confidence-validity-selection/v1"
    assert manifest["protocol_id"] == "VAL-116-heldout-confidence-validity-separation"
    assert manifest["frozen_before_dense_evaluation"] is True
    assert manifest["selection_policy"]["dense_output_used_for_selection"] is False
    assert manifest["selection_policy"]["rgb_match_used_for_selection"] is False
    assert manifest["selection_policy"]["range_reference_used_for_selection"] is False
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == EXPECTED_MANIFEST_SHA256


def test_heldout_manifest_is_balanced_and_has_no_view_reuse() -> None:
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


def test_heldout_manifest_excludes_all_prior_pairs_and_views() -> None:
    selected_pairs, selected_views = _identifiers(json.loads(MANIFEST.read_text()))
    prior_pairs: set[str] = set()
    prior_views: set[str] = set()
    for path in EXCLUDED:
        pairs, views = _identifiers(json.loads(path.read_text()))
        prior_pairs.update(pairs)
        prior_views.update(views)

    assert selected_pairs.isdisjoint(prior_pairs)
    assert selected_views.isdisjoint(prior_views)


def test_manifest_records_exact_source_audits_and_exclusions() -> None:
    manifest = json.loads(MANIFEST.read_text())

    assert manifest["source_audits"] == {
        "pair_analysis_sha256": (
            "bc80164baece362ad406e9732dd19e1e70917c3a73abb06e2640211d43ad3f54"
        ),
        "registered_cloud_overlap_sha256": (
            "134229851264939062c236f410f9f8f50b23dd86a0ac9425253c799d243dd628"
        ),
    }
    assert manifest["excluded_manifests"] == [
        {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in EXCLUDED
    ]
