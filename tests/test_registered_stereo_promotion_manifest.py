from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.panoramic_stereo.build_registered_stereo_promotion_manifest import (
    PRIOR_MANIFESTS,
    PROTOCOL,
    build_promotion_manifest,
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _asset(root: Path, group: str, name: str) -> None:
    for folder, suffix in (("rgb", ".png"), ("depth", ".npy"), ("poses", ".txt")):
        path = root / folder / group / f"{name}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def _synthetic_inventory(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    overlap_rows: list[dict[str, Any]] = []
    analysis_rows: list[dict[str, Any]] = []
    roots = {
        "stanford2d3d": tmp_path / "stanford",
        "matterport360": tmp_path / "matterport",
    }

    candidates: list[tuple[str, str, int]] = []
    for group in ("area_2", "area_3", "area_4"):
        candidates.extend(("stanford2d3d", group, index) for index in range(2))
    candidates.extend(("matterport360", f"building_{index}", 0) for index in range(6))

    for sequence, (dataset, group, index) in enumerate(candidates):
        reference = f"new_{dataset}_{group}_{index}_reference"
        source = f"new_{dataset}_{group}_{index}_source"
        pair_id = f"post-protocol-{sequence:02d}"
        _asset(roots[dataset], group, reference)
        _asset(roots[dataset], group, source)
        overlap_rows.append(
            {
                "dataset_id": dataset,
                "pair_id": pair_id,
                "from_view_id": f"{dataset}::{group}::{reference}",
                "to_view_id": f"{dataset}::{group}::{source}",
                "spatial_group_id": f"{dataset}::{group}",
                "minimum_directional_cloud_overlap": 0.9 - sequence * 0.001,
            }
        )
        analysis_rows.append({"pair_id": pair_id, "capture": {"baseline_m": 0.5}})

    overlap_path = tmp_path / "overlap.jsonl"
    analysis_path = tmp_path / "analysis.jsonl"
    _write_jsonl(overlap_path, overlap_rows)
    _write_jsonl(analysis_path, analysis_rows)
    return overlap_path, analysis_path, roots


def _canonical_sha(payload: dict[str, Any]) -> str:
    without_sha = dict(payload)
    without_sha.pop("canonical_payload_sha256")
    return hashlib.sha256(
        json.dumps(without_sha, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_builder_selects_new_balanced_output_blind_cohort(tmp_path: Path) -> None:
    overlap, analysis, roots = _synthetic_inventory(tmp_path)

    first = build_promotion_manifest(
        overlap_path=overlap,
        analysis_path=analysis,
        roots=roots,
        frozen_at="2026-10-17T03:30:00Z",
    )
    second = build_promotion_manifest(
        overlap_path=overlap,
        analysis_path=analysis,
        roots=roots,
        frozen_at="2026-10-17T03:30:00Z",
    )

    assert first == second
    assert first["schema"] == "panorai-registered-stereo-promotion-selection/v1"
    assert first["frozen_before_dense_evaluation"] is True
    assert first["canonical_payload_sha256"] == _canonical_sha(first)
    assert Counter(pair["dataset_id"] for pair in first["pairs"]) == {
        "stanford2d3d": 6,
        "matterport360": 6,
    }
    assert Counter(
        pair["group"] for pair in first["pairs"] if pair["dataset_id"] == "stanford2d3d"
    ) == {"area_2": 2, "area_3": 2, "area_4": 2}
    assert (
        len(
            {
                pair["group"]
                for pair in first["pairs"]
                if pair["dataset_id"] == "matterport360"
            }
        )
        == 6
    )
    views = [pair[key] for pair in first["pairs"] for key in ("reference", "source")]
    assert len(views) == len(set(views))
    policy = first["selection_policy"]
    assert policy["dense_output_used_for_selection"] is False
    assert policy["rgb_match_used_for_selection"] is False
    assert policy["range_reference_used_for_selection"] is False
    assert policy["post_protocol_holdout"] is True
    assert policy["all_pre_protocol_public_stereo_manifests_excluded"] is True
    assert policy["prediction_independent_non_sky_required_during_evaluation"] is True
    assert policy["fixed_denominator_required_during_evaluation"] is True


def test_builder_binds_every_forbidden_prior_manifest() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    forbidden = set(protocol["policy"]["forbidden_cohort_manifest_sha256"])
    observed = {
        hashlib.sha256(path.read_bytes()).hexdigest() for path in PRIOR_MANIFESTS
    }

    assert observed == forbidden
    assert len(PRIOR_MANIFESTS) == 7


def test_builder_rejects_timestamp_before_protocol(tmp_path: Path) -> None:
    overlap, analysis, roots = _synthetic_inventory(tmp_path)

    with pytest.raises(ValueError, match="after the promotion protocol"):
        build_promotion_manifest(
            overlap_path=overlap,
            analysis_path=analysis,
            roots=roots,
            frozen_at="2026-10-16T23:59:59Z",
        )


def test_builder_fails_when_prior_manifest_drifted(tmp_path: Path) -> None:
    overlap, analysis, roots = _synthetic_inventory(tmp_path)
    changed = tmp_path / "changed-prior.json"
    changed.write_text("{}\n")
    priors = (changed, *PRIOR_MANIFESTS[1:])

    with pytest.raises(ValueError, match="frozen forbidden cohort"):
        build_promotion_manifest(
            overlap_path=overlap,
            analysis_path=analysis,
            roots=roots,
            frozen_at="2026-10-17T03:30:00Z",
            prior_manifests=priors,
        )


def test_builder_fails_closed_when_new_inventory_is_insufficient(
    tmp_path: Path,
) -> None:
    overlap, analysis, roots = _synthetic_inventory(tmp_path)
    rows = overlap.read_text().splitlines()
    overlap.write_text("\n".join(rows[:-1]) + "\n")

    with pytest.raises(RuntimeError, match="Matterport pairs; selected=5"):
        build_promotion_manifest(
            overlap_path=overlap,
            analysis_path=analysis,
            roots=roots,
            frozen_at="2026-10-17T03:30:00Z",
        )
