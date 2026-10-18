from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.panoramic_stereo.build_registered_stereo_promotion_manifest import (
    build_promotion_manifest,
)
from benchmarks.panoramic_stereo.run_registered_stereo_heldout import (
    CORPUS_NAMES,
    PROFILE_ID,
    aggregate_domain,
    evidence_records,
    load_manifest,
    verify_source_state,
)
from scripts.evaluate_spherical_application_promotion import evaluate


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "contracts" / "registered-full-sphere-stereo-promotion-v1.json"


def _pair_record(dataset: str, index: int) -> dict[str, Any]:
    return {
        "dataset_id": dataset,
        "pair_id": f"pair-{dataset}-{index}",
        "metrics": {
            "overall": {
                "population_weight": 100.0,
                "predicted_valid_weight": 50.0,
                "delta_1_25": 0.30,
            }
        },
        "population_audit": {"policy": "fixed"},
        "gross_gt_2x": {
            "fixed_population_fraction": 0.10,
            "conditional_fraction": 0.20,
        },
        "prediction_independent_non_sky": True,
        "fixed_denominator": True,
        "provided_metric_pose": True,
        "invalid_values_are_nan": True,
        "finite_exactly_equals_validity": True,
        "deterministic_replay": True,
        "array_payload_sha256": f"array-{dataset}-{index}",
        "evaluation_population_sha256": f"population-{dataset}-{index}",
    }


def test_domain_aggregation_keeps_missing_predictions_in_denominator() -> None:
    records = [_pair_record("stanford2d3d", index) for index in range(6)]

    metrics = aggregate_domain(records)

    assert metrics == {
        "pair_count": 6,
        "fixed_delta_1_25": pytest.approx(0.30),
        "coverage": pytest.approx(0.50),
        "conditional_gross_gt_2x": pytest.approx(0.20),
        "prediction_independent_non_sky": True,
        "fixed_denominator": True,
        "provided_metric_pose": True,
        "invalid_values_are_nan": True,
        "deterministic_replay": True,
    }


def test_external_records_match_frozen_protocol_selector_and_checks() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    digest = __import__("hashlib").sha256(PROTOCOL.read_bytes()).hexdigest()
    records = [
        _pair_record(dataset, index) for dataset in CORPUS_NAMES for index in range(6)
    ]
    external = evidence_records(
        records,
        source_commit="a" * 40,
        cohort_manifest_sha256="new-post-protocol-cohort",
    )
    result = evaluate(
        protocol,
        {
            "contract": protocol["contract"],
            "protocol_sha256": digest,
            "records": external,
        },
        protocol_sha256=digest,
    )

    assert {item["corpus"] for item in external} == {
        "Matterport3D",
        "Stanford2D3D",
    }
    assert all(item["profile_id"] == PROFILE_ID for item in external)
    heldout = next(
        item
        for item in result["requirements"]
        if item["requirement"] == "registered-stereo-new-public-heldout"
    )
    assert heldout["passed"] is True
    assert result["passed"] is False  # other promotion gates are intentionally absent


def test_manifest_loader_rejects_a_forbidden_prior_cohort() -> None:
    prior = (
        ROOT
        / "benchmarks"
        / "panoramic_stereo"
        / "public_angular_neighborhood_manifest.json"
    )
    with pytest.raises(ValueError, match="previously observed evidence"):
        load_manifest(prior)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _valid_selection(tmp_path: Path) -> Path:
    roots = {
        "stanford2d3d": tmp_path / "stanford",
        "matterport360": tmp_path / "matterport",
    }
    overlap_rows = []
    analysis_rows = []
    specs = []
    for area in ("area_2", "area_3", "area_4"):
        specs.extend(("stanford2d3d", area, index) for index in range(2))
    specs.extend(("matterport360", f"site_{index}", 0) for index in range(6))
    for sequence, (dataset, group, index) in enumerate(specs):
        reference = f"fresh-{sequence}-reference"
        source = f"fresh-{sequence}-source"
        for name in (reference, source):
            for folder, suffix in (
                ("rgb", ".png"),
                ("depth", ".npy"),
                ("poses", ".txt"),
            ):
                asset = roots[dataset] / folder / group / f"{name}{suffix}"
                asset.parent.mkdir(parents=True, exist_ok=True)
                asset.touch()
        pair_id = f"fresh-pair-{sequence}"
        overlap_rows.append(
            {
                "dataset_id": dataset,
                "pair_id": pair_id,
                "from_view_id": f"{dataset}::{group}::{reference}",
                "to_view_id": f"{dataset}::{group}::{source}",
                "spatial_group_id": f"{dataset}::{group}",
                "minimum_directional_cloud_overlap": 0.9,
            }
        )
        analysis_rows.append(
            {"pair_id": pair_id, "capture": {"baseline_m": 0.5 + index * 0.1}}
        )
    overlap = tmp_path / "overlap.jsonl"
    analysis = tmp_path / "analysis.jsonl"
    _write_jsonl(overlap, overlap_rows)
    _write_jsonl(analysis, analysis_rows)
    payload = build_promotion_manifest(
        overlap_path=overlap,
        analysis_path=analysis,
        roots=roots,
        frozen_at="2026-10-17T03:30:00Z",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return manifest


def test_manifest_loader_accepts_builder_output_and_detects_drift(
    tmp_path: Path,
) -> None:
    manifest = _valid_selection(tmp_path)
    payload, digest = load_manifest(manifest)

    assert len(payload["pairs"]) == 12
    assert len(digest) == 64

    payload["pairs"][0]["baseline_m"] = 1.25
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="canonical payload hash changed"):
        load_manifest(manifest)


def test_source_state_must_match_head_and_be_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(("a" * 40 + "\n", ""))

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        return type("Result", (), {"stdout": next(responses)})()

    monkeypatch.setattr(
        "benchmarks.panoramic_stereo.run_registered_stereo_heldout.subprocess.run",
        fake_run,
    )
    verify_source_state("a" * 40)

    mismatch = iter(("b" * 40 + "\n",))

    def fake_mismatch(*args: Any, **kwargs: Any) -> Any:
        return type("Result", (), {"stdout": next(mismatch)})()

    monkeypatch.setattr(
        "benchmarks.panoramic_stereo.run_registered_stereo_heldout.subprocess.run",
        fake_mismatch,
    )
    with pytest.raises(ValueError, match="does not match checkout HEAD"):
        verify_source_state("a" * 40)
