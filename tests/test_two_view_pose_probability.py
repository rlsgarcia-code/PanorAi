from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.two_view_pose_probability.run_census import (
    audit_split_integrity,
    load_sources,
    run_census,
    standardize,
    validate_pairs,
)


def _row(
    pair: str,
    first: str,
    second: str,
    *,
    dataset: str = "dataset-a",
    group: str = "group-a",
) -> dict[str, str]:
    return {
        "pair_id": pair,
        "dataset_id": dataset,
        "spatial_group_id": group,
        "from_view_id": first,
        "to_view_id": second,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_load_sources_skips_prediction_manifest_header(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    _write_jsonl(path, [{"manifest": {"pair_count": 1}}, _row("p1", "a", "b")])

    rows, sources = load_sources([path])

    assert rows == [_row("p1", "a", "b")]
    assert sources[0]["pair_rows"] == 1
    assert sources[0]["skipped_manifest_rows"] == 1


def test_reversed_or_duplicate_pairs_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate or reversed"):
        validate_pairs([_row("p1", "a", "b"), _row("p2", "b", "a")])


def test_self_pair_and_duplicate_pair_id_are_rejected() -> None:
    with pytest.raises(ValueError, match="self-pair"):
        standardize([_row("p1", "a", "a")], seed="test")
    with pytest.raises(ValueError, match="duplicate dataset/pair IDs"):
        validate_pairs([_row("p1", "a", "b"), _row("p1", "c", "d")])


def test_shared_image_merges_nominal_groups_into_one_split() -> None:
    rows = [
        _row("p1", "a", "b", group="g1"),
        _row("p2", "b", "c", group="g2"),
        _row("p3", "d", "e", group="g3"),
        _row("p4", "f", "g", group="g4"),
    ]

    standardized, census = standardize(rows, seed="fixed")
    by_pair = {row["pair_id"]: row for row in standardized}

    assert (
        by_pair["p1"]["independence_component_id"]
        == by_pair["p2"]["independence_component_id"]
    )
    assert by_pair["p1"]["split"] == by_pair["p2"]["split"]
    assert census["dataset-a"]["spatial_groups"] == 4
    assert census["dataset-a"]["independence_components"] == 3
    audit_split_integrity(standardized)


def test_split_is_deterministic_and_has_no_image_or_group_leakage() -> None:
    rows = [
        _row(f"p{index}", f"a{index}", f"b{index}", group=f"g{index}")
        for index in range(20)
    ]

    first, census = standardize(rows, seed="fixed")
    second, _ = standardize(list(reversed(rows)), seed="fixed")

    assert [(row["pair_id"], row["split"]) for row in first] == [
        (row["pair_id"], row["split"]) for row in second
    ]
    assert census["dataset-a"]["components_by_split"] == {
        "development": 14,
        "calibration": 3,
        "evaluation": 3,
    }
    audit_split_integrity(first)


def test_run_census_writes_identity_only_reproducible_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "output"
    _write_jsonl(
        source,
        [
            _row("p1", "a", "b", group="g1"),
            _row("p2", "c", "d", group="g2"),
            _row("p3", "e", "f", group="g3"),
        ],
    )

    payload = run_census([source], output, seed="fixed")

    assert payload["totals"] == {
        "datasets": 1,
        "unique_pairs": 3,
        "unique_images": 6,
        "spatial_groups": 3,
        "independence_components": 3,
    }
    pairs = [
        json.loads(line)
        for line in (output / "standardized-pairs.jsonl").read_text().splitlines()
    ]
    assert {row["split"] for row in pairs} == {
        "development",
        "calibration",
        "evaluation",
    }
    assert not any("reference" in row or "quality_report" in row for row in pairs)
    assert (output / "split-manifest.jsonl").is_file()
    assert (output / "census.json").is_file()
