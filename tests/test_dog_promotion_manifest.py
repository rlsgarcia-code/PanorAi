from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.spherical_dog.build_reviewed_holdout_manifest import (
    EXCLUSIONS,
    PROTOCOL,
    build_manifest,
)
from benchmarks.spherical_dog.run_reviewed_holdout import _validate_manifest


def _asset(root: Path, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _view(root: Path, view_id: str, *, center_x: float) -> dict[str, Any]:
    rgb = f"rgb/{view_id}.png"
    radial_range = f"range/{view_id}.npy"
    _asset(root, rgb)
    _asset(root, radial_range)
    return {
        "view_id": view_id,
        "rgb": rgb,
        "range": radial_range,
        "R_world_to_camera": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "center_world": [center_x, 0.0, 0.0],
    }


def _inventory(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    roots = {
        "Matterport3D": tmp_path / "matterport",
        "Stanford2D3D": tmp_path / "stanford",
    }
    rows: list[dict[str, Any]] = []
    for corpus, prefix in (("Matterport3D", "mp"), ("Stanford2D3D", "st")):
        for index in range(8):
            scene = f"new-{prefix}-scene-{index}"
            rows.append(
                {
                    "corpus": corpus,
                    "scene_id": scene,
                    "pair_id": f"{prefix}-pair-{index}",
                    "directional_cloud_overlap": 0.75,
                    "reference": _view(
                        roots[corpus], f"{prefix}-r-{index}", center_x=0.0
                    ),
                    "source": _view(roots[corpus], f"{prefix}-s-{index}", center_x=0.5),
                }
            )
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return inventory, roots


def _canonical_sha(payload: dict[str, Any]) -> str:
    value = dict(payload)
    value.pop("canonical_payload_sha256")
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_selector_is_deterministic_scene_disjoint_and_output_blind(tmp_path) -> None:
    inventory, roots = _inventory(tmp_path)

    first = build_manifest(
        inventory_path=inventory,
        roots=roots,
        frozen_at="2026-10-19T22:00:00Z",
    )
    second = build_manifest(
        inventory_path=inventory,
        roots=roots,
        frozen_at="2026-10-19T22:00:00Z",
    )

    assert first == second
    assert first["canonical_payload_sha256"] == _canonical_sha(first)
    assert Counter(item["corpus"] for item in first["pairs"]) == {
        "Matterport3D": 6,
        "Stanford2D3D": 6,
    }
    scenes = [(item["corpus"], item["scene_id"]) for item in first["pairs"]]
    views = [
        item[key]["view_id"]
        for item in first["pairs"]
        for key in ("reference", "source")
    ]
    assert len(scenes) == len(set(scenes)) == 12
    assert len(views) == len(set(views)) == 24
    assert first["selection_policy"] == {
        "ordering": "sha256(corpus + NUL + scene_id + NUL + pair_id)",
        "pairs_per_corpus": 6,
        "one_pair_per_scene": True,
        "view_reuse": False,
        "minimum_directional_cloud_overlap": 0.5,
        "minimum_metric_baseline_m": 0.15,
        "maximum_metric_baseline_m": 2.5,
        "rgb_pixels_read": False,
        "range_values_read": False,
        "features_or_matches_read": False,
        "development_groups_excluded": True,
    }


def test_selector_binds_frozen_protocol_and_development_ledger() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    binding = protocol["policy"]["development_exclusion_ledger"]

    assert binding["path"] == EXCLUSIONS.name
    assert binding["sha256"] == hashlib.sha256(EXCLUSIONS.read_bytes()).hexdigest()
    ledger = json.loads(EXCLUSIONS.read_text())
    assert "area_2" in ledger["excluded_groups"]["Stanford2D3D"]
    assert "17drp5sb8fy" in ledger["excluded_groups"]["Matterport3D"]


def test_selector_excludes_development_groups(tmp_path) -> None:
    inventory, roots = _inventory(tmp_path)
    rows = [json.loads(line) for line in inventory.read_text().splitlines()]
    rows[0]["scene_id"] = "17drp5sb8fy"
    rows[8]["scene_id"] = "area_2"
    inventory.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = build_manifest(
        inventory_path=inventory,
        roots=roots,
        frozen_at="2026-10-19T22:00:00Z",
    )

    assert "17drp5sb8fy" not in {item["scene_id"] for item in result["pairs"]}
    assert "area_2" not in {item["scene_id"] for item in result["pairs"]}


def test_selector_fails_closed_on_insufficient_unseen_scenes(tmp_path) -> None:
    inventory, roots = _inventory(tmp_path)
    rows = [json.loads(line) for line in inventory.read_text().splitlines()]
    rows = [
        item
        for item in rows
        if item["corpus"] != "Matterport3D"
        or int(item["pair_id"].rsplit("-", 1)[1]) < 5
    ]
    inventory.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(RuntimeError, match="insufficient unseen Matterport3D"):
        build_manifest(
            inventory_path=inventory,
            roots=roots,
            frozen_at="2026-10-19T22:00:00Z",
        )


def test_selector_applies_frozen_overlap_and_metric_baseline(tmp_path) -> None:
    inventory, roots = _inventory(tmp_path)
    rows = [json.loads(line) for line in inventory.read_text().splitlines()]
    rows[0]["directional_cloud_overlap"] = 0.49
    rows[1]["source"]["center_world"] = [2.51, 0.0, 0.0]
    inventory.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = build_manifest(
        inventory_path=inventory,
        roots=roots,
        frozen_at="2026-10-19T22:00:00Z",
    )

    selected = {item["pair_id"] for item in result["pairs"]}
    assert rows[0]["pair_id"] not in selected
    assert rows[1]["pair_id"] not in selected


def test_selector_rejects_pre_protocol_timestamp_and_ledger_drift(tmp_path) -> None:
    inventory, roots = _inventory(tmp_path)
    with pytest.raises(ValueError, match="after the promotion protocol"):
        build_manifest(
            inventory_path=inventory,
            roots=roots,
            frozen_at="2026-10-18T23:59:59Z",
        )

    changed = tmp_path / EXCLUSIONS.name
    changed.write_text(EXCLUSIONS.read_text().replace("area_2", "area_x", 1))
    with pytest.raises(ValueError, match="ledger has drifted"):
        build_manifest(
            inventory_path=inventory,
            roots=roots,
            frozen_at="2026-10-19T22:00:00Z",
            exclusions_path=changed,
        )


def test_evaluator_revalidates_development_exclusions(tmp_path) -> None:
    inventory, roots = _inventory(tmp_path)
    manifest = build_manifest(
        inventory_path=inventory,
        roots=roots,
        frozen_at="2026-10-19T22:00:00Z",
    )
    manifest["pairs"][0]["scene_id"] = "17drp5sb8fy"
    manifest["canonical_payload_sha256"] = _canonical_sha(manifest)
    path = tmp_path / "holdout.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="reuses a development scene"):
        _validate_manifest(path, PROTOCOL, EXCLUSIONS)
