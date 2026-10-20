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
from benchmarks.spherical_dog.run_reviewed_holdout import (
    _validate_manifest,
    _validate_manifest_assets,
)


FROZEN_AFTER_PROTOCOL = "2026-10-20T00:00:33Z"


def _asset(root: Path, relative: str) -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(relative.encode())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _view(
    root: Path, view_id: str, *, center_x: float, exclusion_group_id: str
) -> dict[str, Any]:
    rgb = f"rgb/{exclusion_group_id}/{view_id}.png"
    radial_range = f"range/{exclusion_group_id}/{view_id}.npy"
    rgb_sha256 = _asset(root, rgb)
    range_sha256 = _asset(root, radial_range)
    _asset(root, f"poses/{exclusion_group_id}/{view_id}.txt")
    return {
        "view_id": view_id,
        "rgb": rgb,
        "rgb_sha256": rgb_sha256,
        "range": radial_range,
        "range_sha256": range_sha256,
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
            exclusion_group = f"m{index:010d}" if corpus == "Matterport3D" else "area_1"
            scene = (
                exclusion_group
                if corpus == "Matterport3D"
                else f"{exclusion_group}/room_{index}"
            )
            rows.append(
                {
                    "corpus": corpus,
                    "scene_id": scene,
                    "exclusion_group_id": exclusion_group,
                    "pair_id": f"{prefix}-pair-{index}",
                    "directional_cloud_overlap": 0.75,
                    "reference": _view(
                        roots[corpus],
                        f"{prefix}-r-{index}",
                        center_x=0.0,
                        exclusion_group_id=exclusion_group,
                    ),
                    "source": _view(
                        roots[corpus],
                        f"{prefix}-s-{index}",
                        center_x=0.5,
                        exclusion_group_id=exclusion_group,
                    ),
                }
            )
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return inventory, roots


def _rehome_group(row: dict[str, Any], root: Path, group: str) -> None:
    old_group = row["exclusion_group_id"]
    row["exclusion_group_id"] = group
    row["scene_id"] = (
        group
        if row["corpus"] == "Matterport3D"
        else f"{group}/{row['scene_id'].split('/', 1)[-1]}"
    )
    for view_name in ("reference", "source"):
        view_id = row[view_name]["view_id"]
        old_pose = root / "poses" / old_group / f"{view_id}.txt"
        new_pose = root / "poses" / group / f"{view_id}.txt"
        new_pose.parent.mkdir(parents=True, exist_ok=True)
        old_pose.replace(new_pose)
        for asset in ("rgb", "range"):
            old_relative = Path(row[view_name][asset])
            new_relative = old_relative.parent.parent / group / old_relative.name
            new_path = root / new_relative
            new_path.parent.mkdir(parents=True, exist_ok=True)
            new_path.write_bytes((root / old_relative).read_bytes())
            row[view_name][asset] = new_relative.as_posix()


def _validate(path: Path, inventory: Path, roots: dict[str, Path]) -> dict[str, Any]:
    return _validate_manifest(
        path,
        PROTOCOL,
        EXCLUSIONS,
        inventory_path=inventory,
        roots=roots,
    )


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
        frozen_at=FROZEN_AFTER_PROTOCOL,
    )
    second = build_manifest(
        inventory_path=inventory,
        roots=roots,
        frozen_at=FROZEN_AFTER_PROTOCOL,
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
        "asset_reuse": False,
        "asset_identity": "precomputed SHA-256 verified before inference",
        "minimum_directional_cloud_overlap": 0.5,
        "minimum_metric_baseline_m": 0.15,
        "maximum_metric_baseline_m": 2.5,
        "rgb_pixels_read": False,
        "range_values_read": False,
        "features_or_matches_read": False,
        "development_groups_excluded": True,
        "separate_scene_and_exclusion_group_ids": True,
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
    _rehome_group(rows[0], roots["Matterport3D"], "17drp5sb8fy")
    _rehome_group(rows[8], roots["Stanford2D3D"], "area_2")
    inventory.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = build_manifest(
        inventory_path=inventory,
        roots=roots,
        frozen_at=FROZEN_AFTER_PROTOCOL,
    )

    assert "17drp5sb8fy" not in {item["exclusion_group_id"] for item in result["pairs"]}
    assert "area_2" not in {item["exclusion_group_id"] for item in result["pairs"]}


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
            frozen_at=FROZEN_AFTER_PROTOCOL,
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
        frozen_at=FROZEN_AFTER_PROTOCOL,
    )

    selected = {item["pair_id"] for item in result["pairs"]}
    assert rows[0]["pair_id"] not in selected
    assert rows[1]["pair_id"] not in selected


@pytest.mark.parametrize(
    "rotation",
    (
        [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ),
)
def test_selector_rejects_non_so3_camera_rotations(tmp_path, rotation) -> None:
    inventory, roots = _inventory(tmp_path)
    rows = [json.loads(line) for line in inventory.read_text().splitlines()]
    rows[0]["reference"]["R_world_to_camera"] = rotation
    inventory.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="orthonormal rotation|determinant \\+1"):
        build_manifest(
            inventory_path=inventory,
            roots=roots,
            frozen_at=FROZEN_AFTER_PROTOCOL,
        )


def test_selector_enforces_asset_disjointness_across_corpora(tmp_path) -> None:
    inventory, roots = _inventory(tmp_path)
    rows = [json.loads(line) for line in inventory.read_text().splitlines()]
    matterport = [row for row in rows if row["corpus"] == "Matterport3D"]
    stanford = [row for row in rows if row["corpus"] == "Stanford2D3D"]
    for source, mirrored in zip(matterport, stanford, strict=True):
        for view in ("reference", "source"):
            for asset in ("rgb", "range"):
                source_path = roots["Matterport3D"] / source[view][asset]
                mirrored_path = roots["Stanford2D3D"] / mirrored[view][asset]
                mirrored_path.write_bytes(source_path.read_bytes())
                mirrored[view][f"{asset}_sha256"] = source[view][f"{asset}_sha256"]
    inventory.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(RuntimeError, match="insufficient unseen Stanford2D3D"):
        build_manifest(
            inventory_path=inventory,
            roots=roots,
            frozen_at=FROZEN_AFTER_PROTOCOL,
        )


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
            frozen_at=FROZEN_AFTER_PROTOCOL,
            exclusions_path=changed,
        )


def test_evaluator_revalidates_development_exclusions(tmp_path) -> None:
    inventory, roots = _inventory(tmp_path)
    manifest = build_manifest(
        inventory_path=inventory,
        roots=roots,
        frozen_at=FROZEN_AFTER_PROTOCOL,
    )
    matterport = next(
        item for item in manifest["pairs"] if item["corpus"] == "Matterport3D"
    )
    _rehome_group(matterport, roots["Matterport3D"], "17drp5sb8fy")
    manifest["canonical_payload_sha256"] = _canonical_sha(manifest)
    path = tmp_path / "holdout.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="reuses a development scene"):
        _validate(path, inventory, roots)


def test_exclusion_group_cannot_be_bypassed_by_a_finer_scene_id(tmp_path) -> None:
    inventory, roots = _inventory(tmp_path)
    rows = [json.loads(line) for line in inventory.read_text().splitlines()]
    _rehome_group(rows[8], roots["Stanford2D3D"], "area_2")
    rows[8]["scene_id"] = "area_2/office_4"
    inventory.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = build_manifest(
        inventory_path=inventory,
        roots=roots,
        frozen_at=FROZEN_AFTER_PROTOCOL,
    )

    assert rows[8]["pair_id"] not in {item["pair_id"] for item in result["pairs"]}


def test_selector_rejects_spoofed_exclusion_group_identity(tmp_path) -> None:
    inventory, roots = _inventory(tmp_path)
    rows = [json.loads(line) for line in inventory.read_text().splitlines()]
    rows[8]["exclusion_group_id"] = "area_5a"
    inventory.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="canonical dataset identity"):
        build_manifest(
            inventory_path=inventory,
            roots=roots,
            frozen_at=FROZEN_AFTER_PROTOCOL,
        )


def test_selector_rejects_assets_copied_to_a_spoofed_group(tmp_path) -> None:
    inventory, roots = _inventory(tmp_path)
    rows = [json.loads(line) for line in inventory.read_text().splitlines()]
    row = rows[8]
    row["scene_id"] = "area_5a/room_0"
    row["exclusion_group_id"] = "area_5a"
    for view_name in ("reference", "source"):
        for asset in ("rgb", "range"):
            old_relative = Path(row[view_name][asset])
            new_relative = old_relative.parent.parent / "area_5a" / old_relative.name
            target = roots["Stanford2D3D"] / new_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((roots["Stanford2D3D"] / old_relative).read_bytes())
            row[view_name][asset] = new_relative.as_posix()
    inventory.write_text("".join(json.dumps(item) + "\n" for item in rows))

    with pytest.raises(ValueError, match="authoritative dataset assets"):
        build_manifest(
            inventory_path=inventory,
            roots=roots,
            frozen_at=FROZEN_AFTER_PROTOCOL,
        )


def test_evaluator_rejects_manifest_group_relabeling(tmp_path) -> None:
    inventory, roots = _inventory(tmp_path)
    manifest = build_manifest(
        inventory_path=inventory,
        roots=roots,
        frozen_at=FROZEN_AFTER_PROTOCOL,
    )
    stanford = next(
        item for item in manifest["pairs"] if item["corpus"] == "Stanford2D3D"
    )
    stanford["exclusion_group_id"] = "area_5a"
    manifest["canonical_payload_sha256"] = _canonical_sha(manifest)
    path = tmp_path / "holdout.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="canonical dataset identity"):
        _validate(path, inventory, roots)


def test_evaluator_reproduces_selection_from_exact_inventory(tmp_path) -> None:
    inventory, roots = _inventory(tmp_path)
    manifest = build_manifest(
        inventory_path=inventory,
        roots=roots,
        frozen_at=FROZEN_AFTER_PROTOCOL,
    )
    manifest["pairs"].reverse()
    manifest["canonical_payload_sha256"] = _canonical_sha(manifest)
    path = tmp_path / "holdout.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="differs from deterministic selection"):
        _validate(path, inventory, roots)


def test_evaluator_rejects_inventory_byte_drift(tmp_path) -> None:
    inventory, roots = _inventory(tmp_path)
    manifest = build_manifest(
        inventory_path=inventory,
        roots=roots,
        frozen_at=FROZEN_AFTER_PROTOCOL,
    )
    path = tmp_path / "holdout.json"
    path.write_text(json.dumps(manifest))
    inventory.write_text(inventory.read_text() + "\n")

    with pytest.raises(ValueError, match="inventory SHA-256 has drifted"):
        _validate(path, inventory, roots)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("directional_cloud_overlap", 0.49, "ineligible overlap"),
        ("baseline_m", 0.0, "baseline does not match camera centres"),
    ),
)
def test_evaluator_revalidates_frozen_pair_eligibility(
    tmp_path, field, value, message
) -> None:
    inventory, roots = _inventory(tmp_path)
    manifest = build_manifest(
        inventory_path=inventory,
        roots=roots,
        frozen_at=FROZEN_AFTER_PROTOCOL,
    )
    manifest["pairs"][0][field] = value
    manifest["canonical_payload_sha256"] = _canonical_sha(manifest)
    path = tmp_path / "holdout.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=message):
        _validate(path, inventory, roots)


def test_evaluator_rejects_non_so3_rotation_with_valid_manifest_hash(tmp_path) -> None:
    inventory, roots = _inventory(tmp_path)
    manifest = build_manifest(
        inventory_path=inventory,
        roots=roots,
        frozen_at=FROZEN_AFTER_PROTOCOL,
    )
    manifest["pairs"][0]["source"]["R_world_to_camera"] = [
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    manifest["canonical_payload_sha256"] = _canonical_sha(manifest)
    path = tmp_path / "holdout.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="determinant \\+1"):
        _validate(path, inventory, roots)


def test_selector_requires_strictly_post_protocol_timestamp(tmp_path) -> None:
    inventory, roots = _inventory(tmp_path)
    protocol = json.loads(PROTOCOL.read_text())

    with pytest.raises(ValueError, match="after the promotion protocol"):
        build_manifest(
            inventory_path=inventory,
            roots=roots,
            frozen_at=protocol["frozen_at"],
        )


def test_evaluator_revalidates_strictly_post_protocol_timestamp(tmp_path) -> None:
    inventory, roots = _inventory(tmp_path)
    manifest = build_manifest(
        inventory_path=inventory,
        roots=roots,
        frozen_at=FROZEN_AFTER_PROTOCOL,
    )
    protocol = json.loads(PROTOCOL.read_text())
    manifest["frozen_at"] = protocol["frozen_at"]
    manifest["canonical_payload_sha256"] = _canonical_sha(manifest)
    path = tmp_path / "holdout.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="frozen after the protocol"):
        _validate(path, inventory, roots)


def test_evaluator_rejects_a_future_freeze_timestamp(tmp_path) -> None:
    inventory, roots = _inventory(tmp_path)
    manifest = build_manifest(
        inventory_path=inventory,
        roots=roots,
        frozen_at=FROZEN_AFTER_PROTOCOL,
    )
    manifest["frozen_at"] = "2999-01-01T00:00:00Z"
    manifest["canonical_payload_sha256"] = _canonical_sha(manifest)
    path = tmp_path / "holdout.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="freeze timestamp is in the future"):
        _validate(path, inventory, roots)


def test_evaluator_rejects_reused_assets_under_distinct_ids(tmp_path) -> None:
    inventory, roots = _inventory(tmp_path)
    manifest = build_manifest(
        inventory_path=inventory,
        roots=roots,
        frozen_at=FROZEN_AFTER_PROTOCOL,
    )
    first = manifest["pairs"][0]["reference"]
    copied = manifest["pairs"][1]["reference"]
    copied["rgb"] = first["rgb"]
    copied["rgb_sha256"] = first["rgb_sha256"]
    manifest["canonical_payload_sha256"] = _canonical_sha(manifest)
    path = tmp_path / "holdout.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="asset-disjoint"):
        _validate(path, inventory, roots)


def test_evaluator_rejects_mirrored_assets_across_corpora(tmp_path) -> None:
    inventory, roots = _inventory(tmp_path)
    manifest = build_manifest(
        inventory_path=inventory,
        roots=roots,
        frozen_at=FROZEN_AFTER_PROTOCOL,
    )
    matterport = next(
        item for item in manifest["pairs"] if item["corpus"] == "Matterport3D"
    )
    stanford = next(
        item for item in manifest["pairs"] if item["corpus"] == "Stanford2D3D"
    )
    stanford["reference"]["rgb_sha256"] = matterport["reference"]["rgb_sha256"]
    manifest["canonical_payload_sha256"] = _canonical_sha(manifest)
    path = tmp_path / "holdout.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="asset-disjoint"):
        _validate(path, inventory, roots)


def test_evaluator_rejects_reused_paths_even_with_forged_hashes(tmp_path) -> None:
    inventory, roots = _inventory(tmp_path)
    manifest = build_manifest(
        inventory_path=inventory,
        roots=roots,
        frozen_at=FROZEN_AFTER_PROTOCOL,
    )
    first = manifest["pairs"][0]["reference"]
    copied = manifest["pairs"][1]["reference"]
    copied["rgb"] = first["rgb"]
    copied["rgb_sha256"] = "f" * 64
    manifest["canonical_payload_sha256"] = _canonical_sha(manifest)
    path = tmp_path / "holdout.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="reuse asset paths"):
        _validate(path, inventory, roots)


def test_evaluator_verifies_all_asset_bytes_before_inference(tmp_path) -> None:
    inventory, roots = _inventory(tmp_path)
    manifest = build_manifest(
        inventory_path=inventory,
        roots=roots,
        frozen_at=FROZEN_AFTER_PROTOCOL,
    )
    manifest["pairs"][0]["reference"]["rgb_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="asset SHA-256 does not match"):
        _validate_manifest_assets(manifest, roots)


def test_evaluator_verifies_global_byte_disjointness_before_inference(tmp_path) -> None:
    inventory, roots = _inventory(tmp_path)
    manifest = build_manifest(
        inventory_path=inventory,
        roots=roots,
        frozen_at=FROZEN_AFTER_PROTOCOL,
    )
    matterport = next(
        item for item in manifest["pairs"] if item["corpus"] == "Matterport3D"
    )
    stanford = next(
        item for item in manifest["pairs"] if item["corpus"] == "Stanford2D3D"
    )
    source = roots["Matterport3D"] / matterport["reference"]["rgb"]
    mirrored = roots["Stanford2D3D"] / stanford["reference"]["rgb"]
    mirrored.write_bytes(source.read_bytes())
    stanford["reference"]["rgb_sha256"] = matterport["reference"]["rgb_sha256"]

    with pytest.raises(ValueError, match="reused asset bytes"):
        _validate_manifest_assets(manifest, roots)
