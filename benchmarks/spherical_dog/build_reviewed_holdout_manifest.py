#!/usr/bin/env python3
"""Freeze a scene-disjoint DoG cohort before feature inference.

The selector reads identifiers, registered poses and asset *presence* only.
It never opens RGB pixels, radial-range values, descriptors, matches or method
outputs.  Dataset bytes and the resulting manifest remain outside artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

if __package__:
    from ._dataset_identity import canonical_exclusion_group  # type: ignore[import-not-found]
else:
    from _dataset_identity import canonical_exclusion_group


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "contracts" / "dog-reviewed-profile-promotion-v1.json"
EXCLUSIONS = ROOT / "contracts" / "dog-development-exclusions-v1.json"
CORPUS_KEYS = {"Matterport3D": "matterport", "Stanford2D3D": "stanford"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("frozen_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("frozen_at must include a timezone")
    return parsed.astimezone(timezone.utc)


def _finite_vector(value: Any, length: int, path: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{path} must contain {length} numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{path} must be finite")
    return result


def _validated_rotation(value: Any, path: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{path} must be 3x3")
    rows = [list(_finite_vector(row, 3, path)) for row in value]
    for row_index, left in enumerate(rows):
        for column_index, right in enumerate(rows):
            expected = 1.0 if row_index == column_index else 0.0
            dot = sum(a * b for a, b in zip(left, right, strict=True))
            if not math.isclose(dot, expected, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(f"{path} must be an orthonormal rotation")
    determinant = (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )
    if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{path} must have determinant +1")
    return rows


def _validate_view(value: Any, *, corpus_root: Path, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    required = {
        "view_id",
        "rgb",
        "rgb_sha256",
        "range",
        "range_sha256",
        "R_world_to_camera",
        "center_world",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{path} is missing {missing}")
    view_id = value["view_id"]
    if not isinstance(view_id, str) or not view_id:
        raise ValueError(f"{path}.view_id must be a non-empty string")
    normalized = dict(value)
    for asset in ("rgb", "range"):
        relative = value[asset]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError(f"{path}.{asset} must be a safe relative path")
        if not (corpus_root / relative).is_file():
            raise FileNotFoundError(f"{path}.{asset} is missing: {relative}")
        digest = value[f"{asset}_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{path}.{asset}_sha256 must be a lowercase SHA-256")
    normalized["R_world_to_camera"] = _validated_rotation(
        value["R_world_to_camera"], f"{path}.R_world_to_camera"
    )
    normalized["center_world"] = list(
        _finite_vector(value["center_world"], 3, f"{path}.center_world")
    )
    return normalized


def _load_inventory(path: Path, roots: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"inventory line {line_number} must be an object")
        corpus = value.get("corpus")
        if corpus not in CORPUS_KEYS or corpus not in roots:
            raise ValueError(f"inventory line {line_number} has unknown corpus")
        pair_id = value.get("pair_id")
        scene_id = value.get("scene_id")
        exclusion_group_id = value.get("exclusion_group_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"inventory line {line_number} has invalid pair_id")
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError(f"inventory line {line_number} has invalid scene_id")
        if not isinstance(exclusion_group_id, str) or not exclusion_group_id:
            raise ValueError(
                f"inventory line {line_number} has invalid exclusion_group_id"
            )
        normalized = dict(value)
        normalized["reference"] = _validate_view(
            value.get("reference"),
            corpus_root=roots[corpus],
            path=f"inventory[{line_number}].reference",
        )
        normalized["source"] = _validate_view(
            value.get("source"),
            corpus_root=roots[corpus],
            path=f"inventory[{line_number}].source",
        )
        if normalized["reference"]["view_id"] == normalized["source"]["view_id"]:
            raise ValueError(f"inventory line {line_number} repeats one view")
        overlap = value.get("directional_cloud_overlap")
        if isinstance(overlap, bool) or not isinstance(overlap, (int, float)):
            raise ValueError(
                f"inventory line {line_number} has invalid directional overlap"
            )
        overlap = float(overlap)
        if not math.isfinite(overlap) or not 0.0 <= overlap <= 1.0:
            raise ValueError(
                f"inventory line {line_number} has invalid directional overlap"
            )
        normalized["directional_cloud_overlap"] = overlap
        normalized["baseline_m"] = math.dist(
            normalized["reference"]["center_world"],
            normalized["source"]["center_world"],
        )
        normalized["exclusion_group_id"] = canonical_exclusion_group(
            normalized, corpus_root=roots[corpus]
        )
        rows.append(normalized)
    return rows


def _selection_key(row: dict[str, Any]) -> tuple[str, str, str]:
    identity = "\0".join((row["corpus"], row["scene_id"], row["pair_id"]))
    return (
        hashlib.sha256(identity.encode()).hexdigest(),
        row["scene_id"],
        row["pair_id"],
    )


def build_manifest(
    *,
    inventory_path: Path,
    roots: dict[str, Path],
    frozen_at: str,
    protocol_path: Path = PROTOCOL,
    exclusions_path: Path = EXCLUSIONS,
    pairs_per_corpus: int = 6,
) -> dict[str, Any]:
    """Select exactly one pair per previously unseen scene in each corpus."""

    if pairs_per_corpus != 6:
        raise ValueError("reviewed-v1 requires exactly six pairs per corpus")
    protocol = _load_json(protocol_path)
    if (
        protocol.get("contract")
        != "panorai-spherical-dog-reviewed-profile-promotion/v1"
    ):
        raise ValueError("unexpected DoG promotion protocol")
    exclusions = _load_json(exclusions_path)
    if exclusions.get("contract") != "panorai-dog-development-exclusions/v1":
        raise ValueError("unexpected DoG development exclusion ledger")
    expected_ledger = protocol["policy"]["development_exclusion_ledger"]
    if exclusions_path.name != expected_ledger["path"]:
        raise ValueError("protocol references a different exclusion ledger")
    if _sha256(exclusions_path) != expected_ledger["sha256"]:
        raise ValueError("development exclusion ledger has drifted")

    timestamp = _parse_timestamp(frozen_at)
    protocol_date = _parse_timestamp(str(protocol["frozen_at"]))
    if timestamp <= protocol_date:
        raise ValueError("cohort must be frozen after the promotion protocol")
    if timestamp > datetime.now(timezone.utc):
        raise ValueError("cohort freeze timestamp cannot be in the future")

    inventory = _load_inventory(inventory_path, roots)
    excluded_groups = exclusions["excluded_groups"]
    selection = protocol["policy"]["cohort_selection"]
    minimum_overlap = float(selection["minimum_directional_cloud_overlap"])
    minimum_baseline = float(selection["minimum_metric_baseline_m"])
    maximum_baseline = float(selection["maximum_metric_baseline_m"])
    selected: list[dict[str, Any]] = []
    used_views: set[tuple[str, str]] = set()
    used_assets: set[str] = set()
    for corpus in CORPUS_KEYS:
        denied = set(excluded_groups[corpus])
        eligible = sorted(
            (
                row
                for row in inventory
                if row["corpus"] == corpus
                and row["exclusion_group_id"] not in denied
                and row["directional_cloud_overlap"] >= minimum_overlap
                and minimum_baseline <= row["baseline_m"] <= maximum_baseline
            ),
            key=_selection_key,
        )
        used_scenes: set[str] = set()
        corpus_rows: list[dict[str, Any]] = []
        for row in eligible:
            views = {
                (corpus, row["reference"]["view_id"]),
                (corpus, row["source"]["view_id"]),
            }
            assets = {
                row[view][f"{asset}_sha256"]
                for view in ("reference", "source")
                for asset in ("rgb", "range")
            }
            if (
                row["scene_id"] in used_scenes
                or views & used_views
                or assets & used_assets
                or len(assets) != 4
            ):
                continue
            corpus_rows.append(row)
            used_scenes.add(row["scene_id"])
            used_views.update(views)
            used_assets.update(assets)
            if len(corpus_rows) == pairs_per_corpus:
                break
        if len(corpus_rows) != pairs_per_corpus:
            raise RuntimeError(
                f"insufficient unseen {corpus} scenes; selected={len(corpus_rows)}"
            )
        selected.extend(corpus_rows)

    payload: dict[str, Any] = {
        "schema": "panorai-spherical-dog-reviewed-holdout/v1",
        "profile_id": protocol["scope"]["profile_id"],
        "protocol": {"path": protocol_path.name, "sha256": _sha256(protocol_path)},
        "development_exclusions": {
            "path": exclusions_path.name,
            "sha256": _sha256(exclusions_path),
        },
        "inventory": {
            "sha256": _sha256(inventory_path),
            "record_count": len(inventory),
        },
        "frozen_at": timestamp.isoformat().replace("+00:00", "Z"),
        "frozen_before_inference": True,
        "selection_policy": {
            "ordering": selection["ordering"],
            "pairs_per_corpus": pairs_per_corpus,
            "one_pair_per_scene": True,
            "view_reuse": False,
            "asset_reuse": False,
            "asset_identity": "precomputed SHA-256 verified before inference",
            "minimum_directional_cloud_overlap": minimum_overlap,
            "minimum_metric_baseline_m": minimum_baseline,
            "maximum_metric_baseline_m": maximum_baseline,
            "rgb_pixels_read": False,
            "range_values_read": False,
            "features_or_matches_read": False,
            "development_groups_excluded": True,
            "separate_scene_and_exclusion_group_ids": True,
        },
        "pairs": selected,
    }
    payload["canonical_payload_sha256"] = _canonical_sha(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--matterport-root", type=Path, required=True)
    parser.add_argument("--stanford-root", type=Path, required=True)
    parser.add_argument("--frozen-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite an already frozen manifest: {args.output}"
        )
    payload = build_manifest(
        inventory_path=args.inventory,
        roots={
            "Matterport3D": args.matterport_root,
            "Stanford2D3D": args.stanford_root,
        },
        frozen_at=args.frozen_at,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"frozen={args.output} sha256={_sha256(args.output)} "
        f"pairs={len(payload['pairs'])}"
    )


if __name__ == "__main__":
    main()
