"""Canonical dataset identities used by the prospective DoG selector."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
from typing import Any


def _authoritative_pose_group(corpus_root: Path, view_id: str) -> str:
    matches = sorted((corpus_root / "poses").glob(f"*/{view_id}.txt"))
    matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        raise ValueError(
            f"view_id {view_id!r} must resolve to exactly one authoritative pose"
        )
    resolved_root = corpus_root.resolve()
    try:
        relative = matches[0].resolve().relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("authoritative pose escapes the corpus root") from exc
    if len(relative.parts) < 3 or relative.parts[0] != "poses":
        raise ValueError("authoritative pose has an invalid corpus layout")
    return relative.parts[1].lower()


def _stanford_room_id(view_id: str) -> str:
    match = re.fullmatch(
        r"camera_[0-9a-f]{32}_(.+)_frame_equirectangular_domain",
        view_id,
    )
    if match is None or not match.group(1):
        raise ValueError("Stanford2D3D view_id has no authoritative room identity")
    return match.group(1).lower()


def canonical_exclusion_group(row: dict[str, Any], *, corpus_root: Path) -> str:
    """Derive the spatial unit from the corpus's authoritative pose index."""

    corpus = row.get("corpus")
    scene_id = row.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError("dataset scene_id must be a non-empty string")
    if scene_id != scene_id.lower():
        raise ValueError("dataset scene_id must use its canonical lowercase identity")
    scene_parts = PurePosixPath(scene_id).parts
    if not scene_parts:
        raise ValueError("dataset scene_id has no canonical group")
    group = scene_parts[0].lower()
    if corpus == "Matterport3D":
        if re.fullmatch(r"[a-z0-9]{11}", group) is None or len(scene_parts) != 1:
            raise ValueError(
                "Matterport3D scene_id must be its 11-character building ID"
            )
    elif corpus == "Stanford2D3D":
        if re.fullmatch(r"area_[1-6](?:[ab])?", group) is None or len(scene_parts) != 2:
            raise ValueError(
                "Stanford2D3D scene_id must be its canonical area/room identity"
            )
    else:
        raise ValueError("unknown corpus for canonical exclusion identity")

    authoritative_groups: set[str] = set()
    authoritative_scenes: set[str] = set()
    asset_groups: set[str] = set()
    resolved_root = corpus_root.resolve()
    for view_name in ("reference", "source"):
        view = row.get(view_name)
        if not isinstance(view, dict):
            raise ValueError(f"{view_name} view is missing")
        view_id = view.get("view_id")
        if not isinstance(view_id, str) or not view_id:
            raise ValueError(f"{view_name} view_id is invalid")
        authoritative_group = _authoritative_pose_group(corpus_root, view_id)
        authoritative_groups.add(authoritative_group)
        if corpus == "Stanford2D3D":
            authoritative_scenes.add(
                f"{authoritative_group}/{_stanford_room_id(view_id)}"
            )
        for asset in ("rgb", "range"):
            relative = view.get(asset)
            if not isinstance(relative, str):
                raise ValueError(f"{view_name}.{asset} path is invalid")
            parts = PurePosixPath(relative).parts
            if len(parts) < 3 or PurePosixPath(relative).is_absolute():
                raise ValueError(f"{view_name}.{asset} must use kind/group/view layout")
            try:
                resolved = (corpus_root / relative).resolve().relative_to(resolved_root)
            except ValueError as exc:
                raise ValueError(
                    f"{view_name}.{asset} escapes the corpus root"
                ) from exc
            if len(resolved.parts) < 3:
                raise ValueError(f"{view_name}.{asset} has an invalid corpus layout")
            asset_groups.add(resolved.parts[1].lower())
            if PurePosixPath(relative).stem != view_id:
                raise ValueError(f"{view_name}.{asset} filename does not match view_id")
    if authoritative_groups != {group} or asset_groups != authoritative_groups:
        raise ValueError("scene identity does not match authoritative dataset assets")
    if corpus == "Stanford2D3D" and authoritative_scenes != {scene_id}:
        raise ValueError("scene_id does not match authoritative Stanford room identity")
    claimed = row.get("exclusion_group_id")
    if not isinstance(claimed, str) or claimed.lower() != group:
        raise ValueError("exclusion_group_id does not match canonical dataset identity")
    return group


__all__ = ["canonical_exclusion_group"]
