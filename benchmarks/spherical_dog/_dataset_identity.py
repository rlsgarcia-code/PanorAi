"""Canonical dataset identities used by the prospective DoG selector."""

from __future__ import annotations

from pathlib import PurePosixPath
import re
from typing import Any


def canonical_exclusion_group(row: dict[str, Any]) -> str:
    """Derive the spatial exclusion unit from canonical corpus asset paths."""

    corpus = row.get("corpus")
    scene_id = row.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError("dataset scene_id must be a non-empty string")
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
        if re.fullmatch(r"area_[1-6](?:[ab])?", group) is None or len(scene_parts) < 2:
            raise ValueError("Stanford2D3D scene_id must begin with its canonical area")
    else:
        raise ValueError("unknown corpus for canonical exclusion identity")

    asset_groups: set[str] = set()
    for view_name in ("reference", "source"):
        view = row.get(view_name)
        if not isinstance(view, dict):
            raise ValueError(f"{view_name} view is missing")
        view_id = view.get("view_id")
        if not isinstance(view_id, str) or not view_id:
            raise ValueError(f"{view_name} view_id is invalid")
        for asset in ("rgb", "range"):
            relative = view.get(asset)
            if not isinstance(relative, str):
                raise ValueError(f"{view_name}.{asset} path is invalid")
            parts = PurePosixPath(relative).parts
            if len(parts) < 3 or PurePosixPath(relative).is_absolute():
                raise ValueError(f"{view_name}.{asset} must use kind/group/view layout")
            asset_groups.add(parts[1].lower())
            if PurePosixPath(relative).stem != view_id:
                raise ValueError(f"{view_name}.{asset} filename does not match view_id")
    if asset_groups != {group}:
        raise ValueError("scene identity does not match canonical dataset asset paths")
    claimed = row.get("exclusion_group_id")
    if not isinstance(claimed, str) or claimed.lower() != group:
        raise ValueError("exclusion_group_id does not match canonical dataset identity")
    return group


__all__ = ["canonical_exclusion_group"]
