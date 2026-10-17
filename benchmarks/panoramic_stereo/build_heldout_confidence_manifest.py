#!/usr/bin/env python3
"""Build the outcome-blind VAL-116 held-out stereo pair manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


DATASETS = ("stanford2d3d", "matterport360")
STANFORD_GROUPS = ("area_2", "area_3", "area_4")
MINIMUM_OVERLAP = 0.35
MINIMUM_BASELINE_M = 0.2
MAXIMUM_BASELINE_M = 5.0
PAIR_COUNT_PER_DATASET = 6
STANFORD_PAIRS_PER_GROUP = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            yield value


def _walk_identifiers(value: Any) -> tuple[set[str], set[str]]:
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


def _view_parts(view_id: str) -> tuple[str, str, str]:
    parts = view_id.split("::", 2)
    if len(parts) != 3 or parts[0] not in DATASETS:
        raise ValueError(f"unexpected public view identifier: {view_id}")
    return parts[0], parts[1], parts[2]


def _assets_exist(root: Path, group: str, name: str) -> bool:
    return all(
        path.is_file()
        for path in (
            root / "rgb" / group / f"{name}.png",
            root / "depth" / group / f"{name}.npy",
            root / "poses" / group / f"{name}.txt",
        )
    )


def _select_stanford_pairs(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_views: set[str] = set()
    for group in STANFORD_GROUPS:
        group_count = 0
        for candidate in candidates:
            candidate_views = {str(candidate["reference"]), str(candidate["source"])}
            if candidate["group"] != group or candidate_views & used_views:
                continue
            selected.append(candidate)
            used_views.update(candidate_views)
            group_count += 1
            if group_count == STANFORD_PAIRS_PER_GROUP:
                break
        if group_count != STANFORD_PAIRS_PER_GROUP:
            raise RuntimeError(
                f"could not select {STANFORD_PAIRS_PER_GROUP} held-out "
                f"Stanford pairs in {group}; selected={group_count}"
            )
    return selected


def _select_matterport_pairs(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_groups: set[str] = set()
    used_views: set[str] = set()
    for candidate in candidates:
        candidate_views = {str(candidate["reference"]), str(candidate["source"])}
        if candidate["group"] in used_groups or candidate_views & used_views:
            continue
        selected.append(candidate)
        used_groups.add(str(candidate["group"]))
        used_views.update(candidate_views)
        if len(selected) == PAIR_COUNT_PER_DATASET:
            break
    if len(selected) != PAIR_COUNT_PER_DATASET:
        raise RuntimeError(
            f"could not select {PAIR_COUNT_PER_DATASET} held-out Matterport "
            f"pairs; selected={len(selected)}"
        )
    return selected


def build_manifest(
    *,
    overlap_path: Path,
    analysis_path: Path,
    exclude_manifests: tuple[Path, ...],
    roots: dict[str, Path],
) -> dict[str, Any]:
    """Select pairs using registered geometry and asset presence only."""

    excluded_pairs: set[str] = set()
    excluded_views: set[str] = set()
    for path in exclude_manifests:
        pairs, views = _walk_identifiers(json.loads(path.read_text()))
        excluded_pairs.update(pairs)
        excluded_views.update(views)

    baselines = {
        str(row["pair_id"]): float(row["capture"]["baseline_m"])
        for row in _jsonl(analysis_path)
    }

    candidates: dict[str, list[dict[str, Any]]] = {
        dataset_id: [] for dataset_id in DATASETS
    }
    for row in _jsonl(overlap_path):
        dataset_id = str(row["dataset_id"])
        if dataset_id not in DATASETS:
            continue
        pair_id = str(row["pair_id"])
        overlap = float(row["minimum_directional_cloud_overlap"])
        baseline = baselines.get(pair_id)
        if (
            pair_id in excluded_pairs
            or baseline is None
            or not MINIMUM_BASELINE_M <= baseline <= MAXIMUM_BASELINE_M
            or overlap < MINIMUM_OVERLAP
        ):
            continue

        reference_id = str(row["from_view_id"])
        source_id = str(row["to_view_id"])
        ref_dataset, ref_group, reference = _view_parts(reference_id)
        src_dataset, src_group, source = _view_parts(source_id)
        if ref_dataset != dataset_id or src_dataset != dataset_id:
            raise ValueError(f"dataset mismatch for {pair_id}")
        if ref_group != src_group:
            continue
        if reference in excluded_views or source in excluded_views:
            continue
        if not _assets_exist(roots[dataset_id], ref_group, reference):
            continue
        if not _assets_exist(roots[dataset_id], src_group, source):
            continue

        candidates[dataset_id].append(
            {
                "dataset_id": dataset_id,
                "pair_id": pair_id,
                "group": ref_group,
                "spatial_group_id": str(row["spatial_group_id"]),
                "reference": reference,
                "source": source,
                "baseline_m": baseline,
                "minimum_directional_cloud_overlap": overlap,
            }
        )

    for dataset_candidates in candidates.values():
        dataset_candidates.sort(
            key=lambda item: (
                -float(item["minimum_directional_cloud_overlap"]),
                str(item["pair_id"]),
            )
        )

    selected = _select_stanford_pairs(candidates["stanford2d3d"])
    selected.extend(_select_matterport_pairs(candidates["matterport360"]))

    return {
        "schema": "panorai-public-confidence-validity-selection/v1",
        "protocol_id": "VAL-116-heldout-confidence-validity-separation",
        "frozen_before_dense_evaluation": True,
        "source_audits": {
            "registered_cloud_overlap_sha256": _sha256(overlap_path),
            "pair_analysis_sha256": _sha256(analysis_path),
        },
        "excluded_manifests": [
            {"path": path.name, "sha256": _sha256(path)} for path in exclude_manifests
        ],
        "selection_policy": {
            "dense_output_used_for_selection": False,
            "rgb_match_used_for_selection": False,
            "range_reference_used_for_selection": False,
            "pair_count_per_dataset": PAIR_COUNT_PER_DATASET,
            "minimum_directional_cloud_overlap": MINIMUM_OVERLAP,
            "minimum_baseline_m": MINIMUM_BASELINE_M,
            "maximum_baseline_m": MAXIMUM_BASELINE_M,
            "ranking": "descending eligible overlap then pair id",
            "stanford_balance": "two pairs from each of area_2, area_3 and area_4",
            "matterport_balance": "six distinct spatial groups",
            "diversity": "exclude all prior manifest pairs/views and greedily prevent new view reuse",
            "orientation": "registered overlap from_view_id is reference and to_view_id is source",
            "asset_policy": "complete RGB, radial-range and pose files required",
        },
        "pairs": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlap", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--stanford-root", type=Path, required=True)
    parser.add_argument("--matterport-root", type=Path, required=True)
    args = parser.parse_args()
    payload = build_manifest(
        overlap_path=args.overlap,
        analysis_path=args.analysis,
        exclude_manifests=tuple(args.exclude_manifest),
        roots={
            "stanford2d3d": args.stanford_root,
            "matterport360": args.matterport_root,
        },
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
