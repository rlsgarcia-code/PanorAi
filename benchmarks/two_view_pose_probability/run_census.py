#!/usr/bin/env python3
"""Audit two-view pair independence and freeze leakage-safe splits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

SCHEMA = "panorai-two-view-pose-census/v1"
PAIR_SCHEMA = "panorai-two-view-pose-pair/v1"
SPLIT_SCHEMA = "panorai-two-view-pose-split/v1"
DEFAULT_SEED = "panorai-val018-e0-v1"
SPLIT_ORDER = ("development", "calibration", "evaluation")


class UnionFind:
    """Small deterministic disjoint-set structure."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, first: str, second: str) -> None:
        root_a = self.find(first)
        root_b = self.find(second)
        if root_a == root_b:
            return
        low, high = sorted((root_a, root_b))
        self.parent[high] = low


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_digest(value: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise TypeError(f"{path}:{line_number} is not a JSON object")
        records.append(item)
    return records


def _normalize_record(item: dict[str, Any], source_path: Path) -> dict[str, str] | None:
    """Return identity-only fields and ignore a prediction manifest header."""

    required = (
        "pair_id",
        "dataset_id",
        "spatial_group_id",
        "from_view_id",
        "to_view_id",
    )
    if not all(key in item for key in required):
        if "manifest" in item and len(item) == 1:
            return None
        missing = [key for key in required if key not in item]
        raise ValueError(f"{source_path} record lacks identity fields: {missing}")
    values = {key: str(item[key]).strip() for key in required}
    if any(not value for value in values.values()):
        raise ValueError(f"{source_path} record has an empty identity field")
    if values["from_view_id"] == values["to_view_id"]:
        raise ValueError(f"self-pair is forbidden: {values['pair_id']}")
    return values


def load_sources(
    paths: Iterable[Path],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    rows: list[dict[str, str]] = []
    sources = []
    for path in paths:
        resolved = path.resolve()
        source_rows = 0
        skipped_headers = 0
        for item in _read_jsonl(resolved):
            normalized = _normalize_record(item, resolved)
            if normalized is None:
                skipped_headers += 1
                continue
            rows.append(normalized)
            source_rows += 1
        sources.append(
            {
                "path": str(resolved),
                "sha256": _sha256(resolved),
                "pair_rows": source_rows,
                "skipped_manifest_rows": skipped_headers,
            }
        )
    if not rows:
        raise ValueError("sources contain no pair rows")
    return rows, sources


def _canonical_pair(row: dict[str, str]) -> tuple[str, str, str]:
    first, second = sorted((row["from_view_id"], row["to_view_id"]))
    return row["dataset_id"], first, second


def validate_pairs(rows: list[dict[str, str]]) -> None:
    self_pairs = [
        row["pair_id"] for row in rows if row["from_view_id"] == row["to_view_id"]
    ]
    if self_pairs:
        raise ValueError(f"self-pair is forbidden: {self_pairs[:5]}")
    pair_ids: dict[tuple[str, str], int] = Counter(
        (row["dataset_id"], row["pair_id"]) for row in rows
    )
    duplicate_ids = [key for key, count in pair_ids.items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"duplicate dataset/pair IDs: {duplicate_ids[:5]}")
    unordered: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in rows:
        unordered[_canonical_pair(row)].append(row["pair_id"])
    duplicates = {key: ids for key, ids in unordered.items() if len(ids) > 1}
    if duplicates:
        first = next(iter(sorted(duplicates.items())))
        raise ValueError(f"duplicate or reversed image pair: {first}")


def _dataset_components(rows: list[dict[str, str]]) -> dict[str, str]:
    """Join images and groups so neither can cross a frozen split."""

    union = UnionFind()
    for row in rows:
        dataset = row["dataset_id"]
        group = f"{dataset}|group|{row['spatial_group_id']}"
        first = f"{dataset}|image|{row['from_view_id']}"
        second = f"{dataset}|image|{row['to_view_id']}"
        union.union(group, first)
        union.union(group, second)
    members: dict[str, list[str]] = defaultdict(list)
    for item in union.parent:
        members[union.find(item)].append(item)
    labels = {
        root: hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()[:16]
        for root, values in members.items()
    }
    result = {}
    for row in rows:
        key = f"{row['dataset_id']}|image|{row['from_view_id']}"
        result[f"{row['dataset_id']}|{row['pair_id']}"] = labels[union.find(key)]
    return result


def _split_counts(count: int) -> dict[str, int]:
    if count < 3:
        return {"development": count, "calibration": 0, "evaluation": 0}
    if count == 3:
        return {name: 1 for name in SPLIT_ORDER}
    evaluation = max(1, int(round(0.15 * count)))
    calibration = max(1, int(round(0.15 * count)))
    development = count - calibration - evaluation
    if development < 1:
        development = 1
        evaluation = max(1, count - development - calibration)
    return {
        "development": development,
        "calibration": calibration,
        "evaluation": evaluation,
    }


def assign_splits(
    rows: list[dict[str, str]], components: dict[str, str], *, seed: str
) -> dict[tuple[str, str], str]:
    by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_dataset[row["dataset_id"]].add(
            components[f"{row['dataset_id']}|{row['pair_id']}"]
        )
    unit_splits: dict[tuple[str, str], str] = {}
    for dataset, units in sorted(by_dataset.items()):
        ordered = sorted(
            units, key=lambda value: _stable_digest(f"{dataset}|{value}", seed)
        )
        counts = _split_counts(len(ordered))
        start = 0
        for split in SPLIT_ORDER:
            stop = start + counts[split]
            for unit in ordered[start:stop]:
                unit_splits[(dataset, unit)] = split
            start = stop
        if start != len(ordered):
            raise RuntimeError("split assignment did not consume every unit")
    return unit_splits


def standardize(
    rows: list[dict[str, str]], *, seed: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_pairs(rows)
    components = _dataset_components(rows)
    assignments = assign_splits(rows, components, seed=seed)
    degree: Counter[tuple[str, str]] = Counter()
    for row in rows:
        degree[(row["dataset_id"], row["from_view_id"])] += 1
        degree[(row["dataset_id"], row["to_view_id"])] += 1
    standardized = []
    for row in rows:
        component = components[f"{row['dataset_id']}|{row['pair_id']}"]
        first, second = sorted((row["from_view_id"], row["to_view_id"]))
        standardized.append(
            {
                "schema": PAIR_SCHEMA,
                **row,
                "unordered_view_ids": [first, second],
                "independence_component_id": component,
                "split": assignments[(row["dataset_id"], component)],
            }
        )
    standardized.sort(
        key=lambda row: (row["dataset_id"], row["spatial_group_id"], row["pair_id"])
    )

    datasets = {}
    for dataset in sorted({row["dataset_id"] for row in rows}):
        selected = [row for row in standardized if row["dataset_id"] == dataset]
        images = {
            view
            for row in selected
            for view in (row["from_view_id"], row["to_view_id"])
        }
        image_degrees = [degree[(dataset, view)] for view in images]
        groups = {row["spatial_group_id"] for row in selected}
        units = {row["independence_component_id"] for row in selected}
        split_pairs = Counter(row["split"] for row in selected)
        split_units = {
            split: len(
                {
                    row["independence_component_id"]
                    for row in selected
                    if row["split"] == split
                }
            )
            for split in SPLIT_ORDER
        }
        datasets[dataset] = {
            "unique_pairs": len(selected),
            "unique_images": len(images),
            "spatial_groups": len(groups),
            "independence_components": len(units),
            "images_reused_across_pairs": sum(value > 1 for value in image_degrees),
            "maximum_pairs_per_image": max(image_degrees),
            "median_pairs_per_image": float(_median(image_degrees)),
            "pairs_by_split": {split: split_pairs[split] for split in SPLIT_ORDER},
            "components_by_split": split_units,
        }
    audit_split_integrity(standardized)
    return standardized, datasets


def _median(values: list[int]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def audit_split_integrity(rows: list[dict[str, Any]]) -> None:
    image_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    component_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        dataset = str(row["dataset_id"])
        split = str(row["split"])
        image_splits[(dataset, str(row["from_view_id"]))].add(split)
        image_splits[(dataset, str(row["to_view_id"]))].add(split)
        group_splits[(dataset, str(row["spatial_group_id"]))].add(split)
        component_splits[(dataset, str(row["independence_component_id"]))].add(split)
    for label, values in (
        ("image", image_splits),
        ("group", group_splits),
        ("component", component_splits),
    ):
        leaking = [key for key, splits in values.items() if len(splits) > 1]
        if leaking:
            raise RuntimeError(f"{label} leakage across splits: {leaking[:5]}")


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
    )


def run_census(paths: list[Path], output_dir: Path, *, seed: str) -> dict[str, Any]:
    rows, sources = load_sources(paths)
    standardized, datasets = standardize(rows, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = output_dir / "standardized-pairs.jsonl"
    splits_path = output_dir / "split-manifest.jsonl"
    _atomic_text(pairs_path, _jsonl(standardized))
    split_rows = [
        {
            "schema": SPLIT_SCHEMA,
            "dataset_id": row["dataset_id"],
            "pair_id": row["pair_id"],
            "independence_component_id": row["independence_component_id"],
            "split": row["split"],
        }
        for row in standardized
    ]
    _atomic_text(splits_path, _jsonl(split_rows))
    payload = {
        "schema": SCHEMA,
        "seed": seed,
        "sources": sources,
        "totals": {
            "datasets": len(datasets),
            "unique_pairs": len(standardized),
            "unique_images": sum(value["unique_images"] for value in datasets.values()),
            "spatial_groups": sum(
                value["spatial_groups"] for value in datasets.values()
            ),
            "independence_components": sum(
                value["independence_components"] for value in datasets.values()
            ),
        },
        "datasets": datasets,
        "outputs": {
            "standardized_pairs": str(pairs_path),
            "standardized_pairs_sha256": _sha256(pairs_path),
            "split_manifest": str(splits_path),
            "split_manifest_sha256": _sha256(splits_path),
        },
        "audit": {
            "duplicate_pair_ids": 0,
            "duplicate_or_reversed_pairs": 0,
            "self_pairs": 0,
            "image_split_leaks": 0,
            "group_split_leaks": 0,
            "component_split_leaks": 0,
        },
    }
    census_path = output_dir / "census.json"
    _atomic_text(
        census_path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        required=True,
        help="JSONL containing pair identity fields; repeat for multiple corpora",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    return parser


def main() -> int:
    args = _parser().parse_args()
    run_census(args.source, args.output_dir.resolve(), seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
