#!/usr/bin/env python3
"""Freeze an outcome-blind overlap-stratified timing sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


SCHEMA = "panorai-two-view-controlled-timing-selection/v1"
MANIFEST_SCHEMA = "panorai-two-view-controlled-timing-manifest/v1"
SEED = "panorai-val018-controlled-timing-v1"
DATASETS = ("matterport360", "stanford2d3d", "p74_native_polar")
OVERLAP_BINS = (
    ("lt-10", 0.0, 0.10),
    ("10-25", 0.10, 0.25),
    ("25-50", 0.25, 0.50),
    ("50-70", 0.50, 0.70),
    ("ge-70", 0.70, 1.0000001),
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _bin(overlap: float) -> tuple[str, float, float]:
    for label, lower, upper in OVERLAP_BINS:
        if lower <= overlap < upper:
            return label, lower, min(upper, 1.0)
    raise ValueError(f"overlap outside [0, 1]: {overlap}")


def _order_key(dataset: str, pair_id: str) -> str:
    return hashlib.sha256(f"{SEED}|{dataset}|{pair_id}".encode()).hexdigest()


def _group_order_key(dataset: str, component_id: str) -> str:
    return hashlib.sha256(
        f"{SEED}|group|{dataset}|{component_id}".encode()
    ).hexdigest()


def _group_round_robin(cell: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in cell:
        by_group.setdefault(row["independence_component_id"], []).append(row)
    for group_rows in by_group.values():
        group_rows.sort(key=lambda row: (row["selection_key"], row["pair_id"]))
    ordered_groups = sorted(
        by_group,
        key=lambda group: (_group_order_key(cell[0]["dataset_id"], group), group),
    )
    return [
        by_group[group][round_index]
        for round_index in range(max(map(len, by_group.values()), default=0))
        for group in ordered_groups
        if round_index < len(by_group[group])
    ]


def select(rows: list[dict[str, Any]], *, pairs_per_cell: int) -> list[dict[str, Any]]:
    if pairs_per_cell < 1:
        raise ValueError("pairs_per_cell must be positive")
    seen = set()
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        dataset = str(row["dataset_id"])
        if dataset not in DATASETS:
            continue
        pair_id = str(row["pair_id"])
        key = (dataset, pair_id)
        if key in seen:
            raise ValueError(f"duplicate pair identity: {key}")
        seen.add(key)
        overlap = float(row["capture"]["registered_cloud_overlap_min"])
        label, lower, upper = _bin(overlap)
        candidates.setdefault((dataset, label), []).append(
            {
                "schema": SCHEMA,
                "dataset_id": dataset,
                "pair_id": pair_id,
                "independence_component_id": str(row["independence_component_id"]),
                "overlap_bin": label,
                "overlap_lower": lower,
                "overlap_upper": upper,
                "registered_cloud_overlap_min": overlap,
                "selection_key": _order_key(dataset, pair_id),
                "selection_uses_algorithm_outcome": False,
            }
        )
    selected = []
    for dataset in DATASETS:
        for label, _, _ in OVERLAP_BINS:
            cell = _group_round_robin(candidates.get((dataset, label), []))
            selected.extend(cell[:pairs_per_cell])
    return selected


def run(args: argparse.Namespace) -> dict[str, Any]:
    features_path = args.features.resolve()
    rows = _read_jsonl(features_path)
    selected = select(rows, pairs_per_cell=args.pairs_per_cell)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "timing-selection.jsonl"
    _atomic_text(
        selection_path,
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in selected
        ),
    )
    cells = []
    for dataset in DATASETS:
        for label, lower, upper in OVERLAP_BINS:
            cell = [
                row
                for row in selected
                if row["dataset_id"] == dataset and row["overlap_bin"] == label
            ]
            cells.append(
                {
                    "dataset_id": dataset,
                    "overlap_bin": label,
                    "overlap_lower": lower,
                    "overlap_upper": min(upper, 1.0),
                    "selected_pairs": len(cell),
                    "independence_components": len(
                        {row["independence_component_id"] for row in cell}
                    ),
                }
            )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "frozen outcome-blind controlled-timing sample",
        "seed": SEED,
        "pairs_per_cell_maximum": args.pairs_per_cell,
        "selection_fields": [
            "dataset_id",
            "pair_id",
            "independence_component_id",
            "registered_cloud_overlap_min",
        ],
        "selection_strategy": (
            "dataset-by-overlap cells; deterministic group round-robin; "
            "deterministic pair fill"
        ),
        "forbidden_selection_fields": [
            "returned",
            "accepted",
            "matches",
            "pose_error",
            "runtime",
            "memory",
        ],
        "source_features": {
            "path": str(features_path),
            "sha256": _sha256(features_path),
            "rows": len(rows),
        },
        "selection": {
            "path": str(selection_path),
            "sha256": _sha256(selection_path),
            "rows": len(selected),
        },
        "cells": cells,
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--pairs-per-cell", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
