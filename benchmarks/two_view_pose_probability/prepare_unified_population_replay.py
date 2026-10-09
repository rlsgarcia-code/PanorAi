#!/usr/bin/env python3
"""Prepare every frozen VAL-018 pair for one aligned-frontend replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from benchmarks.two_view_pose_probability.prepare_unified_taxonomy_replay import (
        _atomic_text,
        _index,
        _read_jsonl,
        _sha256,
        _write_jsonl,
    )
except ModuleNotFoundError:
    from prepare_unified_taxonomy_replay import (  # type: ignore[no-redef]
        _atomic_text,
        _index,
        _read_jsonl,
        _sha256,
        _write_jsonl,
    )


def _stable_interleave(
    rows: list[dict[str, Any]], *, order_seed: str
) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset_id"], []).append(row)
    for dataset, values in by_dataset.items():
        values.sort(
            key=lambda row: hashlib.sha256(
                f"{order_seed}\0{dataset}\0{row['pair_id']}".encode()
            ).hexdigest()
        )
    ordered = []
    datasets = sorted(by_dataset)
    index = 0
    while True:
        appended = False
        for dataset in datasets:
            values = by_dataset[dataset]
            if index < len(values):
                ordered.append(values[index])
                appended = True
        if not appended:
            break
        index += 1
    return ordered


def _audit_pairs(rows: list[dict[str, Any]]) -> dict[str, int]:
    ids: set[tuple[str, str]] = set()
    unordered: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row["dataset_id"], row["pair_id"])
        if key in ids:
            raise ValueError(f"duplicate pair ID: {key}")
        ids.add(key)
        first, second = sorted((row["from_view_id"], row["to_view_id"]))
        pair = (row["dataset_id"], first, second)
        if pair in unordered:
            raise ValueError(f"duplicate or reversed image pair: {pair}")
        unordered.add(pair)
    return {"duplicate_pair_ids": 0, "duplicate_or_reversed_pairs": 0}


def run(args: argparse.Namespace) -> dict[str, Any]:
    split_rows = _read_jsonl(args.split_manifest)
    split_index = _index(split_rows, ("dataset_id", "pair_id"))
    public_predictions = _index(
        _read_jsonl(args.public_predictions), ("dataset_id", "pair_id")
    )
    public_evaluations = _index(
        _read_jsonl(args.public_evaluation), ("dataset_id", "pair_id")
    )
    public_views = _index(_read_jsonl(args.public_views), ("view_id",))
    p74_inputs = _index(_read_jsonl(args.p74_inputs), ("dataset_id", "pair_id"))
    p74_evaluations = _index(
        _read_jsonl(args.p74_evaluation), ("dataset_id", "pair_id")
    )
    source_keys = set(public_predictions) | set(p74_inputs)
    evaluation_keys = set(public_evaluations) | set(p74_evaluations)
    if source_keys != evaluation_keys:
        raise ValueError("method/evaluation pair keys differ")
    if source_keys != set(split_index):
        missing = sorted(source_keys ^ set(split_index))[:5]
        raise ValueError(f"source/split pair keys differ; examples: {missing}")

    methods = []
    evaluations = []
    for key in source_keys:
        split = split_index[key]
        if key[0] == "p74_native_polar":
            method = dict(p74_inputs[key])
            evaluation = dict(p74_evaluations[key])
        else:
            source = public_predictions[key]
            reference = public_evaluations[key]
            from_view = public_views[(str(source["from_view_id"]),)]
            to_view = public_views[(str(source["to_view_id"]),)]
            method = {
                "schema": "panorai-relative-pose-frontend-method-input/v1",
                "pair_id": key[1],
                "dataset_id": key[0],
                "partition": split["split"],
                "spatial_group_id": split["spatial_group_id"],
                "independence_component_id": split["independence_component_id"],
                "from_view_id": source["from_view_id"],
                "to_view_id": source["to_view_id"],
                "from_rgb_path": from_view["rgb"]["path"],
                "to_rgb_path": to_view["rgb"]["path"],
                "from_mask_path": None,
                "to_mask_path": None,
                "input_adapter": "canonical-full-erp/v1",
            }
            evaluation = dict(reference)
        method["partition"] = split["split"]
        method["independence_component_id"] = split["independence_component_id"]
        evaluation["partition"] = split["split"]
        evaluation["independence_component_id"] = split["independence_component_id"]
        for field in ("from_rgb_path", "to_rgb_path"):
            if not Path(method[field]).is_file():
                raise FileNotFoundError(method[field])
        methods.append(method)
        evaluations.append(evaluation)

    audit = _audit_pairs(methods)
    methods = _stable_interleave(methods, order_seed=args.order_seed)
    evaluation_by_key = {
        (row["dataset_id"], row["pair_id"]): row for row in evaluations
    }
    evaluations = [
        evaluation_by_key[(row["dataset_id"], row["pair_id"])] for row in methods
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = args.output_dir / "inputs.jsonl"
    evaluation_path = args.output_dir / "evaluation.jsonl"
    _write_jsonl(inputs_path, methods)
    _write_jsonl(evaluation_path, evaluations)
    dataset_counts = {
        dataset: sum(row["dataset_id"] == dataset for row in methods)
        for dataset in sorted({row["dataset_id"] for row in methods})
    }
    result = {
        "schema": "panorai-unified-population-replay-preparation/v1",
        "pair_count": len(methods),
        "dataset_counts": dataset_counts,
        "order": {
            "algorithm": "stable-sha256-within-dataset round-robin across datasets",
            "seed": args.order_seed,
        },
        "audit": audit,
        "inputs": {"path": str(inputs_path.resolve()), "sha256": _sha256(inputs_path)},
        "evaluation": {
            "path": str(evaluation_path.resolve()),
            "sha256": _sha256(evaluation_path),
        },
        "sources": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in (
                ("split_manifest", args.split_manifest),
                ("public_predictions", args.public_predictions),
                ("public_evaluation", args.public_evaluation),
                ("public_views", args.public_views),
                ("p74_inputs", args.p74_inputs),
                ("p74_evaluation", args.p74_evaluation),
            )
        },
    }
    manifest = args.output_dir / "manifest.json"
    _atomic_text(
        manifest,
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--public-predictions", type=Path, required=True)
    parser.add_argument("--public-evaluation", type=Path, required=True)
    parser.add_argument("--public-views", type=Path, required=True)
    parser.add_argument("--p74-inputs", type=Path, required=True)
    parser.add_argument("--p74-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--order-seed", default="panorai-val018-population-replay-v1")
    return parser


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
