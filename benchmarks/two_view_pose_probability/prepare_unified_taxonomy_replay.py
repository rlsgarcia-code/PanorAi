#!/usr/bin/env python3
"""Prepare the representative taxonomy pairs for one unified frontend replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _index(
    rows: Iterable[dict[str, Any]], fields: tuple[str, ...]
) -> dict[tuple[str, ...], dict[str, Any]]:
    result = {}
    for row in rows:
        if not all(field in row for field in fields):
            continue
        key = tuple(str(row[field]) for field in fields)
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = row
    return result


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
        ),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    selected = {
        (row["dataset_id"], row["pair_id"]): row for row in taxonomy["representatives"]
    }
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
    methods = []
    evaluations = []
    for key, representative in selected.items():
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
                "partition": representative["split"],
                "spatial_group_id": reference["spatial_group_id"],
                "from_view_id": source["from_view_id"],
                "to_view_id": source["to_view_id"],
                "from_rgb_path": from_view["rgb"]["path"],
                "to_rgb_path": to_view["rgb"]["path"],
                "from_mask_path": None,
                "to_mask_path": None,
                "input_adapter": "canonical-full-erp/v1",
            }
            evaluation = {
                "schema": "panorai-relative-pose-frontend-evaluation/v1",
                "pair_id": key[1],
                "dataset_id": key[0],
                "partition": representative["split"],
                "spatial_group_id": reference["spatial_group_id"],
                "from_view_id": source["from_view_id"],
                "to_view_id": source["to_view_id"],
                "reference": reference["reference"],
                "covariates": reference["covariates"],
            }
        method["taxonomy_category"] = representative["category"]
        evaluation["taxonomy_category"] = representative["category"]
        for field in ("from_rgb_path", "to_rgb_path"):
            if not Path(method[field]).is_file():
                raise FileNotFoundError(method[field])
        methods.append(method)
        evaluations.append(evaluation)
    methods.sort(key=lambda row: row["taxonomy_category"])
    evaluations.sort(key=lambda row: row["taxonomy_category"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = args.output_dir / "inputs.jsonl"
    evaluation_path = args.output_dir / "evaluation.jsonl"
    _write_jsonl(inputs_path, methods)
    _write_jsonl(evaluation_path, evaluations)
    result = {
        "schema": "panorai-unified-taxonomy-replay-preparation/v1",
        "pairs": len(methods),
        "datasets": sorted({row["dataset_id"] for row in methods}),
        "inputs": {"path": str(inputs_path.resolve()), "sha256": _sha256(inputs_path)},
        "evaluation": {
            "path": str(evaluation_path.resolve()),
            "sha256": _sha256(evaluation_path),
        },
        "sources": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in (
                ("taxonomy", args.taxonomy),
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
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--public-predictions", type=Path, required=True)
    parser.add_argument("--public-evaluation", type=Path, required=True)
    parser.add_argument("--public-views", type=Path, required=True)
    parser.add_argument("--p74-inputs", type=Path, required=True)
    parser.add_argument("--p74-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
