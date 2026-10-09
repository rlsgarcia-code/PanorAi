#!/usr/bin/env python3
"""Build a deterministic real-pair failure taxonomy and local contact sheet."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "panorai-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = ("matterport360", "stanford2d3d", "p74_native_polar")
LABELS = {
    "matterport360": "Matterport360",
    "stanford2d3d": "Stanford2D3D",
    "p74_native_polar": "P74",
}
SCHEMA = "panorai-two-view-failure-taxonomy/v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _post_probabilities(
    path: Path, model_id: str
) -> dict[tuple[str, str], float]:
    result = {}
    for row in _read_jsonl(path):
        if row["model_id"] == model_id:
            result[(row["dataset_id"], row["pair_id"])] = float(row["probability"])
    return result


def _categories() -> tuple[dict[str, Any], ...]:
    return (
        {
            "id": "no-return-low-overlap",
            "title": "No pose: negligible shared scene",
            "role": "negative eligibility control",
            "preferred_dataset": "p74_native_polar",
            "predicate": lambda row: (
                not row["outcomes"]["returned"]
                and row["capture"]["registered_cloud_overlap_min"] < 0.10
            ),
            "rank": lambda row: row["capture"]["registered_cloud_overlap_min"],
        },
        {
            "id": "returned-but-rejected",
            "title": "Eligible-overlap pose returned, then rejected",
            "role": "eligible estimator outcome",
            "preferred_dataset": "matterport360",
            "predicate": lambda row: (
                row["outcomes"]["returned"] and not row["outcomes"]["accepted"]
                and row["capture"]["registered_cloud_overlap_min"] >= 0.50
            ),
            "rank": lambda row: (
                -max(
                    row["outcomes"]["rotation_error_deg"] or 0.0,
                    row["outcomes"]["translation_direction_error_deg"] or 0.0,
                )
            ),
        },
        {
            "id": "catastrophic-accepted",
            "title": "Catastrophic translation accepted with high confidence",
            "role": "eligible estimator outcome",
            "preferred_dataset": "stanford2d3d",
            "predicate": lambda row: (
                row["outcomes"]["catastrophic_accepted"]
                and row["capture"]["registered_cloud_overlap_min"] >= 0.50
            ),
            "rank": lambda row: -(row["post_probability"] or 0.0),
        },
        {
            "id": "overconfident-near-miss",
            "title": "Eligible-overlap high confidence outside precise bound",
            "role": "eligible estimator outcome",
            "preferred_dataset": "p74_native_polar",
            "predicate": lambda row: (
                row["outcomes"]["returned"]
                and not row["outcomes"]["precise"]
                and (row["post_probability"] or 0.0) >= 0.80
                and row["capture"]["registered_cloud_overlap_min"] >= 0.50
            ),
            "rank": lambda row: -(row["post_probability"] or 0.0),
        },
        {
            "id": "supported-success",
            "title": "Supported high-overlap precise pose",
            "role": "eligible estimator outcome",
            "preferred_dataset": "stanford2d3d",
            "predicate": lambda row: (
                row["outcomes"]["usable"]
                and row["capture"]["registered_cloud_overlap_min"] >= 0.50
            ),
            "rank": lambda row: -row["capture"]["registered_cloud_overlap_min"],
        },
    )


def _select_representative(
    rows: list[dict[str, Any]], category: dict[str, Any]
) -> dict[str, Any]:
    predicate: Callable[[dict[str, Any]], bool] = category["predicate"]
    candidates = [row for row in rows if predicate(row)]
    if not candidates:
        raise RuntimeError(f"category has no candidates: {category['id']}")
    preferred = [
        row
        for row in candidates
        if row["dataset_id"] == category["preferred_dataset"]
        and row["split"] == "evaluation"
    ]
    pool = preferred or [row for row in candidates if row["split"] == "evaluation"]
    pool = pool or candidates
    rank: Callable[[dict[str, Any]], float] = category["rank"]
    return min(pool, key=lambda row: (rank(row), row["pair_id"]))


def _pair_views(
    public_predictions: Path, p74_pairs: Path
) -> dict[tuple[str, str], tuple[str, str]]:
    result = {}
    for path in (public_predictions, p74_pairs):
        for row in _read_jsonl(path):
            if "pair_id" not in row:
                continue
            result[(row["dataset_id"], row["pair_id"])] = (
                str(row["from_view_id"]),
                str(row["to_view_id"]),
            )
    return result


def _view_paths(public_views: Path, p74_pairs: Path) -> dict[str, Path]:
    result = {}
    for row in _read_jsonl(public_views):
        result[str(row["view_id"])] = Path(row["rgb"]["path"])
    for row in _read_jsonl(p74_pairs):
        if "pair_id" not in row:
            continue
        result[str(row["from_view_id"])] = Path(row["from_rgb_path"])
        result[str(row["to_view_id"])] = Path(row["to_rgb_path"])
    return result


def _summary(row: dict[str, Any]) -> str:
    outcomes = row["outcomes"]
    rotation = outcomes["rotation_error_deg"]
    translation = outcomes["translation_direction_error_deg"]
    errors = (
        "pose not returned"
        if rotation is None or translation is None
        else f"R={rotation:.2f}°, t={translation:.2f}°"
    )
    probability = row["post_probability"]
    post = "p_post=n/a" if probability is None else f"p_post={probability:.3f}"
    return (
        f"overlap={100 * row['capture']['registered_cloud_overlap_min']:.1f}%  "
        f"matches={row['post']['match_count']}  {post}  {errors}"
    )


def _render(
    representatives: list[dict[str, Any]],
    view_paths: dict[str, Path],
    output: Path,
) -> None:
    figure = plt.figure(figsize=(13.5, 12.8))
    grid = figure.add_gridspec(
        2 * len(representatives),
        2,
        height_ratios=[0.22, 1.0] * len(representatives),
        hspace=0.10,
        wspace=0.02,
    )
    for row_index, representative in enumerate(representatives):
        title_axis = figure.add_subplot(grid[2 * row_index, :])
        title_axis.set_axis_off()
        title_axis.text(
            0.5,
            0.5,
            (
                f"{chr(65 + row_index)}. {representative['category_title']} — "
                f"{LABELS[representative['dataset_id']]}\n"
                f"{representative['summary']}"
            ),
            ha="center",
            va="center",
            fontsize=10,
            weight="bold" if row_index in (0, 2) else "normal",
        )
        for column, view_id in enumerate(representative["view_ids"]):
            path = view_paths[view_id]
            if not path.is_file():
                raise FileNotFoundError(path)
            image_axis = figure.add_subplot(grid[2 * row_index + 1, column])
            image_axis.imshow(plt.imread(path))
            image_axis.set_axis_off()
            image_axis.set_title(
                "FROM" if column == 0 else "TO", fontsize=8, loc="left"
            )
    figure.suptitle(
        "Representative real two-view outcomes",
        fontsize=14,
        weight="bold",
        y=0.998,
    )
    figure.subplots_adjust(top=0.965, bottom=0.015)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_jsonl(args.analysis_table)
    post = _post_probabilities(args.probability_predictions, args.post_model)
    if not post:
        raise ValueError(
            f"no probability rows found for requested post model: {args.post_model}"
        )
    for row in rows:
        row["post_probability"] = post.get((row["dataset_id"], row["pair_id"]))
    views_by_pair = _pair_views(args.public_predictions, args.p74_pairs)
    view_paths = _view_paths(args.public_views, args.p74_pairs)
    representatives = []
    counts = []
    for category in _categories():
        predicate: Callable[[dict[str, Any]], bool] = category["predicate"]
        candidates = [row for row in rows if predicate(row)]
        by_dataset = Counter(row["dataset_id"] for row in candidates)
        counts.append(
            {
                "category": category["id"],
                "total": len(candidates),
                **{dataset: by_dataset[dataset] for dataset in DATASETS},
            }
        )
        if not candidates:
            continue
        selected = _select_representative(rows, category)
        key = (selected["dataset_id"], selected["pair_id"])
        if key not in views_by_pair:
            raise KeyError(f"missing view IDs for {key}")
        representatives.append(
            {
                "category": category["id"],
                "category_title": category["title"],
                "category_role": category["role"],
                "dataset_id": selected["dataset_id"],
                "pair_id": selected["pair_id"],
                "split": selected["split"],
                "view_ids": list(views_by_pair[key]),
                "summary": _summary(selected),
                "capture": selected["capture"],
                "post": selected["post"],
                "post_probability": selected["post_probability"],
                "outcomes": selected["outcomes"],
            }
        )
    if not representatives:
        raise RuntimeError("no failure-taxonomy category has a representative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / "representative-real-pairs.png"
    _render(representatives, view_paths, figure_path)
    counts_path = args.output_dir / "failure-taxonomy-counts.csv"
    with counts_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(counts[0]))
        writer.writeheader()
        writer.writerows(counts)
    result = {
        "schema": SCHEMA,
        "status": "retrospective descriptive taxonomy",
        "post_probability_model": args.post_model,
        "category_counts": counts,
        "representatives": representatives,
        "figure": {
            "path": str(figure_path.resolve()),
            "sha256": _sha256(figure_path),
            "distribution": "local evidence only; source-dataset image licenses not audited for redistribution",
        },
        "sources": {
            "analysis_table": {
                "path": str(args.analysis_table.resolve()),
                "sha256": _sha256(args.analysis_table),
            },
            "probability_predictions": {
                "path": str(args.probability_predictions.resolve()),
                "sha256": _sha256(args.probability_predictions),
            },
            "public_predictions": {
                "path": str(args.public_predictions.resolve()),
                "sha256": _sha256(args.public_predictions),
            },
            "public_views": {
                "path": str(args.public_views.resolve()),
                "sha256": _sha256(args.public_views),
            },
            "p74_pairs": {
                "path": str(args.p74_pairs.resolve()),
                "sha256": _sha256(args.p74_pairs),
            },
        },
    }
    result_path = args.output_dir / "failure-taxonomy.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-table", type=Path, required=True)
    parser.add_argument("--probability-predictions", type=Path, required=True)
    parser.add_argument(
        "--post-model",
        default="post-precise-raw-score",
        help="Probability model used only to rank confidence-based categories.",
    )
    parser.add_argument("--public-predictions", type=Path, required=True)
    parser.add_argument("--public-views", type=Path, required=True)
    parser.add_argument("--p74-pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
