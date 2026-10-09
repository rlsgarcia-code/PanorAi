#!/usr/bin/env python3
"""Render deterministic paper figures and overlap-response tables for VAL-018."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "panorai-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUMMARY_SCHEMA = "panorai-two-view-pose-paper-results/v1"
DATASET_ORDER = ("matterport360", "stanford2d3d", "p74_native_polar")
DATASET_LABELS = {
    "matterport360": "Matterport360",
    "stanford2d3d": "Stanford2D3D",
    "p74_native_polar": "P74",
}
OVERLAP_BINS = (
    ("<10%", 0.0, 0.10),
    ("10–25%", 0.10, 0.25),
    ("25–50%", 0.25, 0.50),
    ("50–70%", 0.50, 0.70),
    ("≥70%", 0.70, 1.0000001),
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def wilson_interval(
    successes: int, count: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if count == 0:
        return 0.0, 1.0
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def summarize_overlap(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for dataset in DATASET_ORDER:
        dataset_rows = [row for row in rows if row["dataset_id"] == dataset]
        for label, lower, upper in OVERLAP_BINS:
            selected = [
                row
                for row in dataset_rows
                if lower <= row["capture"]["registered_cloud_overlap_min"] < upper
            ]
            record: dict[str, Any] = {
                "dataset_id": dataset,
                "overlap_bin": label,
                "overlap_lower": lower,
                "overlap_upper": min(upper, 1.0),
                "pairs": len(selected),
                "independence_components": len(
                    {row["independence_component_id"] for row in selected}
                ),
            }
            for outcome in ("returned", "accepted", "precise", "usable"):
                successes = sum(row["outcomes"][outcome] for row in selected)
                interval = wilson_interval(successes, len(selected))
                record[f"{outcome}_count"] = successes
                record[f"{outcome}_rate"] = (
                    successes / len(selected) if selected else None
                )
                record[f"{outcome}_wilson95_low"] = interval[0]
                record[f"{outcome}_wilson95_high"] = interval[1]
            record["catastrophic_accepted"] = sum(
                row["outcomes"]["catastrophic_accepted"] for row in selected
            )
            result.append(record)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _overlap_figure(summary: list[dict[str, Any]], path: Path) -> None:
    colors = {"returned": "#2F6B8A", "accepted": "#2B8C7F", "usable": "#D28B27"}
    markers = {"returned": "o", "accepted": "s", "usable": "^"}
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.9), sharey=True)
    x = np.arange(len(OVERLAP_BINS))
    for axis, dataset in zip(axes, DATASET_ORDER, strict=True):
        rows = [row for row in summary if row["dataset_id"] == dataset]
        for outcome in ("returned", "accepted", "usable"):
            values = np.asarray([row[f"{outcome}_rate"] for row in rows], dtype=float)
            low = np.asarray([row[f"{outcome}_wilson95_low"] for row in rows])
            high = np.asarray([row[f"{outcome}_wilson95_high"] for row in rows])
            axis.errorbar(
                x,
                values,
                yerr=np.vstack(
                    (np.maximum(values - low, 0.0), np.maximum(high - values, 0.0))
                ),
                color=colors[outcome],
                marker=markers[outcome],
                linewidth=1.8,
                markersize=5,
                capsize=2.5,
                label=outcome.capitalize(),
            )
        axis.set_title(DATASET_LABELS[dataset], fontsize=11, weight="bold")
        axis.set_xticks(
            x, [row["overlap_bin"] for row in rows], rotation=28, ha="right"
        )
        axis.set_xlabel("minimum bidirectional cloud overlap")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(axis="y", alpha=0.22)
        for index, row in enumerate(rows):
            axis.text(
                index,
                0.015,
                f"n={row['pairs']}\ng={row['independence_components']}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#4A5560",
            )
    axes[0].set_ylabel("observed pair rate")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _calibration_figure(evaluation: dict[str, Any], path: Path) -> None:
    rows = (
        ("capture-accept-overlap-baseline", "Capture: acceptance"),
        ("post-precise-common", "Post: precise pose"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(11.5, 7.0), sharex=True, sharey=True)
    for row_index, (model_id, row_label) in enumerate(rows):
        datasets = evaluation["models"][model_id]["datasets"]
        for column, dataset in enumerate(DATASET_ORDER):
            axis = axes[row_index, column]
            bins = datasets[dataset]["reliability_bins"]
            points = [item for item in bins if item["count"]]
            predicted = np.asarray([item["predicted_mean"] for item in points])
            observed = np.asarray([item["observed_rate"] for item in points])
            sizes = np.asarray([item["count"] for item in points], dtype=float)
            sizes = 25.0 + 120.0 * sizes / max(float(np.max(sizes)), 1.0)
            axis.plot((0.0, 1.0), (0.0, 1.0), "--", color="#8D99A6", linewidth=1)
            axis.scatter(
                predicted,
                observed,
                s=sizes,
                color="#2F7F78" if row_index == 0 else "#3B6FB6",
                edgecolor="white",
                linewidth=0.7,
                alpha=0.9,
            )
            axis.set_xlim(-0.03, 1.03)
            axis.set_ylim(-0.03, 1.03)
            axis.grid(alpha=0.18)
            if row_index == 0:
                axis.set_title(DATASET_LABELS[dataset], fontsize=11, weight="bold")
            if column == 0:
                axis.set_ylabel(f"{row_label}\nobserved")
            if row_index == 1:
                axis.set_xlabel("predicted probability")
            axis.text(
                0.04,
                0.94,
                f"Brier={datasets[dataset]['brier']:.3f}\nECE={datasets[dataset]['ece_10']:.3f}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _ablation_figure(evaluation: dict[str, Any], path: Path) -> None:
    models = (
        ("post-precise-raw-score", "Raw score"),
        ("post-precise-support", "+ support"),
        ("post-precise-common", "+ overlap/geometry"),
        ("post-precise-public-full", "Public full"),
    )
    figure, axis = plt.subplots(figsize=(8.5, 4.2))
    width = 0.22
    x = np.arange(len(models))
    colors = ("#3B6FB6", "#2F8C82", "#D28B27")
    for offset, (dataset, color) in enumerate(zip(DATASET_ORDER, colors, strict=True)):
        values = []
        for model_id, _ in models:
            item = (
                evaluation["models"].get(model_id, {}).get("datasets", {}).get(dataset)
            )
            values.append(np.nan if item is None else item["brier"])
        axis.bar(
            x + (offset - 1) * width,
            values,
            width,
            label=DATASET_LABELS[dataset],
            color=color,
        )
    axis.set_xticks(x, [label for _, label in models], rotation=15, ha="right")
    axis.set_ylabel("Brier score (lower is better)")
    axis.set_ylim(0.0, 0.38)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _transfer_figure(
    evaluation: dict[str, Any], lodo_evaluation: dict[str, Any], path: Path
) -> None:
    comparisons = (
        ("capture-accept-overlap-baseline", "Capture acceptance"),
        ("post-precise-common", "Post-process precision"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.1), sharey=True)
    x = np.arange(len(DATASET_ORDER))
    width = 0.34
    for axis, (model_id, label) in zip(axes, comparisons, strict=True):
        within = [
            evaluation["models"][model_id]["datasets"][dataset]["brier"]
            for dataset in DATASET_ORDER
        ]
        transfer = [
            lodo_evaluation["models"][f"{model_id}--lodo-{dataset}"]["datasets"][
                dataset
            ]["brier"]
            for dataset in DATASET_ORDER
        ]
        axis.bar(
            x - width / 2,
            within,
            width,
            color="#3975A8",
            label="Component-held-out",
        )
        axis.bar(
            x + width / 2,
            transfer,
            width,
            color="#D08A2E",
            label="Leave-one-dataset-out",
        )
        axis.set_xticks(x, [DATASET_LABELS[item] for item in DATASET_ORDER])
        axis.tick_params(axis="x", rotation=18)
        axis.set_title(label, fontsize=11, weight="bold")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Brier score (lower is better)")
    axes[0].set_ylim(0.0, 0.60)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_jsonl(args.analysis_table)
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    summary = summarize_overlap(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "overlap-response.csv"
    _write_csv(csv_path, summary)
    figures = {
        "overlap_response": args.output_dir / "overlap-response.png",
        "calibration": args.output_dir / "calibration-heldout.png",
        "post_ablation": args.output_dir / "post-model-ablation.png",
    }
    _overlap_figure(summary, figures["overlap_response"])
    _calibration_figure(evaluation, figures["calibration"])
    _ablation_figure(evaluation, figures["post_ablation"])
    if args.lodo_evaluation is not None:
        lodo_evaluation = json.loads(args.lodo_evaluation.read_text(encoding="utf-8"))
        figures["cross_dataset_transfer"] = (
            args.output_dir / "cross-dataset-transfer.png"
        )
        _transfer_figure(evaluation, lodo_evaluation, figures["cross_dataset_transfer"])
    payload = {
        "schema": SUMMARY_SCHEMA,
        "status": "post-hoc descriptive response plus frozen held-out model evaluation",
        "overlap_response": summary,
        "figures": {name: str(path.resolve()) for name, path in figures.items()},
        "source_analysis_table": str(args.analysis_table.resolve()),
        "source_evaluation": str(args.evaluation.resolve()),
        "source_lodo_evaluation": (
            str(args.lodo_evaluation.resolve())
            if args.lodo_evaluation is not None
            else None
        ),
    }
    output_path = args.output_dir / "paper-results.json"
    output_path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["figures"], indent=2, sort_keys=True))
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-table", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--lodo-evaluation", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
