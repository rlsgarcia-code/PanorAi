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
ALIGNED_POST_MODEL = "post-precise-aligned-orientation"
HISTORICAL_POST_MODEL = "post-precise-common"
CAPTURE_ACCEPT_MODEL = "capture-accept-overlap-baseline"
CAPTURE_PRECISE_MODEL = "capture-precise-given-accept-overlap-baseline"
MINIMUM_SURFACE_COMPONENTS = 5


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


def primary_post_model(evaluation: dict[str, Any]) -> str:
    """Prefer the frozen aligned model while preserving historical rendering."""
    models = evaluation.get("models", {})
    if ALIGNED_POST_MODEL in models:
        return ALIGNED_POST_MODEL
    if HISTORICAL_POST_MODEL in models:
        return HISTORICAL_POST_MODEL
    raise ValueError("evaluation contains no supported primary post model")


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


def summarize_runtime_overlap(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for dataset in DATASET_ORDER:
        dataset_rows = [row for row in rows if row["dataset_id"] == dataset]
        for label, lower, upper in OVERLAP_BINS:
            selected = [
                row
                for row in dataset_rows
                if lower <= row["capture"]["registered_cloud_overlap_min"] < upper
            ]
            times = _post_values(selected, "pair_total_seconds")
            result.append(
                {
                    "dataset_id": dataset,
                    "overlap_bin": label,
                    "overlap_lower": lower,
                    "overlap_upper": min(upper, 1.0),
                    "pairs": len(selected),
                    "independence_components": len(
                        {row["independence_component_id"] for row in selected}
                    ),
                    "pair_total_median_seconds": (
                        float(np.median(times)) if len(times) else None
                    ),
                    "pair_total_p95_seconds": (
                        float(np.quantile(times, 0.95)) if len(times) else None
                    ),
                }
            )
    return result


def _post_values(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row["post"].get(field)
        if value is None and field == "pair_total_seconds":
            value = row["post"].get("elapsed_seconds")
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def _distribution(values: np.ndarray, *, include_maximum: bool = False) -> dict:
    result: dict[str, Any] = {"available_pairs": int(len(values))}
    if not len(values):
        result.update({"median": None, "p95": None})
        if include_maximum:
            result["maximum"] = None
        return result
    result.update(
        {
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
        }
    )
    if include_maximum:
        result["maximum"] = float(np.max(values))
    return result


def summarize_engineering(rows: list[dict[str, Any]]) -> dict[str, Any]:
    timing_fields = (
        "detection_pair_seconds",
        "patches_pair_seconds",
        "descriptor_pair_seconds",
        "matching_seconds",
        "pose_seconds",
        "pair_total_seconds",
    )
    datasets = {}
    for dataset in DATASET_ORDER:
        selected = [row for row in rows if row["dataset_id"] == dataset]
        timing = {}
        for field in timing_fields:
            timing[field] = _distribution(_post_values(selected, field))
        memory = _post_values(selected, "peak_rss_mib")
        keypoints = _post_values(selected, "keypoint_count_min")
        datasets[dataset] = {
            "pairs": len(selected),
            "timing_seconds": timing,
            "peak_rss_mib": _distribution(memory, include_maximum=True),
            "minimum_pair_keypoints": {
                "available_pairs": int(len(keypoints)),
                "median": float(np.median(keypoints)) if len(keypoints) else None,
                "p05": float(np.quantile(keypoints, 0.05)) if len(keypoints) else None,
            },
        }
    return {
        "resolution": "1024x2048",
        "unit": "two-panorama pair",
        "benchmark_valid": False,
        "measurement_status": (
            "shared-host replay wall time under observed contention; diagnostic "
            "only, not a controlled performance benchmark"
        ),
        "datasets": datasets,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _capture_probability(
    card: dict[str, Any], *, overlap: float, baseline_m: float
) -> float:
    operational = {
        "capture.registered_cloud_overlap_min": overlap,
        "capture.baseline_m": baseline_m,
    }
    values = []
    for feature in card["features"]:
        path = feature["path"]
        if path not in operational:
            raise ValueError(f"capture surface cannot supply feature: {path}")
        number = float(operational[path])
        transform = feature["transform"]
        if transform == "identity":
            transformed = number
        elif transform == "log1p":
            transformed = math.log1p(number)
        elif transform == "logit":
            clipped = min(max(number, 1e-4), 1.0 - 1e-4)
            transformed = math.log(clipped / (1.0 - clipped))
        else:
            raise ValueError(f"unknown capture-surface transform: {transform}")
        values.append(transformed)
    standardized = (
        np.asarray(values, dtype=np.float64)
        - np.asarray(card["scaler_mean"], dtype=np.float64)
    ) / np.asarray(card["scaler_scale"], dtype=np.float64)
    parameters = np.asarray(card["parameters"], dtype=np.float64)
    raw = _sigmoid(float(parameters[0] + standardized @ parameters[1:]))
    raw_logit = math.log(
        min(max(raw, 1e-6), 1.0 - 1e-6)
        / (1.0 - min(max(raw, 1e-6), 1.0 - 1e-6))
    )
    platt = np.asarray(card["platt_parameters"], dtype=np.float64)
    return _sigmoid(float(platt[0] + platt[1] * raw_logit))


def capture_probability_surface(
    rows: list[dict[str, Any]], model_card: dict[str, Any]
) -> list[dict[str, Any]]:
    cards = {card["model_id"]: card for card in model_card["models"]}
    accept_card = cards[CAPTURE_ACCEPT_MODEL]
    precise_card = cards[CAPTURE_PRECISE_MODEL]
    training = [row for row in rows if row["split"] in {"development", "calibration"}]
    baselines = np.asarray(
        [float(row["capture"]["baseline_m"]) for row in training], dtype=np.float64
    )
    if not len(baselines):
        raise ValueError("capture surface has no development/calibration rows")
    baseline_edges = np.unique(np.quantile(baselines, (0.0, 0.25, 0.5, 0.75, 1.0)))
    if len(baseline_edges) < 2:
        value = float(baseline_edges[0])
        baseline_edges = np.asarray([max(0.0, value - 0.5), value + 0.5])
    surface = []
    for baseline_index, (baseline_low, baseline_high) in enumerate(
        zip(baseline_edges[:-1], baseline_edges[1:], strict=True)
    ):
        for overlap_index, (label, overlap_low, overlap_high) in enumerate(
            OVERLAP_BINS
        ):
            is_last_baseline_bin = baseline_index == len(baseline_edges) - 2
            selected = [
                row
                for row in training
                if overlap_low
                <= float(row["capture"]["registered_cloud_overlap_min"])
                < overlap_high
                and baseline_low <= float(row["capture"]["baseline_m"])
                and (
                    float(row["capture"]["baseline_m"]) <= baseline_high
                    if is_last_baseline_bin
                    else float(row["capture"]["baseline_m"]) < baseline_high
                )
            ]
            accepted = [row for row in selected if row["outcomes"]["accepted"]]
            all_groups = len(
                {str(row["independence_component_id"]) for row in selected}
            )
            accepted_groups = len(
                {str(row["independence_component_id"]) for row in accepted}
            )
            support_groups = min(all_groups, accepted_groups)
            supported = support_groups >= MINIMUM_SURFACE_COMPONENTS
            if selected:
                overlap_reference = float(
                    np.median(
                        [
                            row["capture"]["registered_cloud_overlap_min"]
                            for row in selected
                        ]
                    )
                )
                baseline_reference = float(
                    np.median([row["capture"]["baseline_m"] for row in selected])
                )
            else:
                overlap_reference = None
                baseline_reference = None
            p_accept = (
                _capture_probability(
                    accept_card,
                    overlap=overlap_reference,
                    baseline_m=baseline_reference,
                )
                if supported
                else None
            )
            p_precise = (
                _capture_probability(
                    precise_card,
                    overlap=overlap_reference,
                    baseline_m=baseline_reference,
                )
                if supported
                else None
            )
            surface.append(
                {
                    "baseline_bin_index": baseline_index,
                    "overlap_bin_index": overlap_index,
                    "baseline_low_m": float(baseline_low),
                    "baseline_high_m": float(baseline_high),
                    "overlap_bin": label,
                    "overlap_low": overlap_low,
                    "overlap_high": min(overlap_high, 1.0),
                    "pairs": len(selected),
                    "accepted_pairs": len(accepted),
                    "independence_components": all_groups,
                    "accepted_independence_components": accepted_groups,
                    "support_components": support_groups,
                    "minimum_support_components": MINIMUM_SURFACE_COMPONENTS,
                    "supported": supported,
                    "reference_overlap": overlap_reference,
                    "reference_baseline_m": baseline_reference,
                    "p_accept": p_accept,
                    "p_precise_given_accept": p_precise,
                    "p_usable": (
                        p_accept * p_precise
                        if p_accept is not None and p_precise is not None
                        else None
                    ),
                }
            )
    return surface


def _capture_surface_figure(
    surface: list[dict[str, Any]], path: Path
) -> None:
    baseline_bins = 1 + max(row["baseline_bin_index"] for row in surface)
    overlap_bins = len(OVERLAP_BINS)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#E7E9EB")
    image = None
    for axis, (field, title) in zip(
        axes,
        (
            ("p_accept", "P(accepted | capture)"),
            ("p_precise_given_accept", "P(precise | accepted, capture)"),
            ("p_usable", "P(accepted and precise | capture)"),
        ),
        strict=True,
    ):
        values = np.full((baseline_bins, overlap_bins), np.nan)
        for row in surface:
            value = row[field]
            if value is not None:
                values[row["baseline_bin_index"], row["overlap_bin_index"]] = value
        image = axis.imshow(
            np.ma.masked_invalid(values),
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            cmap=cmap,
        )
        for row in surface:
            value = row[field]
            label = (
                f"{value:.2f}\ng={row['support_components']}"
                if value is not None
                else f"unsupported\ng={row['support_components']}"
            )
            axis.text(
                row["overlap_bin_index"],
                row["baseline_bin_index"],
                label,
                ha="center",
                va="center",
                fontsize=6.5,
                color=("white" if value is not None and value < 0.55 else "#26333D"),
            )
        axis.set_title(title, fontsize=10.5, weight="bold")
        axis.set_xticks(
            np.arange(overlap_bins), [label for label, _, _ in OVERLAP_BINS]
        )
        axis.tick_params(axis="x", rotation=28)
        axis.set_xlabel("minimum registered-cloud overlap")
    baseline_labels = []
    for index in range(baseline_bins):
        row = next(item for item in surface if item["baseline_bin_index"] == index)
        baseline_labels.append(
            f"{row['baseline_low_m']:.2f}–{row['baseline_high_m']:.2f} m"
        )
    axes[0].set_yticks(np.arange(baseline_bins), baseline_labels)
    axes[0].set_ylabel("baseline quartile")
    figure.colorbar(
        image,
        ax=axes,
        fraction=0.022,
        pad=0.02,
        shrink=0.88,
        label="calibrated probability",
    )
    figure.suptitle(
        "Capture-model response; grey cells have fewer than five independent groups",
        fontsize=12,
        weight="bold",
    )
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


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


def _runtime_overlap_figure(summary: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), constrained_layout=True)
    x = np.arange(len(OVERLAP_BINS))
    for axis, dataset in zip(axes, DATASET_ORDER, strict=True):
        rows = [row for row in summary if row["dataset_id"] == dataset]
        medians = np.asarray(
            [row["pair_total_median_seconds"] for row in rows], dtype=float
        )
        p95 = np.asarray([row["pair_total_p95_seconds"] for row in rows], dtype=float)
        lower = np.zeros_like(medians)
        upper = np.maximum(p95 - medians, 0.0)
        axis.errorbar(
            x,
            medians,
            yerr=np.vstack((lower, upper)),
            marker="o",
            linewidth=1.8,
            capsize=3,
            color="#3B6FB6",
            ecolor="#D28B27",
        )
        axis.set_title(DATASET_LABELS[dataset], fontsize=11, weight="bold")
        axis.set_xticks(x, [row["overlap_bin"] for row in rows], rotation=28)
        axis.set_xlabel("minimum bidirectional cloud overlap")
        axis.grid(axis="y", alpha=0.22)
        for index, row in enumerate(rows):
            if row["pair_total_median_seconds"] is not None:
                axis.text(
                    index,
                    row["pair_total_p95_seconds"] * 1.015,
                    f"n={row['pairs']}\ng={row['independence_components']}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="#4A5560",
                )
    axes[0].set_ylabel("complete pair time (seconds)")
    figure.suptitle(
        "Shared-host replay wall time — diagnostic only; marker=median, whisker=P95",
        fontsize=12,
        weight="bold",
    )
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _calibration_figure(evaluation: dict[str, Any], path: Path) -> None:
    post_model = primary_post_model(evaluation)
    rows = (
        ("capture-accept-overlap-baseline", "Capture: acceptance"),
        (post_model, "Post: precise pose"),
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
    models = [
        ("post-precise-raw-score", "Raw score"),
        ("post-precise-support", "+ support"),
        ("post-precise-common", "+ overlap/geometry"),
        ("post-precise-public-full", "Public full"),
    ]
    if ALIGNED_POST_MODEL in evaluation.get("models", {}):
        models.append((ALIGNED_POST_MODEL, "Aligned orientation"))
    figure, axis = plt.subplots(figsize=(9.2, 4.2))
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
    finite_values = [
        evaluation["models"][model_id]["datasets"][dataset]["brier"]
        for model_id, _ in models
        for dataset in DATASET_ORDER
        if dataset in evaluation["models"].get(model_id, {}).get("datasets", {})
    ]
    axis.set_ylim(0.0, max(0.38, max(finite_values, default=0.0) * 1.12))
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _transfer_figure(
    evaluation: dict[str, Any], lodo_evaluation: dict[str, Any], path: Path
) -> None:
    post_model = primary_post_model(evaluation)
    comparisons = (
        ("capture-accept-overlap-baseline", "Capture acceptance"),
        (post_model, "Post-process precision"),
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


def _selective_rule_figure(
    rule: dict[str, Any], evaluation: dict[str, Any], path: Path
) -> None:
    calibration = rule["chosen_calibration_result"]
    entries = [
        (
            "Calibration\nMatterport360",
            calibration,
            calibration["selected_components"],
        )
    ]
    for dataset in DATASET_ORDER:
        metrics = evaluation["datasets"][dataset]
        entries.append(
            (
                f"Evaluation\n{DATASET_LABELS[dataset]}",
                metrics,
                metrics["selected_components"],
            )
        )
    labels = [entry[0] for entry in entries]
    precision = np.asarray([entry[1]["selected_precision"] for entry in entries])
    lower = np.asarray([entry[1]["exact_one_sided_95_lower"] for entry in entries])
    coverage = np.asarray([entry[1]["pair_coverage"] for entry in entries])
    colors = ("#2F8C82", "#3B6FB6", "#D28B27", "#9B6AA5")
    x = np.arange(len(entries))
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    axes[0].bar(x, precision, color=colors, alpha=0.92)
    axes[0].errorbar(
        x,
        precision,
        yerr=np.vstack((precision - lower, np.zeros(len(entries)))),
        fmt="none",
        ecolor="#28343E",
        capsize=4,
        linewidth=1.4,
    )
    axes[0].axhline(
        rule["selection_targets"]["minimum_precision"],
        linestyle="--",
        color="#9E3D3D",
        linewidth=1.2,
        label="precision target",
    )
    axes[0].axhline(
        rule["selection_targets"]["minimum_exact_one_sided_95_lower"],
        linestyle=":",
        color="#6B3F3F",
        linewidth=1.2,
        label="lower-bound target",
    )
    axes[0].set_ylabel("selected-pose precision")
    axes[0].set_ylim(0.0, 1.04)
    axes[0].set_title("Frozen rule: precision and one-sided bound", weight="bold")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")

    axes[1].bar(x, coverage, color=colors, alpha=0.92)
    axes[1].set_ylabel("coverage over eligible pairs")
    axes[1].set_ylim(0.0, max(0.20, float(np.max(coverage)) * 1.35))
    axes[1].set_title("Coverage, support, and catastrophic accepts", weight="bold")
    for index, (_, metrics, components) in enumerate(entries):
        axes[1].text(
            index,
            coverage[index] + 0.008,
            (
                f"n={metrics['selected_pairs']}  g={components}\n"
                f"cat={metrics['catastrophic_accepted']}"
            ),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=("#9E3D3D" if metrics["catastrophic_accepted"] else "#36434E"),
        )
    for axis in axes:
        axis.set_xticks(x, labels, rotation=16, ha="right")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Retrospective selective rule — evaluation verdict: "
        f"{evaluation['verdict'].replace('_', ' ')}",
        fontsize=12,
        weight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
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
    runtime_summary = summarize_runtime_overlap(rows)
    engineering_summary = summarize_engineering(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "overlap-response.csv"
    _write_csv(csv_path, summary)
    _write_csv(args.output_dir / "runtime-overlap-response.csv", runtime_summary)
    figures = {
        "overlap_response": args.output_dir / "overlap-response.png",
        "calibration": args.output_dir / "calibration-heldout.png",
        "post_ablation": args.output_dir / "post-model-ablation.png",
        "runtime_overlap_response": args.output_dir / "runtime-overlap-response.png",
    }
    _overlap_figure(summary, figures["overlap_response"])
    _calibration_figure(evaluation, figures["calibration"])
    _ablation_figure(evaluation, figures["post_ablation"])
    _runtime_overlap_figure(runtime_summary, figures["runtime_overlap_response"])
    capture_surface = None
    if args.model_card is not None:
        model_card = json.loads(args.model_card.read_text(encoding="utf-8"))
        capture_surface = capture_probability_surface(rows, model_card)
        _write_csv(
            args.output_dir / "capture-probability-surface.csv", capture_surface
        )
        figures["capture_probability_surface"] = (
            args.output_dir / "capture-probability-surface.png"
        )
        _capture_surface_figure(
            capture_surface, figures["capture_probability_surface"]
        )
    if args.lodo_evaluation is not None:
        lodo_evaluation = json.loads(args.lodo_evaluation.read_text(encoding="utf-8"))
        figures["cross_dataset_transfer"] = (
            args.output_dir / "cross-dataset-transfer.png"
        )
        _transfer_figure(evaluation, lodo_evaluation, figures["cross_dataset_transfer"])
    if args.release_rule is not None or args.release_evaluation is not None:
        if args.release_rule is None or args.release_evaluation is None:
            raise ValueError(
                "--release-rule and --release-evaluation must be provided together"
            )
        release_rule = json.loads(args.release_rule.read_text(encoding="utf-8"))
        release_evaluation = json.loads(
            args.release_evaluation.read_text(encoding="utf-8")
        )
        figures["selective_rule"] = args.output_dir / "selective-rule-evaluation.png"
        _selective_rule_figure(
            release_rule, release_evaluation, figures["selective_rule"]
        )
    payload = {
        "schema": SUMMARY_SCHEMA,
        "status": "post-hoc descriptive response plus frozen held-out model evaluation",
        "overlap_response": summary,
        "runtime_overlap_response": runtime_summary,
        "engineering_summary": engineering_summary,
        "primary_post_model": primary_post_model(evaluation),
        "capture_probability_surface": capture_surface,
        "figures": {name: str(path.resolve()) for name, path in figures.items()},
        "source_analysis_table": str(args.analysis_table.resolve()),
        "source_evaluation": str(args.evaluation.resolve()),
        "source_model_card": (
            str(args.model_card.resolve()) if args.model_card is not None else None
        ),
        "source_lodo_evaluation": (
            str(args.lodo_evaluation.resolve())
            if args.lodo_evaluation is not None
            else None
        ),
        "source_release_rule": (
            str(args.release_rule.resolve()) if args.release_rule is not None else None
        ),
        "source_release_evaluation": (
            str(args.release_evaluation.resolve())
            if args.release_evaluation is not None
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
    parser.add_argument("--model-card", type=Path)
    parser.add_argument("--lodo-evaluation", type=Path)
    parser.add_argument("--release-rule", type=Path)
    parser.add_argument("--release-evaluation", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
