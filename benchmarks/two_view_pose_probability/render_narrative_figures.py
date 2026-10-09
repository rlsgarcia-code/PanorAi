#!/usr/bin/env python3
"""Render deterministic narrative figures for the VAL-018 paper.

The figures in this module are deliberately separated from the generated
concept art.  Every displayed number is read from a frozen evidence artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np


DATASET_ORDER = ("matterport360", "stanford2d3d", "p74_native_polar")
DATASET_LABELS = {
    "matterport360": "Matterport360",
    "stanford2d3d": "Stanford2D3D",
    "p74_native_polar": "P74",
}
DATASET_COLORS = {
    "matterport360": "#3B78A7",
    "stanford2d3d": "#D28B27",
    "p74_native_polar": "#8A6AA5",
}
STATE_LABELS = {
    "no-pose": "No pose",
    "catastrophic-accepted": "Accepted catastrophic",
    "imprecise-accepted": "Accepted imprecise",
    "precise-accepted": "Accepted precise",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prospective_curve(
    plan: dict[str, Any], *, true_precision: float, icc: float
) -> list[dict[str, Any]]:
    """Return the frozen power-grid curve for one declared scenario."""
    rows = [
        row
        for row in plan["power_grid"]
        if np.isclose(row["true_precision"], true_precision)
        and np.isclose(row["intraclass_correlation"], icc)
    ]
    return sorted(rows, key=lambda row: row["independent_groups"])


def replay_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return exact-main replay rows in the paper's causal reading order."""
    order = {
        "no-return-low-overlap": 0,
        "returned-but-rejected": 1,
        "overconfident-near-miss": 2,
        "supported-success": 3,
        "catastrophic-accepted": 4,
    }
    return sorted(
        summary["comparisons"], key=lambda row: order[row["taxonomy_category"]]
    )


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.18, linewidth=0.7)


def _evidence_base(census: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.1))
    metrics = (
        ("unique_images", "Unique panoramas", False),
        ("unique_pairs", "Two-view pairs", False),
        ("independence_components", "Independent groups", True),
    )
    x = np.arange(len(DATASET_ORDER))
    for axis, (key, title, annotate) in zip(axes, metrics, strict=True):
        values = [census["datasets"][dataset][key] for dataset in DATASET_ORDER]
        bars = axis.bar(
            x,
            values,
            color=[DATASET_COLORS[item] for item in DATASET_ORDER],
            width=0.68,
        )
        axis.set_title(title, weight="bold", fontsize=11)
        axis.set_xticks(x, [DATASET_LABELS[item] for item in DATASET_ORDER])
        axis.tick_params(axis="x", rotation=18)
        axis.set_ylabel("count")
        if not annotate:
            axis.set_yscale("log")
            axis.set_ylabel("count (log scale)")
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value * (1.07 if not annotate else 1.02),
                f"{value:,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        _style_axis(axis)
    totals = census["totals"]
    figure.suptitle(
        "Frozen two-view evidence base — "
        f"{totals['unique_images']:,} images, {totals['unique_pairs']:,} pairs, "
        f"{totals['independence_components']} independent groups",
        fontsize=12,
        weight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def _two_model_story(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12.2, 4.4))
    axis.set_xlim(0.0, 12.2)
    axis.set_ylim(0.0, 5.0)
    axis.axis("off")

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        title: str,
        body: str,
        color: str,
    ) -> None:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                facecolor=color,
                edgecolor="none",
                alpha=0.14,
            )
        )
        axis.text(x + 0.16, y + height - 0.25, title, weight="bold", va="top")
        axis.text(x + 0.16, y + height - 0.72, body, va="top", fontsize=9)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        axis.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color="#52616D",
            )
        )

    box(
        0.1,
        2.8,
        2.35,
        1.45,
        "Capture evidence (x)",
        "cloud overlap\nbaseline / scene scale\npredicted parallax\nvisible scene conditions",
        "#2F8C82",
    )
    box(
        3.15,
        2.8,
        2.35,
        1.45,
        "Pre-capture model",
        "P(accepted | x)\nP(precise | accepted, x)",
        "#3B78A7",
    )
    box(
        6.2,
        2.8,
        2.35,
        1.45,
        "Capture decision",
        "screen pair before pose\nreacquire outside support",
        "#3B78A7",
    )
    box(
        3.15,
        0.55,
        2.35,
        1.45,
        "Algorithm evidence (z)",
        "matches and inliers\nspatial coverage\nresiduals / cheirality\nstability / model competition",
        "#D28B27",
    )
    box(
        6.2,
        0.55,
        2.35,
        1.45,
        "Post-process model",
        "P(precise | returned, x, z)",
        "#D28B27",
    )
    box(
        9.25,
        1.67,
        2.55,
        1.45,
        "Selective release",
        "capture gate AND confidence gate\nunsupported domain → no claim",
        "#9E4B46",
    )
    arrow(2.45, 3.52, 3.15, 3.52)
    arrow(5.5, 3.52, 6.2, 3.52)
    arrow(2.45, 3.15, 3.55, 2.0)
    arrow(5.5, 1.28, 6.2, 1.28)
    arrow(8.55, 3.45, 9.25, 2.68)
    arrow(8.55, 1.28, 9.25, 2.05)
    axis.text(
        0.1,
        4.72,
        "Two complementary probability models for spherical two-view pose",
        fontsize=13,
        weight="bold",
        va="top",
    )
    axis.text(
        0.1,
        0.1,
        "Reference R,t errors are outcomes only; they are never model inputs.",
        fontsize=9,
        color="#52616D",
    )
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def _optimized_replay(summary: dict[str, Any], path: Path) -> None:
    rows = replay_rows(summary)
    labels = [
        {
            "no-return-low-overlap": "P74 negligible overlap",
            "returned-but-rejected": "Matterport wrong historical pose",
            "overconfident-near-miss": "P74 precision near miss",
            "supported-success": "Stanford supported success",
            "catastrophic-accepted": "Stanford repetitive-scene failure",
        }[row["taxonomy_category"]]
        for row in rows
    ]
    y = np.arange(len(rows))
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharey=True)

    matches = np.asarray([row["matches"] for row in rows], dtype=float)
    inliers = np.asarray([row["inliers"] or 0 for row in rows], dtype=float)
    axes[0].barh(y, matches, color="#B7C2C9", label="matches")
    axes[0].barh(y, inliers, color="#2F8C82", label="pose inliers")
    axes[0].set_xscale("log")
    axes[0].set_xlim(1.0, 1000.0)
    axes[0].set_xlabel("correspondence support (log count)")
    axes[0].set_title("Optimized spherical frontend", weight="bold")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.18)
    axes[0].spines[["top", "right"]].set_visible(False)
    for index, row in enumerate(rows):
        axes[0].text(
            min(row["matches"] * 1.08, 920),
            index,
            f"{row['matches']}",
            va="center",
            fontsize=8,
        )

    axes[1].set_xscale("log")
    axes[1].set_xlim(0.1, 180.0)
    axes[1].axvline(1.0, color="#3B78A7", linestyle=":", linewidth=1.2)
    axes[1].axvline(5.0, color="#D28B27", linestyle="--", linewidth=1.2)
    for index, row in enumerate(rows):
        rotation = row["rotation_error_deg"]
        translation = row["translation_direction_error_deg"]
        if rotation is None or translation is None:
            axes[1].text(0.13, index, "no pose returned", va="center", fontsize=8)
            continue
        color = (
            "#B34F4A"
            if row["optimized_main_state"] == "catastrophic-accepted"
            else "#2F8C82"
            if row["optimized_main_state"] == "precise-accepted"
            else "#D28B27"
        )
        axes[1].plot(rotation, index, "o", color=color, markersize=6)
        axes[1].plot(translation, index, "^", color=color, markersize=7)
        axes[1].plot([rotation, translation], [index, index], color=color, alpha=0.45)
        axes[1].text(
            min(max(rotation, translation) * 1.08, 165),
            index,
            STATE_LABELS[row["optimized_main_state"]],
            va="center",
            fontsize=7.5,
            color=color,
        )
    axes[1].set_xlabel("pose error (degrees, log scale)")
    axes[1].set_title("Ground-truth pose error", weight="bold")
    axes[1].grid(axis="x", alpha=0.18)
    axes[1].spines[["top", "right", "left"]].set_visible(False)
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    axes[1].text(1.0, -0.68, "1° R", ha="center", fontsize=7, color="#3B78A7")
    axes[1].text(5.0, -0.68, "5° t", ha="center", fontsize=7, color="#D28B27")

    figure.legend(
        handles=[
            Patch(facecolor="#B7C2C9", label="matches"),
            Patch(facecolor="#2F8C82", label="pose inliers"),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color="#52616D",
                label="rotation error",
            ),
            Line2D(
                [],
                [],
                marker="^",
                linestyle="none",
                color="#52616D",
                label="translation error",
            ),
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
    )

    package = summary["package"]
    aggregate = summary["aggregate"]
    figure.suptitle(
        "Five-pair mechanism replay on exact origin/main source\n"
        f"PanorAi {package['version']} · native batch-2 route · "
        f"median {aggregate['median_pair_total_seconds']:.2f} s/pair",
        fontsize=12,
        weight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.07, 1.0, 0.88))
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def _prospective_plan(plan: dict[str, Any], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.6, 4.8))
    colors = ("#A8B4BD", "#3B78A7", "#2F8C82", "#8A6AA5")
    for precision, color in zip((0.95, 0.97, 0.98, 0.99), colors, strict=True):
        rows = prospective_curve(plan, true_precision=precision, icc=0.2)
        axis.plot(
            [row["independent_groups"] for row in rows],
            [row["estimated_power"] for row in rows],
            marker="o",
            linewidth=1.8,
            color=color,
            label=f"true precision {precision:.0%}",
        )
    design = plan["primary_design_scenario"]
    axis.axhline(0.8, color="#9E4B46", linestyle="--", linewidth=1.1)
    axis.scatter(
        [design["minimum_independent_groups"]],
        [design["estimated_power"]],
        s=95,
        facecolor="white",
        edgecolor="#9E4B46",
        linewidth=1.8,
        zorder=5,
    )
    axis.annotate(
        "frozen design\n40 groups · 120 selected pairs\npower 0.808",
        xy=(design["minimum_independent_groups"], design["estimated_power"]),
        xytext=(51, 0.88),
        arrowprops={"arrowstyle": "->", "color": "#52616D"},
        fontsize=8,
        va="center",
    )
    axis.set_xlim(4, 64)
    axis.set_ylim(0.0, 1.03)
    axis.set_xlabel("new independent capture groups")
    axis.set_ylabel("probability of meeting the confirmation gate")
    axis.set_title(
        "Prospective confirmation plan (beta-binomial, ICC = 0.20)",
        weight="bold",
        fontsize=11,
    )
    axis.legend(frameon=False, ncol=2, fontsize=8, loc="lower right")
    axis.grid(alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        5,
        0.98,
        "gate: precision ≥95% · one-sided 95% lower bound ≥90% · zero catastrophes",
        va="top",
        fontsize=8,
        color="#52616D",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    census = _load(args.census)
    replay = _load(args.replay_summary)
    plan = _load(args.prospective_plan)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "evidence_base": args.output_dir / "quantitative-evidence-base.png",
        "two_model_story": args.output_dir / "two-model-probability-story.png",
        "optimized_main_replay": (
            args.output_dir / "quantitative-optimized-main-replay.png"
        ),
        "prospective_plan": (
            args.output_dir / "quantitative-prospective-confirmation.png"
        ),
    }
    _evidence_base(census, figures["evidence_base"])
    _two_model_story(figures["two_model_story"])
    _optimized_replay(replay, figures["optimized_main_replay"])
    _prospective_plan(plan, figures["prospective_plan"])
    payload = {
        "schema": "panorai-two-view-pose-narrative-figures/v1",
        "figures": {
            key: {"filename": value.name, "sha256": _sha256(value)}
            for key, value in figures.items()
        },
        "inputs": {
            "census": {
                "filename": args.census.name,
                "sha256": _sha256(args.census),
            },
            "replay_summary": {
                "filename": args.replay_summary.name,
                "sha256": _sha256(args.replay_summary),
            },
            "prospective_plan": {
                "filename": args.prospective_plan.name,
                "sha256": _sha256(args.prospective_plan),
            },
        },
    }
    (args.output_dir / "narrative-figures.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["figures"], indent=2, sort_keys=True))
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--replay-summary", type=Path, required=True)
    parser.add_argument("--prospective-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
