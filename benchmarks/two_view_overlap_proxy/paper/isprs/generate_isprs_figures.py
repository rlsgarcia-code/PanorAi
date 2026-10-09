"""Generate figures sized for the 2024 ISPRS full-paper layout."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT.parent / "evidence.json"
FIGURES = ROOT / "figures"

NAVY = "#17324D"
BLUE = "#2C6E9F"
TEAL = "#2A9D8F"
GOLD = "#E9A23B"
RED = "#C84C4C"
GRAY = "#687783"
LIGHT = "#EAF0F4"
DOMAIN_COLORS = [BLUE, GOLD, TEAL]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.5,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.4,
            "legend.fontsize": 6.8,
            "axes.edgecolor": "#9CA8B0",
            "axes.linewidth": 0.6,
            "axes.grid": True,
            "grid.color": "#DCE3E8",
            "grid.linewidth": 0.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def add_box(ax, x, y, w, h, text, fill, edge, bold=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0.9,
            edgecolor=edge,
            facecolor=fill,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.8, fontweight="bold" if bold else "normal", linespacing=1.1)


def add_arrow(ax, x1, y1, x2, y2, color=GRAY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color=color))


def decision_path() -> None:
    fig, ax = plt.subplots(figsize=(3.15, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_box(ax, 0.10, 0.83, 0.80, 0.11, "Two EQR panoramas + masks", "#F0F6FA", BLUE, True)
    add_box(ax, 0.10, 0.65, 0.80, 0.12, "Native spherical DoG batch 2\nTangent RootSIFT + matching", "#EAF5F4", TEAL, True)
    add_box(ax, 0.08, 0.45, 0.39, 0.12, "RGB overlap proxy\n(advisory)", "#F0F6FA", BLUE)
    add_box(ax, 0.53, 0.45, 0.39, 0.12, "Spherical R,t\n+ public quality", "#F0F6FA", NAVY, True)
    add_box(ax, 0.10, 0.28, 0.80, 0.11, "Post model P(precise | evidence)", "#FFF7E7", GOLD, True)
    add_box(ax, 0.08, 0.08, 0.39, 0.11, "Release\nR, t_dir", "#E9F6EF", TEAL, True)
    add_box(ax, 0.53, 0.08, 0.39, 0.11, "Abstain /\nreacquire", "#FBECEC", RED, True)
    add_arrow(ax, 0.50, 0.83, 0.50, 0.77)
    add_arrow(ax, 0.40, 0.65, 0.28, 0.57)
    add_arrow(ax, 0.60, 0.65, 0.72, 0.57)
    add_arrow(ax, 0.28, 0.45, 0.38, 0.39)
    add_arrow(ax, 0.72, 0.45, 0.62, 0.39)
    add_arrow(ax, 0.40, 0.28, 0.28, 0.19, TEAL)
    add_arrow(ax, 0.60, 0.28, 0.72, 0.19, RED)
    ax.text(0.5, 0.01, "Metric translation = t_dir x independent baseline", ha="center", va="bottom", fontsize=6.2, color=GRAY)
    save(fig, "isprs-figure-1-decision-path.png")


def overlap_response(data: dict) -> None:
    bins = data["overlap_response"]["bins"]
    x = np.arange(len(bins))
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.45), sharey=True)
    metrics = [("return_rate", "Pose returned"), ("accept_rate", "Public quality accepted"), ("usable_rate", "Accepted and precise")]
    for ax, (key, title) in zip(axes, metrics, strict=True):
        for idx, domain in enumerate(data["overlap_response"]["domains"]):
            ax.plot(x, domain[key], marker="o", linewidth=1.35, markersize=3.0, color=DOMAIN_COLORS[idx], label=domain["name"])
        ax.axvspan(2.5, 4.5, color=TEAL, alpha=0.07)
        ax.axvline(2.5, color=TEAL, linestyle="--", linewidth=0.7)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x, bins, rotation=25, ha="right")
        ax.set_ylim(-0.03, 1.04)
        ax.set_xlabel("Registered-cloud overlap")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Observed rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=False)
    fig.subplots_adjust(wspace=0.16, top=0.82)
    save(fig, "isprs-figure-2-overlap-response.png")


def probability_models(data: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 2.75))
    ax1, ax2, ax3, ax4 = axes.ravel()
    splits = data["overlap_proxy"]["splits"]
    names = ["Dev.", "Cal.", "Held-out"]
    x = np.arange(3)
    width = 0.2
    for offset, key, label, color in [(-1.5, "mae", "MAE", GRAY), (-0.5, "brier", "Brier", BLUE), (0.5, "precision", "Precision", TEAL), (1.5, "recall", "Recall", GOLD)]:
        ax1.bar(x + offset * width, [s[key] for s in splits], width, label=label, color=color)
    ax1.set_xticks(x, names)
    ax1.set_ylim(0, 0.9)
    ax1.set_ylabel("Metric value")
    ax1.set_title("RGB overlap proxy (overlap >= 50%)", fontweight="bold")
    ax1.legend(ncol=2, frameon=False, loc="upper left")
    ax1.spines[["top", "right"]].set_visible(False)

    pooled = data["post_policy"]["pooled"]
    groups = [{"name": "Pooled", **pooled}, *data["post_policy"]["domains"]]
    names2 = [g["name"].replace("Domain ", "D") for g in groups]
    x2 = np.arange(4)
    w = 0.21
    ax2.bar(x2 - w, [g["precision"] for g in groups], w, label="Precision", color=TEAL)
    ax2.bar(x2, [g["recall"] for g in groups], w, label="Recall", color=BLUE)
    ax2.bar(x2 + w, [g["coverage"] for g in groups], w, label="Coverage", color=GOLD)
    ax2.scatter(x2 - w, [g["precision_exact_one_sided_95_lower"] for g in groups], marker="_", s=150, linewidth=1.5, color=RED, label="Precision lower 95%")
    ax2.axhline(0.90, color=RED, linestyle="--", linewidth=0.7)
    ax2.set_xticks(x2, names2)
    ax2.set_ylim(0, 1.07)
    ax2.set_title("Selective post-policy on held-out pairs", fontweight="bold")
    ax2.legend(ncol=2, frameon=False, loc="lower left")
    ax2.spines[["top", "right"]].set_visible(False)

    timing = data["controlled_timing"]
    bins = timing["overlap_bins"]
    x3 = np.arange(len(bins))
    for idx, domain in enumerate(timing["runtime_by_overlap"]):
        median = np.asarray(domain["median_s"])
        ax3.plot(x3, median, marker="o", linewidth=1.15, markersize=2.5, color=DOMAIN_COLORS[idx], label=domain["name"])
    ax3.axhline(timing["reference"]["complete_pair_s"], color=RED, linestyle="--", linewidth=0.7, label="8.99 s reference")
    ax3.set_xticks(x3, bins, rotation=20, ha="right")
    ax3.set_ylabel("Median pair time (s)")
    ax3.set_title("Runtime is not monotonic in overlap", fontweight="bold")
    ax3.legend(ncol=2, frameon=False, fontsize=5.6)
    ax3.spines[["top", "right"]].set_visible(False)

    domains = timing["domain_aggregate"]
    names4 = [d["name"].replace("Domain ", "D") for d in domains]
    y4 = np.arange(3)
    ax4.barh(y4 + 0.15, [d["total_median_s"] for d in domains], height=0.26, color=BLUE, label="Complete pair")
    ax4.barh(y4 - 0.15, [d["pose_median_s"] for d in domains], height=0.26, color=TEAL, label="R,t stage")
    ax4.axvline(timing["reference"]["complete_pair_s"], color=RED, linestyle="--", linewidth=0.7)
    ax4.set_yticks(y4, names4)
    ax4.set_xlabel("Median time (s)")
    ax4.set_title("Complete pair versus R,t stage", fontweight="bold")
    ax4.legend(frameon=False, fontsize=5.8)
    ax4.spines[["top", "right"]].set_visible(False)

    fig.subplots_adjust(wspace=0.24, hspace=0.56)
    save(fig, "isprs-figure-3-evidence-summary.png")


def controlled_runtime(data: dict) -> None:
    timing = data["controlled_timing"]
    bins = timing["overlap_bins"]
    x = np.arange(len(bins))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.05, 2.5), gridspec_kw={"width_ratios": [1.35, 1]})
    for idx, domain in enumerate(timing["runtime_by_overlap"]):
        median = np.asarray(domain["median_s"])
        p95 = np.asarray(domain["p95_s"])
        ax1.plot(x, median, marker="o", linewidth=1.3, markersize=3.0, color=DOMAIN_COLORS[idx], label=domain["name"])
        ax1.fill_between(x, median, p95, color=DOMAIN_COLORS[idx], alpha=0.10)
    ax1.axhline(timing["reference"]["complete_pair_s"], color=RED, linestyle="--", linewidth=0.7, label="8.99 s reference")
    ax1.set_xticks(x, bins, rotation=25, ha="right")
    ax1.set_ylabel("Complete-pair time (s)")
    ax1.set_xlabel("Registered-cloud overlap")
    ax1.set_title("Runtime is not monotonic in overlap", fontweight="bold")
    ax1.legend(ncol=2, frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)

    domains = timing["domain_aggregate"]
    names = [d["name"].replace("Domain ", "D") for d in domains]
    y = np.arange(3)
    ax2.barh(y + 0.16, [d["total_median_s"] for d in domains], height=0.28, color=BLUE, label="Complete pair")
    ax2.barh(y - 0.16, [d["pose_median_s"] for d in domains], height=0.28, color=TEAL, label="R,t stage")
    ax2.axvline(timing["reference"]["complete_pair_s"], color=RED, linestyle="--", linewidth=0.7)
    ax2.set_yticks(y, names)
    ax2.set_xlabel("Median time (s)")
    ax2.set_title("Complete pair versus R,t stage", fontweight="bold")
    ax2.legend(frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(wspace=0.27)
    save(fig, "isprs-figure-4-controlled-runtime.png")


def main() -> None:
    style()
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    decision_path()
    overlap_response(data)
    probability_models(data)
    print(f"Generated 3 ISPRS figures in {FIGURES}")


if __name__ == "__main__":
    main()
