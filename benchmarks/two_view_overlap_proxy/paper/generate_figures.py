"""Generate the anonymized quantitative figures used by the R,t paper."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence.json"
FIGURES = ROOT / "figures"

COLORS = {
    "navy": "#17324D",
    "blue": "#2C6E9F",
    "teal": "#2A9D8F",
    "gold": "#E9A23B",
    "red": "#C84C4C",
    "gray": "#687783",
    "light": "#EAF0F4",
    "ink": "#17232D",
}
DOMAIN_COLORS = [COLORS["blue"], COLORS["gold"], COLORS["teal"]]


def load_evidence() -> dict:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.edgecolor": "#A7B2BA",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#DCE3E8",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.85,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, filename: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / filename, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def box(ax, x, y, w, h, text, fill, edge, size=9.2, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        color=COLORS["ink"],
        fontsize=size,
        fontweight=weight,
        linespacing=1.25,
    )


def arrow(ax, x1, y1, x2, y2, color=None):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.25,
            color=color or COLORS["gray"],
        )
    )


def make_pipeline_figure() -> None:
    fig, ax = plt.subplots(figsize=(12.0, 4.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.02, 0.58, 0.16, 0.22, "Two EQR images\n+ explicit masks", "#F0F6FA", COLORS["blue"], weight="bold")
    box(ax, 0.02, 0.19, 0.16, 0.18, "Estimated baseline\nmean + uncertainty", "#FFF7E7", COLORS["gold"])
    box(ax, 0.25, 0.55, 0.18, 0.28, "Optimized spherical frontend\nnative DoG batch 2\ntangent RootSIFT\nmatching", "#EAF5F4", COLORS["teal"], weight="bold")
    box(ax, 0.50, 0.61, 0.17, 0.20, "RGB overlap proxy\nadvisory latent\noverlap posterior", "#F0F6FA", COLORS["blue"])
    box(ax, 0.50, 0.28, 0.17, 0.20, "Spherical R,t\nrobust estimation\npublic quality gate", "#F0F6FA", COLORS["navy"], weight="bold")
    box(ax, 0.73, 0.44, 0.12, 0.22, "Post model\nP(precise |\nevidence)", "#FFF7E7", COLORS["gold"], weight="bold")
    box(ax, 0.89, 0.55, 0.09, 0.18, "Release\nR, t_dir", "#E9F6EF", COLORS["teal"], weight="bold")
    box(ax, 0.89, 0.26, 0.09, 0.18, "Abstain /\nreacquire", "#FBECEC", COLORS["red"], weight="bold")

    arrow(ax, 0.18, 0.69, 0.25, 0.69)
    arrow(ax, 0.43, 0.71, 0.50, 0.71)
    arrow(ax, 0.43, 0.62, 0.50, 0.40)
    arrow(ax, 0.18, 0.28, 0.50, 0.36, COLORS["gold"])
    arrow(ax, 0.67, 0.71, 0.73, 0.58)
    arrow(ax, 0.67, 0.39, 0.73, 0.52)
    arrow(ax, 0.85, 0.58, 0.89, 0.64, COLORS["teal"])
    arrow(ax, 0.85, 0.51, 0.89, 0.36, COLORS["red"])

    ax.text(0.5, 0.94, "Two-view decision path: geometry first, probability as a safety layer", ha="center", va="center", fontsize=14, fontweight="bold", color=COLORS["navy"])
    ax.text(0.50, 0.08, "Metric translation = estimated direction x independently supplied baseline. Dense stereo is outside this pipeline.", ha="center", va="center", fontsize=9.5, color=COLORS["gray"])
    save(fig, "figure-1-two-view-decision-path.png")


def make_overlap_response_figure(data: dict) -> None:
    bins = data["overlap_response"]["bins"]
    x = np.arange(len(bins))
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    metrics = [("return_rate", "Pose returned"), ("accept_rate", "Public quality accepted"), ("usable_rate", "Accepted and precise")]

    for ax, (key, title) in zip(axes, metrics, strict=True):
        for idx, domain in enumerate(data["overlap_response"]["domains"]):
            ax.plot(x, domain[key], marker="o", linewidth=2.1, markersize=5.2, color=DOMAIN_COLORS[idx], label=domain["name"])
            if key == "usable_rate":
                for xi, yi, n in zip(x, domain[key], domain["pairs"], strict=True):
                    ax.annotate(f"n={n}", (xi, yi), xytext=(0, 6 + idx * 8), textcoords="offset points", ha="center", fontsize=6.5, color=DOMAIN_COLORS[idx])
        ax.axvspan(2.5, 4.5, color=COLORS["teal"], alpha=0.08)
        ax.axvline(2.5, color=COLORS["teal"], linestyle="--", linewidth=1.1)
        ax.set_title(title, fontweight="bold", color=COLORS["navy"])
        ax.set_xticks(x, bins, rotation=25, ha="right")
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("Registered-cloud overlap")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Observed rate")
    axes[1].text(2.58, 0.04, "supported eligibility region", fontsize=7.5, color=COLORS["teal"], rotation=90, va="bottom")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3)
    fig.suptitle("Estimator response rises with spatial overlap, but domain shift remains", fontsize=13.5, fontweight="bold", color=COLORS["navy"], y=1.14)
    fig.subplots_adjust(wspace=0.13)
    save(fig, "figure-2-response-versus-overlap.png")


def make_post_policy_figure(data: dict) -> None:
    pooled = data["post_policy"]["pooled"]
    groups = [{"name": "Pooled", **pooled}, *data["post_policy"]["domains"]]
    names = [g["name"] for g in groups]
    x = np.arange(len(names))
    width = 0.22
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.2), gridspec_kw={"width_ratios": [1.55, 1]})

    ax1.bar(x - width, [g["precision"] for g in groups], width, label="Precision", color=COLORS["teal"])
    ax1.bar(x, [g["recall"] for g in groups], width, label="Recall", color=COLORS["blue"])
    ax1.bar(x + width, [g["coverage"] for g in groups], width, label="Coverage", color=COLORS["gold"])
    lower = [g["precision_exact_one_sided_95_lower"] for g in groups]
    ax1.scatter(x - width, lower, marker="_", s=360, linewidth=2.2, color=COLORS["red"], label="Precision one-sided 95% lower")
    ax1.axhline(0.90, color=COLORS["red"], linestyle="--", linewidth=1.1)
    ax1.text(3.48, 0.905, "0.90 target", ha="right", va="bottom", fontsize=8, color=COLORS["red"])
    ax1.set_xticks(x, names)
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Rate")
    ax1.set_title("Selective policy on held-out pairs", fontweight="bold", color=COLORS["navy"])
    ax1.legend(loc="lower left", fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    funnel_labels = ["Held-out\npairs", "Public quality\naccepted", "Probability\nselected", "Precise\nselected"]
    funnel_values = [pooled["pairs"], data["post_policy"]["public_quality_accepted"], pooled["selected"], pooled["true_positive"]]
    colors = [COLORS["gray"], COLORS["blue"], COLORS["gold"], COLORS["teal"]]
    y = np.arange(len(funnel_labels))[::-1]
    ax2.barh(y, funnel_values, color=colors, height=0.62)
    for yi, value in zip(y, funnel_values, strict=True):
        ax2.text(value + 8, yi, f"{value}", va="center", fontsize=10, fontweight="bold", color=COLORS["ink"])
    ax2.set_yticks(y, funnel_labels)
    ax2.set_xlim(0, 485)
    ax2.set_title("Evidence funnel", fontweight="bold", color=COLORS["navy"])
    ax2.set_xlabel("Pairs")
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.grid(axis="y", visible=False)

    fig.suptitle("High pooled precision does not erase domain-level uncertainty", fontsize=13.5, fontweight="bold", color=COLORS["navy"], y=1.04)
    fig.subplots_adjust(wspace=0.34)
    save(fig, "figure-3-post-policy-evaluation.png")


def make_proxy_figure(data: dict) -> None:
    splits = data["overlap_proxy"]["splits"]
    names = [s["name"].replace(" ", "\n", 1) for s in splits]
    x = np.arange(len(names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    series = [
        ("mae", "Overlap MAE", COLORS["gray"]),
        ("brier", "Brier", COLORS["blue"]),
        ("precision", "Precision at 0.5", COLORS["teal"]),
        ("recall", "Recall at 0.5", COLORS["gold"]),
    ]
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
    for offset, (key, label, color) in zip(offsets, series, strict=True):
        vals = [s[key] for s in splits]
        bars = ax.bar(x + offset, vals, width, label=label, color=color)
        for bar, value in zip(bars, vals, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.2f}", ha="center", va="bottom", fontsize=7.5, rotation=90)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 0.92)
    ax.set_ylabel("Metric value")
    ax.set_title("RGB overlap proxy: conservative detection of overlap >= 50%", fontweight="bold", color=COLORS["navy"])
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(2.0, 0.84, "Held-out precision 0.81\nHeld-out recall 0.37", ha="center", va="center", color=COLORS["navy"], fontsize=10, fontweight="bold", bbox={"boxstyle": "round,pad=0.45", "facecolor": "#F0F6FA", "edgecolor": COLORS["blue"]})
    save(fig, "figure-4-overlap-proxy.png")


def make_runtime_figure(data: dict) -> None:
    timing = data["controlled_timing"]
    bins = timing["overlap_bins"]
    x = np.arange(len(bins))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.2), gridspec_kw={"width_ratios": [1.45, 1]})
    for idx, domain in enumerate(timing["runtime_by_overlap"]):
        median = np.asarray(domain["median_s"])
        p95 = np.asarray(domain["p95_s"])
        ax1.plot(x, median, marker="o", linewidth=2.1, color=DOMAIN_COLORS[idx], label=f"{domain['name']} median")
        ax1.fill_between(x, median, p95, color=DOMAIN_COLORS[idx], alpha=0.12)
    ax1.axhline(timing["reference"]["complete_pair_s"], color=COLORS["red"], linestyle="--", linewidth=1.1, label="8.99 s reference")
    ax1.set_xticks(x, bins, rotation=25, ha="right")
    ax1.set_ylabel("Complete pair time (s)")
    ax1.set_xlabel("Registered-cloud overlap")
    ax1.set_title("Runtime is not monotonic in overlap", fontweight="bold", color=COLORS["navy"])
    ax1.legend(fontsize=7.8, ncol=2)
    ax1.spines[["top", "right"]].set_visible(False)

    domains = timing["domain_aggregate"]
    names = [d["name"] for d in domains]
    y = np.arange(len(names))
    ax2.barh(y + 0.17, [d["total_median_s"] for d in domains], height=0.3, color=COLORS["blue"], label="Complete pair")
    ax2.barh(y - 0.17, [d["pose_median_s"] for d in domains], height=0.3, color=COLORS["teal"], label="R,t stage")
    ax2.axvline(timing["reference"]["complete_pair_s"], color=COLORS["red"], linestyle="--", linewidth=1.1)
    ax2.set_yticks(y, names)
    ax2.set_xlabel("Median time (s)")
    ax2.set_title("Total versus R,t stage", fontweight="bold", color=COLORS["navy"])
    ax2.legend(fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Controlled performance: 75 pairs x 3 repetitions at 1024 x 2048", fontsize=13.5, fontweight="bold", color=COLORS["navy"], y=1.04)
    fig.subplots_adjust(wspace=0.28)
    save(fig, "figure-5-controlled-runtime.png")


def main() -> None:
    apply_style()
    data = load_evidence()
    make_pipeline_figure()
    make_overlap_response_figure(data)
    make_post_policy_figure(data)
    make_proxy_figure(data)
    make_runtime_figure(data)
    print(f"Generated 5 figures in {FIGURES}")


if __name__ == "__main__":
    main()
