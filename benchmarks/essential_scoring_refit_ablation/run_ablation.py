#!/usr/bin/env python3
"""Ablate spherical RANSAC ranking and all-inlier Essential refit."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np
import scipy
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.translation_orientation.run_benchmark import (  # noqa: E402
    CONDITIONS,
    Condition,
    _direction_error,
    _options,
    _rotation_error,
    _scene,
)
from panorai.estimators import (  # noqa: E402
    RelativePoseAcceptancePolicy,
    estimate_relative_pose,
)

SCHEMA = "panorai-essential-scoring-refit-ablation/v1"
REFIT_CAP = 100
WEAK_OUTLIERS = Condition(
    "weak-parallax-with-outliers",
    0.25,
    400,
    5,
    (1.0, 3.0),
    (500.0, 2000.0),
    0.15,
    0.05,
)
ABLATION_CONDITIONS = (*CONDITIONS, WEAK_OUTLIERS)


@dataclass(frozen=True)
class Variant:
    name: str
    ranking: str
    refit: bool


VARIANTS = tuple(
    Variant(
        f"{ranking}-{'refit' if refit else 'no-refit'}",
        ranking,
        refit,
    )
    for ranking in ("count-first", "msac-first", "scale-marginal-first")
    for refit in (False, True)
)
BASELINE = "count-first-no-refit"


def _projective_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(min(np.linalg.norm(first - second), np.linalg.norm(first + second)))


def _case(condition: Condition, seed: int, variant: Variant) -> dict[str, Any]:
    scene = _scene(condition, seed)
    base = _options("parallax-weighted", seed)
    options = replace(
        base,
        hypothesis_ranking=variant.ranking,
        nonminimal_refit_max_steps=REFIT_CAP if variant.refit else 0,
        local_optimization_steps=25,
        robust_refinement_steps=2,
        refinement_max_nfev=100,
    )
    policy = RelativePoseAcceptancePolicy(require_stability=False)
    started = time.perf_counter_ns()
    result = estimate_relative_pose(
        scene["bearings_a"],
        scene["bearings_b"],
        options=options,
        quality_policy=policy,
    )
    runtime_ms = (time.perf_counter_ns() - started) / 1e6
    if result is None:
        return {
            "condition": condition.name,
            "seed": seed,
            "variant": variant.name,
            "ranking": variant.ranking,
            "refit": variant.refit,
            "returned": False,
            "accepted": False,
            "strict_correct": False,
            "rotation_error_deg": math.nan,
            "translation_error_deg": math.nan,
            "translation_axis_error_deg": math.nan,
            "essential_projective_distance": math.nan,
            "num_inliers": 0,
            "num_trials": 0,
            "consensus_refit_steps": 0,
            "refit_cap_reached": False,
            "normalized_msac_cost": math.nan,
            "runtime_ms": runtime_ms,
        }
    rotation_error = _rotation_error(result.R, scene["rotation"])
    translation_error = _direction_error(result.t, scene["translation"])
    axis_error = min(translation_error, 180.0 - translation_error)
    threshold = math.radians(options.max_angular_error_deg)
    finite = np.isfinite(result.residuals_rad)
    normalized = result.residuals_rad[finite] / threshold
    msac_cost = float(np.minimum(normalized * normalized, 1.0).sum())
    return {
        "condition": condition.name,
        "seed": seed,
        "variant": variant.name,
        "ranking": variant.ranking,
        "refit": variant.refit,
        "returned": True,
        "accepted": result.quality_report.accepted,
        "strict_correct": rotation_error < 5.0 and translation_error < 10.0,
        "rotation_error_deg": rotation_error,
        "translation_error_deg": translation_error,
        "translation_axis_error_deg": axis_error,
        "essential_projective_distance": _projective_distance(
            result.essential_matrix, scene["essential"]
        ),
        "num_inliers": result.num_inliers,
        "num_trials": result.num_trials,
        "consensus_refit_steps": result.consensus_refit_steps,
        "refit_cap_reached": result.consensus_refit_steps >= REFIT_CAP,
        "normalized_msac_cost": msac_cost,
        "runtime_ms": runtime_ms,
    }


def _finite(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if math.isfinite(float(row[key]))]


def _p95(values: list[float]) -> float:
    return float(np.quantile(values, 0.95, method="higher"))


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for condition in sorted({str(row["condition"]) for row in rows}):
        by_condition = [row for row in rows if row["condition"] == condition]
        baseline = {
            int(row["seed"]): row for row in by_condition if row["variant"] == BASELINE
        }
        variants: dict[str, Any] = {}
        for variant in VARIANTS:
            selected = [row for row in by_condition if row["variant"] == variant.name]
            rotation = _finite(selected, "rotation_error_deg")
            translation = _finite(selected, "translation_error_deg")
            axis = _finite(selected, "translation_axis_error_deg")
            runtime = _finite(selected, "runtime_ms")
            steps = _finite(selected, "consensus_refit_steps")
            gains = 0
            losses = 0
            rotation_delta = []
            translation_delta = []
            for row in selected:
                reference = baseline[int(row["seed"])]
                gains += bool(row["strict_correct"]) and not bool(
                    reference["strict_correct"]
                )
                losses += bool(reference["strict_correct"]) and not bool(
                    row["strict_correct"]
                )
                if math.isfinite(float(row["rotation_error_deg"])) and math.isfinite(
                    float(reference["rotation_error_deg"])
                ):
                    rotation_delta.append(
                        float(row["rotation_error_deg"])
                        - float(reference["rotation_error_deg"])
                    )
                    translation_delta.append(
                        float(row["translation_error_deg"])
                        - float(reference["translation_error_deg"])
                    )
            discordant = gains + losses
            variants[variant.name] = {
                "count": len(selected),
                "return_rate": sum(bool(row["returned"]) for row in selected)
                / len(selected),
                "strict_accuracy": sum(bool(row["strict_correct"]) for row in selected)
                / len(selected),
                "rotation_error_median_deg": float(np.median(rotation)),
                "rotation_error_p95_deg": _p95(rotation),
                "translation_error_median_deg": float(np.median(translation)),
                "translation_error_p95_deg": _p95(translation),
                "axis_error_median_deg": float(np.median(axis)),
                "essential_projective_distance_median": float(
                    np.median(_finite(selected, "essential_projective_distance"))
                ),
                "num_inliers_median": float(
                    np.median(_finite(selected, "num_inliers"))
                ),
                "runtime_median_ms": float(np.median(runtime)),
                "runtime_p95_ms": _p95(runtime),
                "refit_steps_median": float(np.median(steps)),
                "refit_steps_max": int(max(steps)),
                "refit_cap_rate": sum(
                    bool(row["refit_cap_reached"]) for row in selected
                )
                / len(selected),
                "strict_gains_vs_baseline": gains,
                "strict_losses_vs_baseline": losses,
                "strict_net_percentage_points": 100.0
                * (gains - losses)
                / len(selected),
                "two_sided_exact_mcnemar_p": (
                    float(binomtest(gains, discordant, 0.5).pvalue)
                    if discordant
                    else 1.0
                ),
                "paired_rotation_delta_median_deg": float(np.median(rotation_delta)),
                "paired_translation_delta_median_deg": float(
                    np.median(translation_delta)
                ),
            }
        output[condition] = variants
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    return subprocess.run(
        ("git", *args), check=True, capture_output=True, text=True
    ).stdout.strip()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else 1024 * value


def _report(payload: dict[str, Any]) -> str:
    lines = [
        "# Essential scoring and all-inlier refit ablation",
        "",
        "The table compares identical deterministic five-point proposal streams.",
        "Refit variants use an unweighted all-inlier SVD, calibrated Essential",
        "projection and a 100-step stabilization cap before bounded nonlinear",
        "pose refinement.",
        "",
        "| Condition | Variant | Strict | R median / P95 (deg) | t median / P95 (deg) | Refit median / max | Runtime median (ms) | Gains / losses |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition, variants in payload["results"].items():
        for name, value in variants.items():
            lines.append(
                f"| {condition} | {name} | {100 * value['strict_accuracy']:.1f}% | "
                f"{value['rotation_error_median_deg']:.3f} / "
                f"{value['rotation_error_p95_deg']:.3f} | "
                f"{value['translation_error_median_deg']:.3f} / "
                f"{value['translation_error_p95_deg']:.3f} | "
                f"{value['refit_steps_median']:.1f} / {value['refit_steps_max']} | "
                f"{value['runtime_median_ms']:.1f} | "
                f"{value['strict_gains_vs_baseline']} / "
                f"{value['strict_losses_vs_baseline']} |"
            )
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Schema: `{payload['schema']}`",
            f"- Commit: `{payload['source']['commit']}`",
            f"- Dirty source: `{payload['source']['dirty']}`",
            f"- Seed base: `{payload['configuration']['seed_base']}`",
            f"- Cases per condition: `{payload['configuration']['cases_per_condition']}`",
            f"- Python: `{payload['environment']['python']}`",
            f"- NumPy: `{payload['environment']['numpy']}`",
            f"- SciPy: `{payload['environment']['scipy']}`",
            f"- Platform: `{payload['environment']['platform']}`",
            f"- Peak process RSS: `{payload['measurements']['peak_rss_bytes']}` bytes",
            f"- Raw CSV SHA-256: `{payload['measurements']['raw_csv_sha256']}`",
            "",
            "## Limitations",
            "",
            "This is synthetic source-checkout evidence. The ranking and refit",
            "controls preserve the default behavior but are Experimental. A lower",
            "internal score is not a calibrated accuracy probability, and no result",
            "here establishes performance on real feature correspondences.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=1_100_000)
    args = parser.parse_args()
    if args.cases <= 0:
        parser.error("cases must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    _case(ABLATION_CONDITIONS[0], args.seed_base - 1, VARIANTS[0])
    rows = []
    for condition_index, condition in enumerate(ABLATION_CONDITIONS):
        for offset in range(args.cases):
            seed = args.seed_base + 10_000 * (condition_index + 1) + offset
            for variant in VARIANTS:
                rows.append(_case(condition, seed, variant))

    raw_path = args.output / "raw.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema": SCHEMA,
        "source": {
            "commit": _git_value("rev-parse", "HEAD"),
            "describe": _git_value("describe", "--tags", "--always", "--dirty"),
            "dirty": bool(_git_value("status", "--porcelain")),
        },
        "configuration": {
            "cases_per_condition": args.cases,
            "seed_base": args.seed_base,
            "development_seed_ranges": ["810000-series"],
            "conditions": [asdict(condition) for condition in ABLATION_CONDITIONS],
            "variants": [asdict(variant) for variant in VARIANTS],
            "refit_cap": REFIT_CAP,
            "local_optimization_cap": 25,
            "warmup_calls": 1,
            "p95": "nearest-rank via numpy quantile(method='higher')",
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "measurements": {
            "peak_rss_bytes": _peak_rss_bytes(),
            "peak_rss_scope": "complete process; not isolated per variant",
            "raw_csv_sha256": _sha256(raw_path),
        },
        "source_fingerprints": {
            "relative_pose.py": _sha256(ROOT / "panorai/estimators/relative_pose.py"),
            "run_ablation.py": _sha256(Path(__file__)),
            "scene_generator.py": _sha256(
                ROOT / "benchmarks/translation_orientation/run_benchmark.py"
            ),
        },
        "results": _aggregate(rows),
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "ABLATION_REPORT.md").write_text(_report(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
