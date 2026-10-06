#!/usr/bin/env python3
"""Benchmark decoupled spherical rotation and translation estimation."""

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.essential_scoring_refit_ablation.run_ablation import (  # noqa: E402
    WEAK_OUTLIERS,
)
from benchmarks.translation_orientation.run_benchmark import (  # noqa: E402
    CONDITIONS,
    _direction_error,
    _options,
    _rotation_error,
    _scene,
)
from panorai.estimators import (  # noqa: E402
    RelativePoseAcceptancePolicy,
    estimate_relative_pose,
)

SCHEMA = "panorai-decoupled-rotation-translation-benchmark/v1"
BENCHMARK_CONDITIONS = (*CONDITIONS, WEAK_OUTLIERS)
BASELINE = "count-first"


@dataclass(frozen=True)
class Variant:
    name: str
    ranking: str = "count-first"
    decoupled: bool = False
    margin: float = 0.15


VARIANTS = (
    Variant(BASELINE),
    Variant("msac-first", ranking="msac-first"),
    Variant("decoupled-unguarded", decoupled=True, margin=0.0),
    Variant("decoupled-guarded", decoupled=True, margin=0.15),
)


def _policy() -> RelativePoseAcceptancePolicy:
    """Permit every numerically valid pose except an explicit GEO-015 abstention."""

    return RelativePoseAcceptancePolicy(
        min_inliers=5,
        min_inlier_ratio=0.0,
        min_occupied_cells=1,
        min_coverage_entropy=0.0,
        max_median_residual_deg=90.0,
        min_median_parallax_deg=0.0,
        min_cheirality_ratio=0.0,
        min_translation_orientation_margin=0.0,
        max_stability_rotation_p90_deg=180.0,
        max_stability_translation_p90_deg=180.0,
        min_essential_score_margin=-1.0,
        require_essential_preferred=False,
        require_stability=False,
    )


def _case(condition: Any, seed: int, variant: Variant) -> dict[str, Any]:
    scene = _scene(condition, seed)
    options = replace(
        _options("parallax-weighted", seed),
        hypothesis_ranking=variant.ranking,
        pose_refinement_method="decoupled" if variant.decoupled else "joint",
        decoupled_translation_min_score_margin=variant.margin,
    )
    started = time.perf_counter_ns()
    result = estimate_relative_pose(
        scene["bearings_a"],
        scene["bearings_b"],
        options=options,
        quality_policy=_policy(),
    )
    runtime_ms = (time.perf_counter_ns() - started) / 1e6
    if result is None:
        return {
            "condition": condition.name,
            "seed": seed,
            "variant": variant.name,
            "returned": False,
            "accepted": False,
            "strict_correct": False,
            "high_precision_correct": False,
            "rotation_error_deg": math.nan,
            "translation_error_deg": math.nan,
            "num_inliers": 0,
            "decoupled_applied": False,
            "decoupled_reason": "not-returned",
            "rotation_consensus_ratio": math.nan,
            "translation_pool_size": 0,
            "translation_consensus_size": 0,
            "translation_score_margin": math.nan,
            "runtime_ms": runtime_ms,
        }
    rotation_error = _rotation_error(result.R, scene["rotation"])
    translation_error = _direction_error(result.t, scene["translation"])
    report = result.decoupled_pose_report
    return {
        "condition": condition.name,
        "seed": seed,
        "variant": variant.name,
        "returned": True,
        "accepted": result.quality_report.accepted,
        "strict_correct": rotation_error < 5.0 and translation_error < 10.0,
        "high_precision_correct": rotation_error < 0.1 and translation_error < 5.0,
        "rotation_error_deg": rotation_error,
        "translation_error_deg": translation_error,
        "num_inliers": result.num_inliers,
        "decoupled_applied": bool(report and report.applied),
        "decoupled_reason": "not-requested" if report is None else report.reason,
        "rotation_consensus_ratio": (
            math.nan if report is None else report.rotation_inlier_ratio
        ),
        "translation_pool_size": 0 if report is None else report.translation_pool_size,
        "translation_consensus_size": (
            0 if report is None else report.translation_consensus_size
        ),
        "translation_score_margin": (
            math.nan if report is None else report.translation_score_margin
        ),
        "runtime_ms": runtime_ms,
    }


def _finite(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if math.isfinite(float(row[key]))]


def _percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(values, quantile, method="higher"))


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for condition in sorted({str(row["condition"]) for row in rows}):
        selected_condition = [row for row in rows if row["condition"] == condition]
        baseline = {
            int(row["seed"]): row
            for row in selected_condition
            if row["variant"] == BASELINE
        }
        variants = {}
        for variant in VARIANTS:
            selected = [
                row for row in selected_condition if row["variant"] == variant.name
            ]
            accepted = [row for row in selected if row["accepted"]]
            rotation = _finite(selected, "rotation_error_deg")
            translation = _finite(selected, "translation_error_deg")
            accepted_rotation = _finite(accepted, "rotation_error_deg")
            accepted_translation = _finite(accepted, "translation_error_deg")
            gains = losses = 0
            for row in selected:
                reference = baseline[int(row["seed"])]
                gains += bool(row["strict_correct"]) and not bool(
                    reference["strict_correct"]
                )
                losses += bool(reference["strict_correct"]) and not bool(
                    row["strict_correct"]
                )
            reasons: dict[str, int] = {}
            for row in selected:
                reason = str(row["decoupled_reason"])
                reasons[reason] = reasons.get(reason, 0) + 1
            variants[variant.name] = {
                "count": len(selected),
                "return_rate": sum(bool(row["returned"]) for row in selected)
                / len(selected),
                "accepted_rate": len(accepted) / len(selected),
                "strict_accuracy": sum(bool(row["strict_correct"]) for row in selected)
                / len(selected),
                "high_precision_accuracy": sum(
                    bool(row["high_precision_correct"]) for row in selected
                )
                / len(selected),
                "accepted_strict_precision": (
                    sum(bool(row["strict_correct"]) for row in accepted) / len(accepted)
                    if accepted
                    else math.nan
                ),
                "accepted_high_precision": (
                    sum(bool(row["high_precision_correct"]) for row in accepted)
                    / len(accepted)
                    if accepted
                    else math.nan
                ),
                "false_accept_rate": sum(
                    bool(row["accepted"]) and not bool(row["strict_correct"])
                    for row in selected
                )
                / len(selected),
                "rotation_error_median_deg": float(np.median(rotation)),
                "rotation_error_p95_deg": _percentile(rotation, 0.95),
                "translation_error_median_deg": float(np.median(translation)),
                "translation_error_p95_deg": _percentile(translation, 0.95),
                "accepted_rotation_error_median_deg": (
                    float(np.median(accepted_rotation))
                    if accepted_rotation
                    else math.nan
                ),
                "accepted_rotation_error_p95_deg": (
                    _percentile(accepted_rotation, 0.95)
                    if accepted_rotation
                    else math.nan
                ),
                "accepted_translation_error_median_deg": (
                    float(np.median(accepted_translation))
                    if accepted_translation
                    else math.nan
                ),
                "accepted_translation_error_p95_deg": (
                    _percentile(accepted_translation, 0.95)
                    if accepted_translation
                    else math.nan
                ),
                "runtime_median_ms": float(np.median(_finite(selected, "runtime_ms"))),
                "runtime_p95_ms": _percentile(_finite(selected, "runtime_ms"), 0.95),
                "decoupled_applied_rate": sum(
                    bool(row["decoupled_applied"]) for row in selected
                )
                / len(selected),
                "strict_gains_vs_baseline": gains,
                "strict_losses_vs_baseline": losses,
                "reasons": reasons,
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


def _report(payload: dict[str, Any]) -> str:
    lines = [
        "# Decoupled rotation/translation benchmark",
        "",
        "| Condition | Variant | Accepted | Strict | High precision | Accepted R med / P95 | Accepted t med / P95 | False accept | Runtime med (ms) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition, variants in payload["results"].items():
        for name, value in variants.items():
            lines.append(
                f"| {condition} | {name} | {100 * value['accepted_rate']:.1f}% | "
                f"{100 * value['strict_accuracy']:.1f}% | "
                f"{100 * value['high_precision_accuracy']:.1f}% | "
                f"{value['accepted_rotation_error_median_deg']:.3f} / "
                f"{value['accepted_rotation_error_p95_deg']:.3f} | "
                f"{value['accepted_translation_error_median_deg']:.3f} / "
                f"{value['accepted_translation_error_p95_deg']:.3f} | "
                f"{100 * value['false_accept_rate']:.1f}% | "
                f"{value['runtime_median_ms']:.1f} |"
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
            f"- Raw CSV SHA-256: `{payload['measurements']['raw_csv_sha256']}`",
            f"- Peak process RSS: `{payload['measurements']['peak_rss_bytes']}` bytes",
            "",
            "## Limitations",
            "",
            "Synthetic source-checkout evidence only. The permissive benchmark policy",
            "isolates GEO-015 abstention; it is not the default public acceptance policy.",
            "No result establishes performance on real feature correspondences.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=1_400_000)
    args = parser.parse_args()
    if args.cases <= 0:
        parser.error("cases must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    _case(BENCHMARK_CONDITIONS[0], args.seed_base - 1, VARIANTS[0])
    rows = []
    for condition_index, condition in enumerate(BENCHMARK_CONDITIONS):
        for offset in range(args.cases):
            seed = args.seed_base + 10_000 * (condition_index + 1) + offset
            for variant in VARIANTS:
                rows.append(_case(condition, seed, variant))

    raw_path = args.output / "raw.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        peak_rss *= 1024
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
            "development_seed_upper_bound_exclusive": 1_300_000,
            "conditions": [asdict(item) for item in BENCHMARK_CONDITIONS],
            "variants": [asdict(item) for item in VARIANTS],
            "strict_correct": "R < 5 deg and oriented t < 10 deg",
            "high_precision_correct": "R < 0.1 deg and oriented t < 5 deg",
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "measurements": {
            "peak_rss_bytes": peak_rss,
            "peak_rss_scope": "complete process; not isolated per variant",
            "raw_csv_sha256": _sha256(raw_path),
        },
        "source_fingerprints": {
            "relative_pose.py": _sha256(ROOT / "panorai/estimators/relative_pose.py"),
            "run_benchmark.py": _sha256(Path(__file__)),
            "scene_generator.py": _sha256(
                ROOT / "benchmarks/translation_orientation/run_benchmark.py"
            ),
        },
        "results": _aggregate(rows),
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "BENCHMARK_REPORT.md").write_text(_report(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
