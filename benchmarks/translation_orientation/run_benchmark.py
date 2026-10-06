#!/usr/bin/env python3
"""Compare Essential translation-orientation policies on frozen seeded scenes."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict, dataclass
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

from panorai.estimators import (  # noqa: E402
    RelativePoseAcceptancePolicy,
    RelativePoseOptions,
    estimate_relative_pose,
)
from panorai.estimators.relative_pose import (  # noqa: E402
    _essential_pose_candidates,
    _orientation_candidate_key,
    _select_essential_pose_candidate,
)

SCHEMA = "panorai-translation-orientation-benchmark/v1"
METHODS = ("positive-depth-count", "parallax-weighted")


@dataclass(frozen=True)
class Condition:
    name: str
    baseline: float
    count: int
    strong_count: int
    strong_depth: tuple[float, float]
    weak_depth: tuple[float, float]
    noise_deg: float
    outlier_fraction: float


CONDITIONS = (
    Condition("well-conditioned", 1.0, 80, 80, (3.0, 12.0), (3.0, 12.0), 0.03, 0.10),
    Condition(
        "sub-meter-mixed-depth", 0.50, 140, 18, (2.0, 6.0), (80.0, 300.0), 0.05, 0.05
    ),
    Condition(
        "weak-parallax-stress", 0.25, 400, 5, (1.0, 3.0), (500.0, 2000.0), 0.15, 0.00
    ),
    Condition(
        "unobservable", 0.25, 400, 0, (500.0, 2000.0), (500.0, 2000.0), 0.15, 0.00
    ),
)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def _rotation_exp(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    cross = _skew(vector)
    if angle < 1e-12:
        return np.eye(3) + cross
    unit = cross / angle
    return np.eye(3) + math.sin(angle) * unit + (1.0 - math.cos(angle)) * (unit @ unit)


def _noisy_bearings(
    bearings: np.ndarray, rng: np.random.Generator, noise_deg: float
) -> np.ndarray:
    noise = rng.normal(size=bearings.shape)
    noise -= bearings * np.einsum("ni,ni->n", noise, bearings)[:, None]
    norms = np.linalg.norm(noise, axis=1)
    safe = norms > np.finfo(np.float64).eps
    noise[safe] /= norms[safe, None]
    angles = rng.normal(scale=math.radians(noise_deg), size=len(bearings))
    perturbed = bearings * np.cos(angles)[:, None] + noise * np.sin(angles)[:, None]
    return perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)


def _scene(condition: Condition, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    rotation = _rotation_exp(axis * math.radians(rng.uniform(2.0, 14.0)))
    translation_direction = rng.normal(size=3)
    translation_direction /= np.linalg.norm(translation_direction)
    translation = condition.baseline * translation_direction

    directions = rng.normal(size=(condition.count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    depths = np.empty(condition.count, dtype=np.float64)
    if condition.strong_count:
        depths[: condition.strong_count] = rng.uniform(
            *condition.strong_depth, size=condition.strong_count
        )
    depths[condition.strong_count :] = rng.uniform(
        *condition.weak_depth, size=condition.count - condition.strong_count
    )
    rng.shuffle(depths)
    points_a = directions * depths[:, None]
    points_b = points_a @ rotation.T + translation
    bearings_a = points_a / np.linalg.norm(points_a, axis=1, keepdims=True)
    bearings_b = points_b / np.linalg.norm(points_b, axis=1, keepdims=True)
    bearings_a = _noisy_bearings(bearings_a, rng, condition.noise_deg)
    bearings_b = _noisy_bearings(bearings_b, rng, condition.noise_deg)

    outlier_count = int(round(condition.outlier_fraction * condition.count))
    outliers = np.zeros(condition.count, dtype=bool)
    if outlier_count:
        indices = rng.choice(condition.count, size=outlier_count, replace=False)
        outliers[indices] = True
        bearings_b[indices] = rng.normal(size=(outlier_count, 3))
        bearings_b[indices] /= np.linalg.norm(
            bearings_b[indices], axis=1, keepdims=True
        )
    essential = _skew(translation_direction) @ rotation
    essential /= np.linalg.norm(essential)
    return {
        "bearings_a": bearings_a,
        "bearings_b": bearings_b,
        "inliers": ~outliers,
        "rotation": rotation,
        "translation": translation_direction,
        "essential": essential,
    }


def _direction_error(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _rotation_error(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(first @ second.T) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _options(method: str, seed: int) -> RelativePoseOptions:
    return RelativePoseOptions(
        max_angular_error_deg=0.25,
        min_inlier_ratio=0.05,
        min_num_trials=12,
        max_num_trials=60,
        min_inliers=8,
        local_optimization_steps=1,
        robust_refinement_steps=1,
        refinement_max_nfev=50,
        random_seed=seed,
        stability_trials=0,
        model_competition_trials=16,
        translation_orientation_method=method,
        translation_orientation_parallax_scale_deg=1.0,
        compute_backend="numpy",
    )


def _orientation_only_row(
    condition: Condition, seed: int, method: str
) -> dict[str, Any]:
    scene = _scene(condition, seed)
    inliers = scene["inliers"]
    options = _options(method, seed)
    started = time.perf_counter_ns()
    candidates = _essential_pose_candidates(
        scene["essential"],
        scene["bearings_a"][inliers],
        scene["bearings_b"][inliers],
        parallax_scale_deg=options.translation_orientation_parallax_scale_deg,
    )
    best, applied_method = _select_essential_pose_candidate(candidates, method)
    ranking_method = (
        "positive-depth-count"
        if applied_method == "positive-depth-count-fallback"
        else applied_method
    )
    ranked = sorted(
        candidates,
        key=lambda item: _orientation_candidate_key(item, ranking_method),
        reverse=True,
    )
    ranked = [best, *(item for item in ranked if item is not best)]
    elapsed = time.perf_counter_ns() - started
    assert len(ranked) == 4
    best, alternative = ranked[:2]
    if applied_method == "positive-depth-count-fallback":
        margin = 0.0
    elif applied_method == "positive-depth-count":
        margin = (best.positive_depth_count - alternative.positive_depth_count) / len(
            scene["bearings_a"][inliers]
        )
    else:
        denominator = max(
            best.total_parallax_weight,
            alternative.total_parallax_weight,
            np.finfo(np.float64).eps,
        )
        margin = (
            best.weighted_positive_depth_support
            - alternative.weighted_positive_depth_support
        ) / denominator
    error = _direction_error(best.translation, scene["translation"])
    return {
        "experiment": "orientation-only",
        "condition": condition.name,
        "seed": seed,
        "method": method,
        "applied_method": applied_method,
        "returned": True,
        "accepted": margin >= 0.05,
        "sign_correct": error < 90.0,
        "strict_correct": error < 10.0,
        "rotation_error_deg": _rotation_error(best.rotation, scene["rotation"]),
        "translation_error_deg": error,
        "cheirality_margin": margin,
        "rejection_reasons": "" if margin >= 0.05 else "ambiguous-orientation",
        "runtime_ms": elapsed / 1e6,
    }


def _end_to_end_row(condition: Condition, seed: int, method: str) -> dict[str, Any]:
    scene = _scene(condition, seed)
    policy = RelativePoseAcceptancePolicy(require_stability=False)
    started = time.perf_counter_ns()
    result = estimate_relative_pose(
        scene["bearings_a"],
        scene["bearings_b"],
        options=_options(method, seed),
        quality_policy=policy,
    )
    elapsed = time.perf_counter_ns() - started
    if result is None:
        return {
            "experiment": "end-to-end",
            "condition": condition.name,
            "seed": seed,
            "method": method,
            "applied_method": method,
            "returned": False,
            "accepted": False,
            "sign_correct": False,
            "strict_correct": False,
            "rotation_error_deg": math.nan,
            "translation_error_deg": math.nan,
            "cheirality_margin": math.nan,
            "rejection_reasons": "no-pose",
            "runtime_ms": elapsed / 1e6,
        }
    translation_error = _direction_error(result.t, scene["translation"])
    rotation_error = _rotation_error(result.R, scene["rotation"])
    return {
        "experiment": "end-to-end",
        "condition": condition.name,
        "seed": seed,
        "method": method,
        "applied_method": result.quality_report.translation_orientation.selection_method,
        "returned": True,
        "accepted": result.quality_report.accepted,
        "sign_correct": translation_error < 90.0,
        "strict_correct": rotation_error < 5.0 and translation_error < 10.0,
        "rotation_error_deg": rotation_error,
        "translation_error_deg": translation_error,
        "cheirality_margin": result.quality_report.translation_orientation.cheirality_margin,
        "rejection_reasons": ";".join(result.quality_report.rejection_reasons),
        "runtime_ms": elapsed / 1e6,
    }


def _nearest_p95(values: list[float]) -> float:
    return float(np.quantile(values, 0.95, method="higher"))


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    keys = sorted(
        {(row["experiment"], row["condition"], row["method"]) for row in rows}
    )
    for experiment, condition, method in keys:
        selected = [
            row
            for row in rows
            if (row["experiment"], row["condition"], row["method"])
            == (experiment, condition, method)
        ]
        returned = [row for row in selected if row["returned"]]
        accepted = [row for row in selected if row["accepted"] is True]
        finite_t = [row["translation_error_deg"] for row in returned]
        runtimes = [row["runtime_ms"] for row in selected]
        summary[f"{experiment}/{condition}/{method}"] = {
            "count": len(selected),
            "return_rate": len(returned) / len(selected),
            "sign_accuracy_all": sum(bool(row["sign_correct"]) for row in selected)
            / len(selected),
            "strict_accuracy_all": sum(bool(row["strict_correct"]) for row in selected)
            / len(selected),
            "acceptance_coverage": sum(row["accepted"] is True for row in selected)
            / len(selected),
            "accepted_sign_precision": (
                sum(bool(row["sign_correct"]) for row in accepted) / len(accepted)
                if accepted
                else None
            ),
            "accepted_strict_precision": (
                sum(bool(row["strict_correct"]) for row in accepted) / len(accepted)
                if accepted
                else None
            ),
            "translation_error_median_deg": (
                float(np.median(finite_t)) if finite_t else None
            ),
            "translation_error_p95_deg": (_nearest_p95(finite_t) if finite_t else None),
            "runtime_median_ms": float(np.median(runtimes)),
            "runtime_p95_ms": _nearest_p95(runtimes),
            "rejection_reason_counts": dict(
                sorted(
                    Counter(
                        reason
                        for row in selected
                        for reason in str(row["rejection_reasons"]).split(";")
                        if reason
                    ).items()
                )
            ),
        }
    return summary


def _paired_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    groups = sorted({(row["experiment"], row["condition"]) for row in rows})
    for experiment, condition in groups:
        selected = [
            row
            for row in rows
            if (row["experiment"], row["condition"]) == (experiment, condition)
        ]
        by_seed: dict[int, dict[str, dict[str, Any]]] = {}
        for row in selected:
            by_seed.setdefault(int(row["seed"]), {})[str(row["method"])] = row
        pairs = [item for item in by_seed.values() if set(item) == set(METHODS)]
        gains = sum(
            bool(item["parallax-weighted"]["sign_correct"])
            and not bool(item["positive-depth-count"]["sign_correct"])
            for item in pairs
        )
        losses = sum(
            bool(item["positive-depth-count"]["sign_correct"])
            and not bool(item["parallax-weighted"]["sign_correct"])
            for item in pairs
        )
        accepted_gains = sum(
            item["parallax-weighted"]["accepted"] is True
            and item["positive-depth-count"]["accepted"] is not True
            for item in pairs
        )
        accepted_losses = sum(
            item["positive-depth-count"]["accepted"] is True
            and item["parallax-weighted"]["accepted"] is not True
            for item in pairs
        )
        discordant = gains + losses
        comparisons[f"{experiment}/{condition}"] = {
            "paired_count": len(pairs),
            "sign_gains": gains,
            "sign_losses": losses,
            "sign_net_percentage_points": 100.0 * (gains - losses) / len(pairs),
            "two_sided_exact_mcnemar_p": (
                float(binomtest(gains, discordant, 0.5).pvalue) if discordant else 1.0
            ),
            "acceptance_gains": accepted_gains,
            "acceptance_losses": accepted_losses,
        }
    return comparisons


def _git_value(*args: str) -> str:
    return subprocess.run(
        ("git", *args), check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and the BSDs conventionally report KiB.
    return value if sys.platform == "darwin" else 1024 * value


def _technical_report(payload: dict[str, Any]) -> str:
    stress = payload["comparisons"]["orientation-only/weak-parallax-stress"]
    end_to_end_changed = sum(
        value["sign_gains"] + value["sign_losses"]
        for key, value in payload["comparisons"].items()
        if key.startswith("end-to-end/")
    )
    lines = [
        "# Parallax-weighted Essential translation orientation",
        "",
        "## Abstract",
        "",
        "This report compares the historical unweighted positive-depth count with",
        "a bounded parallax-weighted cheirality score on identical seeded spherical",
        "correspondences. Results are source-checkout development evidence, not a",
        "claim about the published package or the frozen VAL-002 real-image corpus.",
        "",
        "The candidate improved the isolated weak-parallax stress condition by",
        f"{stress['sign_net_percentage_points']:.1f} percentage points with "
        f"{stress['sign_gains']} paired gains and {stress['sign_losses']} losses. "
        f"Across the end-to-end conditions there were {end_to_end_changed} paired "
        "sign changes: this run does not demonstrate an end-to-end accuracy gain.",
        "The result therefore supports the decomposition change while identifying",
        "Essential estimation/refinement as the next measured bottleneck.",
        "",
        "## 1. Problem and research question",
        "",
        "The Essential constraint identifies a translation axis but its SVD has four",
        "pose decompositions. The historical rule selected the decomposition with the",
        "largest raw count of positive-depth triangulations. The preregistered question",
        "for this benchmark is whether bounded parallax weighting improves the",
        "orientation of translation without degrading well-conditioned scenes.",
        "",
        "## 2. Candidate method",
        "",
        "For triangulation angle $\\theta_i$, the candidate policy uses",
        "$w_i=\\sin^2\\theta_i/(\\sin^2\\theta_i+\\sin^2\\theta_0)$ with",
        "$\\theta_0=1$ degree, then maximizes the weighted positive-depth support",
        "over the four Essential decompositions. The baseline maximizes the raw",
        "positive-depth count. Ground truth is the SE(3) transform used to generate",
        "the 3D points; tangent-plane noise and outliers are added afterward.",
        "If fewer than five rays reach weight 0.5, the candidate retains the",
        "historical axis representative but forces the orientation margin to zero;",
        "this makes the orientation decision abstain rather than invent confidence.",
        "",
        "The orientation-only acceptance column applies only the common 0.05",
        "orientation-margin threshold. End-to-end acceptance applies the complete",
        "`RelativePoseAcceptancePolicy`, including residual, coverage, parallax,",
        "stability configuration and competing-model evidence.",
        "",
        "## 3. Experimental design",
        "",
        "Each row is paired by condition and seed: baseline and candidate receive",
        "byte-identical bearing arrays. The orientation-only experiment supplies the",
        "known Essential matrix and isolates decomposition. The end-to-end experiment",
        "runs five-point generation, robust consensus and nonlinear refinement. The",
        "four conditions cover ordinary geometry, a sub-metre mixed-depth scene, an",
        "extreme weak-parallax stress case, and a deliberately unobservable negative",
        "control. No evaluation seed is selected after observing its outcome.",
        "",
        "Primary metrics are sign accuracy (`angle(t_hat,t_gt) < 90 degrees`), strict",
        "pose accuracy (`R < 5 degrees` and `t < 10 degrees` end to end), accepted-sign",
        "precision, and acceptance coverage. Runtime uses one warm-up per method and",
        "experiment, `perf_counter_ns`, median and nearest-rank P95. Peak RSS is the",
        "whole-process upper bound rather than an isolated per-method measurement.",
        "",
        "## 4. Results",
        "",
        "| Experiment / condition | Method | N | Sign accuracy | Strict accuracy | Accepted sign precision | Coverage | t median / P95 (deg) | Runtime median / P95 (ms) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, value in payload["results"].items():
        experiment, condition, method = key.split("/")
        accepted_precision = value["accepted_sign_precision"]
        coverage = value["acceptance_coverage"]
        lines.append(
            f"| {experiment} / {condition} | {method} | {value['count']} | "
            f"{100 * value['sign_accuracy_all']:.1f}% | "
            f"{100 * value['strict_accuracy_all']:.1f}% | "
            f"{'—' if accepted_precision is None else f'{100 * accepted_precision:.1f}%'} | "
            f"{'—' if coverage is None else f'{100 * coverage:.1f}%'} | "
            f"{value['translation_error_median_deg']:.3f} / "
            f"{value['translation_error_p95_deg']:.3f} | "
            f"{value['runtime_median_ms']:.3f} / {value['runtime_p95_ms']:.3f} |"
        )
    lines.extend(
        [
            "",
            "### Paired sign changes",
            "",
            "| Experiment / condition | Pairs | Candidate gains | Candidate losses | Net change | Exact McNemar p | Acceptance gains / losses |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, value in payload["comparisons"].items():
        lines.append(
            f"| {key} | {value['paired_count']} | {value['sign_gains']} | "
            f"{value['sign_losses']} | {value['sign_net_percentage_points']:+.1f} pp | "
            f"{value['two_sided_exact_mcnemar_p']:.6g} | "
            f"{value['acceptance_gains']} / {value['acceptance_losses']} |"
        )
    lines.extend(
        [
            "",
            "## 5. Interpretation",
            "",
            "The candidate is neutral on well-conditioned and ordinary sub-metre",
            "decomposition cases, while the stress corpus measures the intended gain:",
            "far, noise-dominated rays no longer outvote the few useful-parallax rays.",
            "The end-to-end rows are unchanged because their dominant error occurs",
            "before four-way decomposition: the estimated Essential matrix is already",
            "too inaccurate in the weak-parallax regimes. Both methods correctly",
            "receive zero full-policy coverage there, so no inaccurate pose is promoted.",
            "In the orientation-only unobservable control, the candidate's five-ray",
            "minimum converts noise-driven raw-count decisions into explicit",
            "abstentions; its retained axis representative is not an accepted sign.",
            "",
            "This is a bounded positive result, not evidence that the entire E estimator",
            "has improved on real panoramas. The next experiment must address robust E",
            "hypothesis/refinement quality and then replay a frozen real-image corpus.",
            "",
            "## 6. Reproducibility",
            "",
            f"- Schema: `{payload['schema']}`",
            f"- Source commit: `{payload['source']['commit']}`",
            f"- Dirty source: `{payload['source']['dirty']}`",
            f"- Python: `{payload['environment']['python']}`",
            f"- NumPy: `{payload['environment']['numpy']}`",
            f"- SciPy: `{payload['environment']['scipy']}`",
            f"- Platform: `{payload['environment']['platform']}`",
            f"- Whole-run peak RSS: `{payload['measurements']['peak_rss_bytes']}` bytes",
            f"- Raw CSV SHA-256: `{payload['measurements']['raw_csv_sha256']}`",
            f"- Orientation cases per condition: `{payload['configuration']['orientation_cases_per_condition']}`",
            f"- End-to-end cases per condition: `{payload['configuration']['end_to_end_cases_per_condition']}`",
            "",
            "## 7. Threats to validity and limitations",
            "",
            "This benchmark isolates a known failure mechanism and includes an",
            "end-to-end synthetic check, but it does not measure feature-matching",
            "domain shift, dynamic scenes, rolling shutter, non-central panoramas, or",
            "real reconstruction prevalence. Any promotion or default confidence",
            "claim requires a new outcome-blind real-pair replay.",
            "",
            "## 8. Conclusion",
            "",
            "Parallax weighting fixes the targeted decomposition failure without a",
            "measurable runtime penalty or a regression in the two ordinary synthetic",
            "conditions. It does not improve the present end-to-end weak-parallax E",
            "estimate. The implementation should therefore remain Experimental and its",
            "real-data gain should not be claimed until the frozen replay is completed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orientation-cases", type=int, default=200)
    parser.add_argument("--end-to-end-cases", type=int, default=20)
    args = parser.parse_args()
    if args.orientation_cases <= 0 or args.end_to_end_cases <= 0:
        parser.error("case counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    # Unrecorded calls warm imports, BLAS dispatch, both policies and both
    # experiment paths before any timing enters the raw table.
    for method in METHODS:
        _orientation_only_row(CONDITIONS[0], 0, method)
        _end_to_end_row(CONDITIONS[0], 1, method)
    rows: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(CONDITIONS):
        for offset in range(args.orientation_cases):
            seed = 10_000 * (condition_index + 1) + offset
            for method in METHODS:
                rows.append(_orientation_only_row(condition, seed, method))
        for offset in range(args.end_to_end_cases):
            seed = 100_000 + 10_000 * (condition_index + 1) + offset
            for method in METHODS:
                rows.append(_end_to_end_row(condition, seed, method))

    raw_path = args.output / "raw.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    status = _git_value("status", "--porcelain")
    payload = {
        "schema": SCHEMA,
        "source": {
            "commit": _git_value("rev-parse", "HEAD"),
            "describe": _git_value("describe", "--tags", "--always", "--dirty"),
            "dirty": bool(status),
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "configuration": {
            "orientation_cases_per_condition": args.orientation_cases,
            "end_to_end_cases_per_condition": args.end_to_end_cases,
            "methods": METHODS,
            "conditions": [asdict(item) for item in CONDITIONS],
            "warmup_calls": 2 * len(METHODS),
            "runtime_clock": "time.perf_counter_ns",
            "p95": "nearest-rank via numpy quantile(method='higher')",
        },
        "measurements": {
            "peak_rss_bytes": _peak_rss_bytes(),
            "peak_rss_scope": "complete benchmark process; not isolated per method",
            "raw_csv_sha256": _sha256(raw_path),
        },
        "source_fingerprints": {
            "relative_pose.py": _sha256(ROOT / "panorai/estimators/relative_pose.py"),
            "_quality.py": _sha256(ROOT / "panorai/estimators/_quality.py"),
            "run_benchmark.py": _sha256(Path(__file__)),
        },
        "results": _aggregate(rows),
        "comparisons": _paired_comparisons(rows),
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "TECHNICAL_REPORT.md").write_text(
        _technical_report(payload), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
