#!/usr/bin/env python3
"""Map boundary conditions for the public spherical relative-pose estimator."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panorai.estimators import (  # noqa: E402
    RelativePoseAcceptancePolicy,
    RelativePoseOptions,
    SpatiallyWeightedFivePointSampler,
    estimate_relative_pose,
)

SCHEMA = "panorai-relative-pose-operating-envelope/v1"
STRICT_R_DEG = 5.0
STRICT_T_DEG = 10.0
PRECISE_R_DEG = 1.0
PRECISE_T_DEG = 5.0


@dataclass(frozen=True, slots=True)
class Condition:
    family: str
    name: str
    count: int
    target_parallax_deg: float
    inlier_ratio: float
    noise_deg: float
    coverage: str
    observable_translation: bool = True


PARALLAX_VALUES = (0.10, 0.25, 0.50, 1.0, 2.0, 5.0)
INLIER_VALUES = (0.25, 0.40, 0.55, 0.70, 0.85, 1.0)
COUNT_VALUES = (12, 20, 40, 80, 160)
COVERAGE_VALUES = ("cap-15", "equatorial-band-10", "full-sphere")
NOISE_VALUES = (0.02, 0.10, 0.25, 0.50)


def development_conditions() -> tuple[Condition, ...]:
    """Return the preregistered condition grid in deterministic order."""

    conditions: list[Condition] = []
    for parallax in PARALLAX_VALUES:
        for inlier_ratio in INLIER_VALUES:
            conditions.append(
                Condition(
                    family="parallax-vs-inliers",
                    name=f"p{parallax:g}-i{inlier_ratio:g}",
                    count=80,
                    target_parallax_deg=parallax,
                    inlier_ratio=inlier_ratio,
                    noise_deg=0.10,
                    coverage="full-sphere",
                )
            )
    for count in COUNT_VALUES:
        for coverage in COVERAGE_VALUES:
            conditions.append(
                Condition(
                    family="count-vs-coverage",
                    name=f"n{count}-{coverage}",
                    count=count,
                    target_parallax_deg=1.0,
                    inlier_ratio=0.70,
                    noise_deg=0.10,
                    coverage=coverage,
                )
            )
    for noise in NOISE_VALUES:
        for parallax in (0.25, 0.50, 1.0, 2.0):
            conditions.append(
                Condition(
                    family="noise-vs-parallax",
                    name=f"noise{noise:g}-p{parallax:g}",
                    count=80,
                    target_parallax_deg=parallax,
                    inlier_ratio=0.70,
                    noise_deg=noise,
                    coverage="full-sphere",
                )
            )
    conditions.extend(
        (
            Condition(
                family="negative-control",
                name="pure-rotation",
                count=80,
                target_parallax_deg=0.0,
                inlier_ratio=1.0,
                noise_deg=0.10,
                coverage="full-sphere",
                observable_translation=False,
            ),
            Condition(
                family="negative-control",
                name="all-outliers",
                count=80,
                target_parallax_deg=1.0,
                inlier_ratio=0.0,
                noise_deg=0.10,
                coverage="full-sphere",
                observable_translation=False,
            ),
            Condition(
                family="negative-control",
                name="great-circle-band",
                count=80,
                target_parallax_deg=1.0,
                inlier_ratio=0.70,
                noise_deg=0.10,
                coverage="equatorial-band-1",
            ),
            Condition(
                family="negative-control",
                name="clustered-cap",
                count=80,
                target_parallax_deg=1.0,
                inlier_ratio=0.70,
                noise_deg=0.10,
                coverage="cap-5",
            ),
        )
    )
    return tuple(conditions)


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


def _unit_rows(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _directions(count: int, coverage: str, rng: np.random.Generator) -> np.ndarray:
    if coverage == "full-sphere":
        return _unit_rows(rng.normal(size=(count, 3)))
    if coverage.startswith("cap-"):
        half_angle = math.radians(float(coverage.split("-", 1)[1]))
        cos_theta = rng.uniform(math.cos(half_angle), 1.0, size=count)
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        phi = rng.uniform(-math.pi, math.pi, size=count)
        local = np.column_stack(
            (sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta)
        )
        center = _unit_rows(rng.normal(size=(1, 3)))[0]
        z_axis = np.asarray((0.0, 0.0, 1.0))
        cross = np.cross(z_axis, center)
        dot = float(np.clip(np.dot(z_axis, center), -1.0, 1.0))
        if np.linalg.norm(cross) < 1e-12:
            rotation = np.eye(3) if dot > 0.0 else np.diag((1.0, -1.0, -1.0))
        else:
            rotation = _rotation_exp(cross / np.linalg.norm(cross) * math.acos(dot))
        return local @ rotation.T
    if coverage.startswith("equatorial-band-"):
        half_width = math.radians(float(coverage.rsplit("-", 1)[1]))
        longitude = rng.uniform(-math.pi, math.pi, size=count)
        latitude = rng.uniform(-half_width, half_width, size=count)
        cos_latitude = np.cos(latitude)
        return np.column_stack(
            (
                np.sin(longitude) * cos_latitude,
                np.sin(latitude),
                np.cos(longitude) * cos_latitude,
            )
        )
    raise ValueError(f"unknown coverage mode {coverage!r}")


def _parallax_deg(
    points_a: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    bearing_a = _unit_rows(points_a)
    bearing_b = _unit_rows(points_a @ rotation.T + translation)
    derotated_a = bearing_a @ rotation.T
    cosine = np.clip(np.einsum("ni,ni->n", derotated_a, bearing_b), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _depth_scale_for_parallax(
    directions: np.ndarray,
    relative_depths: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    target_deg: float,
) -> float:
    if target_deg <= 0.0 or np.linalg.norm(translation) == 0.0:
        return 100.0

    def median_for(scale: float) -> float:
        points = directions * (relative_depths * scale)[:, None]
        return float(np.median(_parallax_deg(points, rotation, translation)))

    low, high = 0.2, 5000.0
    if median_for(low) < target_deg or median_for(high) > target_deg:
        raise RuntimeError("target parallax is outside the synthetic depth bracket")
    for _ in range(64):
        middle = math.sqrt(low * high)
        if median_for(middle) > target_deg:
            low = middle
        else:
            high = middle
    return math.sqrt(low * high)


def _add_tangent_noise(
    bearings: np.ndarray, rng: np.random.Generator, noise_deg: float
) -> np.ndarray:
    if noise_deg == 0.0:
        return bearings.copy()
    tangent = rng.normal(size=bearings.shape)
    tangent -= bearings * np.einsum("ni,ni->n", tangent, bearings)[:, None]
    tangent = _unit_rows(tangent)
    angle = rng.normal(scale=math.radians(noise_deg), size=len(bearings))
    result = bearings * np.cos(angle)[:, None] + tangent * np.sin(angle)[:, None]
    return _unit_rows(result)


def generate_scene(condition: Condition, seed: int) -> dict[str, Any]:
    """Generate bearings from known SE(3), then inject noise and outliers."""

    rng = np.random.default_rng(seed)
    axis = _unit_rows(rng.normal(size=(1, 3)))[0]
    rotation = _rotation_exp(axis * math.radians(rng.uniform(2.0, 14.0)))
    translation_direction = _unit_rows(rng.normal(size=(1, 3)))[0]
    translation = (
        translation_direction.copy()
        if condition.observable_translation or condition.name == "all-outliers"
        else np.zeros(3)
    )
    directions = _directions(condition.count, condition.coverage, rng)
    relative_depths = rng.uniform(0.8, 1.2, size=condition.count)
    depth_scale = _depth_scale_for_parallax(
        directions,
        relative_depths,
        rotation,
        translation,
        condition.target_parallax_deg,
    )
    points_a = directions * (relative_depths * depth_scale)[:, None]
    points_b = points_a @ rotation.T + translation
    clean_a = _unit_rows(points_a)
    clean_b = _unit_rows(points_b)
    clean_parallax = _parallax_deg(points_a, rotation, translation)
    bearings_a = _add_tangent_noise(clean_a, rng, condition.noise_deg)
    bearings_b = _add_tangent_noise(clean_b, rng, condition.noise_deg)

    inlier_count = int(round(condition.inlier_ratio * condition.count))
    inlier_mask = np.zeros(condition.count, dtype=bool)
    if inlier_count:
        inlier_mask[rng.choice(condition.count, size=inlier_count, replace=False)] = (
            True
        )
    outlier_mask = ~inlier_mask
    if np.any(outlier_mask):
        bearings_b[outlier_mask] = _unit_rows(
            rng.normal(size=(int(outlier_mask.sum()), 3))
        )
    return {
        "bearings_a": bearings_a,
        "bearings_b": bearings_b,
        "true_inliers": inlier_mask,
        "rotation": rotation,
        "translation": translation_direction,
        "actual_median_parallax_deg": (
            float(np.median(clean_parallax[inlier_mask]))
            if np.any(inlier_mask) and np.linalg.norm(translation) > 0.0
            else 0.0
        ),
        "depth_scale": depth_scale,
    }


def _rotation_error_deg(estimated: np.ndarray, expected: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(estimated @ expected.T) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _direction_error_deg(estimated: np.ndarray, expected: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(estimated, expected), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def release_options(seed: int, *, fast: bool = False) -> RelativePoseOptions:
    """Return the frozen best profile; ``fast`` exists only for unit tests."""

    return RelativePoseOptions(
        max_angular_error_deg=1.0,
        min_num_trials=8 if fast else 32,
        max_num_trials=40 if fast else 1000,
        min_inliers=8,
        local_optimization_steps=1 if fast else 3,
        robust_refinement_steps=1 if fast else 2,
        refinement_max_nfev=30 if fast else 100,
        random_seed=seed,
        stability_trials=0 if fast else 6,
        stability_ransac_trials=8 if fast else 24,
        model_competition_trials=8 if fast else 128,
        hypothesis_ranking="msac-first",
        nonminimal_refit_max_steps=20 if fast else 100,
        pose_refinement_method="joint",
        compute_backend="auto",
    )


def evaluate_case(
    condition: Condition,
    seed: int,
    *,
    fast: bool = False,
) -> dict[str, Any]:
    scene = generate_scene(condition, seed)
    options = release_options(seed, fast=fast)
    policy = RelativePoseAcceptancePolicy()
    started = time.perf_counter_ns()
    result = estimate_relative_pose(
        scene["bearings_a"],
        scene["bearings_b"],
        options=options,
        sampler=SpatiallyWeightedFivePointSampler(),
        quality_policy=policy,
    )
    runtime_ms = (time.perf_counter_ns() - started) / 1e6
    base: dict[str, Any] = {
        **asdict(condition),
        "seed": seed,
        "actual_median_parallax_deg": scene["actual_median_parallax_deg"],
        "true_inlier_count": int(scene["true_inliers"].sum()),
        "runtime_ms": runtime_ms,
        "returned": result is not None,
        "accepted": False,
        "strict_correct": False,
        "precise_correct": False,
        "false_accept": False,
        "rotation_error_deg": None,
        "translation_error_deg": None,
        "rejection_reasons": ["no-pose"] if result is None else [],
        "options": options.to_dict(),
        "policy": policy.to_dict(),
    }
    if result is None:
        return base

    quality = result.quality_report
    base.update(
        {
            "accepted": bool(quality.accepted),
            "rejection_reasons": list(quality.rejection_reasons),
            "estimated_num_inliers": result.num_inliers,
            "estimated_inlier_ratio": quality.inlier_ratio,
            "estimated_median_residual_deg": quality.median_residual_deg,
            "estimated_median_parallax_deg": quality.median_parallax_deg,
            "coverage_entropy_a": quality.coverage_entropy_a,
            "coverage_entropy_b": quality.coverage_entropy_b,
            "occupied_cells_a": quality.occupied_cells_a,
            "occupied_cells_b": quality.occupied_cells_b,
            "cheirality_ratio": quality.cheirality_ratio,
            "translation_orientation_margin": quality.translation_orientation.cheirality_margin,
            "stability_rotation_p90_deg": quality.stability.rotation_p90_deg,
            "stability_translation_p90_deg": quality.stability.translation_p90_deg,
            "preferred_model": quality.model_competition.preferred_model,
            "essential_score_margin": quality.model_competition.essential_score_margin,
            "quality_score": quality.raw_quality_score,
            "num_trials": result.num_trials,
            "compute_backend": result.compute_backend,
        }
    )
    if condition.observable_translation:
        rotation_error = _rotation_error_deg(result.R, scene["rotation"])
        translation_error = _direction_error_deg(result.t, scene["translation"])
        strict = rotation_error <= STRICT_R_DEG and translation_error <= STRICT_T_DEG
        precise = rotation_error <= PRECISE_R_DEG and translation_error <= PRECISE_T_DEG
        base.update(
            {
                "rotation_error_deg": rotation_error,
                "translation_error_deg": translation_error,
                "strict_correct": strict,
                "precise_correct": precise,
                "false_accept": bool(quality.accepted and not strict),
            }
        )
    else:
        base["false_accept"] = bool(quality.accepted)
    return base


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["family"]), str(row["name"]))].append(row)
    summary = []
    for (family, name), selected in sorted(grouped.items()):
        observable = bool(selected[0]["observable_translation"])
        returned = [row for row in selected if row["returned"]]
        accepted = [row for row in selected if row["accepted"]]
        summary.append(
            {
                "family": family,
                "name": name,
                "count": len(selected),
                "correspondence_count": selected[0]["count"],
                "target_parallax_deg": selected[0]["target_parallax_deg"],
                "inlier_ratio": selected[0]["inlier_ratio"],
                "noise_deg": selected[0]["noise_deg"],
                "coverage": selected[0]["coverage"],
                "observable_translation": observable,
                "return_rate": _rate(selected, "returned"),
                "acceptance_rate": _rate(selected, "accepted"),
                "strict_rate_all": _rate(selected, "strict_correct"),
                "precise_rate_all": _rate(selected, "precise_correct"),
                "false_accept_rate": _rate(selected, "false_accept"),
                "accepted_strict_precision": (
                    _rate(accepted, "strict_correct")
                    if observable and accepted
                    else None
                ),
                "median_rotation_error_deg": (
                    float(np.median([row["rotation_error_deg"] for row in returned]))
                    if observable and returned
                    else None
                ),
                "median_translation_error_deg": (
                    float(np.median([row["translation_error_deg"] for row in returned]))
                    if observable and returned
                    else None
                ),
                "median_runtime_ms": float(
                    np.median([row["runtime_ms"] for row in selected])
                ),
                "rejection_reason_counts": dict(
                    sorted(
                        Counter(
                            reason
                            for row in selected
                            for reason in row["rejection_reasons"]
                        ).items()
                    )
                ),
            }
        )
    return summary


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _technical_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    negatives = [row for row in summary if row["family"] == "negative-control"]
    false_accepts = sum(row["false_accept_rate"] * row["count"] for row in negatives)
    lines = [
        "# Spherical relative-pose operating envelope",
        "",
        "## Scope",
        "",
        "This source-checkout experiment isolates the public two-view R,t estimator.",
        "It does not include detection, description, matching or dense stereo.",
        "Spatial scene overlap is therefore not an estimator input; it is an upstream",
        "cause of match count, inlier ratio, angular coverage and parallax.",
        "",
        "## Frozen correctness criteria",
        "",
        f"- Strict: R <= {STRICT_R_DEG:g} deg and oriented t <= {STRICT_T_DEG:g} deg.",
        f"- Precise: R <= {PRECISE_R_DEG:g} deg and oriented t <= {PRECISE_T_DEG:g} deg.",
        "- Robust acceptance: the unchanged public RelativePoseAcceptancePolicy.",
        "- Negative controls are successful only when not accepted.",
        "",
        "## Summary",
        "",
        "| Family / condition | N | Return | Accept | Strict | Precise | False accept | Accepted strict precision | Runtime median (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        precision = row["accepted_strict_precision"]
        lines.append(
            f"| {row['family']} / {row['name']} | {row['count']} | "
            f"{100 * row['return_rate']:.1f}% | {100 * row['acceptance_rate']:.1f}% | "
            f"{100 * row['strict_rate_all']:.1f}% | {100 * row['precise_rate_all']:.1f}% | "
            f"{100 * row['false_accept_rate']:.1f}% | "
            f"{'—' if precision is None else f'{100 * precision:.1f}%'} | "
            f"{row['median_runtime_ms']:.1f} |"
        )
    lines.extend(
        (
            "",
            "## Interpretation rule",
            "",
            "A development cell is a candidate robust region only when accepted-pose",
            "strict precision is 100%, false acceptance is zero, and acceptance coverage",
            "is at least two thirds across the three preregistered seeds. Three seeds are",
            "sufficient to locate transitions, not to certify a release probability.",
            "The final release boundary requires frozen real-pair replay and a one-sided",
            "confidence bound on accepted-pose precision.",
            "",
            "## Negative controls",
            "",
            f"Observed false acceptances: {false_accepts:g} across "
            f"{sum(row['count'] for row in negatives)} negative-control trials.",
            "",
            "## Reproducibility",
            "",
            f"- Commit: `{payload['source']['commit']}`",
            f"- Dirty source: `{payload['source']['dirty']}`",
            f"- Python: `{payload['environment']['python']}`",
            f"- NumPy: `{payload['environment']['numpy']}`",
            f"- Platform: `{payload['environment']['platform']}`",
            f"- Seeds per condition: `{payload['configuration']['seeds']}`",
            f"- Total cases: `{len(payload['rows'])}`",
            "",
            "## Limitations",
            "",
            "The grid uses ideal central-camera bearings with controlled independent",
            "noise and random outliers. It does not reproduce descriptor aliasing, repeated",
            "industrial texture, dynamic objects, calibration bias, non-central capture or",
            "correlated mismatches. These are measured separately on frozen real pairs.",
        )
    )
    return "\n".join(lines) + "\n"


def run(
    output_dir: Path,
    *,
    seeds: int,
    families: Iterable[str] | None = None,
) -> dict[str, Any]:
    if seeds < 1:
        raise ValueError("seeds must be positive")
    allowed = None if families is None else set(families)
    conditions = tuple(
        condition
        for condition in development_conditions()
        if allowed is None or condition.family in allowed
    )
    if not conditions:
        raise ValueError("no conditions selected")
    rows = []
    total = len(conditions) * seeds
    for index, condition in enumerate(conditions, start=1):
        for seed_offset in range(seeds):
            seed = 17000 + index * 100 + seed_offset
            row = evaluate_case(condition, seed)
            rows.append(row)
            print(
                f"[{len(rows):03d}/{total:03d}] {condition.family}/{condition.name} "
                f"seed={seed} returned={row['returned']} accepted={row['accepted']} "
                f"strict={row['strict_correct']} runtime={row['runtime_ms'] / 1000:.2f}s",
                flush=True,
            )
    payload = {
        "schema": SCHEMA,
        "source": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "configuration": {
            "seeds": seeds,
            "families": sorted({condition.family for condition in conditions}),
            "condition_count": len(conditions),
            "strict_thresholds_deg": {
                "rotation": STRICT_R_DEG,
                "translation": STRICT_T_DEG,
            },
            "precise_thresholds_deg": {
                "rotation": PRECISE_R_DEG,
                "translation": PRECISE_T_DEG,
            },
            "conditions": [asdict(condition) for condition in conditions],
        },
        "rows": rows,
        "summary": aggregate(rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "results.json", payload)
    (output_dir / "REPORT.md").write_text(_technical_report(payload), encoding="utf-8")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--families",
        help="comma-separated subset of parallax-vs-inliers,count-vs-coverage,noise-vs-parallax,negative-control",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    families = None if args.families is None else args.families.split(",")
    payload = run(args.output_dir.resolve(), seeds=args.seeds, families=families)
    print(f"wrote {len(payload['rows'])} cases to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
