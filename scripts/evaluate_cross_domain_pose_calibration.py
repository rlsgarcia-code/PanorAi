#!/usr/bin/env python3
"""Evaluate domain-disjoint calibration of frozen relative-pose scores.

The evaluator consumes the frozen VAL-002 prediction and post-freeze outcome
tables. It never opens panorama data. Calibration is conditional on a returned
finite pose because only those records carry a quality report.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from panorai.estimators import (
    ModelCompetitionReport,
    ModelEvidence,
    PoseStabilityReport,
    RelativePoseConfidenceCalibrator,
    RelativePoseQualityReport,
    TranslationOrientationReport,
)

SCHEMA = "panorai-cross-domain-pose-calibration/v1"
EXPECTED_PREDICTIONS_SHA256 = (
    "2b31b60e223537136d6b7e8e1f5f3b0825b1864fbc3fce9ceba13fa63fecbc61"
)
EXPECTED_EVALUATED_SHA256 = (
    "4faad4726a2028a51d588dd8efdaf2bf2d31f9da6d77e942d68565108c606463"
)
DOMAIN_ORDER = ("matterport360", "stanford2d3d")


@dataclass(frozen=True)
class CalibrationRow:
    sample_id: str
    domain: str
    group_id: str
    success: bool
    report: RelativePoseQualityReport


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"expected CSV boolean, got {value!r}")


def _model_evidence(value: dict[str, Any]) -> ModelEvidence:
    return ModelEvidence(
        model=str(value["model"]),
        num_inliers=int(value["num_inliers"]),
        inlier_ratio=float(value["inlier_ratio"]),
        normalized_robust_score=float(value["normalized_robust_score"]),
        median_residual_deg=float(value["median_residual_deg"]),
    )


def _quality_report(value: dict[str, Any]) -> RelativePoseQualityReport:
    competition = value["model_competition"]
    orientation = value.get("translation_orientation")
    if orientation is None:
        # VAL-002 predates this diagnostic. The confidence calibrator consumes
        # only raw_quality_score; the explicit placeholder prevents invention
        # of cheirality evidence while retaining the historical score exactly.
        orientation_report = TranslationOrientationReport(
            hypothesis_count=0,
            provisional_correspondence_count=0,
            best_positive_depth_count=0,
            alternative_positive_depth_count=0,
            positive_depth_fraction=float(value["cheirality_ratio"]),
            cheirality_margin=0.0,
            median_triangulation_angle_deg=float(value["median_parallax_deg"]),
            ambiguous=True,
        )
    else:
        orientation_report = TranslationOrientationReport(**orientation)
    return RelativePoseQualityReport(
        num_correspondences=int(value["num_correspondences"]),
        num_inliers=int(value["num_inliers"]),
        inlier_ratio=float(value["inlier_ratio"]),
        occupied_cells_a=int(value["occupied_cells_a"]),
        occupied_cells_b=int(value["occupied_cells_b"]),
        coverage_entropy_a=float(value["coverage_entropy_a"]),
        coverage_entropy_b=float(value["coverage_entropy_b"]),
        median_residual_deg=float(value["median_residual_deg"]),
        p90_residual_deg=float(value["p90_residual_deg"]),
        median_parallax_deg=float(value["median_parallax_deg"]),
        cheirality_ratio=float(value["cheirality_ratio"]),
        translation_orientation=orientation_report,
        stability=PoseStabilityReport(**value["stability"]),
        model_competition=ModelCompetitionReport(
            essential=_model_evidence(competition["essential"]),
            rotation_only=_model_evidence(competition["rotation_only"]),
            spherical_homography=_model_evidence(competition["spherical_homography"]),
            preferred_model=str(competition["preferred_model"]),
            essential_score_margin=float(competition["essential_score_margin"]),
        ),
        raw_quality_score=float(value["raw_quality_score"]),
        accepted=bool(value["accepted"]),
        rejection_reasons=tuple(value["rejection_reasons"]),
        interface=str(value.get("interface", "panorai-relative-pose-quality/v1")),
    )


def load_rows(
    predictions_path: Path,
    evaluated_path: Path,
    *,
    expected_predictions_sha256: str | None = EXPECTED_PREDICTIONS_SHA256,
    expected_evaluated_sha256: str | None = EXPECTED_EVALUATED_SHA256,
) -> tuple[list[CalibrationRow], dict[str, Any]]:
    prediction_sha = sha256(predictions_path)
    evaluated_sha = sha256(evaluated_path)
    if expected_predictions_sha256 and prediction_sha != expected_predictions_sha256:
        raise ValueError("frozen prediction SHA-256 does not match VAL-002")
    if expected_evaluated_sha256 and evaluated_sha != expected_evaluated_sha256:
        raise ValueError("evaluated-pair SHA-256 does not match VAL-002")

    lines = [line for line in predictions_path.read_text().splitlines() if line]
    if not lines:
        raise ValueError("prediction file is empty")
    manifest = json.loads(lines[0]).get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("prediction file lacks its frozen manifest")
    predictions = [json.loads(line) for line in lines[1:]]
    outcomes = list(csv.DictReader(evaluated_path.open(newline="")))
    prediction_by_id = {row["pair_id"]: row for row in predictions}
    outcome_by_id = {row["pair_id"]: row for row in outcomes}
    if len(prediction_by_id) != len(predictions):
        raise ValueError("prediction sample IDs are not unique")
    if len(outcome_by_id) != len(outcomes):
        raise ValueError("evaluated sample IDs are not unique")
    if prediction_by_id.keys() != outcome_by_id.keys():
        raise ValueError("prediction and evaluated sample IDs differ")

    rows: list[CalibrationRow] = []
    for sample_id, prediction in prediction_by_id.items():
        outcome = outcome_by_id[sample_id]
        if prediction["dataset_id"] != outcome["dataset_id"]:
            raise ValueError(f"domain mismatch for {sample_id}")
        if prediction["spatial_group_id"] != outcome["spatial_group_id"]:
            raise ValueError(f"group mismatch for {sample_id}")
        prediction_valid = bool(prediction.get("method_valid"))
        outcome_valid = _as_bool(outcome["method_valid"])
        if prediction_valid != outcome_valid:
            raise ValueError(f"validity mismatch for {sample_id}")
        if not prediction_valid:
            continue
        quality = prediction.get("quality_report")
        if not isinstance(quality, dict):
            raise ValueError(f"valid pose lacks quality report: {sample_id}")
        rows.append(
            CalibrationRow(
                sample_id=sample_id,
                domain=str(prediction["dataset_id"]),
                group_id=str(prediction["spatial_group_id"]),
                success=_as_bool(outcome["primary_success"]),
                report=_quality_report(quality),
            )
        )
    if {row.domain for row in rows} != set(DOMAIN_ORDER):
        raise ValueError("expected both frozen evaluation domains")
    return rows, {
        "prediction_sha256": prediction_sha,
        "evaluated_sha256": evaluated_sha,
        "frozen_pair_count": len(predictions),
        "returned_pose_count": len(rows),
        "source_manifest": manifest,
    }


def _ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 5) -> float:
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        selected = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1]
            if index == bins - 1
            else probabilities < edges[index + 1]
        )
        if selected.any():
            result += float(selected.mean()) * abs(
                float(probabilities[selected].mean() - labels[selected].mean())
            )
    return result


def _metric_vector(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
        "expected_calibration_error": _ece(probabilities, labels),
        "calibration_gap": abs(float(probabilities.mean() - labels.mean())),
        "predicted_mean": float(probabilities.mean()),
        "observed_success_rate": float(labels.mean()),
        "accuracy_at_half": float(
            np.mean((probabilities >= 0.5) == labels.astype(bool))
        ),
    }


def _confusion(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    selected = probabilities >= 0.5
    positive = labels.astype(bool)
    tp = int(np.sum(selected & positive))
    fp = int(np.sum(selected & ~positive))
    tn = int(np.sum(~selected & ~positive))
    fn = int(np.sum(~selected & positive))
    return {
        "threshold": 0.5,
        "selected": int(selected.sum()),
        "coverage": float(selected.mean()),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": None if tp + fp == 0 else float(tp / (tp + fp)),
        "recall": None if tp + fn == 0 else float(tp / (tp + fn)),
    }


def _reliability(probabilities: np.ndarray, labels: np.ndarray) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, 6)
    rows: list[dict[str, Any]] = []
    for index in range(5):
        selected = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1]
            if index == 4
            else probabilities < edges[index + 1]
        )
        rows.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": int(selected.sum()),
                "predicted_mean": (
                    None
                    if not selected.any()
                    else float(probabilities[selected].mean())
                ),
                "observed_success_rate": (
                    None if not selected.any() else float(labels[selected].mean())
                ),
            }
        )
    return rows


def _bootstrap(
    probabilities: np.ndarray,
    raw_scores: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[str],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    unique_groups = tuple(sorted(set(groups)))
    by_group = {
        group: np.flatnonzero(np.asarray(groups) == group) for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    values = np.empty((repetitions, 5), dtype=np.float64)
    for repetition in range(repetitions):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([by_group[str(group)] for group in sampled])
        calibrated = _metric_vector(probabilities[indices], labels[indices])
        raw = _metric_vector(raw_scores[indices], labels[indices])
        values[repetition] = (
            calibrated["brier_score"],
            calibrated["expected_calibration_error"],
            calibrated["calibration_gap"],
            calibrated["brier_score"] - raw["brier_score"],
            calibrated["expected_calibration_error"]
            - raw["expected_calibration_error"],
        )
    names = (
        "brier_score",
        "expected_calibration_error",
        "calibration_gap",
        "brier_delta_vs_raw_score",
        "ece_delta_vs_raw_score",
    )
    return {
        "unit": "spatial_group",
        "group_count": len(unique_groups),
        "repetitions": repetitions,
        "seed": seed,
        "percentile_95": {
            name: [
                float(np.quantile(values[:, index], 0.025)),
                float(np.quantile(values[:, index], 0.975)),
            ]
            for index, name in enumerate(names)
        },
    }


def _fit_and_evaluate(
    calibration_rows: Sequence[CalibrationRow],
    evaluation_rows: Sequence[CalibrationRow],
    *,
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    calibration_ids = [row.sample_id for row in calibration_rows]
    evaluation_ids = [row.sample_id for row in evaluation_rows]
    if set(calibration_ids).intersection(evaluation_ids):
        raise ValueError("calibration and evaluation sample IDs overlap")
    calibrator = RelativePoseConfidenceCalibrator.fit(
        [row.report for row in calibration_rows],
        [row.success for row in calibration_rows],
        sample_ids=calibration_ids,
    )
    reports = [row.report for row in evaluation_rows]
    labels = np.asarray([row.success for row in evaluation_rows], dtype=np.float64)
    probabilities = np.asarray(calibrator.predict_proba(reports), dtype=np.float64)
    raw_scores = np.asarray(
        [row.report.raw_quality_score for row in evaluation_rows], dtype=np.float64
    )
    public_evaluation = calibrator.evaluate(
        reports,
        [row.success for row in evaluation_rows],
        sample_ids=evaluation_ids,
    )
    calibrated = _metric_vector(probabilities, labels)
    if calibrated["brier_score"] != public_evaluation.brier_score:
        raise AssertionError("independent Brier calculation differs from public API")
    if (
        calibrated["expected_calibration_error"]
        != public_evaluation.expected_calibration_error
    ):
        raise AssertionError("independent ECE calculation differs from public API")
    training_prevalence = float(
        np.mean([row.success for row in calibration_rows], dtype=np.float64)
    )
    return {
        "calibration_count": len(calibration_rows),
        "calibration_group_count": len({row.group_id for row in calibration_rows}),
        "evaluation_count": len(evaluation_rows),
        "evaluation_group_count": len({row.group_id for row in evaluation_rows}),
        "sample_id_overlap": 0,
        "training_prevalence": training_prevalence,
        "isotonic_knots": len(calibrator.score_upper_bounds),
        "calibrated": calibrated,
        "raw_score_as_probability": _metric_vector(raw_scores, labels),
        "constant_training_prevalence": _metric_vector(
            np.full(len(labels), training_prevalence, dtype=np.float64), labels
        ),
        "threshold_at_half": _confusion(probabilities, labels),
        "reliability_bins": _reliability(probabilities, labels),
        "spatial_group_bootstrap": _bootstrap(
            probabilities,
            raw_scores,
            labels,
            [row.group_id for row in evaluation_rows],
            repetitions=bootstrap_repetitions,
            seed=seed,
        ),
    }


def _group_held_out(rows: Sequence[CalibrationRow]) -> dict[str, Any]:
    groups = tuple(sorted({row.group_id for row in rows}))
    probabilities = np.empty(len(rows), dtype=np.float64)
    labels = np.asarray([row.success for row in rows], dtype=np.float64)
    for group in groups:
        held_indices = [
            index for index, row in enumerate(rows) if row.group_id == group
        ]
        train = [row for row in rows if row.group_id != group]
        held = [rows[index] for index in held_indices]
        calibrator = RelativePoseConfidenceCalibrator.fit(
            [row.report for row in train],
            [row.success for row in train],
            sample_ids=[row.sample_id for row in train],
        )
        fold = np.asarray(
            calibrator.predict_proba([row.report for row in held]), dtype=np.float64
        )
        probabilities[np.asarray(held_indices)] = fold
    return {
        "protocol": "leave-one-spatial-group-out",
        "count": len(rows),
        "group_count": len(groups),
        "metrics": _metric_vector(probabilities, labels),
        "threshold_at_half": _confusion(probabilities, labels),
        "reliability_bins": _reliability(probabilities, labels),
    }


def evaluate_rows(
    rows: Sequence[CalibrationRow],
    *,
    bootstrap_repetitions: int = 10_000,
    seed: int = 6006,
) -> dict[str, Any]:
    if bootstrap_repetitions <= 0:
        raise ValueError("bootstrap_repetitions must be positive")
    domains = {
        domain: [row for row in rows if row.domain == domain] for domain in DOMAIN_ORDER
    }
    if any(not domain_rows for domain_rows in domains.values()):
        raise ValueError("both domains require returned-pose rows")
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "outcome": "primary_success: finite pose with rotation<=15deg and translation<=30deg",
        "population": "conditional on a returned finite pose with quality report",
        "study_status": "post-hoc domain-disjoint evaluation",
        "bootstrap_repetitions": bootstrap_repetitions,
        "seed": seed,
        "domain_counts": {
            domain: {
                "returned_pose_count": len(domain_rows),
                "spatial_group_count": len({row.group_id for row in domain_rows}),
                "success_count": sum(row.success for row in domain_rows),
                "success_rate": float(np.mean([row.success for row in domain_rows])),
            }
            for domain, domain_rows in domains.items()
        },
        "within_domain_group_held_out": {
            domain: _group_held_out(domain_rows)
            for domain, domain_rows in domains.items()
        },
        "cross_domain": {},
    }
    for index, (source, target) in enumerate(
        ((DOMAIN_ORDER[0], DOMAIN_ORDER[1]), (DOMAIN_ORDER[1], DOMAIN_ORDER[0]))
    ):
        result["cross_domain"][f"{source}_to_{target}"] = _fit_and_evaluate(
            domains[source],
            domains[target],
            bootstrap_repetitions=bootstrap_repetitions,
            seed=seed + index,
        )
    return result


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--evaluated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=6006)
    parser.add_argument(
        "--allow-unrecognized-input-hashes",
        action="store_true",
        help="intended only for synthetic evaluator tests, never VAL-002 evidence",
    )
    args = parser.parse_args(argv)
    expected_predictions = (
        None if args.allow_unrecognized_input_hashes else EXPECTED_PREDICTIONS_SHA256
    )
    expected_evaluated = (
        None if args.allow_unrecognized_input_hashes else EXPECTED_EVALUATED_SHA256
    )
    rows, provenance = load_rows(
        args.predictions,
        args.evaluated,
        expected_predictions_sha256=expected_predictions,
        expected_evaluated_sha256=expected_evaluated,
    )
    result = evaluate_rows(
        rows,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    result["provenance"] = provenance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"output": str(args.output), **result["domain_counts"]}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
