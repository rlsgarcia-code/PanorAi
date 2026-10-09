#!/usr/bin/env python3
"""Prepare, but never authorize, an E8 candidate from frozen evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

try:
    from run_prospective_confirmation import CANDIDATE_SCHEMA, EXPECTED_GATE
    from select_release_rule import (
        ALIGNED_POST_MODEL,
        CAPTURE_ACCEPT_MODEL,
        CAPTURE_PRECISE_MODEL,
        RULE_SCHEMA,
        _features,
        _outcomes,
        _predictions,
        _selected,
        exact_one_sided_lower,
    )
except ImportError:
    from benchmarks.two_view_pose_probability.run_prospective_confirmation import (
        CANDIDATE_SCHEMA,
        EXPECTED_GATE,
    )
    from benchmarks.two_view_pose_probability.select_release_rule import (
        ALIGNED_POST_MODEL,
        CAPTURE_ACCEPT_MODEL,
        CAPTURE_PRECISE_MODEL,
        RULE_SCHEMA,
        _features,
        _outcomes,
        _predictions,
        _selected,
        exact_one_sided_lower,
    )


DRAFT_AUTHORIZATION = "DRAFT_ONLY_NOT_AUTHORIZED"
CONFIGURATION_SCHEMA = "panorai-two-view-prospective-configuration/v1"
MODEL_SNAPSHOT_SCHEMA = "panorai-two-view-prospective-model-snapshot/v1"
RULE_AMENDMENT_SCHEMA = "panorai-two-view-prospective-rule-amendment/v1"
READINESS_SCHEMA = "panorai-two-view-prospective-readiness/v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _model_snapshot(
    model_card: Mapping[str, Any], model_ids: Iterable[str], *, role: str
) -> dict[str, Any]:
    wanted = tuple(model_ids)
    indexed = {
        str(model["model_id"]): model
        for model in model_card.get("models", [])
        if isinstance(model, dict) and "model_id" in model
    }
    missing = [model_id for model_id in wanted if model_id not in indexed]
    if missing:
        raise ValueError(f"model card is missing required models: {missing}")
    return {
        "schema": MODEL_SNAPSHOT_SCHEMA,
        "role": role,
        "source_model_card_sha256": None,
        "models": [indexed[model_id] for model_id in wanted],
    }


def _metrics(rows: list[dict[str, Any]], population_count: int) -> dict[str, Any]:
    selected = len(rows)
    precise = sum(bool(row["precise"]) for row in rows)
    return {
        "population_pairs": population_count,
        "selected_pairs": selected,
        "selected_components": len(
            {str(row["independence_component_id"]) for row in rows}
        ),
        "precise_pairs": precise,
        "selected_precision": precise / selected if selected else None,
        "exact_one_sided_95_lower": exact_one_sided_lower(precise, selected),
        "catastrophic_selected": sum(
            bool(row["catastrophic_accepted"]) for row in rows
        ),
        "pair_coverage": selected / population_count if population_count else 0.0,
    }


def _select_rows(
    outcomes: Mapping[tuple[str, str], dict[str, Any]],
    features: Mapping[tuple[str, str], dict[str, Any]],
    predictions: Mapping[tuple[str, str, str], dict[str, Any]],
    *,
    baseline_range: tuple[float, float],
    capture_threshold: float,
    post_threshold: float,
) -> list[dict[str, Any]]:
    return [
        row
        for key, row in outcomes.items()
        if _selected(
            key,
            row,
            features[key],
            predictions,
            baseline_range=baseline_range,
            capture_threshold=capture_threshold,
            post_threshold=post_threshold,
            post_model=ALIGNED_POST_MODEL,
        )
    ]


def _by_dataset(
    outcomes: Mapping[tuple[str, str], dict[str, Any]],
    selected: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    datasets = sorted({str(key[0]) for key in outcomes})
    return {
        dataset: _metrics(
            [row for row in selected if row["dataset_id"] == dataset],
            sum(key[0] == dataset for key in outcomes),
        )
        for dataset in datasets
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError("prospective draft output directory must not exist")
    if not _is_sha256(args.wheel_sha256):
        raise ValueError("wheel SHA-256 is invalid")

    release_rule = _read_json(args.release_rule)
    if release_rule.get("schema") != RULE_SCHEMA:
        raise ValueError("release rule schema is invalid")
    if release_rule.get("inputs", {}).get("features", {}).get("sha256") != _sha256(
        args.features
    ):
        raise ValueError("features do not match the frozen release rule")
    if release_rule.get("inputs", {}).get("predictions", {}).get(
        "sha256"
    ) != _sha256(args.predictions):
        raise ValueError("predictions do not match the frozen release rule")
    if release_rule.get("inputs", {}).get("training_outcomes", {}).get(
        "sha256"
    ) != _sha256(args.training_outcomes):
        raise ValueError("training outcomes do not match the frozen release rule")

    matching_candidates = [
        row
        for row in release_rule.get("candidate_grid", [])
        if math.isclose(
            float(row["capture_usable_probability_threshold"]),
            args.capture_threshold,
        )
        and math.isclose(
            float(row["post_precision_probability_threshold"]),
            args.post_threshold,
        )
    ]
    if len(matching_candidates) != 1 or not matching_candidates[0].get("qualifies"):
        raise ValueError("requested amendment was not a qualifying frozen-grid candidate")

    model_card = _read_json(args.model_card)
    route_validation = _read_json(args.route_validation)
    package = route_validation.get("package", {})
    if package.get("version") != args.expected_package_version:
        raise ValueError("route-validation package version mismatch")
    if package.get("expected_source_commit") != args.expected_source_commit:
        raise ValueError("route-validation source commit mismatch")
    native = route_validation.get("native", {})
    route = route_validation.get("route", {})
    optimized = route.get("route", {})
    if not (
        native.get("convolution_backend") == "native"
        and native.get("numpy_fallback_permitted") is False
        and optimized.get("batch_size") == 2
        and optimized.get("detector_method") == "detect_batch"
        and optimized.get("patch_provider_max_workers") == 4
        and optimized.get("validity_masks") == "explicit-per-panorama"
    ):
        raise ValueError("route validation is not the frozen optimized route")

    features = _features(args.features)
    predictions = _predictions(args.predictions)
    training = _outcomes(args.training_outcomes, {"development", "calibration"})
    evaluation = _outcomes(args.evaluation_outcomes, {"evaluation"})
    if set(training) | set(evaluation) != set(features):
        raise ValueError("outcome partitions do not cover the feature population")
    baseline = tuple(
        float(value)
        for value in release_rule["capture_envelope"]["baseline_m_closed_interval"]
    )
    if len(baseline) != 2:
        raise ValueError("release rule baseline interval is malformed")

    calibration = {
        key: row
        for key, row in training.items()
        if row["split"] == "calibration"
        and key[0] in set(release_rule["supported_domains_at_freeze"])
    }
    calibration_selected = _select_rows(
        calibration,
        features,
        predictions,
        baseline_range=(baseline[0], baseline[1]),
        capture_threshold=args.capture_threshold,
        post_threshold=args.post_threshold,
    )
    calibration_metrics = _metrics(calibration_selected, len(calibration))
    frozen_grid = matching_candidates[0]
    for field in (
        "selected_pairs",
        "precise_pairs",
        "catastrophic_accepted",
    ):
        expected_field = (
            "catastrophic_selected" if field == "catastrophic_accepted" else field
        )
        if calibration_metrics[expected_field] != frozen_grid[field]:
            raise ValueError(f"calibration recomputation differs for {field}")

    evaluation_selected = _select_rows(
        evaluation,
        features,
        predictions,
        baseline_range=(baseline[0], baseline[1]),
        capture_threshold=args.capture_threshold,
        post_threshold=args.post_threshold,
    )
    evaluation_metrics = _metrics(evaluation_selected, len(evaluation))
    selection_rate = evaluation_metrics["pair_coverage"]
    screening_pairs = (
        math.ceil(EXPECTED_GATE["minimum_selected_pairs"] / selection_rate)
        if selection_rate > 0.0
        else None
    )

    configuration = {
        "schema": CONFIGURATION_SCHEMA,
        "panorai": {
            "version": args.expected_package_version,
            "source_commit": args.expected_source_commit,
            "wheel_sha256": args.wheel_sha256,
        },
        "route": route,
    }
    capture_models = _model_snapshot(
        model_card,
        (CAPTURE_ACCEPT_MODEL, CAPTURE_PRECISE_MODEL),
        role="capture",
    )
    post_model = _model_snapshot(
        model_card, (ALIGNED_POST_MODEL,), role="post-processing"
    )
    model_card_hash = _sha256(args.model_card)
    capture_models["source_model_card_sha256"] = model_card_hash
    post_model["source_model_card_sha256"] = model_card_hash

    args.output_dir.mkdir(parents=True)
    configuration_path = args.output_dir / "configuration.json"
    capture_model_path = args.output_dir / "capture-models.json"
    post_model_path = args.output_dir / "post-model.json"
    _atomic_json(configuration_path, configuration)
    _atomic_json(capture_model_path, capture_models)
    _atomic_json(post_model_path, post_model)

    amendment = {
        "schema": RULE_AMENDMENT_SCHEMA,
        "status": "post-evaluation hypothesis; prospective confirmation required",
        "base_release_rule": {
            "artifact_name": args.release_rule.name,
            "sha256": _sha256(args.release_rule),
            "retrospective_verdict": "NO_GO",
        },
        "capture_envelope": release_rule["capture_envelope"],
        "thresholds": {
            "capture_usable_probability": args.capture_threshold,
            "post_precision_probability": args.post_threshold,
            "requires_public_quality_acceptance": True,
        },
        "models": release_rule["models"],
        "calibration_recomputation": calibration_metrics,
        "frozen_grid_candidate": frozen_grid,
        "evaluation_hypothesis_generation": {
            "pooled": evaluation_metrics,
            "datasets": _by_dataset(evaluation, evaluation_selected),
            "disclosure": (
                "The amendment was chosen after viewing retrospective evaluation; "
                "these metrics generate a hypothesis and cannot validate it."
            ),
        },
        "prospective_screening_plan": {
            "observed_retrospective_selection_rate": selection_rate,
            "minimum_selected_pairs": EXPECTED_GATE["minimum_selected_pairs"],
            "estimated_candidate_pairs_for_120_selected": screening_pairs,
            "planning_only": True,
        },
    }
    amendment_path = args.output_dir / "selective-rule-amendment.json"
    _atomic_json(amendment_path, amendment)

    candidate = {
        "schema": CANDIDATE_SCHEMA,
        "authorization": DRAFT_AUTHORIZATION,
        "panorai": {
            "version": args.expected_package_version,
            "source_commit": args.expected_source_commit,
        },
        "wheel_sha256": args.wheel_sha256,
        "configuration_sha256": _sha256(configuration_path),
        "capture_model_sha256": _sha256(capture_model_path),
        "post_model_sha256": _sha256(post_model_path),
        "selective_rule_sha256": _sha256(amendment_path),
        "gate": EXPECTED_GATE,
        "deviations": [],
        "status": "draft only; deliberately rejected by the E8 seal executor",
        "post_hoc_selection_disclosure": amendment[
            "evaluation_hypothesis_generation"
        ]["disclosure"],
    }
    candidate_path = args.output_dir / "candidate-draft.json"
    _atomic_json(candidate_path, candidate)

    report = {
        "schema": READINESS_SCHEMA,
        "status": "DRAFT_REQUIRES_EXPLICIT_AUTHORIZATION_AND_E8",
        "candidate": {
            "artifact_name": candidate_path.name,
            "sha256": _sha256(candidate_path),
            "authorization": DRAFT_AUTHORIZATION,
        },
        "artifacts": {
            "configuration_sha256": _sha256(configuration_path),
            "capture_model_sha256": _sha256(capture_model_path),
            "post_model_sha256": _sha256(post_model_path),
            "selective_rule_sha256": _sha256(amendment_path),
        },
        "calibration": calibration_metrics,
        "retrospective_evaluation_hypothesis": evaluation_metrics,
        "retrospective_evaluation_by_dataset": _by_dataset(
            evaluation, evaluation_selected
        ),
        "estimated_candidate_pairs_for_120_selected": screening_pairs,
        "conclusion": (
            "The 0.90 post threshold is eligible for prospective testing, but "
            "the current release verdict remains NO_GO."
        ),
    }
    _atomic_json(args.output_dir / "readiness-report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--training-outcomes", type=Path, required=True)
    parser.add_argument("--evaluation-outcomes", type=Path, required=True)
    parser.add_argument("--release-rule", type=Path, required=True)
    parser.add_argument("--model-card", type=Path, required=True)
    parser.add_argument("--route-validation", type=Path, required=True)
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument("--expected-package-version", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--capture-threshold", type=float, default=0.50)
    parser.add_argument("--post-threshold", type=float, default=0.90)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    report = prepare(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
