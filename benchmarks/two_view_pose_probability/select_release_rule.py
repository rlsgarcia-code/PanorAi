#!/usr/bin/env python3
"""Freeze and evaluate the VAL-018 selective two-view operating rule."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.stats import beta

RULE_SCHEMA = "panorai-two-view-selective-rule/v1"
EVALUATION_SCHEMA = "panorai-two-view-selective-rule-evaluation/v1"
NO_RULE_SCHEMA = "panorai-two-view-selective-rule-search/v1"
CAPTURE_ACCEPT_MODEL = "capture-accept-overlap-baseline"
CAPTURE_PRECISE_MODEL = "capture-precise-given-accept-overlap-baseline"
POST_MODEL = "post-precise-raw-score"
ALIGNED_POST_MODEL = "post-precise-aligned-orientation"
POST_MODELS = (POST_MODEL, ALIGNED_POST_MODEL)
MINIMUM_OVERLAP = 0.50
THRESHOLD_GRID = (0.50, 0.60, 0.70, 0.80, 0.90)
TARGET_PRECISION = 0.95
TARGET_EXACT_LOWER = 0.90
MINIMUM_COMPONENTS = 5
BOOTSTRAP_REPETITIONS = 10_000


class NoQualifyingRuleError(RuntimeError):
    """Raised after recording that no calibration-grid candidate qualified."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"{path} contains a non-object JSON value")
                rows.append(value)
    return rows


def _index(
    rows: Iterable[dict[str, Any]], fields: tuple[str, ...], label: str
) -> dict[tuple[str, ...], dict[str, Any]]:
    result = {}
    for row in rows:
        key = tuple(str(row[field]) for field in fields)
        if key in result:
            raise ValueError(f"duplicate {label}: {key}")
        result[key] = row
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def exact_one_sided_lower(successes: int, count: int, alpha: float = 0.05) -> float:
    """Clopper-Pearson one-sided lower confidence bound."""
    if not 0 <= successes <= count:
        raise ValueError("successes must be between zero and count")
    if count == 0 or successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, count - successes + 1))


def _seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _cluster_precision_interval(
    rows: list[dict[str, Any]], *, seed: int
) -> dict[str, Any] | None:
    groups = sorted({str(row["independence_component_id"]) for row in rows})
    if len(groups) < MINIMUM_COMPONENTS:
        return None
    by_group = {
        group: [row for row in rows if row["independence_component_id"] == group]
        for group in groups
    }
    generator = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for iteration in range(BOOTSTRAP_REPETITIONS):
        sampled = generator.choice(groups, size=len(groups), replace=True)
        combined = [row for group in sampled for row in by_group[group]]
        values[iteration] = np.mean([bool(row["precise"]) for row in combined])
    return {
        "unit": "independence_component",
        "components": len(groups),
        "repetitions": BOOTSTRAP_REPETITIONS,
        "seed": seed,
        "percentile_90": [float(value) for value in np.quantile(values, (0.05, 0.95))],
    }


def _predictions(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    return _index(
        _read_jsonl(path),
        ("model_id", "dataset_id", "pair_id"),
        "prediction key",
    )


def _features(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return _index(_read_jsonl(path), ("dataset_id", "pair_id"), "feature key")


def _outcomes(path: Path, expected_splits: set[str]) -> dict[tuple[str, str], dict]:
    result = _index(_read_jsonl(path), ("dataset_id", "pair_id"), "outcome key")
    found = {str(row["split"]) for row in result.values()}
    if not found <= expected_splits:
        raise ValueError(
            f"unexpected outcome splits: {sorted(found - expected_splits)}"
        )
    return result


def _probability(
    predictions: Mapping[tuple[str, str, str], dict[str, Any]],
    model_id: str,
    key: tuple[str, str],
) -> float | None:
    row = predictions.get((model_id, *key))
    return None if row is None else float(row["probability"])


def _capture_values(feature: Mapping[str, Any]) -> tuple[float, float]:
    capture = feature["capture"]
    return (
        float(capture["registered_cloud_overlap_min"]),
        float(capture["baseline_m"]),
    )


def _support_domains(
    outcomes: Mapping[tuple[str, str], dict],
    features: Mapping[tuple[str, str], dict],
) -> dict[str, int]:
    groups: dict[str, set[str]] = defaultdict(set)
    for key, outcome in outcomes.items():
        if outcome["split"] != "calibration":
            continue
        overlap, _ = _capture_values(features[key])
        if overlap >= MINIMUM_OVERLAP:
            groups[key[0]].add(str(outcome["independence_component_id"]))
    return {dataset: len(values) for dataset, values in sorted(groups.items())}


def _baseline_support(
    outcomes: Mapping[tuple[str, str], dict],
    features: Mapping[tuple[str, str], dict],
    supported_domains: set[str],
) -> tuple[float, float, int]:
    values = []
    for key in outcomes:
        if key[0] not in supported_domains:
            continue
        overlap, baseline = _capture_values(features[key])
        if overlap >= MINIMUM_OVERLAP:
            values.append(baseline)
    if not values:
        raise RuntimeError("no supported baseline values")
    lower, upper = np.quantile(np.asarray(values), (0.05, 0.95))
    return float(lower), float(upper), len(values)


def _passes_capture_envelope(
    feature: Mapping[str, Any], baseline_range: tuple[float, float]
) -> bool:
    overlap, baseline = _capture_values(feature)
    return (
        overlap >= MINIMUM_OVERLAP
        and baseline_range[0] <= baseline <= baseline_range[1]
    )


def _selected(
    key: tuple[str, str],
    outcome: Mapping[str, Any],
    feature: Mapping[str, Any],
    predictions: Mapping[tuple[str, str, str], dict[str, Any]],
    *,
    baseline_range: tuple[float, float],
    capture_threshold: float,
    post_threshold: float,
    post_model: str,
) -> bool:
    if not bool(outcome["accepted"]):
        return False
    if not _passes_capture_envelope(feature, baseline_range):
        return False
    accept = _probability(predictions, CAPTURE_ACCEPT_MODEL, key)
    conditional = _probability(predictions, CAPTURE_PRECISE_MODEL, key)
    post = _probability(predictions, post_model, key)
    if accept is None or conditional is None or post is None:
        return False
    return accept * conditional >= capture_threshold and post >= post_threshold


def _metrics(rows: list[dict[str, Any]], eligible: list[dict[str, Any]]) -> dict:
    successes = sum(bool(row["precise"]) for row in rows)
    accepted = sum(bool(row["accepted"]) for row in eligible)
    groups = len({str(row["independence_component_id"]) for row in rows})
    count = len(rows)
    return {
        "eligible_pairs": len(eligible),
        "accepted_pairs": accepted,
        "selected_pairs": count,
        "selected_components": groups,
        "pair_coverage": count / len(eligible) if eligible else 0.0,
        "accepted_coverage": count / accepted if accepted else 0.0,
        "precise_pairs": successes,
        "selected_precision": successes / count if count else None,
        "exact_one_sided_95_lower": exact_one_sided_lower(successes, count),
        "catastrophic_accepted": sum(
            bool(row["catastrophic_accepted"]) for row in rows
        ),
    }


def freeze_rule(args: argparse.Namespace) -> dict[str, Any]:
    features_path = args.features.resolve()
    predictions_path = args.predictions.resolve()
    outcomes_path = args.training_outcomes.resolve()
    features = _features(features_path)
    predictions = _predictions(predictions_path)
    outcomes = _outcomes(outcomes_path, {"development", "calibration"})
    if set(outcomes) - set(features):
        raise ValueError("training outcomes contain keys missing from features")

    domain_components = _support_domains(outcomes, features)
    supported_domains = {
        dataset
        for dataset, components in domain_components.items()
        if components >= MINIMUM_COMPONENTS
    }
    if not supported_domains:
        raise RuntimeError("no domain has five calibration components in the envelope")
    baseline_low, baseline_high, baseline_count = _baseline_support(
        outcomes, features, supported_domains
    )
    baseline_range = (baseline_low, baseline_high)

    calibration_keys = [
        key
        for key, outcome in outcomes.items()
        if outcome["split"] == "calibration" and key[0] in supported_domains
    ]
    candidates = []
    for capture_threshold in THRESHOLD_GRID:
        for post_threshold in THRESHOLD_GRID:
            selected_rows = [
                outcomes[key]
                for key in calibration_keys
                if _selected(
                    key,
                    outcomes[key],
                    features[key],
                    predictions,
                    baseline_range=baseline_range,
                    capture_threshold=capture_threshold,
                    post_threshold=post_threshold,
                    post_model=args.post_model,
                )
            ]
            eligible_rows = [outcomes[key] for key in calibration_keys]
            metrics = _metrics(selected_rows, eligible_rows)
            qualifies = (
                metrics["selected_precision"] is not None
                and metrics["selected_precision"] >= TARGET_PRECISION
                and metrics["exact_one_sided_95_lower"] >= TARGET_EXACT_LOWER
                and metrics["catastrophic_accepted"] == 0
                and metrics["selected_components"] >= MINIMUM_COMPONENTS
            )
            candidates.append(
                {
                    "capture_usable_probability_threshold": capture_threshold,
                    "post_precision_probability_threshold": post_threshold,
                    "qualifies": qualifies,
                    **metrics,
                }
            )
    qualifying = [candidate for candidate in candidates if candidate["qualifies"]]
    if not qualifying:
        search_path = args.output.with_name("release-rule-search.json").resolve()
        _atomic_json(
            search_path,
            {
                "schema": NO_RULE_SCHEMA,
                "status": "no candidate satisfies the frozen calibration targets",
                "post_precision_model": args.post_model,
                "supported_domains_at_freeze": sorted(supported_domains),
                "calibration_components_in_envelope_by_dataset": domain_components,
                "candidate_grid": candidates,
                "inputs": {
                    "features": {
                        "path": str(features_path),
                        "sha256": _sha256(features_path),
                    },
                    "predictions": {
                        "path": str(predictions_path),
                        "sha256": _sha256(predictions_path),
                    },
                    "training_outcomes": {
                        "path": str(outcomes_path),
                        "sha256": _sha256(outcomes_path),
                    },
                },
            },
        )
        raise NoQualifyingRuleError(
            f"no candidate satisfies the frozen calibration targets; see {search_path}"
        )
    chosen = min(
        qualifying,
        key=lambda row: (
            -row["selected_pairs"],
            row["capture_usable_probability_threshold"],
            row["post_precision_probability_threshold"],
        ),
    )
    chosen_rows = [
        outcomes[key]
        for key in calibration_keys
        if _selected(
            key,
            outcomes[key],
            features[key],
            predictions,
            baseline_range=baseline_range,
            capture_threshold=chosen["capture_usable_probability_threshold"],
            post_threshold=chosen["post_precision_probability_threshold"],
            post_model=args.post_model,
        )
    ]
    result = {
        "schema": RULE_SCHEMA,
        "status": "retrospectively frozen for untouched-split evaluation; not prospective",
        "scope": "scanner-assisted two-view spherical relative pose",
        "selection_data": "calibration components from supported domains only",
        "models": {
            "capture_acceptance": CAPTURE_ACCEPT_MODEL,
            "capture_conditional_precision": CAPTURE_PRECISE_MODEL,
            "post_precision": args.post_model,
        },
        "capture_envelope": {
            "registered_cloud_overlap_min": MINIMUM_OVERLAP,
            "overlap_origin": "operator requirement fixed before probability-model evaluation",
            "baseline_m_closed_interval": [baseline_low, baseline_high],
            "baseline_support_pairs_development_and_calibration": baseline_count,
            "requires_registered_depth_clouds": True,
        },
        "thresholds": {
            "capture_usable_probability": chosen[
                "capture_usable_probability_threshold"
            ],
            "post_precision_probability": chosen[
                "post_precision_probability_threshold"
            ],
            "requires_public_quality_acceptance": True,
        },
        "selection_targets": {
            "minimum_precision": TARGET_PRECISION,
            "minimum_exact_one_sided_95_lower": TARGET_EXACT_LOWER,
            "maximum_catastrophic_accepted": 0,
            "minimum_selected_components": MINIMUM_COMPONENTS,
        },
        "supported_domains_at_freeze": sorted(supported_domains),
        "calibration_components_in_envelope_by_dataset": domain_components,
        "chosen_calibration_result": chosen,
        "chosen_calibration_component_bootstrap": _cluster_precision_interval(
            chosen_rows, seed=_seed("release-rule", "calibration")
        ),
        "candidate_grid": candidates,
        "inputs": {
            "features": {"path": str(features_path), "sha256": _sha256(features_path)},
            "predictions": {
                "path": str(predictions_path),
                "sha256": _sha256(predictions_path),
            },
            "training_outcomes": {
                "path": str(outcomes_path),
                "sha256": _sha256(outcomes_path),
            },
        },
    }
    _atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


def evaluate_rule(args: argparse.Namespace) -> dict[str, Any]:
    rule_path = args.rule.resolve()
    features_path = args.features.resolve()
    predictions_path = args.predictions.resolve()
    outcomes_path = args.evaluation_outcomes.resolve()
    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    if rule.get("schema") != RULE_SCHEMA:
        raise ValueError("unsupported rule schema")
    features = _features(features_path)
    predictions = _predictions(predictions_path)
    outcomes = _outcomes(outcomes_path, {"evaluation"})
    baseline_range = tuple(rule["capture_envelope"]["baseline_m_closed_interval"])
    capture_threshold = float(rule["thresholds"]["capture_usable_probability"])
    post_threshold = float(rule["thresholds"]["post_precision_probability"])
    post_model = str(rule["models"]["post_precision"])
    if post_model not in POST_MODELS:
        raise ValueError(f"unsupported post-precision model: {post_model}")

    by_dataset = {}
    all_selected = []
    for dataset in sorted({key[0] for key in outcomes}):
        keys = [key for key in outcomes if key[0] == dataset]
        selected = [
            outcomes[key]
            for key in keys
            if _selected(
                key,
                outcomes[key],
                features[key],
                predictions,
                baseline_range=(float(baseline_range[0]), float(baseline_range[1])),
                capture_threshold=capture_threshold,
                post_threshold=post_threshold,
                post_model=post_model,
            )
        ]
        all_selected.extend(selected)
        metrics = _metrics(selected, [outcomes[key] for key in keys])
        metrics["component_bootstrap"] = _cluster_precision_interval(
            selected, seed=_seed("release-rule", dataset)
        )
        metrics["domain_supported_at_freeze"] = dataset in set(
            rule["supported_domains_at_freeze"]
        )
        metrics["passes_release_evidence_gate"] = (
            metrics["domain_supported_at_freeze"]
            and metrics["selected_components"] >= MINIMUM_COMPONENTS
            and metrics["exact_one_sided_95_lower"] >= TARGET_EXACT_LOWER
            and metrics["catastrophic_accepted"] == 0
        )
        by_dataset[dataset] = metrics

    pooled = _metrics(all_selected, list(outcomes.values()))
    pooled["component_bootstrap"] = _cluster_precision_interval(
        all_selected, seed=_seed("release-rule", "pooled")
    )
    supported_results = [
        value for value in by_dataset.values() if value["domain_supported_at_freeze"]
    ]
    verdict = (
        "GO_FOR_PROSPECTIVE_CONFIRMATION"
        if supported_results
        and all(value["passes_release_evidence_gate"] for value in supported_results)
        else "NO_GO"
    )
    result = {
        "schema": EVALUATION_SCHEMA,
        "status": "retrospective untouched-split evaluation; not confirmatory",
        "verdict": verdict,
        "verdict_interpretation": (
            "This rule is not a release claim. A GO result would only authorize E8 "
            "prospective confirmation on new independent groups."
        ),
        "rule": {"path": str(rule_path), "sha256": _sha256(rule_path)},
        "inputs": {
            "features": {"path": str(features_path), "sha256": _sha256(features_path)},
            "predictions": {
                "path": str(predictions_path),
                "sha256": _sha256(predictions_path),
            },
            "evaluation_outcomes": {
                "path": str(outcomes_path),
                "sha256": _sha256(outcomes_path),
            },
        },
        "datasets": by_dataset,
        "pooled_descriptive_only": pooled,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output_dir / "release-rule-evaluation.json", result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--features", type=Path, required=True)
    freeze.add_argument("--predictions", type=Path, required=True)
    freeze.add_argument("--training-outcomes", type=Path, required=True)
    freeze.add_argument("--post-model", choices=POST_MODELS, default=POST_MODEL)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.set_defaults(handler=freeze_rule)
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--rule", type=Path, required=True)
    evaluation.add_argument("--features", type=Path, required=True)
    evaluation.add_argument("--predictions", type=Path, required=True)
    evaluation.add_argument("--evaluation-outcomes", type=Path, required=True)
    evaluation.add_argument("--output-dir", type=Path, required=True)
    evaluation.set_defaults(handler=evaluate_rule)
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
