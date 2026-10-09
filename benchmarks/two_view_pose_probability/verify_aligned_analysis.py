#!/usr/bin/env python3
"""Independently verify a completed sealed VAL-018 aligned analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


DATASET_COUNTS = {
    "matterport360": 1890,
    "stanford2d3d": 450,
    "p74_native_polar": 45,
}
EXPECTED_CENSUS = {
    "matterport360": {
        "unique_images": 3305,
        "unique_pairs": 1890,
        "independence_components": 63,
    },
    "stanford2d3d": {
        "unique_images": 638,
        "unique_pairs": 450,
        "independence_components": 3,
    },
    "p74_native_polar": {
        "unique_images": 74,
        "unique_pairs": 45,
        "independence_components": 3,
    },
}
EXPECTED_CENSUS_TOTALS = {
    "datasets": 3,
    "unique_images": 4017,
    "unique_pairs": 2385,
    "independence_components": 69,
    "spatial_groups": 69,
}
EXPECTED_PAIRS = sum(DATASET_COUNTS.values())
ALIGNED_POST_MODEL = "post-precise-aligned-orientation"
EXPECTED_MODEL_IDS = {
    "capture-accept-overlap-baseline",
    "capture-precise-given-accept-overlap-baseline",
    "post-precise-raw-score",
    "post-precise-support",
    "post-precise-common",
    "post-precise-public-full",
    ALIGNED_POST_MODEL,
}
EXPECTED_MODEL_CONTRACTS = {
    "capture-accept-overlap-baseline": ("accepted", "eligible"),
    "capture-precise-given-accept-overlap-baseline": ("precise", "accepted"),
    "capture-accept-public-full": ("accepted", "eligible"),
    "capture-precise-given-accept-public-full": ("precise", "accepted"),
    "post-precise-raw-score": ("precise", "returned"),
    "post-precise-support": ("precise", "returned"),
    "post-precise-common": ("precise", "returned"),
    "post-precise-public-full": ("precise", "returned"),
    ALIGNED_POST_MODEL: ("precise", "returned"),
}
NATIVE_ROUTE = {
    "convolution_backend": "native",
    "native_filter_available": True,
    "native_pose_kernels_available": True,
    "numpy_fallback_permitted": False,
}
CAPTURE_FEATURE_POLICY = {
    "capture.registered_cloud_overlap_min": {
        "transform": "logit",
        "availability": "capture-system-visible registered depth/cloud",
        "deployment_profile": "scanner-assisted only",
    },
    "capture.baseline_m": {
        "transform": "log1p",
        "availability": "operator-controlled or external-tracker-visible",
        "deployment_profile": "scanner-assisted or tracked RGB",
    },
    "capture.baseline_depth_ratio": {
        "transform": "log1p",
        "availability": "derived from controlled baseline and visible depth",
        "deployment_profile": "depth-assisted only",
    },
    "capture.rgb_similarity": {
        "transform": "logit",
        "availability": "image-visible before matching and pose",
        "deployment_profile": "RGB preview",
    },
}
PAPER_SCOPE_MARKERS = (
    "Use exactly two calibrated spherical panoramas",
    "scanner-assisted capture variable only",
    "must not be advertised as an RGB-only observable",
    "Plan or measure baseline independently of the reference pose",
    "Runtime and memory are engineering outcomes only",
    "Translation magnitude is not scored",
    "Prospective confirmation still required",
    "No result in this document validates dense stereo",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
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


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, condition: bool, evidence: Any) -> None:
        self.checks.append(
            {"name": name, "passed": bool(condition), "evidence": evidence}
        )

    @property
    def passed(self) -> bool:
        return all(check["passed"] for check in self.checks)


def _capture_policy_violations(
    models: list[dict[str, Any]], *, card_label: str
) -> list[dict[str, Any]]:
    violations = []
    for model in models:
        model_id = str(model.get("model_id", ""))
        base_model_id = str(model.get("base_model_id", model_id))
        if not base_model_id.startswith("capture-"):
            continue
        features = model.get("features")
        if not isinstance(features, list) or not features:
            violations.append(
                {
                    "card": card_label,
                    "model_id": model_id,
                    "reason": "missing capture feature declaration",
                }
            )
            continue
        for feature in features:
            path = str(feature.get("path", ""))
            transform = str(feature.get("transform", ""))
            policy = CAPTURE_FEATURE_POLICY.get(path)
            if policy is None:
                violations.append(
                    {
                        "card": card_label,
                        "model_id": model_id,
                        "path": path,
                        "reason": "not operator-controlled/visible allowlist",
                    }
                )
            elif transform != policy["transform"]:
                violations.append(
                    {
                        "card": card_label,
                        "model_id": model_id,
                        "path": path,
                        "reason": "transform differs from frozen policy",
                        "expected": policy["transform"],
                        "actual": transform,
                    }
                )
    return violations


def _model_contract_violations(
    models: list[dict[str, Any]], *, card_label: str
) -> list[dict[str, Any]]:
    violations = []
    for model in models:
        model_id = str(model.get("model_id", ""))
        base_model_id = str(model.get("base_model_id", model_id))
        expected = EXPECTED_MODEL_CONTRACTS.get(base_model_id)
        if expected is None:
            violations.append(
                {
                    "card": card_label,
                    "model_id": model_id,
                    "reason": "unknown probability-model contract",
                }
            )
            continue
        actual = (str(model.get("target")), str(model.get("population")))
        if actual != expected:
            violations.append(
                {
                    "card": card_label,
                    "model_id": model_id,
                    "reason": "target/population contract mismatch",
                    "expected": list(expected),
                    "actual": list(actual),
                }
            )
    return violations


def _missing_paper_scope_markers(document: str) -> list[str]:
    return [marker for marker in PAPER_SCOPE_MARKERS if marker not in document]


def _missing_census_markers(
    document: str, totals: dict[str, int]
) -> list[str]:
    normalized = " ".join(document.split())
    markers = (
        f"{totals['unique_images']:,} unique images",
        f"{totals['unique_pairs']:,} unordered pairs",
        f"{totals['independence_components']} independence components",
    )
    return [marker for marker in markers if marker not in normalized]


def _uncertainty_violations(
    evaluation: dict[str, Any], *, evaluation_label: str
) -> list[dict[str, Any]]:
    violations = []
    collections = {
        "model": evaluation.get("models", {}),
        "usable_product": evaluation.get("usable_products", {}),
    }
    for collection_name, collection in collections.items():
        for model_id, model in collection.items():
            datasets = model.get("datasets", {})
            for dataset, metrics in datasets.items():
                prefix = {
                    "evaluation": evaluation_label,
                    "collection": collection_name,
                    "model_id": model_id,
                    "dataset": dataset,
                }
                required_metrics = ("brier", "log_loss", "ece_10")
                if any(
                    not math.isfinite(float(metrics.get(field, math.nan)))
                    for field in required_metrics
                ):
                    violations.append({**prefix, "reason": "non-finite proper score"})
                count = int(metrics.get("count", -1))
                groups = int(metrics.get("independence_components", -1))
                bins = metrics.get("reliability_bins")
                if (
                    count < 1
                    or groups < 1
                    or not isinstance(bins, list)
                    or len(bins) != 10
                    or sum(int(row.get("count", -1)) for row in bins) != count
                    or any(
                        not 0
                        <= int(row.get("independence_components", -1))
                        <= groups
                        for row in bins
                    )
                ):
                    violations.append(
                        {**prefix, "reason": "invalid reliability-bin support"}
                    )
                bootstrap = metrics.get("component_bootstrap")
                if groups < 5:
                    if bootstrap is not None:
                        violations.append(
                            {
                                **prefix,
                                "reason": "bootstrap reported with fewer than five groups",
                            }
                        )
                    continue
                valid_bootstrap = (
                    isinstance(bootstrap, dict)
                    and bootstrap.get("unit") == "independence_component"
                    and bootstrap.get("component_count") == groups
                    and bootstrap.get("repetitions") == 10_000
                    and isinstance(bootstrap.get("seed"), int)
                )
                intervals = (
                    bootstrap.get("percentile_95", {})
                    if isinstance(bootstrap, dict)
                    else {}
                )
                for field in ("brier", "log_loss"):
                    interval = intervals.get(field)
                    valid_bootstrap = valid_bootstrap and (
                        isinstance(interval, list)
                        and len(interval) == 2
                        and all(math.isfinite(float(value)) for value in interval)
                        and float(interval[0]) <= float(interval[1])
                    )
                if not valid_bootstrap:
                    violations.append(
                        {**prefix, "reason": "missing/invalid component bootstrap"}
                    )
    return violations


def _in_population(outcome: dict[str, Any], population: str) -> bool:
    if population == "eligible":
        return True
    if population == "returned":
        return bool(outcome["returned"])
    if population == "accepted":
        return bool(outcome["accepted"])
    raise ValueError(f"unknown probability population: {population}")


def _independent_probability_metrics(
    samples: list[tuple[float, float, str]]
) -> dict[str, Any]:
    if not samples:
        raise ValueError("probability metric sample is empty")
    target = [sample[0] for sample in samples]
    probability = [sample[1] for sample in samples]
    groups = [sample[2] for sample in samples]
    clipped = [min(max(value, 1e-12), 1.0 - 1e-12) for value in probability]
    count = len(samples)
    reliability = []
    ece = 0.0
    for index in range(10):
        lower = index / 10
        upper = (index + 1) / 10
        selected = [
            position
            for position, value in enumerate(probability)
            if value >= lower and (value <= upper if index == 9 else value < upper)
        ]
        predicted_mean = (
            sum(probability[position] for position in selected) / len(selected)
            if selected
            else None
        )
        observed_rate = (
            sum(target[position] for position in selected) / len(selected)
            if selected
            else None
        )
        if selected:
            ece += (len(selected) / count) * abs(predicted_mean - observed_rate)
        reliability.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(selected),
                "independence_components": len(
                    {groups[position] for position in selected}
                ),
                "predicted_mean": predicted_mean,
                "observed_rate": observed_rate,
            }
        )
    return {
        "count": count,
        "positives": int(sum(target)),
        "prevalence": sum(target) / count,
        "predicted_mean": sum(probability) / count,
        "brier": sum(
            (prediction - observed) ** 2
            for prediction, observed in zip(probability, target, strict=True)
        )
        / count,
        "log_loss": -sum(
            observed * math.log(prediction)
            + (1.0 - observed) * math.log1p(-prediction)
            for prediction, observed in zip(clipped, target, strict=True)
        )
        / count,
        "ece_10": ece,
        "independence_components": len(set(groups)),
        "reliability_bins": reliability,
    }


def _same_number(first: Any, second: Any) -> bool:
    if first is None or second is None:
        return first is second
    try:
        return math.isclose(
            float(first), float(second), rel_tol=1e-10, abs_tol=1e-12
        )
    except (TypeError, ValueError):
        return False


def _metric_mismatches(
    reported: dict[str, Any],
    expected: dict[str, Any],
    *,
    context: dict[str, str],
) -> list[dict[str, Any]]:
    mismatches = []
    for field in (
        "count",
        "positives",
        "prevalence",
        "predicted_mean",
        "brier",
        "log_loss",
        "ece_10",
        "independence_components",
    ):
        if not _same_number(reported.get(field), expected[field]):
            mismatches.append(
                {
                    **context,
                    "field": field,
                    "reported": reported.get(field),
                    "recomputed": expected[field],
                }
            )
    reported_bins = reported.get("reliability_bins")
    if not isinstance(reported_bins, list) or len(reported_bins) != 10:
        mismatches.append({**context, "field": "reliability_bins"})
        return mismatches
    for index, (reported_bin, expected_bin) in enumerate(
        zip(reported_bins, expected["reliability_bins"], strict=True)
    ):
        for field in (
            "lower",
            "upper",
            "count",
            "independence_components",
            "predicted_mean",
            "observed_rate",
        ):
            if not _same_number(reported_bin.get(field), expected_bin[field]):
                mismatches.append(
                    {
                        **context,
                        "field": f"reliability_bins[{index}].{field}",
                        "reported": reported_bin.get(field),
                        "recomputed": expected_bin[field],
                    }
                )
    return mismatches


def _recompute_evaluation_violations(
    evaluation: dict[str, Any],
    predictions: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    *,
    evaluation_label: str,
) -> list[dict[str, Any]]:
    violations = []
    outcome_index = _index(outcomes, ("dataset_id", "pair_id"), "evaluation outcome")
    evaluation_predictions = [row for row in predictions if row["split"] == "evaluation"]
    prediction_index = _index(
        evaluation_predictions,
        ("model_id", "dataset_id", "pair_id"),
        f"{evaluation_label} prediction",
    )
    samples: dict[tuple[str, str], list[tuple[float, float, str]]] = {}
    for prediction in evaluation_predictions:
        key = (str(prediction["dataset_id"]), str(prediction["pair_id"]))
        outcome = outcome_index.get(key)
        context = {
            "evaluation": evaluation_label,
            "model_id": str(prediction["model_id"]),
            "dataset": key[0],
            "pair_id": key[1],
        }
        base_model_id = str(
            prediction.get("base_model_id", prediction["model_id"])
        )
        expected_contract = EXPECTED_MODEL_CONTRACTS.get(base_model_id)
        actual_contract = (
            str(prediction.get("target")),
            str(prediction.get("population")),
        )
        if expected_contract is None or actual_contract != expected_contract:
            violations.append(
                {
                    **context,
                    "reason": "prediction target/population contract mismatch",
                    "base_model_id": base_model_id,
                    "expected": (
                        list(expected_contract) if expected_contract is not None else None
                    ),
                    "actual": list(actual_contract),
                }
            )
            continue
        if outcome is None:
            violations.append({**context, "reason": "prediction has no outcome"})
            continue
        probability = float(prediction["probability"])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            violations.append({**context, "reason": "probability outside [0,1]"})
            continue
        if prediction["independence_component_id"] != outcome[
            "independence_component_id"
        ]:
            violations.append({**context, "reason": "group identity mismatch"})
        if not _in_population(outcome, str(prediction["population"])):
            continue
        target_name = str(prediction["target"])
        if target_name not in outcome:
            violations.append({**context, "reason": "unknown target"})
            continue
        samples.setdefault((str(prediction["model_id"]), key[0]), []).append(
            (
                float(outcome[target_name]),
                probability,
                str(outcome["independence_component_id"]),
            )
        )
    reported_models = evaluation.get("models", {})
    expected_model_ids = {model_id for model_id, _ in samples}
    if set(reported_models) != expected_model_ids:
        violations.append(
            {
                "evaluation": evaluation_label,
                "reason": "reported/recomputed model set differs",
                "reported": sorted(reported_models),
                "recomputed": sorted(expected_model_ids),
            }
        )
    model_metrics: dict[str, list[dict[str, Any]]] = {}
    for (model_id, dataset), values in samples.items():
        expected = _independent_probability_metrics(values)
        model_metrics.setdefault(model_id, []).append(expected)
        reported = reported_models.get(model_id, {}).get("datasets", {}).get(dataset)
        context = {
            "evaluation": evaluation_label,
            "model_id": model_id,
            "dataset": dataset,
        }
        if not isinstance(reported, dict):
            violations.append({**context, "reason": "missing reported metrics"})
            continue
        violations.extend(_metric_mismatches(reported, expected, context=context))
    for model_id, metrics in model_metrics.items():
        reported = reported_models.get(model_id, {})
        for field, metric_field in (
            ("macro_brier", "brier"),
            ("macro_log_loss", "log_loss"),
            ("macro_ece_10", "ece_10"),
        ):
            expected = sum(row[metric_field] for row in metrics) / len(metrics)
            if not _same_number(reported.get(field), expected):
                violations.append(
                    {
                        "evaluation": evaluation_label,
                        "model_id": model_id,
                        "field": field,
                        "reported": reported.get(field),
                        "recomputed": expected,
                    }
                )
    usable_mapping = {
        "capture-usable-overlap-baseline": (
            "capture-accept-overlap-baseline",
            "capture-precise-given-accept-overlap-baseline",
        ),
        "capture-usable-public-full": (
            "capture-accept-public-full",
            "capture-precise-given-accept-public-full",
        ),
    }
    for dataset in DATASET_COUNTS:
        usable_mapping[f"capture-usable--lodo-{dataset}"] = (
            f"capture-accept-overlap-baseline--lodo-{dataset}",
            f"capture-precise-given-accept-overlap-baseline--lodo-{dataset}",
        )
    reported_usable = evaluation.get("usable_products", {})
    expected_usable_ids = {
        usable_id
        for usable_id, (accept_id, precise_id) in usable_mapping.items()
        if any(
            (accept_id, *key) in prediction_index
            and (precise_id, *key) in prediction_index
            for key in outcome_index
        )
    }
    if set(reported_usable) != expected_usable_ids:
        violations.append(
            {
                "evaluation": evaluation_label,
                "reason": "reported/recomputed usable-product set differs",
                "reported": sorted(reported_usable),
                "recomputed": sorted(expected_usable_ids),
            }
        )
    for usable_id, usable in reported_usable.items():
        component_ids = usable_mapping.get(usable_id)
        if component_ids is None:
            violations.append(
                {
                    "evaluation": evaluation_label,
                    "model_id": usable_id,
                    "reason": "unknown usable probability product",
                }
            )
            continue
        if "p_accept * p_precise_given_accept" not in str(usable.get("definition")):
            violations.append(
                {
                    "evaluation": evaluation_label,
                    "model_id": usable_id,
                    "reason": "usable definition is not the frozen probability product",
                }
            )
        accept_id, precise_id = component_ids
        by_dataset: dict[str, list[tuple[float, float, str]]] = {}
        for key, outcome in outcome_index.items():
            accept = prediction_index.get((accept_id, *key))
            precise = prediction_index.get((precise_id, *key))
            if accept is None or precise is None:
                continue
            by_dataset.setdefault(key[0], []).append(
                (
                    float(outcome["usable"]),
                    float(accept["probability"]) * float(precise["probability"]),
                    str(outcome["independence_component_id"]),
                )
            )
        usable_metrics = []
        for dataset, values in by_dataset.items():
            expected = _independent_probability_metrics(values)
            usable_metrics.append(expected)
            reported = usable.get("datasets", {}).get(dataset)
            context = {
                "evaluation": evaluation_label,
                "model_id": usable_id,
                "dataset": dataset,
            }
            if not isinstance(reported, dict):
                violations.append({**context, "reason": "missing usable metrics"})
                continue
            violations.extend(_metric_mismatches(reported, expected, context=context))
        if not usable_metrics:
            violations.append(
                {
                    "evaluation": evaluation_label,
                    "model_id": usable_id,
                    "reason": "usable product has no paired component predictions",
                }
            )
            continue
        for field, metric_field in (
            ("macro_brier", "brier"),
            ("macro_log_loss", "log_loss"),
        ):
            expected = sum(row[metric_field] for row in usable_metrics) / len(
                usable_metrics
            )
            if not _same_number(usable.get(field), expected):
                violations.append(
                    {
                        "evaluation": evaluation_label,
                        "model_id": usable_id,
                        "field": field,
                        "reported": usable.get(field),
                        "recomputed": expected,
                    }
                )
    return violations


def _verify_artifact(
    audit: Audit, label: str, record: dict[str, Any]
) -> Path | None:
    path = Path(record["path"])
    exists = path.is_file()
    audit.check(f"artifact exists: {label}", exists, str(path))
    if not exists:
        return None
    actual = _sha256(path)
    audit.check(
        f"artifact hash: {label}",
        actual == record["sha256"],
        {"expected": record["sha256"], "actual": actual},
    )
    return path


def _verify_raw_results(
    audit: Audit,
    table_manifest: dict[str, Any],
    *,
    expected_package_version: str,
    expected_source_commit: str,
) -> None:
    results_dir = Path(table_manifest["results_dir"])
    expected_hashes = table_manifest["result_sha256"]
    paths = sorted(results_dir.glob("*.json"))
    audit.check(
        "raw result count",
        len(paths) == EXPECTED_PAIRS == len(expected_hashes),
        {"files": len(paths), "manifest_hashes": len(expected_hashes)},
    )
    invalid = []
    for path in paths:
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            route = result.get("route", {}).get("route", {})
            valid = (
                expected_hashes.get(path.name) == _sha256(path)
                and result.get("schema") == "panorai-unified-optimized-pair/v2"
                and result.get("package", {}).get("version")
                == expected_package_version
                and result.get("package", {}).get("expected_source_commit")
                == expected_source_commit
                and result.get("native") == NATIVE_ROUTE
                and route.get("detector_method") == "detect_batch"
                and route.get("patch_provider_max_workers") == 4
                and isinstance(result.get("matching_diagnostics"), dict)
            )
            pose = result.get("pose", {})
            if valid and pose.get("returned"):
                valid = isinstance(
                    (pose.get("quality_report") or {}).get(
                        "translation_orientation"
                    ),
                    dict,
                )
            if not valid:
                invalid.append(path.name)
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            invalid.append(path.name)
    audit.check(
        "all raw results match frozen native route",
        not invalid,
        {"invalid_count": len(invalid), "first_five": invalid[:5]},
    )


def _verify_tables(
    audit: Audit, analysis_dir: Path, census: dict[str, Any]
) -> None:
    table_dir = analysis_dir / "aligned-table"
    features = _read_jsonl(table_dir / "features.jsonl")
    outcomes = _read_jsonl(table_dir / "outcomes.jsonl")
    training = _read_jsonl(table_dir / "outcomes-development-calibration.jsonl")
    evaluation = _read_jsonl(table_dir / "outcomes-evaluation.jsonl")
    tables = _read_jsonl(table_dir / "analysis-table.jsonl")
    feature_index = _index(features, ("dataset_id", "pair_id"), "feature")
    outcome_index = _index(outcomes, ("dataset_id", "pair_id"), "outcome")
    training_index = _index(training, ("dataset_id", "pair_id"), "training outcome")
    evaluation_index = _index(
        evaluation, ("dataset_id", "pair_id"), "evaluation outcome"
    )
    table_index = _index(tables, ("dataset_id", "pair_id"), "analysis row")
    full_keys = set(feature_index)
    audit.check(
        "aligned table denominators",
        len(full_keys) == EXPECTED_PAIRS
        and set(outcome_index) == full_keys
        and set(table_index) == full_keys,
        {
            "features": len(feature_index),
            "outcomes": len(outcome_index),
            "analysis_rows": len(table_index),
        },
    )
    audit.check(
        "training/evaluation outcome partition",
        not (set(training_index) & set(evaluation_index))
        and set(training_index) | set(evaluation_index) == full_keys
        and all(row["split"] != "evaluation" for row in training)
        and all(row["split"] == "evaluation" for row in evaluation),
        {"training": len(training), "evaluation": len(evaluation)},
    )
    forbidden_feature_fields = {
        "outcomes",
        "rotation_error_deg",
        "translation_direction_error_deg",
        "precise",
        "accepted",
    }
    leaks = [
        key
        for key, row in feature_index.items()
        if forbidden_feature_fields & set(row)
    ]
    audit.check(
        "prediction feature table contains no outcomes",
        not leaks,
        {"leak_count": len(leaks), "first_five": leaks[:5]},
    )
    group_splits: dict[str, set[str]] = {}
    for row in outcomes:
        group_splits.setdefault(str(row["independence_component_id"]), set()).add(
            str(row["split"])
        )
    split_groups = [group for group, splits in group_splits.items() if len(splits) > 1]
    audit.check(
        "independence components do not cross splits",
        not split_groups,
        {"crossing_groups": split_groups[:5]},
    )
    audit.check(
        "census leakage audit is clean",
        all(value == 0 for value in census["audit"].values()),
        census["audit"],
    )


def _verify_census(audit: Audit, census: dict[str, Any]) -> None:
    observed = {
        dataset: {
            field: census.get("datasets", {}).get(dataset, {}).get(field)
            for field in values
        }
        for dataset, values in EXPECTED_CENSUS.items()
    }
    audit.check(
        "census identifies exact image, pair and independent-group sample size",
        observed == EXPECTED_CENSUS
        and census.get("totals") == EXPECTED_CENSUS_TOTALS,
        {
            "expected_datasets": EXPECTED_CENSUS,
            "observed_datasets": observed,
            "expected_totals": EXPECTED_CENSUS_TOTALS,
            "observed_totals": census.get("totals"),
        },
    )


def _verify_models(audit: Audit, analysis_dir: Path) -> None:
    models_dir = analysis_dir / "models"
    card = json.loads((models_dir / "model-card.json").read_text(encoding="utf-8"))
    model_ids = {model["model_id"] for model in card["models"]}
    audit.check(
        "aligned model profile and declared models",
        card.get("model_profile") == "aligned" and EXPECTED_MODEL_IDS <= model_ids,
        {"profile": card.get("model_profile"), "models": sorted(model_ids)},
    )
    predictions = _read_jsonl(models_dir / "predictions.jsonl")
    prediction_index = _index(
        predictions, ("model_id", "dataset_id", "pair_id"), "prediction"
    )
    audit.check(
        "model prediction rows are unique",
        len(prediction_index) == len(predictions),
        {"rows": len(predictions)},
    )
    lodo_dir = analysis_dir / "models-lodo"
    lodo_card = json.loads((lodo_dir / "model-card.json").read_text(encoding="utf-8"))
    contract_violations = _model_contract_violations(
        card["models"], card_label="primary"
    ) + _model_contract_violations(lodo_card["models"], card_label="lodo")
    audit.check(
        "probability models preserve frozen target and conditioning populations",
        not contract_violations,
        {
            "violations": contract_violations,
            "contracts": EXPECTED_MODEL_CONTRACTS,
        },
    )
    policy_violations = _capture_policy_violations(
        card["models"], card_label="primary"
    ) + _capture_policy_violations(lodo_card["models"], card_label="lodo")
    audit.check(
        "capture models use only operator-controlled or operator-visible predictors",
        not policy_violations,
        {
            "violations": policy_violations,
            "allowlist": CAPTURE_FEATURE_POLICY,
            "scope_note": (
                "registered-cloud overlap is deployment-visible only in the "
                "scanner-assisted profile; it is a reference difficulty variable "
                "for RGB-only captures"
            ),
        },
    )
    lodo_ok = lodo_card.get("model_profile") == "aligned"
    violations = []
    for model in lodo_card["models"]:
        held_out = model["held_out_dataset"]
        if (
            model.get("target_outcomes_opened_during_fit") is not False
            or held_out in model.get("source_datasets", [])
        ):
            violations.append(model["model_id"])
    audit.check(
        "LODO fitting excludes target outcomes",
        lodo_ok and not violations,
        {"profile": lodo_card.get("model_profile"), "violations": violations},
    )


def _verify_evaluation(audit: Audit, analysis_dir: Path) -> None:
    evaluation = json.loads(
        (analysis_dir / "evaluation" / "evaluation.json").read_text(encoding="utf-8")
    )
    lodo = json.loads(
        (analysis_dir / "evaluation-lodo" / "evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    audit.check(
        "primary evaluation contains aligned post and usable capture models",
        ALIGNED_POST_MODEL in evaluation["models"]
        and "capture-usable-overlap-baseline" in evaluation["usable_products"],
        {
            "models": sorted(evaluation["models"]),
            "usable": sorted(evaluation["usable_products"]),
        },
    )
    expected_lodo = {
        f"{ALIGNED_POST_MODEL}--lodo-{dataset}" for dataset in DATASET_COUNTS
    }
    audit.check(
        "LODO evaluation contains all aligned target domains",
        expected_lodo <= set(lodo["models"]),
        {"expected": sorted(expected_lodo)},
    )
    uncertainty_violations = _uncertainty_violations(
        evaluation, evaluation_label="component-held-out"
    ) + _uncertainty_violations(lodo, evaluation_label="leave-one-dataset-out")
    audit.check(
        "calibration metrics report group-aware uncertainty when supported",
        not uncertainty_violations,
        {
            "violations": uncertainty_violations,
            "bootstrap_minimum_groups": 5,
            "bootstrap_repetitions": 10_000,
            "bootstrap_unit": "independence_component",
        },
    )
    outcomes = _read_jsonl(
        analysis_dir / "aligned-table" / "outcomes-evaluation.jsonl"
    )
    recomputation_violations = _recompute_evaluation_violations(
        evaluation,
        _read_jsonl(analysis_dir / "models" / "predictions.jsonl"),
        outcomes,
        evaluation_label="component-held-out",
    ) + _recompute_evaluation_violations(
        lodo,
        _read_jsonl(analysis_dir / "models-lodo" / "predictions.jsonl"),
        outcomes,
        evaluation_label="leave-one-dataset-out",
    )
    audit.check(
        "probability scores and usable products reproduce independently",
        not recomputation_violations,
        {"violations": recomputation_violations},
    )


def _verify_release_and_paper(
    audit: Audit,
    analysis_dir: Path,
    manifest: dict[str, Any],
    census: dict[str, Any],
) -> None:
    qualified = bool(manifest["rule_qualified_on_calibration"])
    release_dir = analysis_dir / "release-rule"
    if qualified:
        rule = json.loads((release_dir / "release-rule.json").read_text())
        release = json.loads(
            (release_dir / "release-rule-evaluation.json").read_text()
        )
        audit.check(
            "selective rule uses aligned post model",
            rule["models"]["post_precision"] == ALIGNED_POST_MODEL,
            rule["models"],
        )
        audit.check(
            "retrospective release verdict is non-release",
            release["verdict"] in {"NO_GO", "GO_FOR_PROSPECTIVE_CONFIRMATION"},
            release["verdict"],
        )
        audit.check(
            "prospective plan exists after qualifying calibration rule",
            (analysis_dir / "prospective-plan" / "prospective-power-plan.json").is_file(),
            str(analysis_dir / "prospective-plan"),
        )
    else:
        search = json.loads((release_dir / "release-rule-search.json").read_text())
        audit.check(
            "failed rule search retains aligned model and full grid",
            search["post_precision_model"] == ALIGNED_POST_MODEL
            and bool(search["candidate_grid"])
            and not any(row["qualifies"] for row in search["candidate_grid"]),
            {
                "model": search["post_precision_model"],
                "candidates": len(search["candidate_grid"]),
            },
        )
    paper = json.loads(
        (analysis_dir / "paper-results" / "paper-results.json").read_text()
    )
    figures = {name: Path(path) for name, path in paper["figures"].items()}
    missing = [name for name, path in figures.items() if not path.is_file()]
    audit.check(
        "paper results use aligned post model and all figures exist",
        paper["primary_post_model"] == ALIGNED_POST_MODEL and not missing,
        {"primary_post_model": paper["primary_post_model"], "missing": missing},
    )
    document = (analysis_dir / "PAPER_RESULTS.md").read_text(encoding="utf-8")
    audit.check(
        "paper document declares retrospective scope and PanorAi 3.5.0",
        "retrospective validation" in document
        and "PanorAi `3.5.0`" in document
        and "Prospective confirmation still required" in document,
        {"characters": len(document)},
    )
    missing_scope = _missing_paper_scope_markers(document)
    audit.check(
        "paper preserves two-view and deployment-scope boundaries",
        not missing_scope,
        {"missing_markers": missing_scope},
    )
    missing_census = _missing_census_markers(document, census["totals"])
    audit.check(
        "paper states exact unique-image, pair and independent-group sample size",
        not missing_census,
        {"missing_markers": missing_census, "totals": census["totals"]},
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    analysis_dir = args.analysis_dir.resolve()
    census_path = args.census.resolve()
    audit = Audit()
    manifest_path = analysis_dir / "manifest.json"
    status_path = analysis_dir / "status.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    census = json.loads(census_path.read_text(encoding="utf-8"))
    audit.check(
        "runner completed",
        manifest.get("status") == "complete retrospective aligned analysis"
        and status.get("state") == "complete",
        {"manifest": manifest.get("status"), "status": status.get("state")},
    )
    audit.check(
        "frozen package identity",
        manifest.get("expected_package_version") == args.expected_package_version
        and manifest.get("expected_source_commit") == args.expected_source_commit,
        {
            "version": manifest.get("expected_package_version"),
            "commit": manifest.get("expected_source_commit"),
        },
    )
    for label, record in manifest["artifacts"].items():
        _verify_artifact(audit, label, record)
    lock = manifest["analysis_code_lock"]
    _verify_artifact(
        audit,
        "analysis code lock",
        {"path": lock["path"], "sha256": lock["sha256"]},
    )
    table_manifest = json.loads(
        (analysis_dir / "aligned-table" / "manifest.json").read_text()
    )
    audit.check(
        "aligned table population",
        table_manifest["pair_count"] == EXPECTED_PAIRS
        and table_manifest["dataset_counts"] == DATASET_COUNTS
        and table_manifest["expected_package_version"]
        == args.expected_package_version
        and table_manifest["expected_source_commit"] == args.expected_source_commit,
        {
            "pairs": table_manifest.get("pair_count"),
            "datasets": table_manifest.get("dataset_counts"),
        },
    )
    _verify_raw_results(
        audit,
        table_manifest,
        expected_package_version=args.expected_package_version,
        expected_source_commit=args.expected_source_commit,
    )
    _verify_census(audit, census)
    _verify_tables(audit, analysis_dir, census)
    _verify_models(audit, analysis_dir)
    _verify_evaluation(audit, analysis_dir)
    _verify_release_and_paper(audit, analysis_dir, manifest, census)
    result = {
        "schema": "panorai-two-view-pose-aligned-analysis-verification/v1",
        "status": "PASS" if audit.passed else "FAIL",
        "analysis_dir": str(analysis_dir),
        "analysis_manifest_sha256": _sha256(manifest_path),
        "census": {"path": str(census_path), "sha256": _sha256(census_path)},
        "checks": audit.checks,
        "passed_checks": sum(check["passed"] for check in audit.checks),
        "total_checks": len(audit.checks),
    }
    _atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not audit.passed:
        raise RuntimeError("aligned analysis verification failed")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--expected-package-version", default="3.5.0")
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
