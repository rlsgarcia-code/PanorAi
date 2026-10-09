#!/usr/bin/env python3
"""Fit, freeze and evaluate transparent VAL-018 probability models."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

PREDICTION_SCHEMA = "panorai-two-view-probability-prediction/v1"
CARD_SCHEMA = "panorai-two-view-probability-model-card/v1"
EVALUATION_SCHEMA = "panorai-two-view-probability-evaluation/v1"
L2_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0)
LODO_FIXED_L2 = 0.1
FOLD_SEED = "panorai-val018-model-v1"


def _model(
    model_id: str,
    *,
    target: str,
    population: str,
    features: tuple[tuple[str, str], ...],
    datasets: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "target": target,
        "population": population,
        "features": features,
        "datasets": datasets,
    }


MODELS = (
    _model(
        "capture-accept-overlap-baseline",
        target="accepted",
        population="eligible",
        features=(
            ("capture.registered_cloud_overlap_min", "logit"),
            ("capture.baseline_m", "log1p"),
        ),
    ),
    _model(
        "capture-precise-given-accept-overlap-baseline",
        target="precise",
        population="accepted",
        features=(
            ("capture.registered_cloud_overlap_min", "logit"),
            ("capture.baseline_m", "log1p"),
        ),
    ),
    _model(
        "capture-accept-public-full",
        target="accepted",
        population="eligible",
        datasets=("matterport360", "stanford2d3d"),
        features=(
            ("capture.registered_cloud_overlap_min", "logit"),
            ("capture.baseline_m", "log1p"),
            ("capture.baseline_depth_ratio", "log1p"),
            ("capture.rgb_similarity", "logit"),
        ),
    ),
    _model(
        "capture-precise-given-accept-public-full",
        target="precise",
        population="accepted",
        datasets=("matterport360", "stanford2d3d"),
        features=(
            ("capture.registered_cloud_overlap_min", "logit"),
            ("capture.baseline_m", "log1p"),
            ("capture.baseline_depth_ratio", "log1p"),
            ("capture.rgb_similarity", "logit"),
        ),
    ),
    _model(
        "post-precise-raw-score",
        target="precise",
        population="returned",
        features=(("post.raw_quality_score", "logit"),),
    ),
    _model(
        "post-precise-support",
        target="precise",
        population="returned",
        features=(
            ("post.raw_quality_score", "logit"),
            ("post.match_count", "log1p"),
            ("post.inlier_count", "log1p"),
            ("post.inlier_ratio", "logit"),
        ),
    ),
    _model(
        "post-precise-common",
        target="precise",
        population="returned",
        features=(
            ("capture.registered_cloud_overlap_min", "logit"),
            ("capture.baseline_m", "log1p"),
            ("post.raw_quality_score", "logit"),
            ("post.match_count", "log1p"),
            ("post.inlier_count", "log1p"),
            ("post.inlier_ratio", "logit"),
            ("post.median_parallax_deg", "log1p"),
            ("post.cheirality_ratio", "logit"),
        ),
    ),
    _model(
        "post-precise-public-full",
        target="precise",
        population="returned",
        datasets=("matterport360", "stanford2d3d"),
        features=(
            ("capture.registered_cloud_overlap_min", "logit"),
            ("capture.baseline_m", "log1p"),
            ("post.raw_quality_score", "logit"),
            ("post.match_count", "log1p"),
            ("post.inlier_count", "log1p"),
            ("post.inlier_ratio", "logit"),
            ("post.median_parallax_deg", "log1p"),
            ("post.cheirality_ratio", "logit"),
            ("post.median_residual_deg", "log1p"),
            ("post.p90_residual_deg", "log1p"),
            ("post.coverage_entropy_a", "identity"),
            ("post.coverage_entropy_b", "identity"),
            ("post.stability_success_fraction", "logit"),
            ("post.stability_rotation_p90_deg", "log1p"),
            ("post.stability_translation_p90_deg", "log1p"),
            ("post.essential_score_margin", "identity"),
            ("post.preferred_model_essential", "identity"),
        ),
    ),
)

ALIGNED_ONLY_MODELS = (
    _model(
        "post-precise-aligned-orientation",
        target="precise",
        population="returned",
        features=(
            ("post.raw_quality_score", "logit"),
            ("post.match_count", "log1p"),
            ("post.inlier_count", "log1p"),
            ("post.inlier_ratio", "logit"),
            ("post.median_parallax_deg", "log1p"),
            ("post.cheirality_ratio", "logit"),
            ("post.coverage_entropy_a", "identity"),
            ("post.coverage_entropy_b", "identity"),
            ("post.stability_translation_p90_deg", "log1p"),
            ("post.essential_score_margin", "identity"),
            ("post.translation_orientation_cheirality_margin", "identity"),
            ("post.translation_orientation_weighted_margin", "identity"),
            (
                "post.translation_orientation_median_triangulation_angle_deg",
                "log1p",
            ),
            ("post.translation_orientation_ambiguous", "identity"),
        ),
    ),
)

MODEL_PROFILES = {
    "historical": MODELS,
    "aligned": MODELS + ALIGNED_ONLY_MODELS,
}

LODO_DATASETS = ("matterport360", "stanford2d3d", "p74_native_polar")
LODO_MODEL_IDS = {
    "capture-accept-overlap-baseline",
    "capture-precise-given-accept-overlap-baseline",
    "post-precise-raw-score",
    "post-precise-support",
    "post-precise-common",
    "post-precise-aligned-orientation",
}


def models_for_profile(profile: str) -> tuple[dict[str, Any], ...]:
    try:
        return MODEL_PROFILES[profile]
    except KeyError as error:
        raise ValueError(f"unknown model profile: {profile}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise TypeError(f"{path} contains a non-object JSON value")
                rows.append(item)
    return rows


def _index(rows: Iterable[dict[str, Any]], label: str) -> dict[tuple[str, str], dict]:
    result = {}
    for row in rows:
        key = (str(row["dataset_id"]), str(row["pair_id"]))
        if key in result:
            raise ValueError(f"duplicate {label} key: {key}")
        result[key] = row
    return result


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _get(row: Mapping[str, Any], dotted: str) -> Any:
    value: Any = row
    for part in dotted.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _transform(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("feature is not finite")
    if name == "identity":
        return number
    if name == "log1p":
        if number < 0.0:
            raise ValueError("log1p feature is negative")
        return math.log1p(number)
    if name == "logit":
        clipped = min(max(number, 1e-4), 1.0 - 1e-4)
        return math.log(clipped / (1.0 - clipped))
    raise ValueError(f"unknown transform: {name}")


def design_matrix(
    rows: list[dict[str, Any]], features: tuple[tuple[str, str], ...]
) -> tuple[np.ndarray, np.ndarray]:
    values = []
    complete = []
    for row in rows:
        current = []
        ok = True
        for path, transform in features:
            value = _get(row, path)
            if value is None:
                ok = False
                break
            current.append(_transform(value, transform))
        values.append(current if ok else [0.0] * len(features))
        complete.append(ok)
    return np.asarray(values, dtype=np.float64), np.asarray(complete, dtype=bool)


def fit_scaler(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return mean, scale


def fit_logistic(values: np.ndarray, target: np.ndarray, *, l2: float) -> np.ndarray:
    if values.ndim != 2 or len(values) != len(target):
        raise ValueError("invalid logistic design")
    if len(np.unique(target)) != 2:
        raise ValueError("logistic target must contain both classes")
    augmented = np.column_stack((np.ones(len(values)), values))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        logits = augmented @ parameters
        loss = np.mean(np.logaddexp(0.0, logits) - target * logits)
        loss += 0.5 * l2 * float(parameters[1:] @ parameters[1:])
        probabilities = expit(logits)
        gradient = augmented.T @ (probabilities - target) / len(target)
        gradient[1:] += l2 * parameters[1:]
        return float(loss), gradient

    initial = np.zeros(augmented.shape[1], dtype=np.float64)
    prevalence = float(np.mean(target))
    initial[0] = math.log(prevalence / (1.0 - prevalence))
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"logistic fit failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def predict_logistic(values: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    return expit(parameters[0] + values @ parameters[1:])


def _log_loss(target: np.ndarray, probability: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return float(
        -np.mean(target * np.log(clipped) + (1.0 - target) * np.log1p(-clipped))
    )


def _brier(target: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean((probability - target) ** 2))


def _group_macro_metric(
    target: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    return float(
        np.mean(
            [
                metric(target[groups == group], probability[groups == group])
                for group in sorted(set(groups))
            ]
        )
    )


def _fold(group: str, folds: int = 5) -> int:
    digest = hashlib.sha256(f"{FOLD_SEED}|{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def select_l2(
    values: np.ndarray, target: np.ndarray, groups: np.ndarray
) -> tuple[float, list[dict[str, Any]]]:
    fold_ids = np.asarray([_fold(str(group)) for group in groups], dtype=np.int64)
    candidates = []
    for l2 in L2_GRID:
        predictions = np.full(len(target), np.nan, dtype=np.float64)
        used_folds = []
        for fold in range(5):
            validation = fold_ids == fold
            training = ~validation
            if not np.any(validation) or len(np.unique(target[training])) != 2:
                continue
            mean, scale = fit_scaler(values[training])
            parameters = fit_logistic(
                (values[training] - mean) / scale, target[training], l2=l2
            )
            predictions[validation] = predict_logistic(
                (values[validation] - mean) / scale, parameters
            )
            used_folds.append(fold)
        valid = np.isfinite(predictions)
        if not np.all(valid):
            raise RuntimeError(
                f"group CV left {np.count_nonzero(~valid)} rows unpredicted"
            )
        candidates.append(
            {
                "l2": l2,
                "used_folds": used_folds,
                "group_macro_log_loss": _group_macro_metric(
                    target, predictions, groups, _log_loss
                ),
                "group_macro_brier": _group_macro_metric(
                    target, predictions, groups, _brier
                ),
            }
        )
    best = min(
        candidates,
        key=lambda item: (
            item["group_macro_log_loss"],
            item["group_macro_brier"],
            item["l2"],
        ),
    )
    return float(best["l2"]), candidates


def select_l2_or_fixed(
    values: np.ndarray, target: np.ndarray, groups: np.ndarray
) -> tuple[float, list[dict[str, Any]], str]:
    """Tune by group only when at least five independent groups exist."""
    group_count = len(set(groups))
    if group_count < 5:
        return (
            LODO_FIXED_L2,
            [],
            f"fixed preregistered L2 because only {group_count} source groups exist",
        )
    selected, candidates = select_l2(values, target, groups)
    return selected, candidates, "five-fold deterministic group cross-validation"


def _eligible(feature: dict[str, Any], model: dict[str, Any]) -> bool:
    datasets = model["datasets"]
    return datasets is None or feature["dataset_id"] in datasets


def _in_population(outcome: Mapping[str, Any], population: str) -> bool:
    if population == "eligible":
        return True
    if population == "accepted":
        return bool(outcome["accepted"])
    if population == "returned":
        return bool(outcome["returned"])
    raise ValueError(f"unknown population: {population}")


def _platt_fit(probability: np.ndarray, target: np.ndarray) -> np.ndarray:
    logits = np.log(
        np.clip(probability, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - probability, 1e-6, 1.0)
    )
    return fit_logistic(logits[:, None], target, l2=1e-6)


def _platt_predict(probability: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    logits = np.log(
        np.clip(probability, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - probability, 1e-6, 1.0)
    )
    return predict_logistic(logits[:, None], parameters)


def fit_predict(args: argparse.Namespace) -> dict[str, Any]:
    features_path = args.features.resolve()
    outcomes_path = args.training_outcomes.resolve()
    features = _read_jsonl(features_path)
    outcomes = _index(_read_jsonl(outcomes_path), "training outcome")
    if any(row["split"] == "evaluation" for row in outcomes.values()):
        raise ValueError("training outcomes must not contain evaluation rows")
    predictions = []
    cards = []
    models = models_for_profile(args.model_profile)
    for model in models:
        candidate_features = [row for row in features if _eligible(row, model)]
        values, complete = design_matrix(candidate_features, model["features"])
        feature_by_key = {
            (row["dataset_id"], row["pair_id"]): (row, values[index], complete[index])
            for index, row in enumerate(candidate_features)
        }
        training_items = []
        for key, outcome in outcomes.items():
            if key not in feature_by_key or outcome["split"] not in {
                "development",
                "calibration",
            }:
                continue
            feature, value, is_complete = feature_by_key[key]
            if is_complete and _in_population(outcome, model["population"]):
                training_items.append((feature, outcome, value))
        development = [
            item for item in training_items if item[1]["split"] == "development"
        ]
        calibration = [
            item for item in training_items if item[1]["split"] == "calibration"
        ]
        x_development = np.asarray([item[2] for item in development], dtype=np.float64)
        y_development = np.asarray(
            [item[1][model["target"]] for item in development], dtype=np.float64
        )
        groups = np.asarray(
            [item[0]["independence_component_id"] for item in development], dtype=str
        )
        selected_l2, cv = select_l2(x_development, y_development, groups)
        mean, scale = fit_scaler(x_development)
        parameters = fit_logistic(
            (x_development - mean) / scale, y_development, l2=selected_l2
        )
        x_calibration = np.asarray([item[2] for item in calibration], dtype=np.float64)
        y_calibration = np.asarray(
            [item[1][model["target"]] for item in calibration], dtype=np.float64
        )
        raw_calibration = predict_logistic((x_calibration - mean) / scale, parameters)
        platt = _platt_fit(raw_calibration, y_calibration)
        model_predictions = 0
        incomplete = 0
        for feature, value, is_complete in feature_by_key.values():
            if not is_complete:
                incomplete += 1
                continue
            raw = float(
                predict_logistic(((value - mean) / scale)[None, :], parameters)[0]
            )
            probability = float(_platt_predict(np.asarray([raw]), platt)[0])
            predictions.append(
                {
                    "schema": PREDICTION_SCHEMA,
                    "model_id": model["model_id"],
                    "dataset_id": feature["dataset_id"],
                    "pair_id": feature["pair_id"],
                    "independence_component_id": feature["independence_component_id"],
                    "split": feature["split"],
                    "target": model["target"],
                    "population": model["population"],
                    "raw_probability": raw,
                    "probability": probability,
                }
            )
            model_predictions += 1
        cards.append(
            {
                "model_id": model["model_id"],
                "target": model["target"],
                "population": model["population"],
                "datasets": model["datasets"],
                "features": [
                    {"path": path, "transform": transform}
                    for path, transform in model["features"]
                ],
                "development_count": len(development),
                "development_groups": len(set(groups)),
                "development_prevalence": float(np.mean(y_development)),
                "calibration_count": len(calibration),
                "calibration_prevalence": float(np.mean(y_calibration)),
                "selected_l2": selected_l2,
                "cross_validation": cv,
                "scaler_mean": mean.tolist(),
                "scaler_scale": scale.tolist(),
                "parameters": parameters.tolist(),
                "platt_parameters": platt.tolist(),
                "prediction_count": model_predictions,
                "incomplete_feature_rows": incomplete,
            }
        )
    predictions.sort(
        key=lambda row: (row["model_id"], row["dataset_id"], row["pair_id"])
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    _atomic_text(
        predictions_path,
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in predictions
        ),
    )
    card = {
        "schema": CARD_SCHEMA,
        "study_status": "post-hoc validation; evaluation outcomes sealed from fitting command",
        "features": {"path": str(features_path), "sha256": _sha256(features_path)},
        "training_outcomes": {
            "path": str(outcomes_path),
            "sha256": _sha256(outcomes_path),
        },
        "fold_seed": FOLD_SEED,
        "model_profile": args.model_profile,
        "l2_grid": list(L2_GRID),
        "selection_rule": "minimum development group-macro log loss, then Brier, then L2",
        "calibration": "Platt logistic on frozen calibration components",
        "models": cards,
        "predictions": {
            "path": str(predictions_path.resolve()),
            "sha256": _sha256(predictions_path),
            "rows": len(predictions),
        },
    }
    card_path = args.output_dir / "model-card.json"
    _atomic_text(
        card_path, json.dumps(card, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(card, indent=2, sort_keys=True, allow_nan=False))
    return card


def _lodo_model_id(model_id: str, held_out_dataset: str) -> str:
    return f"{model_id}--lodo-{held_out_dataset}"


def fit_predict_lodo(args: argparse.Namespace) -> dict[str, Any]:
    """Fit common models without opening any outcome from the target dataset."""
    features_path = args.features.resolve()
    outcomes_path = args.training_outcomes.resolve()
    features = _read_jsonl(features_path)
    outcomes = _index(_read_jsonl(outcomes_path), "training outcome")
    if any(row["split"] == "evaluation" for row in outcomes.values()):
        raise ValueError("training outcomes must not contain evaluation rows")

    predictions = []
    cards = []
    for held_out_dataset in LODO_DATASETS:
        for model in models_for_profile(args.model_profile):
            if model["model_id"] not in LODO_MODEL_IDS:
                continue
            candidate_features = [row for row in features if _eligible(row, model)]
            values, complete = design_matrix(candidate_features, model["features"])
            feature_by_key = {
                (row["dataset_id"], row["pair_id"]): (
                    row,
                    values[index],
                    complete[index],
                )
                for index, row in enumerate(candidate_features)
            }
            training_items = []
            for key, outcome in outcomes.items():
                if (
                    key not in feature_by_key
                    or outcome["dataset_id"] == held_out_dataset
                ):
                    continue
                feature, value, is_complete = feature_by_key[key]
                if is_complete and _in_population(outcome, model["population"]):
                    training_items.append((feature, outcome, value))
            development = [
                item for item in training_items if item[1]["split"] == "development"
            ]
            calibration = [
                item for item in training_items if item[1]["split"] == "calibration"
            ]
            x_development = np.asarray(
                [item[2] for item in development], dtype=np.float64
            )
            y_development = np.asarray(
                [item[1][model["target"]] for item in development], dtype=np.float64
            )
            groups = np.asarray(
                [item[0]["independence_component_id"] for item in development],
                dtype=str,
            )
            selected_l2, cv, selection_method = select_l2_or_fixed(
                x_development, y_development, groups
            )
            mean, scale = fit_scaler(x_development)
            parameters = fit_logistic(
                (x_development - mean) / scale,
                y_development,
                l2=selected_l2,
            )
            x_calibration = np.asarray(
                [item[2] for item in calibration], dtype=np.float64
            )
            y_calibration = np.asarray(
                [item[1][model["target"]] for item in calibration], dtype=np.float64
            )
            raw_calibration = predict_logistic(
                (x_calibration - mean) / scale, parameters
            )
            if len(np.unique(y_calibration)) == 2:
                platt = _platt_fit(raw_calibration, y_calibration)
                calibration_method = "Platt logistic on source calibration components"
            else:
                platt = np.asarray([0.0, 1.0], dtype=np.float64)
                calibration_method = (
                    "identity; source calibration has one outcome class"
                )

            model_id = _lodo_model_id(model["model_id"], held_out_dataset)
            prediction_count = 0
            incomplete = 0
            for feature, value, is_complete in feature_by_key.values():
                if feature["dataset_id"] != held_out_dataset:
                    continue
                if not is_complete:
                    incomplete += 1
                    continue
                raw = float(
                    predict_logistic(((value - mean) / scale)[None, :], parameters)[0]
                )
                probability = float(_platt_predict(np.asarray([raw]), platt)[0])
                predictions.append(
                    {
                        "schema": PREDICTION_SCHEMA,
                        "model_id": model_id,
                        "base_model_id": model["model_id"],
                        "held_out_dataset": held_out_dataset,
                        "dataset_id": feature["dataset_id"],
                        "pair_id": feature["pair_id"],
                        "independence_component_id": feature[
                            "independence_component_id"
                        ],
                        "split": feature["split"],
                        "target": model["target"],
                        "population": model["population"],
                        "raw_probability": raw,
                        "probability": probability,
                    }
                )
                prediction_count += 1
            source_datasets = sorted({item[0]["dataset_id"] for item in training_items})
            cards.append(
                {
                    "model_id": model_id,
                    "base_model_id": model["model_id"],
                    "target": model["target"],
                    "population": model["population"],
                    "held_out_dataset": held_out_dataset,
                    "source_datasets": source_datasets,
                    "target_outcomes_opened_during_fit": False,
                    "features": [
                        {"path": path, "transform": transform}
                        for path, transform in model["features"]
                    ],
                    "development_count": len(development),
                    "development_groups": len(set(groups)),
                    "development_prevalence": float(np.mean(y_development)),
                    "calibration_count": len(calibration),
                    "calibration_groups": len(
                        {item[0]["independence_component_id"] for item in calibration}
                    ),
                    "calibration_prevalence": float(np.mean(y_calibration)),
                    "selected_l2": selected_l2,
                    "selection_method": selection_method,
                    "cross_validation": cv,
                    "calibration_method": calibration_method,
                    "scaler_mean": mean.tolist(),
                    "scaler_scale": scale.tolist(),
                    "parameters": parameters.tolist(),
                    "platt_parameters": platt.tolist(),
                    "prediction_count": prediction_count,
                    "incomplete_target_feature_rows": incomplete,
                }
            )

    predictions.sort(
        key=lambda row: (row["model_id"], row["dataset_id"], row["pair_id"])
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    _atomic_text(
        predictions_path,
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in predictions
        ),
    )
    card = {
        "schema": CARD_SCHEMA,
        "study_design": "leave-one-dataset-out",
        "study_status": (
            "post-hoc transfer validation; no outcome from a held-out dataset was "
            "opened by this fitting command"
        ),
        "features": {"path": str(features_path), "sha256": _sha256(features_path)},
        "training_outcomes": {
            "path": str(outcomes_path),
            "sha256": _sha256(outcomes_path),
        },
        "fold_seed": FOLD_SEED,
        "model_profile": args.model_profile,
        "l2_grid": list(L2_GRID),
        "fixed_l2_for_fewer_than_five_groups": LODO_FIXED_L2,
        "models": cards,
        "predictions": {
            "path": str(predictions_path.resolve()),
            "sha256": _sha256(predictions_path),
            "rows": len(predictions),
        },
    }
    card_path = args.output_dir / "model-card.json"
    _atomic_text(
        card_path, json.dumps(card, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(card, indent=2, sort_keys=True, allow_nan=False))
    return card


def _ece(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    result = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = (probability >= lower) & (
            probability <= upper if index == bins - 1 else probability < upper
        )
        if np.any(selected):
            result += np.mean(selected) * abs(
                float(np.mean(probability[selected]) - np.mean(target[selected]))
            )
    return float(result)


def _metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    return {
        "count": int(len(target)),
        "positives": int(np.sum(target)),
        "prevalence": float(np.mean(target)),
        "predicted_mean": float(np.mean(probability)),
        "brier": _brier(target, probability),
        "log_loss": _log_loss(target, probability),
        "ece_10": _ece(target, probability),
    }


def _calibration_intercept_slope(
    target: np.ndarray, probability: np.ndarray
) -> tuple[float | None, float | None]:
    if len(np.unique(target)) != 2:
        return None, None
    logits = np.log(
        np.clip(probability, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - probability, 1e-6, 1.0)
    )
    parameters = fit_logistic(logits[:, None], target, l2=1e-6)
    return float(parameters[0]), float(parameters[1])


def _reliability(
    target: np.ndarray, probability: np.ndarray, groups: np.ndarray, bins: int = 10
) -> list[dict[str, Any]]:
    result = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = (probability >= lower) & (
            probability <= upper if index == bins - 1 else probability < upper
        )
        result.append(
            {
                "lower": lower,
                "upper": upper,
                "count": int(np.count_nonzero(selected)),
                "independence_components": len(set(groups[selected])),
                "predicted_mean": (
                    float(np.mean(probability[selected])) if np.any(selected) else None
                ),
                "observed_rate": (
                    float(np.mean(target[selected])) if np.any(selected) else None
                ),
            }
        )
    return result


def _bootstrap_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    repetitions: int = 10_000,
) -> dict[str, Any] | None:
    unique_groups = np.asarray(sorted(set(groups)), dtype=str)
    if len(unique_groups) < 5:
        return None
    indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    generator = np.random.default_rng(seed)
    brier = np.empty(repetitions, dtype=np.float64)
    log_loss = np.empty(repetitions, dtype=np.float64)
    for iteration in range(repetitions):
        sampled_groups = generator.choice(
            unique_groups, size=len(unique_groups), replace=True
        )
        sampled = np.concatenate([indices[group] for group in sampled_groups])
        brier[iteration] = _brier(target[sampled], probability[sampled])
        log_loss[iteration] = _log_loss(target[sampled], probability[sampled])
    return {
        "unit": "independence_component",
        "component_count": int(len(unique_groups)),
        "repetitions": repetitions,
        "seed": seed,
        "percentile_95": {
            "brier": [float(value) for value in np.quantile(brier, (0.025, 0.975))],
            "log_loss": [
                float(value) for value in np.quantile(log_loss, (0.025, 0.975))
            ],
        },
    }


def _seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _detailed_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    intercept, slope = _calibration_intercept_slope(target, probability)
    return {
        **_metrics(target, probability),
        "independence_components": len(set(groups)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "reliability_bins": _reliability(target, probability, groups),
        "component_bootstrap": _bootstrap_metrics(
            target, probability, groups, seed=seed
        ),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    predictions_path = args.predictions.resolve()
    outcomes_path = args.evaluation_outcomes.resolve()
    card_path = args.model_card.resolve()
    predictions = _read_jsonl(predictions_path)
    outcomes = _index(_read_jsonl(outcomes_path), "evaluation outcome")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    cards = {row["model_id"]: row for row in card["models"]}
    if any(row["split"] != "evaluation" for row in outcomes.values()):
        raise ValueError("evaluation outcomes must contain evaluation rows only")
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        if prediction["split"] == "evaluation":
            by_model[prediction["model_id"]].append(prediction)
    results = {}
    for model_id, selected_predictions in sorted(by_model.items()):
        rows = []
        for prediction in selected_predictions:
            key = (prediction["dataset_id"], prediction["pair_id"])
            if key not in outcomes:
                continue
            outcome = outcomes[key]
            if not _in_population(outcome, prediction["population"]):
                continue
            rows.append((prediction, outcome))
        datasets = {}
        for dataset in sorted({row[0]["dataset_id"] for row in rows}):
            subset = [row for row in rows if row[0]["dataset_id"] == dataset]
            target = np.asarray(
                [row[1][row[0]["target"]] for row in subset], dtype=float
            )
            probability = np.asarray([row[0]["probability"] for row in subset])
            groups = np.asarray(
                [row[0]["independence_component_id"] for row in subset], dtype=str
            )
            constant = np.full(len(target), cards[model_id]["calibration_prevalence"])
            model_metrics = _detailed_metrics(
                target,
                probability,
                groups,
                seed=_seed(model_id, dataset, "model"),
            )
            baseline_metrics = _detailed_metrics(
                target,
                constant,
                groups,
                seed=_seed(model_id, dataset, "constant"),
            )
            datasets[dataset] = {
                **model_metrics,
                "constant_calibration_prevalence": baseline_metrics,
                "brier_delta_vs_constant": model_metrics["brier"]
                - baseline_metrics["brier"],
                "log_loss_delta_vs_constant": model_metrics["log_loss"]
                - baseline_metrics["log_loss"],
            }
        results[model_id] = {
            "datasets": datasets,
            "macro_brier": float(
                np.mean([value["brier"] for value in datasets.values()])
            ),
            "macro_log_loss": float(
                np.mean([value["log_loss"] for value in datasets.values()])
            ),
            "macro_ece_10": float(
                np.mean([value["ece_10"] for value in datasets.values()])
            ),
        }

    prediction_index = {
        (row["model_id"], row["dataset_id"], row["pair_id"]): row
        for row in predictions
        if row["split"] == "evaluation"
    }
    usable_models = {}
    for name, accept_id, precise_id in (
        (
            "capture-usable-overlap-baseline",
            "capture-accept-overlap-baseline",
            "capture-precise-given-accept-overlap-baseline",
        ),
        (
            "capture-usable-public-full",
            "capture-accept-public-full",
            "capture-precise-given-accept-public-full",
        ),
    ):
        rows = []
        for key, outcome in outcomes.items():
            accept = prediction_index.get((accept_id, *key))
            precise = prediction_index.get((precise_id, *key))
            if accept is not None and precise is not None:
                rows.append(
                    (
                        key[0],
                        float(accept["probability"] * precise["probability"]),
                        float(outcome["usable"]),
                        str(outcome["independence_component_id"]),
                    )
                )
        if not rows:
            continue
        datasets = {}
        for dataset in sorted({row[0] for row in rows}):
            subset = [row for row in rows if row[0] == dataset]
            target = np.asarray([row[2] for row in subset])
            probability = np.asarray([row[1] for row in subset])
            groups = np.asarray([row[3] for row in subset], dtype=str)
            datasets[dataset] = _detailed_metrics(
                target,
                probability,
                groups,
                seed=_seed(name, dataset, "usable"),
            )
        usable_models[name] = {
            "definition": "p_accept * p_precise_given_accept",
            "datasets": datasets,
            "macro_brier": float(
                np.mean([value["brier"] for value in datasets.values()])
            ),
            "macro_log_loss": float(
                np.mean([value["log_loss"] for value in datasets.values()])
            ),
        }
    if card.get("study_design") == "leave-one-dataset-out":
        for held_out_dataset in LODO_DATASETS:
            name = f"capture-usable--lodo-{held_out_dataset}"
            accept_id = _lodo_model_id(
                "capture-accept-overlap-baseline", held_out_dataset
            )
            precise_id = _lodo_model_id(
                "capture-precise-given-accept-overlap-baseline", held_out_dataset
            )
            rows = []
            for key, outcome in outcomes.items():
                if key[0] != held_out_dataset:
                    continue
                accept = prediction_index.get((accept_id, *key))
                precise = prediction_index.get((precise_id, *key))
                if accept is not None and precise is not None:
                    rows.append(
                        (
                            float(accept["probability"] * precise["probability"]),
                            float(outcome["usable"]),
                            str(outcome["independence_component_id"]),
                        )
                    )
            if not rows:
                continue
            target = np.asarray([row[1] for row in rows])
            probability = np.asarray([row[0] for row in rows])
            groups = np.asarray([row[2] for row in rows], dtype=str)
            usable_models[name] = {
                "definition": "LODO p_accept * p_precise_given_accept",
                "held_out_dataset": held_out_dataset,
                "datasets": {
                    held_out_dataset: _detailed_metrics(
                        target,
                        probability,
                        groups,
                        seed=_seed(name, held_out_dataset, "usable"),
                    )
                },
                "macro_brier": _brier(target, probability),
                "macro_log_loss": _log_loss(target, probability),
            }
    result = {
        "schema": EVALUATION_SCHEMA,
        "study_status": "post-hoc evaluation on frozen component-held-out split",
        "predictions": {
            "path": str(predictions_path),
            "sha256": _sha256(predictions_path),
        },
        "evaluation_outcomes": {
            "path": str(outcomes_path),
            "sha256": _sha256(outcomes_path),
        },
        "model_card": {"path": str(card_path), "sha256": _sha256(card_path)},
        "models": results,
        "usable_products": usable_models,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "evaluation.json"
    _atomic_text(
        output_path,
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit-predict")
    fit.add_argument("--features", type=Path, required=True)
    fit.add_argument("--training-outcomes", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument(
        "--model-profile", choices=sorted(MODEL_PROFILES), default="historical"
    )
    fit.set_defaults(handler=fit_predict)
    lodo = subparsers.add_parser("fit-predict-lodo")
    lodo.add_argument("--features", type=Path, required=True)
    lodo.add_argument("--training-outcomes", type=Path, required=True)
    lodo.add_argument("--output-dir", type=Path, required=True)
    lodo.add_argument(
        "--model-profile", choices=sorted(MODEL_PROFILES), default="historical"
    )
    lodo.set_defaults(handler=fit_predict_lodo)
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--predictions", type=Path, required=True)
    evaluation.add_argument("--model-card", type=Path, required=True)
    evaluation.add_argument("--evaluation-outcomes", type=Path, required=True)
    evaluation.add_argument("--output-dir", type=Path, required=True)
    evaluation.set_defaults(handler=evaluate)
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
