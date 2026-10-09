#!/usr/bin/env python3
"""Build leakage-audited pair features and outcomes for VAL-018."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

FEATURE_SCHEMA = "panorai-two-view-pose-features/v1"
OUTCOME_SCHEMA = "panorai-two-view-pose-outcomes/v1"
TABLE_SCHEMA = "panorai-two-view-pose-analysis-table/v1"
MANIFEST_SCHEMA = "panorai-two-view-pose-table-manifest/v1"

IDENTITY_FIELDS = (
    "pair_id",
    "dataset_id",
    "spatial_group_id",
    "independence_component_id",
    "split",
)

CAPTURE_FIELDS = (
    "baseline_m",
    "representative_scene_distance_m",
    "baseline_depth_ratio",
    "predicted_parallax_deg",
    "rgb_similarity",
    "registered_cloud_overlap_min",
    "registered_cloud_overlap_symmetric",
    "registered_covisible_fraction_symmetric",
)

POST_FIELDS = (
    "match_count",
    "inlier_count",
    "inlier_ratio",
    "raw_quality_score",
    "descriptor_distance_median",
    "median_parallax_deg",
    "cheirality_ratio",
    "median_residual_deg",
    "p90_residual_deg",
    "coverage_entropy_a",
    "coverage_entropy_b",
    "occupied_cells_a",
    "occupied_cells_b",
    "stability_success_fraction",
    "stability_rotation_p90_deg",
    "stability_translation_p90_deg",
    "essential_score_margin",
    "preferred_model_essential",
    "elapsed_seconds",
    "extract_a_seconds",
    "extract_b_seconds",
    "match_seconds",
    "pose_seconds",
)

OUTCOME_FIELDS = (
    "returned",
    "accepted",
    "primary",
    "strict",
    "precise",
    "usable",
    "catastrophic_accepted",
    "rotation_error_deg",
    "translation_direction_error_deg",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            if set(item) == {"manifest"}:
                continue
            records.append(item)
    return records


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _index(
    rows: Iterable[dict[str, Any]], *, label: str
) -> dict[tuple[str, str], dict[str, Any]]:
    indexed = {}
    for row in rows:
        key = (str(row["dataset_id"]), str(row["pair_id"]))
        if key in indexed:
            raise ValueError(f"duplicate {label} key: {key}")
        indexed[key] = row
    return indexed


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _optional_int(value: Any) -> int | None:
    number = _optional_float(value)
    return None if number is None else int(number)


def _optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"not a boolean: {value!r}")


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _first(*values: Any) -> Any:
    return next((value for value in values if value is not None and value != ""), None)


def _predicted_parallax_deg(
    baseline: float | None, distance: float | None
) -> float | None:
    """Fronto-parallel midpoint approximation: 2 atan(b / (2 d))."""

    if baseline is None or distance is None or baseline < 0.0 or distance <= 0.0:
        return None
    return math.degrees(2.0 * math.atan2(baseline, 2.0 * distance))


def _capture_features(
    evaluated: Mapping[str, Any] | None,
    overlap: Mapping[str, Any] | None,
) -> tuple[dict[str, float | None], dict[str, Any]]:
    evaluated = evaluated or {}
    overlap = overlap or {}
    baseline = _optional_float(
        _first(evaluated.get("baseline_m"), overlap.get("baseline_m"))
    )
    distance = _optional_float(evaluated.get("harmonic_median_depth_m"))
    ratio = _optional_float(evaluated.get("baseline_over_harmonic_median_depth"))
    if ratio is None and baseline is not None and distance not in (None, 0.0):
        ratio = baseline / distance
    capture = {
        "baseline_m": baseline,
        "representative_scene_distance_m": distance,
        "baseline_depth_ratio": ratio,
        "predicted_parallax_deg": _predicted_parallax_deg(baseline, distance),
        "rgb_similarity": _optional_float(evaluated.get("rgb_similarity")),
        "registered_cloud_overlap_min": _optional_float(
            overlap.get("minimum_directional_cloud_overlap")
        ),
        "registered_cloud_overlap_symmetric": _optional_float(
            overlap.get("symmetric_cloud_overlap")
        ),
        "registered_covisible_fraction_symmetric": _optional_float(
            overlap.get("symmetric_covisible_fraction")
        ),
    }
    reference_only = {
        "rotation_change_deg": _optional_float(evaluated.get("rotation_change_deg")),
        "overlap_bin": overlap.get("overlap_bin"),
    }
    return capture, reference_only


def _post_features(prediction: Mapping[str, Any]) -> dict[str, Any]:
    quality = prediction.get("quality_report")
    quality = quality if isinstance(quality, Mapping) else {}
    stability = quality.get("stability")
    stability = stability if isinstance(stability, Mapping) else {}
    competition = quality.get("model_competition")
    competition = competition if isinstance(competition, Mapping) else {}
    match_count = _optional_int(
        _first(prediction.get("match_count"), quality.get("num_correspondences"))
    )
    inlier_count = _optional_int(
        _first(prediction.get("inlier_count"), quality.get("num_inliers"))
    )
    inlier_ratio = _optional_float(quality.get("inlier_ratio"))
    if inlier_ratio is None and match_count and inlier_count is not None:
        inlier_ratio = inlier_count / match_count
    requested = _optional_int(stability.get("requested_trials"))
    successful = _optional_int(stability.get("successful_trials"))
    stability_fraction = (
        successful / requested
        if successful is not None and requested is not None and requested > 0
        else None
    )
    preferred = competition.get("preferred_model")
    return {
        "match_count": match_count,
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "raw_quality_score": _optional_float(
            _first(prediction.get("quality_score"), quality.get("raw_quality_score"))
        ),
        "descriptor_distance_median": _optional_float(
            prediction.get("descriptor_distance_median")
        ),
        "median_parallax_deg": _optional_float(
            _first(
                prediction.get("median_parallax_deg"),
                quality.get("median_parallax_deg"),
            )
        ),
        "cheirality_ratio": _optional_float(
            _first(prediction.get("cheirality_ratio"), quality.get("cheirality_ratio"))
        ),
        "median_residual_deg": _optional_float(quality.get("median_residual_deg")),
        "p90_residual_deg": _optional_float(quality.get("p90_residual_deg")),
        "coverage_entropy_a": _optional_float(quality.get("coverage_entropy_a")),
        "coverage_entropy_b": _optional_float(quality.get("coverage_entropy_b")),
        "occupied_cells_a": _optional_int(quality.get("occupied_cells_a")),
        "occupied_cells_b": _optional_int(quality.get("occupied_cells_b")),
        "stability_success_fraction": stability_fraction,
        "stability_rotation_p90_deg": _optional_float(
            stability.get("rotation_p90_deg")
        ),
        "stability_translation_p90_deg": _optional_float(
            stability.get("translation_p90_deg")
        ),
        "essential_score_margin": _optional_float(
            competition.get("essential_score_margin")
        ),
        "preferred_model_essential": None
        if preferred is None
        else preferred == "essential",
        "elapsed_seconds": _optional_float(prediction.get("elapsed_seconds")),
        "extract_a_seconds": _optional_float(prediction.get("extract_a_seconds")),
        "extract_b_seconds": _optional_float(prediction.get("extract_b_seconds")),
        "match_seconds": _optional_float(prediction.get("match_seconds")),
        "pose_seconds": _optional_float(prediction.get("pose_seconds")),
    }


def _outcomes(
    prediction: Mapping[str, Any], evaluated: Mapping[str, Any]
) -> dict[str, Any]:
    returned = bool(
        _optional_bool(
            _first(prediction.get("pose_returned"), evaluated.get("pose_returned"))
        )
    )
    accepted = bool(
        _optional_bool(
            _first(
                prediction.get("quality_accepted"), evaluated.get("quality_accepted")
            )
        )
    )
    rotation = _optional_float(evaluated.get("rotation_error_deg"))
    translation = _optional_float(evaluated.get("translation_direction_error_deg"))
    finite_error = rotation is not None and translation is not None
    primary = bool(finite_error and rotation <= 15.0 and translation <= 30.0)
    strict = bool(finite_error and rotation <= 5.0 and translation <= 10.0)
    precise = bool(finite_error and rotation <= 1.0 and translation <= 5.0)
    catastrophic = bool(accepted and not primary)
    if accepted and not returned:
        raise ValueError("accepted pose must also be returned")
    if precise and not strict:
        raise ValueError("precise pose must also be strict")
    for field, computed in (
        ("primary_success", primary),
        ("strict_success", strict),
        ("catastrophic_accepted", catastrophic),
    ):
        recorded = _optional_bool(evaluated.get(field))
        if recorded is not None and recorded != computed:
            raise ValueError(f"{field} contradicts recomputed thresholds")
    return {
        "returned": returned,
        "accepted": accepted,
        "primary": primary,
        "strict": strict,
        "precise": precise,
        "usable": accepted and precise,
        "catastrophic_accepted": catastrophic,
        "rotation_error_deg": rotation,
        "translation_direction_error_deg": translation,
    }


def source_label_audit(
    evaluated: list[dict[str, Any]], table: list[dict[str, Any]]
) -> dict[str, Any]:
    normalized = _index(table, label="normalized table")
    fields = {
        "primary_success": "primary",
        "strict_success": "strict",
        "precise_success": "precise",
        "catastrophic_accepted": "catastrophic_accepted",
    }
    audit: dict[str, Any] = {}
    for dataset in sorted({str(row["dataset_id"]) for row in evaluated}):
        selected = [row for row in evaluated if str(row["dataset_id"]) == dataset]
        dataset_audit = {}
        for source_field, normalized_field in fields.items():
            compared = 0
            disagreements = []
            for row in selected:
                recorded = _optional_bool(row.get(source_field))
                if recorded is None:
                    continue
                compared += 1
                key = (dataset, str(row["pair_id"]))
                computed = bool(normalized[key]["outcomes"][normalized_field])
                if recorded != computed:
                    disagreements.append(str(row["pair_id"]))
            dataset_audit[source_field] = {
                "compared": compared,
                "disagreements": len(disagreements),
                "disagreement_pair_ids": disagreements,
            }
        audit[dataset] = dataset_audit
    return audit


def build_rows(
    pairs: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    evaluated: list[dict[str, Any]],
    overlap_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pair_index = _index(pairs, label="pair")
    prediction_index = _index(predictions, label="prediction")
    evaluated_index = _index(evaluated, label="evaluation")
    overlap_index = _index(overlap_rows, label="overlap")
    if set(prediction_index) != set(pair_index):
        missing = sorted(set(pair_index) - set(prediction_index))[:5]
        extra = sorted(set(prediction_index) - set(pair_index))[:5]
        raise ValueError(
            f"prediction/pair key mismatch; missing={missing}, extra={extra}"
        )
    if set(evaluated_index) != set(pair_index):
        missing = sorted(set(pair_index) - set(evaluated_index))[:5]
        extra = sorted(set(evaluated_index) - set(pair_index))[:5]
        raise ValueError(
            f"evaluation/pair key mismatch; missing={missing}, extra={extra}"
        )
    if not set(overlap_index).issubset(pair_index):
        raise ValueError("overlap table contains unknown pairs")

    feature_rows = []
    outcome_rows = []
    table_rows = []
    for key in sorted(pair_index):
        pair = pair_index[key]
        prediction = prediction_index[key]
        evaluation = evaluated_index[key]
        overlap = overlap_index.get(key)
        identity = {field: pair[field] for field in IDENTITY_FIELDS}
        capture, reference_only = _capture_features(evaluation, overlap)
        post = _post_features(prediction)
        method = {
            "method_id": str(
                _first(
                    prediction.get("method_version"),
                    prediction.get("variant"),
                    "unknown",
                )
            ),
            "frontend_family": (
                "optimized-spherical-dog"
                if prediction.get("variant")
                == "coarse-spherical-tangent-dog-rootsift-d1p5"
                else "historical-public-rgb"
            ),
        }
        outcomes = _outcomes(prediction, evaluation)
        feature = {
            "schema": FEATURE_SCHEMA,
            **identity,
            "method": method,
            "capture": capture,
            "post": post,
        }
        outcome = {"schema": OUTCOME_SCHEMA, **identity, **outcomes}
        table = {
            "schema": TABLE_SCHEMA,
            **identity,
            "method": method,
            "capture": capture,
            "post": post,
            "outcomes": outcomes,
            "reference_only": reference_only,
        }
        feature_rows.append(feature)
        outcome_rows.append(outcome)
        table_rows.append(table)
    return feature_rows, outcome_rows, table_rows


def missingness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    datasets = {}
    for dataset in sorted({str(row["dataset_id"]) for row in rows}):
        selected = [row for row in rows if row["dataset_id"] == dataset]
        fields = {}
        for namespace, names in (("capture", CAPTURE_FIELDS), ("post", POST_FIELDS)):
            for name in names:
                available = sum(row[namespace][name] is not None for row in selected)
                fields[f"{namespace}.{name}"] = {
                    "available": available,
                    "missing": len(selected) - available,
                    "available_fraction": available / len(selected),
                }
        returned = sum(row["outcomes"]["returned"] for row in selected)
        accepted = sum(row["outcomes"]["accepted"] for row in selected)
        precise = sum(row["outcomes"]["precise"] for row in selected)
        usable = sum(row["outcomes"]["usable"] for row in selected)
        datasets[dataset] = {
            "pairs": len(selected),
            "returned": returned,
            "accepted": accepted,
            "precise": precise,
            "usable": usable,
            "fields": fields,
        }
    return datasets


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


def _jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_paths = {
        "pairs": args.pairs.resolve(),
        "public_predictions": args.public_predictions.resolve(),
        "public_evaluated": args.public_evaluated.resolve(),
        "p74_predictions": args.p74_predictions.resolve(),
        "p74_evaluated": args.p74_evaluated.resolve(),
        "p74_overlap": args.p74_overlap.resolve(),
        "public_overlap": args.public_overlap.resolve(),
    }
    pairs = _read_jsonl(source_paths["pairs"])
    predictions = _read_jsonl(source_paths["public_predictions"]) + _read_jsonl(
        source_paths["p74_predictions"]
    )
    evaluated = _read_csv(source_paths["public_evaluated"]) + _read_jsonl(
        source_paths["p74_evaluated"]
    )
    p74_overlap_rows = _read_jsonl(source_paths["p74_overlap"])
    for row in p74_overlap_rows:
        row.setdefault("dataset_id", "p74_native_polar")
    overlap_rows = _read_jsonl(source_paths["public_overlap"]) + p74_overlap_rows
    features, outcomes, table = build_rows(pairs, predictions, evaluated, overlap_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "features": args.output_dir / "features.jsonl",
        "outcomes": args.output_dir / "outcomes.jsonl",
        "training_outcomes": args.output_dir / "outcomes-development-calibration.jsonl",
        "evaluation_outcomes": args.output_dir / "outcomes-evaluation.jsonl",
        "analysis_table": args.output_dir / "analysis-table.jsonl",
        "missingness": args.output_dir / "missingness.json",
    }
    _atomic_text(output_paths["features"], _jsonl(features))
    _atomic_text(output_paths["outcomes"], _jsonl(outcomes))
    _atomic_text(
        output_paths["training_outcomes"],
        _jsonl(row for row in outcomes if row["split"] != "evaluation"),
    )
    _atomic_text(
        output_paths["evaluation_outcomes"],
        _jsonl(row for row in outcomes if row["split"] == "evaluation"),
    )
    _atomic_text(output_paths["analysis_table"], _jsonl(table))
    missing = missingness(table)
    label_audit = source_label_audit(evaluated, table)
    _atomic_text(
        output_paths["missingness"],
        json.dumps(missing, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "sources": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in source_paths.items()
        },
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in output_paths.items()
        },
        "pair_count": len(table),
        "dataset_counts": {
            dataset: sum(row["dataset_id"] == dataset for row in table)
            for dataset in sorted({row["dataset_id"] for row in table})
        },
        "feature_names": {
            "capture": list(CAPTURE_FIELDS),
            "post": list(POST_FIELDS),
        },
        "outcome_names": list(OUTCOME_FIELDS),
        "outcome_thresholds": {
            "primary": "rotation<=15deg and oriented translation<=30deg",
            "strict": "rotation<=5deg and oriented translation<=10deg",
            "precise": "rotation<=1deg and oriented translation<=5deg",
            "catastrophic_accepted": "accepted and not primary",
        },
        "source_label_audit": label_audit,
        "missingness": missing,
    }
    manifest_path = args.output_dir / "manifest.json"
    _atomic_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--public-predictions", type=Path, required=True)
    parser.add_argument("--public-evaluated", type=Path, required=True)
    parser.add_argument("--p74-predictions", type=Path, required=True)
    parser.add_argument("--p74-evaluated", type=Path, required=True)
    parser.add_argument("--p74-overlap", type=Path, required=True)
    parser.add_argument("--public-overlap", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
