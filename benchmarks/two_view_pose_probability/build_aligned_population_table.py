#!/usr/bin/env python3
"""Build the VAL-018 table after all pairs use one optimized frontend."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


RESULT_SCHEMA = "panorai-unified-optimized-pair/v2"
FEATURE_SCHEMA = "panorai-two-view-pose-aligned-features/v1"
OUTCOME_SCHEMA = "panorai-two-view-pose-aligned-outcomes/v1"
TABLE_SCHEMA = "panorai-two-view-pose-aligned-analysis-table/v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _index(
    rows: Iterable[dict[str, Any]], *, label: str
) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for row in rows:
        key = (str(row["dataset_id"]), str(row["pair_id"]))
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


def _validate_package(
    result: dict[str, Any],
    *,
    path: Path,
    expected_package_version: str,
    expected_source_commit: str,
) -> dict[str, Any]:
    package = result.get("package", {})
    if package.get("version") != expected_package_version:
        raise ValueError(
            f"unexpected PanorAi version in {path}: "
            f"{package.get('version')!r} != {expected_package_version!r}"
        )
    if package.get("expected_source_commit") != expected_source_commit:
        raise ValueError(
            f"unexpected source commit in {path}: "
            f"{package.get('expected_source_commit')!r} != "
            f"{expected_source_commit!r}"
        )
    return package


def _post(result: dict[str, Any]) -> dict[str, Any]:
    pose = result["pose"]
    quality = pose.get("quality_report") or {}
    stability = quality.get("stability") or {}
    competition = quality.get("model_competition") or {}
    orientation = quality.get("translation_orientation") or {}
    requested = stability.get("requested_trials")
    successful = stability.get("successful_trials")
    stability_fraction = (
        successful / requested if requested and successful is not None else None
    )
    preferred = competition.get("preferred_model")
    timings = result["timings_seconds"]
    counts = result["counts"]
    validity = result["validity"]
    matching = result["matching_diagnostics"]
    return {
        "match_count": counts["matches"],
        "inlier_count": pose.get("inlier_count"),
        "inlier_ratio": quality.get("inlier_ratio"),
        "raw_quality_score": quality.get("raw_quality_score"),
        "descriptor_distance_median": matching.get("descriptor_distance_median"),
        "descriptor_distance_p90": matching.get("descriptor_distance_p90"),
        "ratio_score_median": matching.get("ratio_score_median"),
        "median_parallax_deg": quality.get("median_parallax_deg"),
        "cheirality_ratio": quality.get("cheirality_ratio"),
        "median_residual_deg": quality.get("median_residual_deg"),
        "p90_residual_deg": quality.get("p90_residual_deg"),
        "coverage_entropy_a": quality.get("coverage_entropy_a"),
        "coverage_entropy_b": quality.get("coverage_entropy_b"),
        "occupied_cells_a": quality.get("occupied_cells_a"),
        "occupied_cells_b": quality.get("occupied_cells_b"),
        "stability_success_fraction": stability_fraction,
        "stability_rotation_p90_deg": stability.get("rotation_p90_deg"),
        "stability_translation_p90_deg": stability.get("translation_p90_deg"),
        "essential_score_margin": competition.get("essential_score_margin"),
        "preferred_model_essential": (
            None if preferred is None else preferred == "essential"
        ),
        "translation_orientation_cheirality_margin": orientation.get(
            "cheirality_margin"
        ),
        "translation_orientation_weighted_margin": orientation.get(
            "weighted_cheirality_margin"
        ),
        "translation_orientation_median_triangulation_angle_deg": orientation.get(
            "median_triangulation_angle_deg"
        ),
        "translation_orientation_ambiguous": orientation.get("ambiguous"),
        "keypoint_count_min": min(counts["keypoints_a"], counts["keypoints_b"]),
        "valid_fraction_min": min(
            validity["valid_fraction_a"], validity["valid_fraction_b"]
        ),
        "detection_pair_seconds": timings["detection_pair"],
        "patches_pair_seconds": timings["patches_pair"],
        "descriptor_pair_seconds": timings["descriptor_pair"],
        "matching_seconds": timings["matching"],
        "pose_seconds": timings["pose"],
        "pair_total_seconds": timings["pair_total"],
        "peak_rss_mib": result["peak_rss_mib"],
    }


def _outcomes(result: dict[str, Any]) -> dict[str, Any]:
    pose = result["pose"]
    returned = bool(pose.get("returned"))
    accepted = bool(pose.get("quality_accepted"))
    rotation = pose.get("rotation_error_deg")
    translation = pose.get("translation_direction_error_deg")
    primary = bool(returned and rotation <= 15.0 and translation <= 30.0)
    strict = bool(returned and rotation <= 5.0 and translation <= 10.0)
    precise = bool(returned and rotation <= 1.0 and translation <= 5.0)
    catastrophic = bool(accepted and not primary)
    for name, computed in (
        ("primary", primary),
        ("strict", strict),
        ("precise", precise),
        ("catastrophic_accepted", catastrophic),
    ):
        recorded = pose.get(name)
        if recorded is not None and bool(recorded) != computed:
            raise ValueError(f"result {name} contradicts recomputed outcome")
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_rows = _read_jsonl(args.base_analysis_table)
    base = _index(base_rows, label="base pair")
    result_paths = sorted(args.results_dir.glob("*.json"))
    results = []
    hashes = {}
    for path in result_paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("schema") != RESULT_SCHEMA:
            raise ValueError(f"unsupported result schema in {path}")
        _validate_package(
            row,
            path=path,
            expected_package_version=args.expected_package_version,
            expected_source_commit=args.expected_source_commit,
        )
        results.append(row)
        hashes[path.name] = _sha256(path)
    result_index = _index(results, label="aligned result")
    if set(result_index) != set(base):
        missing = sorted(set(base) - set(result_index))[:5]
        extra = sorted(set(result_index) - set(base))[:5]
        raise ValueError(
            f"aligned replay incomplete; missing={missing}, extra={extra}, "
            f"completed={len(result_index)}/{len(base)}"
        )

    features = []
    outcomes = []
    tables = []
    identity_names = (
        "pair_id",
        "dataset_id",
        "spatial_group_id",
        "independence_component_id",
        "split",
    )
    for key in sorted(base):
        old = base[key]
        result = result_index[key]
        identity = {name: old[name] for name in identity_names}
        post = _post(result)
        outcome_values = _outcomes(result)
        method = {
            "method_id": "panorai-origin-main-optimized-spherical-v1",
            "frontend_family": "optimized-spherical-dog-rootsift",
            "package_version": result["package"]["version"],
            "expected_source_commit": result["package"]["expected_source_commit"],
        }
        feature = {
            "schema": FEATURE_SCHEMA,
            **identity,
            "method": method,
            "capture": old["capture"],
            "post": post,
        }
        outcome = {"schema": OUTCOME_SCHEMA, **identity, **outcome_values}
        table = {
            "schema": TABLE_SCHEMA,
            **identity,
            "method": method,
            "capture": old["capture"],
            "post": post,
            "outcomes": outcome_values,
            "reference_only": old["reference_only"],
        }
        features.append(feature)
        outcomes.append(outcome)
        tables.append(table)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "features": args.output_dir / "features.jsonl",
        "outcomes": args.output_dir / "outcomes.jsonl",
        "training_outcomes": args.output_dir / "outcomes-development-calibration.jsonl",
        "evaluation_outcomes": args.output_dir / "outcomes-evaluation.jsonl",
        "analysis_table": args.output_dir / "analysis-table.jsonl",
    }
    _atomic_text(paths["features"], _jsonl(features))
    _atomic_text(paths["outcomes"], _jsonl(outcomes))
    _atomic_text(
        paths["training_outcomes"],
        _jsonl(row for row in outcomes if row["split"] != "evaluation"),
    )
    _atomic_text(
        paths["evaluation_outcomes"],
        _jsonl(row for row in outcomes if row["split"] == "evaluation"),
    )
    _atomic_text(paths["analysis_table"], _jsonl(tables))
    dataset_counts = {
        dataset: sum(row["dataset_id"] == dataset for row in tables)
        for dataset in sorted({row["dataset_id"] for row in tables})
    }
    manifest = {
        "schema": "panorai-two-view-pose-aligned-table-manifest/v1",
        "status": "complete aligned-frontend population table",
        "pair_count": len(tables),
        "dataset_counts": dataset_counts,
        "expected_package_version": args.expected_package_version,
        "expected_source_commit": args.expected_source_commit,
        "base_analysis_table": {
            "path": str(args.base_analysis_table.resolve()),
            "sha256": _sha256(args.base_analysis_table),
        },
        "results_dir": str(args.results_dir.resolve()),
        "result_sha256": hashes,
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }
    _atomic_text(
        args.output_dir / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-analysis-table", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-package-version", default="3.5.0")
    parser.add_argument("--expected-source-commit", required=True)
    return parser


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
