#!/usr/bin/env python3
"""Evaluate the frozen post-protocol registered-stereo holdout.

The licensed datasets and generated arrays stay outside the repository. The
runner streams one pair at a time and emits the two external records expected
by ``panorai-registered-full-sphere-stereo-promotion/v1``.
"""

# ruff: noqa: E402 -- the source-checkout root precedes PanorAi imports

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.panoramic_stereo.build_heldout_confidence_manifest import (
    _walk_identifiers,
)
from benchmarks.panoramic_stereo.build_registered_stereo_promotion_manifest import (
    PRIOR_MANIFESTS,
    PROTOCOL,
    _parse_timestamp,
)
from benchmarks.panoramic_stereo.run_full_sphere_stability_validation import (
    evaluate_result,
    registered_pair,
    solid_angle_weights,
)
from benchmarks.panoramic_stereo.run_public_multiview_validation import (
    _array_payload_sha256,
)
from benchmarks.panoramic_stereo.run_real_domain_validation import (
    json_safe,
    load_public_pair_selection,
    sha256,
)
from panorai.applications import FullSphereDenseStereo


SELECTION_SCHEMA = "panorai-registered-stereo-promotion-selection/v1"
EVIDENCE_CONTRACT = "panorai-registered-full-sphere-stereo-promotion/v1"
PROFILE_ID = "panorai-registered-full-sphere-stereo/reviewed-2026-10-17"
SHAPE_HW = (512, 1024)
MIN_RANGE_M = 0.3
MAX_RANGE_M = 12.0
CORPUS_NAMES = {
    "stanford2d3d": "Stanford2D3D",
    "matterport360": "Matterport3D",
}


def verify_source_state(source_commit: str) -> None:
    """Bind evidence to one clean, committed source checkout."""

    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != source_commit:
        raise ValueError(
            f"source_commit does not match checkout HEAD: {source_commit} != {actual}"
        )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(
            "registered-stereo evidence requires a clean tracked checkout"
        )


def _canonical_payload_sha256(payload: dict[str, Any]) -> str:
    value = dict(payload)
    value.pop("canonical_payload_sha256", None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    """Validate a frozen selection before opening any dataset asset."""

    protocol = json.loads(PROTOCOL.read_text())
    protocol_sha = sha256(PROTOCOL)
    payload = json.loads(path.read_text())
    manifest_sha = sha256(path)
    forbidden = set(protocol["policy"]["forbidden_cohort_manifest_sha256"])
    if manifest_sha in forbidden:
        raise ValueError("cohort manifest reuses previously observed evidence")
    if payload.get("schema") != SELECTION_SCHEMA:
        raise ValueError("unexpected registered-stereo holdout schema")
    if payload.get("protocol_id") != (
        "STAB-136-registered-full-sphere-stereo-promotion"
    ):
        raise ValueError("unexpected registered-stereo holdout protocol")
    if payload.get("frozen_before_dense_evaluation") is not True:
        raise ValueError("cohort was not frozen before dense evaluation")
    protocol_timestamp = _parse_timestamp(f"{protocol['frozen_at']}T00:00:00Z")
    if _parse_timestamp(str(payload.get("frozen_at"))) < protocol_timestamp:
        raise ValueError("cohort predates the registered-stereo promotion protocol")
    if payload.get("promotion_protocol") != {
        "path": PROTOCOL.name,
        "sha256": protocol_sha,
    }:
        raise ValueError("cohort is not bound to the current promotion protocol")
    if payload.get("canonical_payload_sha256") != _canonical_payload_sha256(payload):
        raise ValueError("cohort canonical payload hash changed")

    exclusions = payload.get("excluded_manifests")
    if not isinstance(exclusions, list):
        raise ValueError("cohort exclusions are missing")
    if {item.get("sha256") for item in exclusions} != forbidden:
        raise ValueError("cohort does not exclude every forbidden manifest")
    expected_exclusions = {(prior.name, sha256(prior)) for prior in PRIOR_MANIFESTS}
    observed_exclusions = {
        (str(item.get("path")), str(item.get("sha256"))) for item in exclusions
    }
    if observed_exclusions != expected_exclusions:
        raise ValueError("cohort exclusion names or hashes changed")
    source_audits = payload.get("source_audits")
    if not isinstance(source_audits, dict) or set(source_audits) != {
        "registered_cloud_overlap_sha256",
        "pair_analysis_sha256",
    }:
        raise ValueError("cohort source-audit hashes are missing")
    if any(
        re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
        for value in source_audits.values()
    ):
        raise ValueError("cohort source-audit hashes are malformed")
    prior_pairs: set[str] = set()
    prior_views: set[str] = set()
    for prior in PRIOR_MANIFESTS:
        pairs, views = _walk_identifiers(json.loads(prior.read_text()))
        prior_pairs.update(pairs)
        prior_views.update(views)

    policy = payload.get("selection_policy")
    required_false = (
        "dense_output_used_for_selection",
        "rgb_match_used_for_selection",
        "range_reference_used_for_selection",
    )
    required_true = (
        "post_protocol_holdout",
        "all_pre_protocol_public_stereo_manifests_excluded",
        "prediction_independent_non_sky_required_during_evaluation",
        "fixed_denominator_required_during_evaluation",
    )
    if not isinstance(policy, dict) or any(
        policy.get(key) is not False for key in required_false
    ):
        raise ValueError("cohort selection is not outcome blind")
    if any(policy.get(key) is not True for key in required_true):
        raise ValueError("cohort selection is missing promotion safeguards")

    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 12:
        raise ValueError("cohort must contain exactly twelve pairs")
    counts = Counter(str(item.get("dataset_id")) for item in pairs)
    if counts != Counter({name: 6 for name in CORPUS_NAMES}):
        raise ValueError("cohort must contain six pairs per corpus")
    stanford_groups = Counter(
        str(item.get("group"))
        for item in pairs
        if item.get("dataset_id") == "stanford2d3d"
    )
    if stanford_groups != Counter({"area_2": 2, "area_3": 2, "area_4": 2}):
        raise ValueError("Stanford cohort is not balanced across areas 2/3/4")
    matterport_groups = {
        str(item.get("group"))
        for item in pairs
        if item.get("dataset_id") == "matterport360"
    }
    if len(matterport_groups) != 6:
        raise ValueError("Matterport cohort must use six spatial groups")
    selected_pairs, selected_views = _walk_identifiers(payload)
    if selected_pairs & prior_pairs or selected_views & prior_views:
        raise ValueError("cohort reuses a previously observed pair or view")
    views = [str(item[key]) for item in pairs for key in ("reference", "source")]
    if len(views) != len(set(views)):
        raise ValueError("cohort reuses a view internally")
    if any(
        not 0.2 <= float(item.get("baseline_m", float("nan"))) <= 5.0
        or float(item.get("minimum_directional_cloud_overlap", float("nan"))) < 0.35
        for item in pairs
    ):
        raise ValueError("cohort violates frozen baseline or overlap bounds")
    return payload, manifest_sha


def _equal(first: np.ndarray, second: np.ndarray) -> bool:
    if first.dtype.kind in "fc":
        return bool(np.array_equal(first, second, equal_nan=True))
    return bool(np.array_equal(first, second))


def _result_arrays(result: Any) -> dict[str, np.ndarray]:
    return {
        "estimated_radial_range_m": np.asarray(result.radial_range),
        "prediction_validity": np.asarray(result.validity),
        "confidence": np.asarray(result.confidence),
        "matching_confidence": np.asarray(result.matching_confidence),
        "consistency_confidence": np.asarray(result.consistency_confidence),
        "sampling_support": np.asarray(result.sampling_support),
        "observable": np.asarray(result.observable),
        "observable_source_count": np.asarray(result.observable_source_count),
        "source_count": np.asarray(result.source_count),
    }


def _gross_error(
    truth: np.ndarray,
    prediction: np.ndarray,
    validity: np.ndarray,
    population: np.ndarray,
) -> tuple[float, float]:
    weights = solid_angle_weights(population.shape)
    population_weight = float(weights[population].sum())
    accepted = population & validity & np.isfinite(prediction) & (prediction > 0.0)
    accepted_weight = float(weights[accepted].sum())
    ratio = np.maximum(
        prediction[accepted] / truth[accepted],
        truth[accepted] / prediction[accepted],
    )
    gross_weight = float(weights[accepted][ratio > 2.0].sum())
    return (
        gross_weight / population_weight,
        gross_weight / accepted_weight if accepted_weight else 1.0,
    )


def run_pair(
    selection: dict[str, Any],
    *,
    root: Path,
    candidate: FullSphereDenseStereo,
    output_dir: Path,
) -> dict[str, Any]:
    pair = load_public_pair_selection(
        str(selection["dataset_id"]), root, SHAPE_HW, selection
    )
    registered = registered_pair(pair)
    started = time.perf_counter()
    first = candidate.run(registered)
    elapsed = time.perf_counter() - started
    replay = candidate.run(registered)
    first_arrays = _result_arrays(first)
    replay_arrays = _result_arrays(replay)
    deterministic = all(
        _equal(first_arrays[name], replay_arrays[name]) for name in first_arrays
    )
    if not deterministic:
        raise RuntimeError(f"non-deterministic replay for {pair.pair_id}")

    evaluation, explicit_non_sky, population = evaluate_result(
        pair,
        first,
        min_range_m=MIN_RANGE_M,
        max_range_m=MAX_RANGE_M,
    )
    prediction = np.asarray(first.radial_range)
    validity = np.asarray(first.validity, dtype=bool)
    gross_fixed, gross_conditional = _gross_error(
        pair.reference_range, prediction, validity, population
    )
    invalid_values_are_nan = bool(np.all(np.isnan(prediction[~validity])))
    finite_exactly_equals_validity = bool(
        np.array_equal(np.isfinite(prediction), validity)
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    arrays = {
        **first_arrays,
        "reference_radial_range_m": pair.reference_range,
        "explicit_non_sky": explicit_non_sky,
        "evaluation_population": population,
    }
    arrays_path = output_dir / "result.npz"
    np.savez_compressed(arrays_path, **arrays)
    record = {
        "dataset_id": pair.dataset_id,
        "pair_id": pair.pair_id,
        "selection": dict(selection),
        "elapsed_seconds": elapsed,
        "metrics": evaluation["metrics"],
        "population_audit": evaluation["population_audit"],
        "gross_gt_2x": {
            "fixed_population_fraction": gross_fixed,
            "conditional_fraction": gross_conditional,
        },
        "prediction_independent_non_sky": True,
        "fixed_denominator": True,
        "provided_metric_pose": True,
        "invalid_values_are_nan": invalid_values_are_nan,
        "finite_exactly_equals_validity": finite_exactly_equals_validity,
        "deterministic_replay": deterministic,
        "array_payload_sha256": _array_payload_sha256(arrays),
        "evaluation_population_sha256": _array_payload_sha256(
            {"evaluation_population": population}
        ),
        "assets": pair.provenance["assets"],
        "arrays": str(arrays_path.resolve()),
    }
    (output_dir / "record.json").write_text(
        json.dumps(json_safe(record), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return record


def aggregate_domain(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 6:
        raise ValueError("each corpus requires exactly six pair records")
    overall = [item["metrics"]["overall"] for item in records]
    population_weight = sum(float(item["population_weight"]) for item in overall)
    predicted_weight = sum(float(item["predicted_valid_weight"]) for item in overall)
    correct_weight = sum(
        float(item["delta_1_25"]) * float(item["population_weight"]) for item in overall
    )
    gross_weight = sum(
        float(record["gross_gt_2x"]["fixed_population_fraction"])
        * float(metrics["population_weight"])
        for record, metrics in zip(records, overall, strict=True)
    )
    return {
        "pair_count": len(records),
        "fixed_delta_1_25": correct_weight / population_weight,
        "coverage": predicted_weight / population_weight,
        "conditional_gross_gt_2x": (
            gross_weight / predicted_weight if predicted_weight else 1.0
        ),
        "prediction_independent_non_sky": all(
            item["prediction_independent_non_sky"] for item in records
        ),
        "fixed_denominator": all(item["fixed_denominator"] for item in records),
        "provided_metric_pose": all(item["provided_metric_pose"] for item in records),
        "invalid_values_are_nan": all(
            item["invalid_values_are_nan"] for item in records
        ),
        "deterministic_replay": all(item["deterministic_replay"] for item in records),
    }


def _domain_evidence_sha256(records: list[dict[str, Any]]) -> str:
    payload = [
        {
            key: record[key]
            for key in (
                "dataset_id",
                "pair_id",
                "metrics",
                "population_audit",
                "gross_gt_2x",
                "array_payload_sha256",
                "evaluation_population_sha256",
            )
        }
        for record in records
    ]
    encoded = json.dumps(
        json_safe(payload), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def evidence_records(
    records: list[dict[str, Any]],
    *,
    source_commit: str,
    cohort_manifest_sha256: str,
) -> list[dict[str, Any]]:
    output = []
    for dataset_id, corpus in CORPUS_NAMES.items():
        domain = [item for item in records if item["dataset_id"] == dataset_id]
        output.append(
            {
                "application": "registered-full-sphere-stereo",
                "evaluation": "external",
                "split": "post-protocol-heldout",
                "profile_id": PROFILE_ID,
                "corpus": corpus,
                "source_commit": source_commit,
                "cohort_manifest_sha256": cohort_manifest_sha256,
                "evidence_sha256": _domain_evidence_sha256(domain),
                "metrics": aggregate_domain(domain),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stanford-root", type=Path, required=True)
    parser.add_argument("--matterport-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--filter-backend", choices=("native", "numpy"), default="native"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise ValueError("source_commit must be a full lowercase 40-character SHA")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if args.output_dir.resolve().is_relative_to(ROOT):
        raise ValueError("held-out outputs must be written outside the source checkout")
    verify_source_state(args.source_commit)
    manifest, manifest_sha = load_manifest(args.manifest)
    candidate = FullSphereDenseStereo.reviewed_profile(
        filter_backend=args.filter_backend
    )
    if candidate.describe()["profile_id"] != PROFILE_ID:
        raise RuntimeError("runtime reviewed profile does not match the protocol")

    roots = {
        "stanford2d3d": args.stanford_root,
        "matterport360": args.matterport_root,
    }
    args.output_dir.mkdir(parents=True)
    records = []
    for selection in manifest["pairs"]:
        dataset_id = str(selection["dataset_id"])
        pair_id = str(selection["pair_id"])
        print(f"[{dataset_id}] {pair_id}", flush=True)
        records.append(
            run_pair(
                selection,
                root=roots[dataset_id],
                candidate=candidate,
                output_dir=args.output_dir / pair_id,
            )
        )
    protocol_sha = sha256(PROTOCOL)
    evidence = {
        "contract": EVIDENCE_CONTRACT,
        "protocol_sha256": protocol_sha,
        "records": evidence_records(
            records,
            source_commit=args.source_commit,
            cohort_manifest_sha256=manifest_sha,
        ),
    }
    (args.output_dir / "external-evidence.json").write_text(
        json.dumps(json_safe(evidence), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    summary = {
        "schema": "panorai-registered-stereo-heldout-run/v1",
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": manifest_sha,
            "canonical_payload_sha256": manifest["canonical_payload_sha256"],
        },
        "source_commit": args.source_commit,
        "profile": candidate.describe(),
        "shape_hw": list(SHAPE_HW),
        "filter_backend": args.filter_backend,
        "streaming_policy": "one pair plus deterministic replay at a time",
        "records": records,
        "external_evidence": evidence,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
