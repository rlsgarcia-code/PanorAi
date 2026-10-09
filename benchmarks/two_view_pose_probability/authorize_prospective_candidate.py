#!/usr/bin/env python3
"""Materialize an explicitly authorized E8 candidate without rewriting drafts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

try:
    from plan_prospective_acquisition import SCHEMA as ACQUISITION_SCHEMA
    from prepare_prospective_candidate_draft import (
        CONFIGURATION_SCHEMA,
        DRAFT_AUTHORIZATION,
        MODEL_SNAPSHOT_SCHEMA,
        READINESS_SCHEMA,
        RULE_AMENDMENT_SCHEMA,
    )
    from prepare_prospective_registry import AUTHORIZED_PLAN_STATUS
    from run_prospective_confirmation import (
        AUTHORIZATION,
        CANDIDATE_SCHEMA,
        EXPECTED_GATE,
        _validate_candidate,
    )
except ImportError:
    from benchmarks.two_view_pose_probability.plan_prospective_acquisition import (
        SCHEMA as ACQUISITION_SCHEMA,
    )
    from benchmarks.two_view_pose_probability.prepare_prospective_candidate_draft import (
        CONFIGURATION_SCHEMA,
        DRAFT_AUTHORIZATION,
        MODEL_SNAPSHOT_SCHEMA,
        READINESS_SCHEMA,
        RULE_AMENDMENT_SCHEMA,
    )
    from benchmarks.two_view_pose_probability.prepare_prospective_registry import (
        AUTHORIZED_PLAN_STATUS,
    )
    from benchmarks.two_view_pose_probability.run_prospective_confirmation import (
        AUTHORIZATION,
        CANDIDATE_SCHEMA,
        EXPECTED_GATE,
        _validate_candidate,
    )


AUTHORIZATION_SCHEMA = "panorai-two-view-prospective-authorization/v1"
MANIFEST_SCHEMA = "panorai-two-view-prospective-authorization-manifest/v1"
EXPECTED_SCOPE = {
    "post_precision_probability_threshold": 0.90,
    "capture_usable_probability_threshold": 0.50,
    "domains": 3,
    "groups_per_domain": 20,
    "total_groups": 60,
    "proposed_unique_images_per_group": 16,
    "proposed_unique_images_total": 960,
    "candidates_per_group": 52,
    "total_candidate_pairs": 3120,
    "selected_target_per_domain": 40,
    "selected_target_total": 120,
    "maximum_selected_pairs_per_group": 3,
}


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


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    if _sha256(source) != expected_sha256:
        raise ValueError(f"source hash mismatch: {source}")
    shutil.copyfile(source, destination)
    if _sha256(destination) != expected_sha256:
        raise RuntimeError(f"copied artifact hash mismatch: {destination}")


def _require_close(actual: Any, expected: float, label: str) -> None:
    if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"authorized scope mismatch for {label}")


def authorize(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError("authorization output directory must not exist")
    if not str(args.authorization_id).strip():
        raise ValueError("authorization ID must not be empty")
    if not str(args.authorization_statement).strip():
        raise ValueError("authorization statement must not be empty")
    if not str(args.authorized_at).strip():
        raise ValueError("authorization timestamp must not be empty")

    draft = _read_json(args.candidate_draft)
    plan_draft = _read_json(args.acquisition_plan_draft)
    configuration = _read_json(args.configuration)
    capture_models = _read_json(args.capture_models)
    post_model = _read_json(args.post_model)
    rule = _read_json(args.selective_rule)
    readiness = _read_json(args.readiness_report)

    if draft.get("schema") != CANDIDATE_SCHEMA:
        raise ValueError("candidate draft schema is invalid")
    if draft.get("authorization") != DRAFT_AUTHORIZATION:
        raise ValueError("source candidate is not an unauthorised draft")
    if draft.get("gate") != EXPECTED_GATE or draft.get("deviations") != []:
        raise ValueError("candidate draft differs from the frozen E8 gate")
    package = draft.get("panorai", {})
    if package.get("version") != args.expected_package_version:
        raise ValueError("candidate PanorAi version mismatch")
    if package.get("source_commit") != args.expected_source_commit:
        raise ValueError("candidate PanorAi source commit mismatch")

    artifact_inputs = {
        "configuration": (args.configuration, "configuration_sha256"),
        "capture-models": (args.capture_models, "capture_model_sha256"),
        "post-model": (args.post_model, "post_model_sha256"),
        "selective-rule-amendment": (
            args.selective_rule,
            "selective_rule_sha256",
        ),
    }
    for label, (path, candidate_field) in artifact_inputs.items():
        if _sha256(path) != draft.get(candidate_field):
            raise ValueError(f"{label} does not match the candidate draft")

    if configuration.get("schema") != CONFIGURATION_SCHEMA:
        raise ValueError("configuration schema is invalid")
    if capture_models.get("schema") != MODEL_SNAPSHOT_SCHEMA:
        raise ValueError("capture-model schema is invalid")
    if post_model.get("schema") != MODEL_SNAPSHOT_SCHEMA:
        raise ValueError("post-model schema is invalid")
    if rule.get("schema") != RULE_AMENDMENT_SCHEMA:
        raise ValueError("selective-rule schema is invalid")
    if readiness.get("schema") != READINESS_SCHEMA:
        raise ValueError("readiness-report schema is invalid")
    if readiness.get("candidate", {}).get("sha256") != _sha256(args.candidate_draft):
        raise ValueError("readiness report does not bind the candidate draft")

    thresholds = rule.get("thresholds", {})
    _require_close(
        thresholds.get("post_precision_probability"),
        EXPECTED_SCOPE["post_precision_probability_threshold"],
        "post threshold",
    )
    _require_close(
        thresholds.get("capture_usable_probability"),
        EXPECTED_SCOPE["capture_usable_probability_threshold"],
        "capture threshold",
    )
    if rule.get("evaluation_hypothesis_generation", {}).get("disclosure") != draft.get(
        "post_hoc_selection_disclosure"
    ):
        raise ValueError("post-hoc selection disclosure was not preserved")

    if plan_draft.get("schema") != ACQUISITION_SCHEMA:
        raise ValueError("acquisition-plan schema is invalid")
    if plan_draft.get("status") != "PLANNING_ONLY_NO_COLLECTION_AUTHORIZED":
        raise ValueError("source acquisition plan is not a planning draft")
    if plan_draft.get("candidate_readiness", {}).get("candidate_sha256") != _sha256(
        args.candidate_draft
    ):
        raise ValueError("acquisition plan does not bind the candidate draft")
    if plan_draft.get("candidate_readiness", {}).get("sha256") != _sha256(
        args.readiness_report
    ):
        raise ValueError("acquisition plan does not bind the readiness report")
    if plan_draft.get("scenario_csv_sha256") != _sha256(args.scenario_csv):
        raise ValueError("acquisition plan does not bind the scenario CSV")

    recommended = plan_draft.get("recommended_plan", {})
    for field in (
        "domains",
        "groups_per_domain",
        "total_groups",
        "proposed_unique_images_per_group",
        "proposed_unique_images_total",
        "candidates_per_group",
        "total_candidate_pairs",
        "selected_target_per_domain",
        "selected_target_total",
    ):
        if int(recommended.get(field, -1)) != EXPECTED_SCOPE[field]:
            raise ValueError(f"authorized scope mismatch for {field}")
    if int(plan_draft.get("unit", {}).get("primary_selected_pair_cap_per_group", -1)) != (
        EXPECTED_SCOPE["maximum_selected_pairs_per_group"]
    ):
        raise ValueError("authorized scope mismatch for group cap")

    args.output_dir.mkdir(parents=True)
    copied = {
        "source-candidate-draft.json": (
            args.candidate_draft,
            _sha256(args.candidate_draft),
        ),
        "source-acquisition-plan-draft.json": (
            args.acquisition_plan_draft,
            _sha256(args.acquisition_plan_draft),
        ),
        "configuration.json": (args.configuration, draft["configuration_sha256"]),
        "capture-models.json": (args.capture_models, draft["capture_model_sha256"]),
        "post-model.json": (args.post_model, draft["post_model_sha256"]),
        "selective-rule-amendment.json": (
            args.selective_rule,
            draft["selective_rule_sha256"],
        ),
        "readiness-report.json": (
            args.readiness_report,
            _sha256(args.readiness_report),
        ),
        "acquisition-scenarios.csv": (
            args.scenario_csv,
            _sha256(args.scenario_csv),
        ),
    }
    for name, (source, expected_hash) in copied.items():
        _copy_verified(source, args.output_dir / name, expected_hash)

    authorization_record = {
        "schema": AUTHORIZATION_SCHEMA,
        "authorization": AUTHORIZATION,
        "authorization_id": args.authorization_id,
        "authorized_at": args.authorized_at,
        "authorization_source": args.authorization_source,
        "authorization_statement": args.authorization_statement,
        "scope": EXPECTED_SCOPE,
        "source_drafts": {
            "candidate": {
                "artifact_name": "source-candidate-draft.json",
                "sha256": _sha256(args.output_dir / "source-candidate-draft.json"),
            },
            "acquisition_plan": {
                "artifact_name": "source-acquisition-plan-draft.json",
                "sha256": _sha256(
                    args.output_dir / "source-acquisition-plan-draft.json"
                ),
            },
        },
        "interpretation": (
            "Authorization permits prospective collection and confirmation only; "
            "it does not change the retrospective NO_GO verdict."
        ),
    }
    authorization_path = args.output_dir / "authorization-record.json"
    _atomic_json(authorization_path, authorization_record)

    candidate = dict(draft)
    candidate.update(
        {
            "authorization": AUTHORIZATION,
            "status": "authorized for prospective confirmation; no outcomes opened",
            "authorization_record": {
                "artifact_name": authorization_path.name,
                "sha256": _sha256(authorization_path),
            },
            "source_candidate_draft_sha256": _sha256(args.candidate_draft),
        }
    )
    candidate_path = args.output_dir / "candidate-authorized.json"
    _atomic_json(candidate_path, candidate)
    _validate_candidate(
        candidate,
        expected_version=args.expected_package_version,
        expected_source_commit=args.expected_source_commit,
    )

    plan = dict(plan_draft)
    plan["status"] = AUTHORIZED_PLAN_STATUS
    plan["candidate_readiness"] = {
        **plan_draft["candidate_readiness"],
        "artifact_name": candidate_path.name,
        "candidate_sha256": _sha256(candidate_path),
        "source_readiness_artifact_name": "readiness-report.json",
    }
    plan["authorization_record"] = {
        "artifact_name": authorization_path.name,
        "sha256": _sha256(authorization_path),
    }
    plan["source_acquisition_plan_draft_sha256"] = _sha256(
        args.acquisition_plan_draft
    )
    plan_path = args.output_dir / "acquisition-plan-authorized.json"
    _atomic_json(plan_path, plan)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": AUTHORIZATION,
        "authorization_id": args.authorization_id,
        "artifacts": {
            path.name: {"sha256": _sha256(path)}
            for path in sorted(args.output_dir.iterdir())
            if path.is_file()
        },
        "authorized_candidate_sha256": _sha256(candidate_path),
        "authorized_acquisition_plan_sha256": _sha256(plan_path),
        "next_stage": (
            "register new outcome-blind panoramas and candidate pairs; do not open "
            "reference geometry before the prediction seal"
        ),
    }
    manifest_path = args.output_dir / "authorization-manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-draft", type=Path, required=True)
    parser.add_argument("--acquisition-plan-draft", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--capture-models", type=Path, required=True)
    parser.add_argument("--post-model", type=Path, required=True)
    parser.add_argument("--selective-rule", type=Path, required=True)
    parser.add_argument("--readiness-report", type=Path, required=True)
    parser.add_argument("--scenario-csv", type=Path, required=True)
    parser.add_argument("--expected-package-version", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--authorized-at", required=True)
    parser.add_argument("--authorization-source", required=True)
    parser.add_argument("--authorization-statement", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    manifest = authorize(_parser().parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
