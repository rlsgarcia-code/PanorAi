#!/usr/bin/env python3
"""Score E8 pairs and choose the primary sample without reference geometry."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np

try:
    from run_probability_models import design_matrix, predict_logistic
    from run_prospective_confirmation import (
        AUTHORIZATION,
        CANDIDATE_SCHEMA,
        PREDICTION_SCHEMA,
        REGISTRY_SCHEMA,
    )
    from select_release_rule import (
        ALIGNED_POST_MODEL,
        CAPTURE_ACCEPT_MODEL,
        CAPTURE_PRECISE_MODEL,
    )
except ImportError:
    from benchmarks.two_view_pose_probability.run_probability_models import (
        design_matrix,
        predict_logistic,
    )
    from benchmarks.two_view_pose_probability.run_prospective_confirmation import (
        AUTHORIZATION,
        CANDIDATE_SCHEMA,
        PREDICTION_SCHEMA,
        REGISTRY_SCHEMA,
    )
    from benchmarks.two_view_pose_probability.select_release_rule import (
        ALIGNED_POST_MODEL,
        CAPTURE_ACCEPT_MODEL,
        CAPTURE_PRECISE_MODEL,
    )


FRONTEND_SCHEMA = "panorai-two-view-prospective-frontend/v1"
MODEL_SNAPSHOT_SCHEMA = "panorai-two-view-prospective-model-snapshot/v1"
RULE_AMENDMENT_SCHEMA = "panorai-two-view-prospective-rule-amendment/v1"
ACQUISITION_SCHEMA = "panorai-two-view-prospective-acquisition-plan/v1"
AUTHORIZED_PLAN_STATUS = "AUTHORIZED_FOR_COLLECTION"
REPORT_SCHEMA = "panorai-two-view-prospective-prediction-build/v1"
REFERENCE_FIELDS = {
    "catastrophic_accepted",
    "ground_truth",
    "outcomes",
    "precise",
    "primary",
    "reference_rotation",
    "reference_translation",
    "rotation_error_deg",
    "translation_direction_error_deg",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contains_reference(value: Any, path: str = "$") -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in REFERENCE_FIELDS:
                return f"{path}.{key}"
            violation = _contains_reference(child, f"{path}.{key}")
            if violation is not None:
                return violation
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violation = _contains_reference(child, f"{path}[{index}]")
            if violation is not None:
                return violation
    return None


def _index(rows: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed = {}
    for row in rows:
        pair_id = str(row.get("pair_id", ""))
        if not pair_id or pair_id in indexed:
            raise ValueError(f"missing or duplicate {label} pair_id: {pair_id}")
        indexed[pair_id] = row
    return indexed


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


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _models(snapshot: Mapping[str, Any], role: str) -> dict[str, dict[str, Any]]:
    if snapshot.get("schema") != MODEL_SNAPSHOT_SCHEMA or snapshot.get("role") != role:
        raise ValueError(f"invalid {role} model snapshot")
    result = {}
    for model in snapshot.get("models", []):
        model_id = str(model.get("model_id", ""))
        if not model_id or model_id in result:
            raise ValueError(f"missing or duplicate model in {role} snapshot")
        result[model_id] = model
    return result


def _score(model: Mapping[str, Any], row: dict[str, Any]) -> float:
    specification = tuple(
        (str(feature["path"]), str(feature["transform"]))
        for feature in model["features"]
    )
    values, complete = design_matrix([row], specification)
    if not bool(complete[0]):
        raise ValueError(f"incomplete features for model {model['model_id']}")
    mean = np.asarray(model["scaler_mean"], dtype=np.float64)
    scale = np.asarray(model["scaler_scale"], dtype=np.float64)
    parameters = np.asarray(model["parameters"], dtype=np.float64)
    platt = np.asarray(model["platt_parameters"], dtype=np.float64)
    if (
        values.shape[1] != len(mean)
        or len(mean) != len(scale)
        or len(parameters) != len(mean) + 1
        or len(platt) != 2
        or np.any(scale <= 0.0)
    ):
        raise ValueError(f"invalid serialized parameters for {model['model_id']}")
    raw = float(predict_logistic((values - mean) / scale, parameters)[0])
    clipped = min(max(raw, 1e-6), 1.0 - 1e-6)
    logit = math.log(clipped / (1.0 - clipped))
    probability = float(predict_logistic(np.asarray([[logit]]), platt)[0])
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"invalid model probability for {model['model_id']}")
    return probability


def _stable_tie(candidate_hash: str, pair_id: str) -> str:
    return hashlib.sha256(f"{candidate_hash}|{pair_id}".encode()).hexdigest()


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError("prospective prediction output directory must not exist")
    candidate = _read_json(args.candidate)
    plan = _read_json(args.acquisition_plan)
    capture_snapshot = _read_json(args.capture_models)
    post_snapshot = _read_json(args.post_model)
    rule = _read_json(args.selective_rule)
    if candidate.get("schema") != CANDIDATE_SCHEMA:
        raise ValueError("candidate schema is invalid")
    if plan.get("schema") != ACQUISITION_SCHEMA:
        raise ValueError("acquisition plan schema is invalid")
    if rule.get("schema") != RULE_AMENDMENT_SCHEMA:
        raise ValueError("selective rule schema is invalid")
    candidate_hash = _sha256(args.candidate)
    if plan.get("candidate_readiness", {}).get("candidate_sha256") != candidate_hash:
        raise ValueError("candidate does not match acquisition plan")
    for field, path in (
        ("capture_model_sha256", args.capture_models),
        ("post_model_sha256", args.post_model),
        ("selective_rule_sha256", args.selective_rule),
    ):
        if candidate.get(field) != _sha256(path):
            raise ValueError(f"candidate {field} does not match supplied artifact")

    capture_models = _models(capture_snapshot, "capture")
    post_models = _models(post_snapshot, "post-processing")
    required_capture = {CAPTURE_ACCEPT_MODEL, CAPTURE_PRECISE_MODEL}
    if set(capture_models) != required_capture or set(post_models) != {
        ALIGNED_POST_MODEL
    }:
        raise ValueError("candidate model snapshots contain unexpected models")

    registry = _index(_read_jsonl(args.registry), "registry")
    frontend = _index(_read_jsonl(args.frontend), "frontend")
    if set(registry) != set(frontend):
        raise ValueError("frontend pair IDs differ from the frozen registry")
    plan_hash = _sha256(args.acquisition_plan)
    thresholds = rule["thresholds"]
    capture_threshold = float(thresholds["capture_usable_probability"])
    post_threshold = float(thresholds["post_precision_probability"])
    provisional = []
    for pair_id in sorted(registry):
        registered = registry[pair_id]
        result = frontend[pair_id]
        if registered.get("schema") != REGISTRY_SCHEMA:
            raise ValueError(f"invalid registry schema for {pair_id}")
        if result.get("schema") != FRONTEND_SCHEMA:
            raise ValueError(f"invalid frontend schema for {pair_id}")
        violation = _contains_reference(result)
        if violation is not None:
            raise ValueError(f"frontend result contains reference field: {violation}")
        if (
            result.get("group_id") != registered.get("group_id")
            or result.get("domain_id") != registered.get("domain_id")
        ):
            raise ValueError(f"frontend identity mismatch for {pair_id}")
        if registered.get("candidate_sha256") not in (None, candidate_hash):
            raise ValueError(f"registry candidate mismatch for {pair_id}")
        if registered.get("acquisition_plan_sha256") not in (None, plan_hash):
            raise ValueError(f"registry plan mismatch for {pair_id}")
        returned = result.get("returned")
        accepted = result.get("accepted")
        if not isinstance(returned, bool) or not isinstance(accepted, bool):
            raise ValueError(f"frontend decisions are not boolean for {pair_id}")
        if accepted and not returned:
            raise ValueError(f"accepted frontend result did not return pose: {pair_id}")
        feature = {
            "capture": registered["capture"],
            "post": result.get("post", {}),
        }
        p_accept = _score(capture_models[CAPTURE_ACCEPT_MODEL], feature)
        p_conditional = _score(capture_models[CAPTURE_PRECISE_MODEL], feature)
        p_usable = p_accept * p_conditional
        post_available = returned
        p_post = (
            _score(post_models[ALIGNED_POST_MODEL], feature)
            if post_available
            else 0.0
        )
        selected = bool(
            accepted
            and p_usable >= capture_threshold
            and p_post >= post_threshold
        )
        provisional.append(
            {
                "schema": PREDICTION_SCHEMA,
                "pair_id": pair_id,
                "group_id": registered["group_id"],
                "domain_id": registered["domain_id"],
                "returned": returned,
                "accepted": accepted,
                "selected": selected,
                "primary_selected": False,
                "p_accept_capture": p_accept,
                "p_precise_capture_given_accept": p_conditional,
                "p_usable_capture": p_usable,
                "p_precise_post": p_post,
                "post_probability_available": post_available,
                "failure_reason": (
                    None
                    if selected
                    else "frontend-not-accepted"
                    if not accepted
                    else "capture-probability-below-threshold"
                    if p_usable < capture_threshold
                    else "post-probability-below-threshold"
                ),
            }
        )

    domains = tuple(plan["balanced_domain_design"]["labels"])
    target_per_domain = int(
        plan["balanced_domain_design"]["selected_target_per_domain"]
    )
    cap = int(plan["unit"]["primary_selected_pair_cap_per_group"])
    selected_by_domain = {
        domain: sorted(
            [
                row
                for row in provisional
                if row["domain_id"] == domain and row["selected"]
            ],
            key=lambda row: (
                -float(row["p_precise_post"]),
                -float(row["p_usable_capture"]),
                _stable_tie(candidate_hash, str(row["pair_id"])),
            ),
        )
        for domain in domains
    }
    primary_by_group: Counter[str] = Counter()
    primary_by_domain: Counter[str] = Counter()
    for domain in domains:
        for row in selected_by_domain[domain]:
            if primary_by_domain[domain] >= target_per_domain:
                break
            group = str(row["group_id"])
            if primary_by_group[group] >= cap:
                continue
            row["primary_selected"] = True
            primary_by_group[group] += 1
            primary_by_domain[domain] += 1
    complete = all(primary_by_domain[domain] == target_per_domain for domain in domains)
    authorized = (
        candidate.get("authorization") == AUTHORIZATION
        and plan.get("status") == AUTHORIZED_PLAN_STATUS
    )
    if args.mode == "freeze" and not complete:
        raise ValueError("primary sample does not meet every domain target")
    if args.mode == "freeze" and not authorized:
        raise PermissionError("candidate and acquisition plan are not authorized")

    args.output_dir.mkdir(parents=True)
    output_name = "predictions.jsonl" if args.mode == "freeze" else "predictions-audit.jsonl"
    predictions_path = args.output_dir / output_name
    _atomic_jsonl(predictions_path, provisional)
    report = {
        "schema": REPORT_SCHEMA,
        "mode": args.mode,
        "status": (
            "FROZEN_PREDICTIONS_READY_FOR_SEAL"
            if args.mode == "freeze"
            else "AUDIT_COMPLETE_BUT_NOT_AUTHORIZED"
            if complete and not authorized
            else "AUDIT_COMPLETE"
            if complete
            else "PRIMARY_SAMPLE_INCOMPLETE"
        ),
        "candidate_sha256": candidate_hash,
        "acquisition_plan_sha256": plan_hash,
        "registry_sha256": _sha256(args.registry),
        "frontend_sha256": _sha256(args.frontend),
        "predictions_sha256": _sha256(predictions_path),
        "counts": {
            "candidate_pairs": len(provisional),
            "returned": sum(bool(row["returned"]) for row in provisional),
            "accepted": sum(bool(row["accepted"]) for row in provisional),
            "selected": sum(bool(row["selected"]) for row in provisional),
            "primary_selected": sum(
                bool(row["primary_selected"]) for row in provisional
            ),
            "primary_by_domain": dict(sorted(primary_by_domain.items())),
            "primary_by_group": dict(sorted(primary_by_group.items())),
        },
        "thresholds": {
            "capture_usable_probability": capture_threshold,
            "post_precision_probability": post_threshold,
            "maximum_primary_pairs_per_group": cap,
            "primary_target_per_domain": target_per_domain,
        },
        "ranking": (
            "descending post probability, descending usable capture probability, "
            "then candidate-bound SHA-256 tie break"
        ),
        "complete": complete,
        "authorized": authorized,
        "reference_fields_opened": False,
    }
    _atomic_json(args.output_dir / "prediction-build-report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "freeze"))
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--acquisition-plan", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--capture-models", type=Path, required=True)
    parser.add_argument("--post-model", type=Path, required=True)
    parser.add_argument("--selective-rule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    report = build(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
