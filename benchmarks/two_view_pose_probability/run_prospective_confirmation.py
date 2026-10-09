#!/usr/bin/env python3
"""Seal and evaluate E8 without opening reference geometry before predictions."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np

try:
    from select_release_rule import exact_one_sided_lower  # type: ignore[import-not-found]
except ImportError:
    from benchmarks.two_view_pose_probability.select_release_rule import (
        exact_one_sided_lower,
    )


CANDIDATE_SCHEMA = "panorai-two-view-prospective-candidate/v1"
REGISTRY_SCHEMA = "panorai-two-view-prospective-registry/v1"
PREDICTION_SCHEMA = "panorai-two-view-prospective-prediction/v1"
REFERENCE_SCHEMA = "panorai-two-view-prospective-reference/v1"
SEAL_SCHEMA = "panorai-two-view-prospective-seal/v1"
REPORT_SCHEMA = "panorai-two-view-prospective-report/v1"
AUTHORIZATION = "GO_FOR_PROSPECTIVE_CONFIRMATION"
REFERENCE_ONLY_FIELDS = {
    "rotation_error_deg",
    "translation_direction_error_deg",
    "primary",
    "strict",
    "precise",
    "usable",
    "catastrophic_accepted",
    "reference_rotation",
    "reference_translation",
    "ground_truth",
}
REGISTRY_FORBIDDEN_FIELDS = REFERENCE_ONLY_FIELDS | {
    "returned",
    "accepted",
    "selected",
    "primary_selected",
    "p_accept_capture",
    "p_precise_capture_given_accept",
    "p_usable_capture",
    "p_precise_post",
}
PROBABILITY_FIELDS = (
    "p_accept_capture",
    "p_precise_capture_given_accept",
    "p_usable_capture",
    "p_precise_post",
)
EXPECTED_GATE = {
    "minimum_selected_precision": 0.95,
    "minimum_exact_one_sided_95_lower": 0.90,
    "maximum_catastrophic_accepted": 0,
    "minimum_independent_groups": 40,
    "minimum_selected_pairs": 120,
    "maximum_selected_pairs_per_group": 3,
}
BOOTSTRAP_REPETITIONS = 10_000


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


def _contains_forbidden(value: Any, forbidden: set[str], path: str = "$") -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in forbidden:
                return f"{path}.{key}"
            violation = _contains_forbidden(child, forbidden, f"{path}.{key}")
            if violation is not None:
                return violation
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violation = _contains_forbidden(child, forbidden, f"{path}[{index}]")
            if violation is not None:
                return violation
    return None


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_candidate(
    candidate: dict[str, Any], *, expected_version: str, expected_source_commit: str
) -> None:
    if candidate.get("schema") != CANDIDATE_SCHEMA:
        raise ValueError("invalid prospective candidate schema")
    if candidate.get("authorization") != AUTHORIZATION:
        raise ValueError("candidate is not authorized for prospective confirmation")
    package = candidate.get("panorai", {})
    if package.get("version") != expected_version:
        raise ValueError("prospective candidate package version mismatch")
    if package.get("source_commit") != expected_source_commit:
        raise ValueError("prospective candidate source commit mismatch")
    for field in (
        "wheel_sha256",
        "configuration_sha256",
        "capture_model_sha256",
        "post_model_sha256",
        "selective_rule_sha256",
    ):
        if not _is_sha256(candidate.get(field)):
            raise ValueError(f"candidate {field} is not a SHA-256 digest")
    if candidate.get("gate") != EXPECTED_GATE:
        raise ValueError("prospective candidate gate differs from frozen E8 gate")
    if candidate.get("deviations") != []:
        raise ValueError("confirmatory candidate must declare zero deviations")


def _index(rows: Iterable[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        pair_id = str(row.get("pair_id", ""))
        if not pair_id:
            raise ValueError(f"{label} row has no pair_id")
        if pair_id in result:
            raise ValueError(f"duplicate {label} pair_id: {pair_id}")
        result[pair_id] = row
    return result


def _retrospective_groups(rows: list[dict[str, Any]]) -> set[str]:
    groups = set()
    for row in rows:
        group = str(row.get("group_id", row.get("independence_component_id", "")))
        if not group:
            raise ValueError("retrospective group row has no group identifier")
        groups.add(group)
    return groups


def _validate_registry(
    rows: list[dict[str, Any]], *, forbidden_groups: set[str]
) -> dict[str, dict[str, Any]]:
    indexed = _index(rows, label="registry")
    unordered_pairs = set()
    image_groups: dict[str, str] = {}
    for pair_id, row in indexed.items():
        if row.get("schema") != REGISTRY_SCHEMA:
            raise ValueError(f"invalid registry schema for {pair_id}")
        violation = _contains_forbidden(row, REGISTRY_FORBIDDEN_FIELDS)
        if violation is not None:
            raise ValueError(f"registry contains forbidden field: {violation}")
        group = str(row.get("group_id", ""))
        domain = str(row.get("domain_id", ""))
        images = row.get("image_ids")
        if not group or not domain:
            raise ValueError(f"registry identity is incomplete for {pair_id}")
        if group in forbidden_groups:
            raise ValueError(f"prospective group was used retrospectively: {group}")
        if not isinstance(images, list) or len(images) != 2:
            raise ValueError(f"registry pair must contain exactly two images: {pair_id}")
        image_a, image_b = (str(images[0]), str(images[1]))
        if not image_a or not image_b or image_a == image_b:
            raise ValueError(f"registry pair has invalid image IDs: {pair_id}")
        unordered = tuple(sorted((image_a, image_b)))
        if unordered in unordered_pairs:
            raise ValueError(f"duplicate or reversed registry pair: {pair_id}")
        unordered_pairs.add(unordered)
        for image in unordered:
            previous = image_groups.setdefault(image, group)
            if previous != group:
                raise ValueError(f"image crosses prospective groups: {image}")
    return indexed


def _probability(row: Mapping[str, Any], field: str) -> float:
    value = float(row.get(field, math.nan))
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"invalid probability {field} for {row.get('pair_id')}")
    return value


def _validate_predictions(
    rows: list[dict[str, Any]], registry: Mapping[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    indexed = _index(rows, label="prediction")
    if set(indexed) != set(registry):
        raise ValueError("prediction pair IDs differ from the registered candidate set")
    selected_by_group: Counter[str] = Counter()
    for pair_id, row in indexed.items():
        if row.get("schema") != PREDICTION_SCHEMA:
            raise ValueError(f"invalid prediction schema for {pair_id}")
        violation = _contains_forbidden(row, REFERENCE_ONLY_FIELDS)
        if violation is not None:
            raise ValueError(f"prediction contains reference-only field: {violation}")
        registered = registry[pair_id]
        if row.get("group_id") != registered["group_id"]:
            raise ValueError(f"prediction group mismatch for {pair_id}")
        if row.get("domain_id") != registered["domain_id"]:
            raise ValueError(f"prediction domain mismatch for {pair_id}")
        returned = row.get("returned")
        accepted = row.get("accepted")
        selected = row.get("selected")
        primary_selected = row.get("primary_selected")
        if not all(
            isinstance(value, bool)
            for value in (returned, accepted, selected, primary_selected)
        ):
            raise ValueError(f"prediction decisions must be boolean for {pair_id}")
        if accepted and not returned:
            raise ValueError(f"accepted prediction did not return a pose: {pair_id}")
        if selected and not accepted:
            raise ValueError(f"selected prediction was not accepted: {pair_id}")
        if primary_selected and not selected:
            raise ValueError(f"primary prediction was not selected: {pair_id}")
        probabilities = {field: _probability(row, field) for field in PROBABILITY_FIELDS}
        product = (
            probabilities["p_accept_capture"]
            * probabilities["p_precise_capture_given_accept"]
        )
        if not math.isclose(
            probabilities["p_usable_capture"], product, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"p_usable_capture is not the declared product: {pair_id}")
        if primary_selected:
            selected_by_group[str(row["group_id"])] += 1
    limit = EXPECTED_GATE["maximum_selected_pairs_per_group"]
    if any(count > limit for count in selected_by_group.values()):
        raise ValueError("a prospective group contributes more than three primary pairs")
    return indexed


def seal(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError("prospective seal output must not exist")
    if args.future_references.exists():
        raise FileExistsError("reference geometry exists before prediction sealing")
    candidate = _read_json(args.candidate)
    _validate_candidate(
        candidate,
        expected_version=args.expected_package_version,
        expected_source_commit=args.expected_source_commit,
    )
    retrospective_rows = _read_jsonl(args.retrospective_groups)
    registry_rows = _read_jsonl(args.registry)
    prediction_rows = _read_jsonl(args.predictions)
    registry = _validate_registry(
        registry_rows, forbidden_groups=_retrospective_groups(retrospective_rows)
    )
    predictions = _validate_predictions(prediction_rows, registry)
    primary = [row for row in predictions.values() if row["primary_selected"]]
    groups = {str(row["group_id"]) for row in primary}
    if len(primary) < EXPECTED_GATE["minimum_selected_pairs"]:
        raise ValueError("prospective seal has fewer than 120 primary selected pairs")
    if len(groups) < EXPECTED_GATE["minimum_independent_groups"]:
        raise ValueError("prospective seal has fewer than 40 contributing groups")
    seal_record = {
        "schema": SEAL_SCHEMA,
        "status": "predictions sealed before reference geometry",
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "path": str(args.candidate.resolve()),
            "sha256": _sha256(args.candidate),
            "authorization": AUTHORIZATION,
            "package_version": args.expected_package_version,
            "source_commit": args.expected_source_commit,
        },
        "registry": {
            "path": str(args.registry.resolve()),
            "sha256": _sha256(args.registry),
            "pairs": len(registry),
            "unique_images": len(
                {str(image) for row in registry.values() for image in row["image_ids"]}
            ),
            "groups": len({str(row["group_id"]) for row in registry.values()}),
        },
        "predictions": {
            "path": str(args.predictions.resolve()),
            "sha256": _sha256(args.predictions),
            "returned": sum(bool(row["returned"]) for row in predictions.values()),
            "accepted": sum(bool(row["accepted"]) for row in predictions.values()),
            "selected": sum(bool(row["selected"]) for row in predictions.values()),
            "primary_selected": len(primary),
            "primary_groups": len(groups),
            "maximum_primary_pairs_per_group": max(
                Counter(str(row["group_id"]) for row in primary).values()
            ),
        },
        "retrospective_groups": {
            "path": str(args.retrospective_groups.resolve()),
            "sha256": _sha256(args.retrospective_groups),
            "groups": len(_retrospective_groups(retrospective_rows)),
        },
        "future_references_path": str(args.future_references.resolve()),
        "gate": EXPECTED_GATE,
    }
    _atomic_json(args.output, seal_record)
    return seal_record


def _validate_sealed_file(record: Mapping[str, Any], label: str) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_file() or _sha256(path) != record.get("sha256"):
        raise ValueError(f"sealed {label} file is missing or changed")
    return path


def _bootstrap_precision(
    rows: list[dict[str, Any]], *, seed: int, repetitions: int = BOOTSTRAP_REPETITIONS
) -> dict[str, Any]:
    groups = sorted({str(row["group_id"]) for row in rows})
    by_group = {group: [row for row in rows if row["group_id"] == group] for group in groups}
    generator = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        selected_groups = generator.choice(groups, size=len(groups), replace=True)
        sampled = [row for group in selected_groups for row in by_group[str(group)]]
        values[index] = np.mean([bool(row["precise"]) for row in sampled])
    return {
        "method": "independent-group percentile bootstrap",
        "repetitions": repetitions,
        "seed": seed,
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _reliability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, 6)
    records = []
    for index in range(5):
        low, high = float(edges[index]), float(edges[index + 1])
        members = [
            row
            for row in rows
            if low <= float(row["p_precise_post"])
            and (float(row["p_precise_post"]) < high or index == 4)
        ]
        records.append(
            {
                "lower": low,
                "upper": high,
                "count": len(members),
                "predicted_mean": (
                    float(np.mean([row["p_precise_post"] for row in members]))
                    if members
                    else None
                ),
                "observed_precision": (
                    float(np.mean([row["precise"] for row in members]))
                    if members
                    else None
                ),
            }
        )
    return records


def _capture_ranges(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    values: dict[str, list[float]] = {}
    for row in rows:
        capture = row.get("capture", {})
        if not isinstance(capture, dict):
            continue
        for field, value in capture.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                number = float(value)
                if math.isfinite(number):
                    values.setdefault(str(field), []).append(number)
    return {
        field: {"count": len(numbers), "minimum": min(numbers), "maximum": max(numbers)}
        for field, numbers in sorted(values.items())
    }


def _metrics(
    rows: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    count = len(rows)
    successes = sum(bool(row["precise"]) for row in rows)
    catastrophic = sum(bool(row["catastrophic_accepted"]) for row in rows)
    groups = len({str(row["group_id"]) for row in rows})
    precision = successes / count if count else None
    return {
        "selected_pairs": count,
        "selected_groups": groups,
        "precise_pairs": successes,
        "selected_precision": precision,
        "exact_one_sided_95_lower": exact_one_sided_lower(successes, count),
        "catastrophic_accepted": catastrophic,
        "group_bootstrap_precision": (
            _bootstrap_precision(rows, seed=seed) if rows else None
        ),
        "reliability": _reliability(rows),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError("prospective evaluation output must not exist")
    seal_record = _read_json(args.seal)
    if seal_record.get("schema") != SEAL_SCHEMA:
        raise ValueError("invalid prospective seal schema")
    candidate_path = _validate_sealed_file(seal_record["candidate"], "candidate")
    registry_path = _validate_sealed_file(seal_record["registry"], "registry")
    predictions_path = _validate_sealed_file(seal_record["predictions"], "predictions")
    retrospective_path = _validate_sealed_file(
        seal_record["retrospective_groups"], "retrospective groups"
    )
    if args.references.resolve() != Path(seal_record["future_references_path"]):
        raise ValueError("reference path differs from the path declared before sealing")
    candidate = _read_json(candidate_path)
    _validate_candidate(
        candidate,
        expected_version=seal_record["candidate"]["package_version"],
        expected_source_commit=seal_record["candidate"]["source_commit"],
    )
    registry = _validate_registry(
        _read_jsonl(registry_path),
        forbidden_groups=_retrospective_groups(_read_jsonl(retrospective_path)),
    )
    predictions = _validate_predictions(_read_jsonl(predictions_path), registry)
    references = _index(_read_jsonl(args.references), label="reference")
    if set(references) != set(registry):
        raise ValueError("reference pair IDs differ from the sealed registry")
    joined = []
    for pair_id in sorted(registry):
        reference = references[pair_id]
        if reference.get("schema") != REFERENCE_SCHEMA:
            raise ValueError(f"invalid reference schema for {pair_id}")
        rotation = float(reference.get("rotation_error_deg", math.nan))
        translation = float(
            reference.get("translation_direction_error_deg", math.nan)
        )
        if not all(
            math.isfinite(value) and 0.0 <= value <= 180.0
            for value in (rotation, translation)
        ):
            raise ValueError(f"invalid reference error for {pair_id}")
        prediction = predictions[pair_id]
        accepted = bool(prediction["accepted"])
        primary = rotation <= 15.0 and translation <= 30.0
        precise = rotation <= 1.0 and translation <= 5.0
        joined.append(
            {
                "pair_id": pair_id,
                "group_id": prediction["group_id"],
                "domain_id": prediction["domain_id"],
                "returned": prediction["returned"],
                "accepted": accepted,
                "selected": prediction["selected"],
                "primary_selected": prediction["primary_selected"],
                **{field: prediction[field] for field in PROBABILITY_FIELDS},
                "rotation_error_deg": rotation,
                "translation_direction_error_deg": translation,
                "primary": primary,
                "precise": precise,
                "catastrophic_accepted": accepted and not primary,
                "failure_reason": prediction.get("failure_reason"),
            }
        )
    selected = [row for row in joined if row["primary_selected"]]
    accepted = [row for row in joined if row["accepted"]]
    seal_hash = _sha256(args.seal)
    seed = int.from_bytes(bytes.fromhex(seal_hash[:16]), "big")
    overall = _metrics(selected, seed=seed)
    selected_by_group = Counter(str(row["group_id"]) for row in selected)
    gate_checks = {
        "minimum_selected_precision": bool(
            overall["selected_precision"] is not None
            and overall["selected_precision"] >= EXPECTED_GATE["minimum_selected_precision"]
        ),
        "minimum_exact_one_sided_95_lower": bool(
            overall["exact_one_sided_95_lower"]
            >= EXPECTED_GATE["minimum_exact_one_sided_95_lower"]
        ),
        "maximum_catastrophic_accepted": bool(
            overall["catastrophic_accepted"]
            <= EXPECTED_GATE["maximum_catastrophic_accepted"]
        ),
        "minimum_independent_groups": bool(
            overall["selected_groups"] >= EXPECTED_GATE["minimum_independent_groups"]
        ),
        "minimum_selected_pairs": bool(
            overall["selected_pairs"] >= EXPECTED_GATE["minimum_selected_pairs"]
        ),
        "maximum_selected_pairs_per_group": bool(
            max(selected_by_group.values(), default=0)
            <= EXPECTED_GATE["maximum_selected_pairs_per_group"]
        ),
    }
    domains = {}
    for domain in sorted({str(row["domain_id"]) for row in joined}):
        domain_selected = [row for row in selected if row["domain_id"] == domain]
        domains[domain] = _metrics(
            domain_selected,
            seed=seed
            ^ int.from_bytes(hashlib.sha256(domain.encode()).digest()[:8], "big"),
        )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "PASS" if all(gate_checks.values()) else "FAIL",
        "interpretation": (
            "prospective gate passed; independent review is still required"
            if all(gate_checks.values())
            else "prospective gate failed; no release-reliability claim"
        ),
        "seal": {"path": str(args.seal.resolve()), "sha256": seal_hash},
        "references": {
            "path": str(args.references.resolve()),
            "sha256": _sha256(args.references),
            "pairs": len(references),
        },
        "population": {
            "candidate_pairs": len(joined),
            "unique_images": seal_record["registry"]["unique_images"],
            "groups": seal_record["registry"]["groups"],
            "returned": sum(bool(row["returned"]) for row in joined),
            "accepted": sum(bool(row["accepted"]) for row in joined),
            "frontend_catastrophic_accepted": sum(
                bool(row["catastrophic_accepted"]) for row in accepted
            ),
            "selected": sum(bool(row["selected"]) for row in joined),
            "primary_selected": len(selected),
            "pair_coverage": len(selected) / len(joined) if joined else 0.0,
        },
        "overall": overall,
        "domains": domains,
        "gate": EXPECTED_GATE,
        "gate_checks": gate_checks,
        "capture_ranges": _capture_ranges(registry.values()),
        "failure_reasons": dict(
            sorted(
                Counter(
                    str(row["failure_reason"])
                    for row in joined
                    if row.get("failure_reason")
                ).items()
            )
        ),
        "deviations": candidate["deviations"],
    }
    args.output_dir.mkdir(parents=True)
    _atomic_jsonl(args.output_dir / "prospective-joined.jsonl", joined)
    report["joined_sha256"] = _sha256(
        args.output_dir / "prospective-joined.jsonl"
    )
    _atomic_json(args.output_dir / "prospective-report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal_parser = subparsers.add_parser("seal", help="seal outcome-blind predictions")
    seal_parser.add_argument("--candidate", type=Path, required=True)
    seal_parser.add_argument("--registry", type=Path, required=True)
    seal_parser.add_argument("--predictions", type=Path, required=True)
    seal_parser.add_argument("--retrospective-groups", type=Path, required=True)
    seal_parser.add_argument("--future-references", type=Path, required=True)
    seal_parser.add_argument("--expected-package-version", required=True)
    seal_parser.add_argument("--expected-source-commit", required=True)
    seal_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="open references only after verifying the seal"
    )
    evaluate_parser.add_argument("--seal", type=Path, required=True)
    evaluate_parser.add_argument("--references", type=Path, required=True)
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = seal(args) if args.command == "seal" else evaluate(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
