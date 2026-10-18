#!/usr/bin/env python3
"""Evaluate prospective application evidence against the frozen protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "contracts/spherical-application-promotion-v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return value


def _matches(record: dict[str, Any], selector: dict[str, Any]) -> bool:
    return all(record.get(name) == expected for name, expected in selector.items())


def _number(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: expected a numeric measurement")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path}: measurement must be finite")
    return result


def _check(
    check: dict[str, Any],
    metrics: dict[str, Any],
    *,
    path: str,
) -> tuple[bool, str]:
    metric = check["metric"]
    if metric not in metrics:
        return False, f"{path}.metrics.{metric}: missing"
    value = metrics[metric]
    operator = check["operator"]
    if operator == "true":
        passed = value is True
        return passed, f"{metric} is {value!r}, expected true"

    numeric = _number(value, path=f"{path}.metrics.{metric}")
    if operator == "le_metric":
        threshold_metric = check["threshold_metric"]
        if threshold_metric not in metrics:
            return False, f"{path}.metrics.{threshold_metric}: missing"
        threshold = _number(
            metrics[threshold_metric],
            path=f"{path}.metrics.{threshold_metric}",
        )
    else:
        threshold = _number(check["threshold"], path=f"check.{metric}.threshold")
    if operator in {"le", "le_metric"}:
        passed = numeric <= threshold
        relation = "<="
    elif operator == "ge":
        passed = numeric >= threshold
        relation = ">="
    elif operator == "eq":
        passed = numeric == threshold
        relation = "=="
    else:
        raise ValueError(f"unsupported comparison operator {operator!r}")
    return passed, f"{metric}={numeric:g}, expected {relation} {threshold:g}"


def evaluate(
    protocol: dict[str, Any],
    evidence: dict[str, Any],
    *,
    protocol_sha256: str,
) -> dict[str, Any]:
    if evidence.get("contract") != protocol["contract"]:
        raise ValueError("evidence contract does not match the protocol")
    if evidence.get("protocol_sha256") != protocol_sha256:
        raise ValueError("evidence protocol_sha256 does not match the tracked protocol")
    records = evidence.get("records")
    if not isinstance(records, list) or any(
        not isinstance(item, dict) for item in records
    ):
        raise ValueError("evidence.records must be a list of objects")
    forbidden_cohorts = set(
        protocol.get("policy", {}).get("forbidden_cohort_manifest_sha256", [])
    )
    if any(not isinstance(item, str) or not item for item in forbidden_cohorts):
        raise ValueError(
            "policy.forbidden_cohort_manifest_sha256 must contain non-empty strings"
        )

    results = []
    for requirement in protocol["requirements"]:
        application_records = [
            (index, record)
            for index, record in enumerate(records)
            if record.get("application") == requirement["application"]
            and _matches(record, requirement["selector"])
        ]
        failures: list[str] = []
        minimum = requirement["minimum_records"]
        if len(application_records) < minimum:
            failures.append(
                f"matched {len(application_records)} records; requires at least {minimum}"
            )

        for index, record in application_records:
            cohort = record.get("cohort_manifest_sha256")
            if cohort in forbidden_cohorts:
                failures.append(
                    f"records[{index}]: cohort_manifest_sha256 reuses previously "
                    "observed evidence"
                )

        required_record_fields = requirement.get("required_record_fields", [])
        if any(
            not isinstance(field, str) or not field for field in required_record_fields
        ):
            raise ValueError(
                f"requirement {requirement['id']}: required_record_fields must "
                "contain non-empty strings"
            )
        for index, record in application_records:
            for field in required_record_fields:
                value = record.get(field)
                if value is None or value == "":
                    failures.append(f"records[{index}].{field}: missing")

        distinct = requirement.get("minimum_distinct")
        if distinct is not None:
            values = {
                record.get(distinct["field"]) for _, record in application_records
            }
            values.discard(None)
            if len(values) < distinct["count"]:
                failures.append(
                    f"distinct {distinct['field']} count {len(values)}; "
                    f"requires {distinct['count']}"
                )

        for field, required in requirement.get("required_values", {}).items():
            actual = {record.get(field) for _, record in application_records}
            missing = sorted(set(required) - actual)
            if missing:
                failures.append(f"missing required {field} values: {missing}")

        checked_records = []
        for index, record in application_records:
            metrics = record.get("metrics")
            if not isinstance(metrics, dict):
                failures.append(f"records[{index}].metrics must be an object")
                continue
            checks = []
            for check in requirement["checks"]:
                try:
                    passed, detail = _check(check, metrics, path=f"records[{index}]")
                except ValueError as exc:
                    passed, detail = False, str(exc)
                checks.append(
                    {
                        "metric": check["metric"],
                        "passed": passed,
                        "detail": detail,
                    }
                )
                if not passed:
                    failures.append(f"records[{index}]: {detail}")
            checked_records.append({"index": index, "checks": checks})

        results.append(
            {
                "application": requirement["application"],
                "requirement": requirement["id"],
                "passed": not failures,
                "failures": failures,
                "records": checked_records,
            }
        )

    return {
        "contract": protocol["contract"],
        "protocol_sha256": protocol_sha256,
        "passed": all(item["passed"] for item in results),
        "requirements": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-protocol-sha256", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    digest = _sha256(protocol_path)
    if args.print_protocol_sha256:
        print(digest)
        if args.evidence is None:
            return
    if args.evidence is None:
        parser.error("--evidence is required unless only printing the protocol hash")
    result = evaluate(
        _load_object(protocol_path),
        _load_object(args.evidence.resolve()),
        protocol_sha256=digest,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
