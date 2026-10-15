from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "contracts/spherical-application-promotion-v1.json"
SCRIPT = ROOT / "scripts/evaluate_spherical_application_promotion.py"


def _load_script() -> dict[str, object]:
    namespace: dict[str, object] = {
        "__name__": "promotion_contract_test",
        "__file__": str(SCRIPT),
    }
    exec(compile(SCRIPT.read_text(encoding="utf-8"), str(SCRIPT), "exec"), namespace)
    return namespace


def _passing_evidence() -> dict[str, object]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    digest = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    records = []
    for requirement in protocol["requirements"]:
        selector = requirement["selector"]
        required_values = requirement.get("required_values", {})
        count = requirement["minimum_records"]
        for index in range(count):
            record = {
                "application": requirement["application"],
                **selector,
                "metrics": {},
            }
            for field, values in required_values.items():
                record[field] = values[index % len(values)]
            distinct = requirement.get("minimum_distinct")
            if distinct and distinct["field"] not in record:
                record[distinct["field"]] = f"corpus-{index}"
            for check in requirement["checks"]:
                operator = check["operator"]
                if operator == "true":
                    value = True
                elif operator == "ge":
                    value = check["threshold"]
                elif operator in {"le", "eq"}:
                    value = check["threshold"]
                elif operator == "le_metric":
                    value = 0.25
                    record["metrics"][check["threshold_metric"]] = 0.25
                else:
                    raise AssertionError(operator)
                record["metrics"][check["metric"]] = value
            records.append(record)
    return {
        "contract": protocol["contract"],
        "protocol_sha256": digest,
        "records": records,
    }


def test_frozen_protocol_encodes_the_plan_thresholds() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    serialized = json.dumps(protocol, sort_keys=True)

    assert protocol["contract"] == "panorai-spherical-application-promotion/v1"
    assert protocol["policy"]["threshold_changes_after_heldout"] == "forbidden"
    for value in (
        "isolated_blob_detection_rate",
        "repeatability",
        "median_location_error_deg",
        "verified_inlier_precision",
        "abs_rel",
        "delta1",
        "accepted_coverage",
        "proposal_recall_spherical_iou_0_5",
        "median_fused_mask_spherical_iou",
        "non_abstained_instance_precision",
        "non_abstained_instance_coverage",
    ):
        assert value in serialized


def test_complete_evidence_passes_and_missing_cohorts_fail_closed() -> None:
    namespace = _load_script()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    digest = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    evidence = _passing_evidence()

    passing = namespace["evaluate"](protocol, evidence, protocol_sha256=digest)
    assert passing["passed"] is True

    incomplete = copy.deepcopy(evidence)
    incomplete["records"] = [
        record for record in incomplete["records"] if record["application"] == "slic"
    ]
    result = namespace["evaluate"](protocol, incomplete, protocol_sha256=digest)
    assert result["passed"] is False
    assert any(
        "matched 0 records" in failure
        for requirement in result["requirements"]
        for failure in requirement["failures"]
    )


def test_failed_metric_and_non_finite_value_fail() -> None:
    namespace = _load_script()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    digest = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    evidence = _passing_evidence()
    dog = next(
        record
        for record in evidence["records"]
        if record["application"] == "dog" and record["evaluation"] == "external"
    )
    dog["metrics"]["repeatability"] = float("nan")

    result = namespace["evaluate"](protocol, evidence, protocol_sha256=digest)

    assert result["passed"] is False
    dog_gate = next(
        item
        for item in result["requirements"]
        if item["requirement"] == "dog-external-heldout-corpora"
    )
    assert any("finite" in failure for failure in dog_gate["failures"])


def test_evidence_is_bound_to_exact_protocol_hash() -> None:
    namespace = _load_script()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    evidence = _passing_evidence()

    with pytest.raises(ValueError, match="protocol_sha256"):
        namespace["evaluate"](protocol, evidence, protocol_sha256="changed")


def test_cli_returns_nonzero_for_incomplete_evidence(tmp_path: Path) -> None:
    evidence = _passing_evidence()
    evidence["records"] = []
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--evidence", str(path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert '"passed": false' in completed.stdout
