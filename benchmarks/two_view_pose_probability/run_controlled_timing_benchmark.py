#!/usr/bin/env python3
"""Run the frozen overlap-stratified timing benchmark on an idle host."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
from typing import Any

try:
    from run_resumable_population_replay import (  # type: ignore[import-not-found]
        _atomic_json,
        _load_valid_result,
        _sha256,
        _slug,
    )
except ImportError:
    from benchmarks.two_view_pose_probability.run_resumable_population_replay import (
        _atomic_json,
        _load_valid_result,
        _sha256,
        _slug,
    )


SELECTION_SCHEMA = "panorai-two-view-controlled-timing-selection/v1"
SELECTION_MANIFEST_SCHEMA = "panorai-two-view-controlled-timing-manifest/v1"
HOST_GATE_SCHEMA = "panorai-controlled-timing-host-gate/v1"
STATUS_SCHEMA = "panorai-controlled-timing-status/v1"
EXPECTED_VERSION = "3.5.0"
EXPECTED_WHEEL_SHA256 = (
    "e861dafbaa5991aef77dd512b3ef1bf6fdc7967d10fbd236d850bab5c1a5f8a7"
)
ORDER_SEED = "panorai-val018-controlled-timing-order-v1"
DATASETS = ("matterport360", "stanford2d3d", "p74_native_polar")
OVERLAP_BINS = ("lt-10", "10-25", "25-50", "50-70", "ge-70")
_stop_requested = False


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def deterministic_order(
    rows: list[dict[str, Any]], repetition: int
) -> list[dict[str, Any]]:
    if repetition < 1:
        raise ValueError("repetition must be positive")

    def key(row: dict[str, Any]) -> tuple[str, str, str]:
        identity = f"{row['dataset_id']}|{row['pair_id']}"
        digest = hashlib.sha256(
            f"{ORDER_SEED}|{repetition}|{identity}".encode()
        ).hexdigest()
        return digest, str(row["dataset_id"]), str(row["pair_id"])

    return sorted(rows, key=key)


def validate_host_gate(
    gate: dict[str, Any], *, expected_source_commit: str
) -> None:
    if gate.get("schema") != HOST_GATE_SCHEMA:
        raise ValueError("invalid controlled-timing host-gate schema")
    required_true = (
        "approved",
        "stable_power",
        "output_directory_exclusive",
    )
    for field in required_true:
        if gate.get(field) is not True:
            raise ValueError(f"host gate requires {field}=true")
    if gate.get("unrelated_intensive_processes") is not False:
        raise ValueError("host gate requires unrelated_intensive_processes=false")
    if gate.get("thermal_throttling") is not False:
        raise ValueError("host gate requires thermal_throttling=false")
    free_memory = float(gate.get("free_memory_gib", -1.0))
    required_memory = float(gate.get("required_free_memory_gib", 0.0))
    if required_memory <= 0.0 or free_memory < required_memory:
        raise ValueError("host gate has insufficient free memory")
    route = gate.get("route_validation", {})
    expected_route = {
        "validated": True,
        "package_version": EXPECTED_VERSION,
        "source_commit": expected_source_commit,
        "wheel_sha256": EXPECTED_WHEEL_SHA256,
        "convolution_backend": "native",
        "detector_method": "detect_batch",
        "detector_batch_size": 2,
        "patch_provider_max_workers": 4,
        "numpy_fallback_permitted": False,
        "explicit_validity_masks": True,
    }
    if route != expected_route:
        raise ValueError("host gate route-validation identity does not match protocol")
    system = gate.get("system")
    required_system = {
        "cpu_model",
        "physical_cpu_count",
        "logical_cpu_count",
        "ram_gib",
        "os",
        "python",
        "numpy",
        "opencv",
        "power_source",
        "power_mode",
        "thermal_state",
        "thread_environment",
    }
    if not isinstance(system, dict) or not required_system.issubset(system):
        raise ValueError("host gate is missing required system metadata")


def validate_selection(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    inputs: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> None:
    if manifest.get("schema") != SELECTION_MANIFEST_SCHEMA:
        raise ValueError("invalid timing-selection manifest schema")
    if manifest.get("selection", {}).get("rows") != len(rows):
        raise ValueError("timing-selection row count differs from manifest")
    if len(rows) != len(DATASETS) * len(OVERLAP_BINS) * 5:
        raise ValueError("frozen timing selection must contain exactly 75 rows")
    seen: set[tuple[str, str]] = set()
    input_index = {(row["dataset_id"], row["pair_id"]): row for row in inputs}
    evaluation_index = {
        (row["dataset_id"], row["pair_id"]): row for row in evaluations
    }
    if len(input_index) != len(inputs) or len(evaluation_index) != len(evaluations):
        raise ValueError("input or evaluation population contains duplicate identity")
    expected_fields = {
        "schema",
        "dataset_id",
        "pair_id",
        "independence_component_id",
        "overlap_bin",
        "overlap_lower",
        "overlap_upper",
        "registered_cloud_overlap_min",
        "selection_key",
        "selection_uses_algorithm_outcome",
    }
    for row in rows:
        identity = (str(row["dataset_id"]), str(row["pair_id"]))
        if set(row) != expected_fields:
            raise ValueError(f"unexpected timing-selection field for {identity}")
        if row.get("schema") != SELECTION_SCHEMA:
            raise ValueError(f"invalid selection schema for {identity}")
        if identity in seen:
            raise ValueError(f"duplicate timing-selection identity: {identity}")
        seen.add(identity)
        if row.get("selection_uses_algorithm_outcome") is not False:
            raise ValueError(f"outcome-blind marker is absent for {identity}")
        if identity not in input_index or identity not in evaluation_index:
            raise ValueError(f"selected pair is absent from frozen inputs: {identity}")
        overlap = float(row["registered_cloud_overlap_min"])
        if not 0.0 <= overlap <= 1.0:
            raise ValueError(f"invalid overlap for {identity}: {overlap}")
    if set(input_index) != set(evaluation_index):
        raise ValueError("input and evaluation populations differ")
    cell_counts = Counter(
        (str(row["dataset_id"]), str(row["overlap_bin"])) for row in rows
    )
    expected_cells = {
        (dataset, overlap_bin): 5
        for dataset in DATASETS
        for overlap_bin in OVERLAP_BINS
    }
    if cell_counts != expected_cells:
        raise ValueError("timing selection must contain five rows in every cell")


def _request_stop(_signum: int, _frame: Any) -> None:
    global _stop_requested
    _stop_requested = True


def _status(
    *,
    state: str,
    args: argparse.Namespace,
    completed: int,
    total: int,
    current: dict[str, Any] | None,
    repetition: int | None,
    host_gate_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": STATUS_SCHEMA,
        "state": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "completed_timed_observations": completed,
        "total_timed_observations": total,
        "current_repetition": repetition,
        "current": (
            {
                "dataset_id": current["dataset_id"],
                "pair_id": current["pair_id"],
            }
            if current is not None
            else None
        ),
        "selection": {
            "path": str(args.selection.resolve()),
            "sha256": _sha256(args.selection),
        },
        "selection_manifest": {
            "path": str(args.selection_manifest.resolve()),
            "sha256": _sha256(args.selection_manifest),
        },
        "host_gate": {
            "path": str(args.host_gate.resolve()),
            "sha256": host_gate_sha256,
        },
        "runner": {
            "path": str(args.runner.resolve()),
            "sha256": _sha256(args.runner),
        },
        "python": str(args.python.absolute()),
        "expected_package_version": EXPECTED_VERSION,
        "expected_source_commit": args.expected_source_commit,
        "repetitions": args.repetitions,
        "height": args.height,
        "process_lifecycle": "fresh process per pair",
    }


def _run_pair(
    *,
    args: argparse.Namespace,
    row: dict[str, Any],
    output: Path,
    failure: Path,
) -> bool:
    command = [
        str(args.python),
        str(args.runner),
        "--inputs",
        str(args.inputs),
        "--evaluation",
        str(args.evaluation),
        "--pair-id",
        row["pair_id"],
        "--height",
        str(args.height),
        "--expected-source-commit",
        args.expected_source_commit,
        "--forbidden-checkout",
        str(args.forbidden_checkout),
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        start_new_session=True,
        cwd=args.runner.parent,
    )
    result = _load_valid_result(
        output,
        row,
        expected_package_version=EXPECTED_VERSION,
        expected_source_commit=args.expected_source_commit,
    )
    if completed.returncode == 0 and result is not None:
        return True
    _atomic_json(
        failure,
        {
            "schema": "panorai-controlled-timing-failure/v1",
            "dataset_id": row["dataset_id"],
            "pair_id": row["pair_id"],
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "command": command,
        },
    )
    return False


def run(args: argparse.Namespace) -> int:
    if args.output_dir.exists():
        raise FileExistsError(
            "controlled timing output must not exist; preserve interrupted runs"
        )
    selection = _read_jsonl(args.selection)
    selection_manifest = _read_json(args.selection_manifest)
    if _sha256(args.selection) != selection_manifest.get("selection", {}).get(
        "sha256"
    ):
        raise ValueError("timing-selection hash differs from manifest")
    inputs = _read_jsonl(args.inputs)
    evaluations = _read_jsonl(args.evaluation)
    validate_selection(selection, selection_manifest, inputs, evaluations)
    host_gate = _read_json(args.host_gate)
    validate_host_gate(host_gate, expected_source_commit=args.expected_source_commit)
    host_gate_sha256 = _sha256(args.host_gate)
    args.output_dir.mkdir(parents=True)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    status_path = args.output_dir / "status.json"
    total = len(selection) * args.repetitions
    completed = 0
    _atomic_json(
        status_path,
        _status(
            state="warming-up",
            args=args,
            completed=completed,
            total=total,
            current=None,
            repetition=None,
            host_gate_sha256=host_gate_sha256,
        ),
    )
    for dataset in DATASETS:
        row = next((item for item in selection if item["dataset_id"] == dataset), None)
        if row is None:
            raise ValueError(f"timing selection has no warm-up pair for {dataset}")
        output = args.output_dir / "warmup" / f"{_slug(row)}.json"
        if not _run_pair(
            args=args,
            row=row,
            output=output,
            failure=args.output_dir / "failures" / f"warmup-{_slug(row)}.json",
        ):
            _atomic_json(
                status_path,
                _status(
                    state="failed",
                    args=args,
                    completed=completed,
                    total=total,
                    current=row,
                    repetition=None,
                    host_gate_sha256=host_gate_sha256,
                ),
            )
            return 1
    for repetition in range(1, args.repetitions + 1):
        if _stop_requested:
            break
        validate_host_gate(
            _read_json(args.host_gate),
            expected_source_commit=args.expected_source_commit,
        )
        if _sha256(args.host_gate) != host_gate_sha256:
            raise ValueError("host gate changed after benchmark start")
        for row in deterministic_order(selection, repetition):
            if _stop_requested:
                break
            _atomic_json(
                status_path,
                _status(
                    state="running",
                    args=args,
                    completed=completed,
                    total=total,
                    current=row,
                    repetition=repetition,
                    host_gate_sha256=host_gate_sha256,
                ),
            )
            output = (
                args.output_dir
                / "results"
                / f"repetition-{repetition}"
                / f"{_slug(row)}.json"
            )
            failure = (
                args.output_dir
                / "failures"
                / f"repetition-{repetition}-{_slug(row)}.json"
            )
            if not _run_pair(args=args, row=row, output=output, failure=failure):
                _atomic_json(
                    status_path,
                    _status(
                        state="failed",
                        args=args,
                        completed=completed,
                        total=total,
                        current=row,
                        repetition=repetition,
                        host_gate_sha256=host_gate_sha256,
                    ),
                )
                return 1
            completed += 1
    state = "complete" if completed == total else "interrupted"
    _atomic_json(
        status_path,
        _status(
            state=state,
            args=args,
            completed=completed,
            total=total,
            current=None,
            repetition=None,
            host_gate_sha256=host_gate_sha256,
        ),
    )
    print(json.dumps(_read_json(status_path), indent=2, sort_keys=True))
    return 0 if state == "complete" else 130


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--host-gate", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--forbidden-checkout", type=Path, required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    return args


def main() -> int:
    return run(_parser())


if __name__ == "__main__":
    raise SystemExit(main())
