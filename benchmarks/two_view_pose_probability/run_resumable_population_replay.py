#!/usr/bin/env python3
"""Run an interruptible, resumable exact-wheel replay over frozen pairs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
from typing import Any


RESULT_SCHEMA = "panorai-unified-optimized-pair/v2"
_stop_requested = False


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def _slug(row: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        f"{row['dataset_id']}\0{row['pair_id']}".encode()
    ).hexdigest()[:20]
    return f"{row['dataset_id']}-{digest}"


def _load_valid_result(
    path: Path,
    row: dict[str, Any],
    *,
    expected_package_version: str = "3.5.0",
    expected_source_commit: str | None = None,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("dataset_id") != row["dataset_id"]
        or result.get("pair_id") != row["pair_id"]
        or result.get("package", {}).get("version") != expected_package_version
        or result.get("native")
        != {
            "convolution_backend": "native",
            "native_filter_available": True,
            "native_pose_kernels_available": True,
            "numpy_fallback_permitted": False,
        }
    ):
        return None
    if (
        expected_source_commit is not None
        and result.get("package", {}).get("expected_source_commit")
        != expected_source_commit
    ):
        return None
    route = result.get("route", {}).get("route", {})
    if route.get("detector_method") != "detect_batch":
        return None
    if route.get("patch_provider_max_workers") != 4:
        return None
    if not isinstance(result.get("matching_diagnostics"), dict):
        return None
    pose = result.get("pose", {})
    if pose.get("returned"):
        quality = pose.get("quality_report")
        if not isinstance(quality, dict) or not isinstance(
            quality.get("translation_orientation"), dict
        ):
            return None
    return result


def _counts(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    expected_package_version: str = "3.5.0",
    expected_source_commit: str | None = None,
) -> dict[str, Any]:
    by_dataset: dict[str, dict[str, int]] = {}
    completed = 0
    for row in rows:
        values = by_dataset.setdefault(row["dataset_id"], {"total": 0, "completed": 0})
        values["total"] += 1
        path = output_dir / "results" / f"{_slug(row)}.json"
        if (
            _load_valid_result(
                path,
                row,
                expected_package_version=expected_package_version,
                expected_source_commit=expected_source_commit,
            )
            is not None
        ):
            values["completed"] += 1
            completed += 1
    return {
        "total": len(rows),
        "completed": completed,
        "remaining": len(rows) - completed,
        "by_dataset": by_dataset,
    }


def _status(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    counts: dict[str, Any],
    state: str,
    current: dict[str, Any] | None,
    attempts_this_run: int,
    failures_this_run: int,
) -> dict[str, Any]:
    return {
        "schema": "panorai-resumable-population-replay-status/v1",
        "state": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "counts": counts,
        "current": (
            {"dataset_id": current["dataset_id"], "pair_id": current["pair_id"]}
            if current is not None
            else None
        ),
        "attempts_this_run": attempts_this_run,
        "failures_this_run": failures_this_run,
        "inputs": {"path": str(args.inputs.resolve()), "sha256": _sha256(args.inputs)},
        "evaluation": {
            "path": str(args.evaluation.resolve()),
            "sha256": _sha256(args.evaluation),
        },
        "runner": {"path": str(args.runner.resolve()), "sha256": _sha256(args.runner)},
        "python": {
            "requested_executable": str(args.python.absolute()),
            "resolved_binary": str(args.python.resolve()),
        },
        "expected_source_commit": args.expected_source_commit,
        "expected_package_version": args.expected_package_version,
        "height": args.height,
        "stop_file": str(args.stop_file.resolve()),
    }


def _request_stop(_signum: int, _frame: Any) -> None:
    global _stop_requested
    _stop_requested = True


def run(args: argparse.Namespace) -> int:
    rows = _read_jsonl(args.inputs)
    evaluations = _read_jsonl(args.evaluation)
    if [(row["dataset_id"], row["pair_id"]) for row in rows] != [
        (row["dataset_id"], row["pair_id"]) for row in evaluations
    ]:
        raise ValueError("input and evaluation rows are not identically ordered")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_dir / "replay.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another replay owns {lock_path}") from error
        lock.seek(0)
        lock.truncate()
        lock.write(f"pid={os.getpid()}\n")
        lock.flush()
        signal.signal(signal.SIGINT, _request_stop)
        signal.signal(signal.SIGTERM, _request_stop)
        attempts = 0
        failures = 0
        counts = _counts(
            rows,
            args.output_dir,
            expected_package_version=args.expected_package_version,
            expected_source_commit=args.expected_source_commit,
        )
        status_path = args.output_dir / "status.json"
        _atomic_json(
            status_path,
            _status(
                args=args,
                rows=rows,
                counts=counts,
                state="running",
                current=None,
                attempts_this_run=attempts,
                failures_this_run=failures,
            ),
        )
        for row in rows:
            result_path = args.output_dir / "results" / f"{_slug(row)}.json"
            if (
                _load_valid_result(
                    result_path,
                    row,
                    expected_package_version=args.expected_package_version,
                    expected_source_commit=args.expected_source_commit,
                )
                is not None
            ):
                continue
            if _stop_requested or args.stop_file.exists():
                break
            if args.max_pairs is not None and attempts >= args.max_pairs:
                break
            attempts += 1
            _atomic_json(
                status_path,
                _status(
                    args=args,
                    rows=rows,
                    counts=counts,
                    state="running",
                    current=row,
                    attempts_this_run=attempts,
                    failures_this_run=failures,
                ),
            )
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
                str(result_path),
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                start_new_session=True,
                cwd=args.runner.parent,
            )
            if (
                completed.returncode != 0
                or _load_valid_result(
                    result_path,
                    row,
                    expected_package_version=args.expected_package_version,
                    expected_source_commit=args.expected_source_commit,
                )
                is None
            ):
                failures += 1
                _atomic_json(
                    args.output_dir / "failures" / f"{_slug(row)}.json",
                    {
                        "schema": "panorai-population-replay-failure/v1",
                        "dataset_id": row["dataset_id"],
                        "pair_id": row["pair_id"],
                        "returncode": completed.returncode,
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                        "command": command,
                    },
                )
                if args.fail_fast:
                    break
            else:
                counts["completed"] += 1
                counts["remaining"] -= 1
                counts["by_dataset"][row["dataset_id"]]["completed"] += 1
                (args.output_dir / "failures" / f"{_slug(row)}.json").unlink(
                    missing_ok=True
                )
        final_counts = _counts(
            rows,
            args.output_dir,
            expected_package_version=args.expected_package_version,
            expected_source_commit=args.expected_source_commit,
        )
        if final_counts["remaining"] == 0:
            state = "complete"
        elif failures and args.fail_fast:
            state = "failed"
        else:
            state = "paused"
        _atomic_json(
            status_path,
            _status(
                args=args,
                rows=rows,
                counts=final_counts,
                state=state,
                current=None,
                attempts_this_run=attempts,
                failures_this_run=failures,
            ),
        )
        print(json.dumps(json.loads(status_path.read_text()), indent=2, sort_keys=True))
        return 0 if state in {"complete", "paused"} else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-package-version", default="3.5.0")
    parser.add_argument("--forbidden-checkout", type=Path, required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.stop_file is None:
        args.stop_file = args.output_dir / "STOP"
    if args.max_pairs is not None and args.max_pairs < 1:
        parser.error("--max-pairs must be positive")
    return args


def main() -> int:
    return run(_parser())


if __name__ == "__main__":
    raise SystemExit(main())
