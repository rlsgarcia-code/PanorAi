#!/usr/bin/env python3
"""Seal the exact installed-wheel environment used by a live population replay."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any


SCHEMA = "panorai-two-view-population-environment-seal/v1"
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _installed_tree(path: Path) -> dict[str, Any]:
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and "__pycache__" not in candidate.parts
        and candidate.suffix != ".pyc"
    )
    digest = hashlib.sha256()
    records = []
    for candidate in files:
        relative = candidate.relative_to(path).as_posix()
        file_hash = _sha256(candidate)
        size = candidate.stat().st_size
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(file_hash.encode())
        digest.update(b"\0")
        records.append({"path": relative, "sha256": file_hash, "size_bytes": size})
    return {
        "root": str(path.resolve()),
        "file_count": len(records),
        "total_bytes": sum(record["size_bytes"] for record in records),
        "tree_sha256": digest.hexdigest(),
        "files": records,
    }


def _probe(python: Path, cwd: Path) -> dict[str, Any]:
    program = (
        "import cv2, importlib.metadata as m, json, numpy, panorai, platform, sys; "
        "from panorai.image_processing import native_filter_available; "
        "from panorai.estimators import native_kernels_available; "
        "print(json.dumps({'version':m.version('panorai'),"
        "'package_file':panorai.__file__,'python':sys.version,"
        "'python_executable':sys.executable,'platform':platform.platform(),"
        "'numpy':numpy.__version__,'opencv':cv2.__version__,"
        "'opencv_threads':cv2.getNumThreads(),"
        "'native_filter_available':native_filter_available(),"
        "'native_pose_kernels_available':native_kernels_available()}))"
    )
    completed = subprocess.run(
        [str(python), "-c", program],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "installed-wheel probe failed: "
            f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )
    lines = [line for line in completed.stdout.splitlines() if line]
    if len(lines) != 1:
        raise RuntimeError(f"installed-wheel probe emitted unexpected output: {lines}")
    return json.loads(lines[0])


def _command_value(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _hardware_identity() -> dict[str, Any]:
    cpu_model = None
    physical_cpu_count = None
    memory_bytes = None
    if platform.system() == "Darwin":
        cpu_model = _command_value(["sysctl", "-n", "machdep.cpu.brand_string"])
        physical = _command_value(["sysctl", "-n", "hw.physicalcpu"])
        memory = _command_value(["sysctl", "-n", "hw.memsize"])
        physical_cpu_count = int(physical) if physical else None
        memory_bytes = int(memory) if memory else None
    elif Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.lower().startswith(("model name", "hardware")):
                cpu_model = line.partition(":")[2].strip() or None
                if cpu_model:
                    break
        memory = None
        if Path("/proc/meminfo").is_file():
            for line in Path("/proc/meminfo").read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                if line.startswith("MemTotal:"):
                    memory = line.split()[1]
                    break
        memory_bytes = int(memory) * 1024 if memory else None
    cpu_model = cpu_model or platform.processor() or platform.machine()
    return {
        "cpu_model": cpu_model,
        "physical_cpu_count": physical_cpu_count,
        "logical_cpu_count": os.cpu_count(),
        "memory_bytes": memory_bytes,
    }


def _validate_status(
    status: dict[str, Any],
    *,
    python: Path,
    runner: Path,
    expected_source_commit: str,
    expected_runner_sha256: str,
) -> None:
    if status.get("schema") != "panorai-resumable-population-replay-status/v1":
        raise ValueError("unsupported replay status schema")
    if status.get("state") not in {"running", "paused", "complete"}:
        raise ValueError("replay status is not sealable")
    if status.get("expected_source_commit") != expected_source_commit:
        raise ValueError("replay source commit differs")
    requested_python = Path(
        status.get("python", {}).get("requested_executable", "")
    ).absolute()
    if requested_python != python.absolute():
        raise ValueError("replay Python differs")
    resolved_binary = status.get("python", {}).get("resolved_binary")
    if resolved_binary and Path(resolved_binary).resolve() != python.resolve():
        raise ValueError("replay resolved Python binary differs")
    if Path(status.get("runner", {}).get("path", "")).resolve() != runner.resolve():
        raise ValueError("replay runner path differs")
    if status.get("runner", {}).get("sha256") != expected_runner_sha256:
        raise ValueError("replay runner hash differs")


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    wheel = args.wheel.resolve()
    runner = args.runner.resolve()
    # Preserve the venv entry point for execution. Resolving this symlink before
    # the probe would invoke the base interpreter and silently lose site-packages.
    python = args.python.absolute()
    probe_cwd = args.probe_cwd.resolve()
    status_path = args.status.resolve()
    for path in (wheel, runner, python, status_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not probe_cwd.is_dir():
        raise NotADirectoryError(probe_cwd)
    wheel_hash = _sha256(wheel)
    runner_hash = _sha256(runner)
    if wheel_hash != args.expected_wheel_sha256:
        raise ValueError("wheel hash differs")
    if runner_hash != args.expected_runner_sha256:
        raise ValueError("runner hash differs")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    _validate_status(
        status,
        python=python,
        runner=runner,
        expected_source_commit=args.expected_source_commit,
        expected_runner_sha256=args.expected_runner_sha256,
    )
    probe = _probe(python, probe_cwd)
    package_file = Path(probe["package_file"]).resolve()
    forbidden = args.forbidden_checkout.resolve()
    if package_file.is_relative_to(forbidden):
        raise ValueError("probe imported PanorAi from the forbidden checkout")
    if "site-packages" not in package_file.parts:
        raise ValueError("probe did not import an installed distribution")
    if probe["version"] != args.expected_version:
        raise ValueError("installed PanorAi version differs")
    if not probe["native_filter_available"] or not probe[
        "native_pose_kernels_available"
    ]:
        raise ValueError("installed native kernels are unavailable")
    package_tree = _installed_tree(package_file.parent)
    dist_info = package_file.parent.parent / f"panorai-{args.expected_version}.dist-info"
    record = dist_info / "RECORD"
    if not record.is_file():
        raise FileNotFoundError(record)
    result = {
        "schema": SCHEMA,
        "status": "sealed while replay process was active",
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "release": {
            "version": args.expected_version,
            "tag": args.release_tag,
            "tag_object": args.release_tag_object,
            "source_commit": args.expected_source_commit,
            "source_tree": args.expected_source_tree,
        },
        "wheel": {
            "path": str(wheel),
            "sha256": wheel_hash,
            "size_bytes": wheel.stat().st_size,
        },
        "runner": {
            "path": str(runner),
            "sha256": runner_hash,
        },
        "probe": {**probe, "cwd": str(probe_cwd)},
        "hardware": _hardware_identity(),
        "installed_package_tree": package_tree,
        "distribution_record": {
            "path": str(record.resolve()),
            "sha256": _sha256(record),
        },
        "replay_status": {
            "path": str(status_path),
            "sha256_at_seal": _sha256(status_path),
            "pid": status.get("pid"),
            "state": status.get("state"),
            "completed_at_seal": status.get("counts", {}).get("completed"),
            "total": status.get("counts", {}).get("total"),
            "failures_this_run": status.get("failures_this_run"),
            "runner": status.get("runner"),
            "python": status.get("python"),
        },
        "thread_environment": {
            name: os.environ.get(name) for name in THREAD_VARIABLES
        },
        "scope": (
            "Environment identity only; contains no aggregate pose accuracy and "
            "does not replace the final 2,385-result verification."
        ),
    }
    _atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--probe-cwd", type=Path, required=True)
    parser.add_argument("--forbidden-checkout", type=Path, required=True)
    parser.add_argument("--expected-version", default="3.5.0")
    parser.add_argument("--release-tag", default="v3.5.0")
    parser.add_argument("--release-tag-object", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--expected-wheel-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
