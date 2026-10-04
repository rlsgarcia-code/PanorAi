#!/usr/bin/env python3
"""Install one exact wheel into a temporary target and run a consumer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--consumer", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--extra-site-packages",
        action="append",
        default=[],
        type=Path,
        help="optional external dependency directory added after the wheel target",
    )
    parser.add_argument("consumer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    wheel = args.wheel.resolve()
    consumer = args.consumer.resolve()
    source_root = args.source_root.resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        parser.error(f"--wheel must identify an existing .whl file: {wheel}")
    if not consumer.is_file():
        parser.error(f"--consumer must identify an existing file: {consumer}")
    extra_sites = [path.resolve() for path in args.extra_site_packages]
    for path in extra_sites:
        if not path.is_dir():
            parser.error(f"--extra-site-packages must identify a directory: {path}")

    forwarded = list(args.consumer_args)
    if forwarded[:1] == ["--"]:
        forwarded.pop(0)

    with tempfile.TemporaryDirectory(prefix="panorai-consumer-") as directory:
        root = Path(directory)
        target = root / "site-packages"
        run_directory = root / "run"
        run_directory.mkdir()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--target",
                str(target),
                str(wheel),
            ],
            check=True,
        )
        environment = os.environ.copy()
        environment.pop("PYTHONHOME", None)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(target), *(str(path) for path in extra_sites)]
        )
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [
                sys.executable,
                str(consumer),
                "--source-root",
                str(source_root),
                *forwarded,
            ],
            cwd=run_directory,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "consumer failed with exit code "
                f"{completed.returncode}\nstdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        output_lines = [line for line in completed.stdout.splitlines() if line]
        if not output_lines:
            raise RuntimeError("consumer produced no output")
        consumer_result = json.loads(output_lines[-1])
        if consumer_result.get("status") != "ok":
            raise RuntimeError(f"consumer did not report success: {consumer_result}")
        report = {
            "status": "ok",
            "wheel": wheel.name,
            "wheel_sha256": _sha256(wheel),
            "consumer": str(consumer),
            "extra_site_packages": [str(path) for path in extra_sites],
            "consumer_result": consumer_result,
        }
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
