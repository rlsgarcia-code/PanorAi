#!/usr/bin/env python3
"""Measure cold PanorAi imports in fresh Python subprocesses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("panorai", "panorai.geometry")


CHILD = r"""
import importlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

target, source_root_text, require_installed_text = sys.argv[1:]
source_root = Path(source_root_text).resolve()
start = time.perf_counter()
module = importlib.import_module(target)
elapsed = time.perf_counter() - start
package = importlib.import_module("panorai")
origin = Path(package.__file__).resolve()
if require_installed_text == "true":
    try:
        origin.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise AssertionError(f"panorai resolved inside source checkout: {origin}")

raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_bytes = raw_rss if sys.platform == "darwin" else raw_rss * 1024
print(json.dumps({
    "elapsed_seconds": elapsed,
    "module_count": len(sys.modules),
    "panorai_modules": sorted(
        name for name in sys.modules
        if name == "panorai" or name.startswith("panorai.")
    ),
    "top_level_modules": sorted({name.split(".")[0] for name in sys.modules}),
    "rss_bytes": rss_bytes,
    "origin": str(origin),
    "version": package.__version__,
    "python": platform.python_version(),
    "platform": platform.platform(),
}))
"""


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _profile(
    target: str,
    repetitions: int,
    *,
    working_directory: Path,
    source_root: Path,
    require_installed: bool,
) -> dict:
    runs = []
    for _ in range(repetitions):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                CHILD,
                target,
                str(source_root),
                str(require_installed).lower(),
            ],
            cwd=working_directory,
            text=True,
            capture_output=True,
            check=True,
        )
        runs.append(json.loads(completed.stdout))

    elapsed = [run["elapsed_seconds"] for run in runs]
    rss = [run["rss_bytes"] for run in runs]
    representative = runs[-1]
    return {
        "target": target,
        "repetitions": repetitions,
        "cold_subprocess": True,
        "elapsed_seconds": {
            "raw": elapsed,
            "median": statistics.median(elapsed),
            "p95_nearest_rank": _percentile(elapsed, 0.95),
        },
        "rss_bytes": {
            "raw": rss,
            "median": statistics.median(rss),
            "maximum": max(rss),
        },
        "module_count": representative["module_count"],
        "panorai_modules": representative["panorai_modules"],
        "top_level_modules": representative["top_level_modules"],
        "origin": representative["origin"],
        "version": representative["version"],
        "python": representative["python"],
        "platform": representative["platform"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--target", action="append", choices=TARGETS)
    parser.add_argument("--working-directory", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--require-installed",
        action="store_true",
        help="fail if the measured package resolves inside --source-root",
    )
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    working_directory = args.working_directory.resolve()
    if not working_directory.is_dir():
        parser.error("--working-directory must name an existing directory")

    targets = args.target or list(TARGETS)
    report = {
        "schema": "panorai-import-profile-v1",
        "driver_python": platform.python_version(),
        "driver_platform": platform.platform(),
        "source_root": str(args.source_root.resolve()),
        "working_directory": str(working_directory),
        "require_installed": args.require_installed,
        "results": [
            _profile(
                target,
                args.repetitions,
                working_directory=working_directory,
                source_root=args.source_root,
                require_installed=args.require_installed,
            )
            for target in targets
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
