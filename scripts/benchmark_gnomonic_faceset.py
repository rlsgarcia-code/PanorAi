#!/usr/bin/env python3
"""Reproducible benchmark for N-view GnomonicFaceSet reconstruction.

Each measured route runs in a fresh process. Results separate the first
selective call from plan reuse and retain every raw timing sample plus peak RSS.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "raw_seconds": values,
        "median_seconds": statistics.median(values),
        "p95_seconds": _p95(values),
    }


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _layout_arguments(layout: str, count: int) -> dict[str, int]:
    return {"count": count} if layout in {"fibonacci", "spiral"} else {}


def _full_workflow(views):
    from panorai.data._workflow import back_project_modality, blend_reprojected

    workflow = views._workflow
    values = []
    masks = []
    specs = []
    for face in views:
        value, _, valid = back_project_modality(
            face.image,
            face._workflow_metadata["image"]["validity"],
            face.spec,
            workflow["erp_shape"],
            kind="image",
            depth_policy=workflow["depth_policy"],
            min_valid_weight=workflow["min_valid_weight"],
            projector_template=workflow["projector"],
        )
        values.append(value)
        masks.append(valid)
        specs.append(face.spec)
    return blend_reprojected(values, masks, specs, workflow["erp_shape"], "average")[0]


def _full_legacy(views, shape):
    from panorai.blenders.average import AverageBlender

    projected = [face.to_equirectangular(shape, return_mask=True) for face in views]
    return views.blend_channels(
        [item[0] for item in projected],
        False,
        AverageBlender(),
        masks=[item[1] for item in projected],
    )


def _worker(args) -> dict[str, Any]:
    import numpy as np
    import panorai as pa

    origin = Path(pa.__file__).resolve()
    if args.require_installed and origin.is_relative_to(REPOSITORY_ROOT):
        raise RuntimeError(f"panorai resolved inside the checkout: {origin}")
    rng = np.random.default_rng(908)
    source = rng.random((args.height, args.width, 3), dtype=np.float32)
    layout_kwargs = _layout_arguments(args.layout, args.count)
    panorama = pa.EquirectangularImage(source)
    if args.api == "workflow":
        views = panorama.views(
            args.layout,
            size=(args.face_height, args.face_width),
            fov=(args.hfov, args.vfov),
            **layout_kwargs,
        )
        operation = (
            _full_workflow
            if args.route == "full"
            else lambda item: item.reconstruct(modalities="image", blend="average")
        )
    else:
        sampler_kwargs = (
            {"n_points": args.count} if args.layout in {"fibonacci", "spiral"} else {}
        )
        panorama.attach_sampler(args.layout, **sampler_kwargs)
        views = panorama.to_gnomonic_face_set(fov=args.hfov)
        operation = (
            (lambda item: _full_legacy(item, (args.height, args.width)))
            if args.route == "full"
            else lambda item: item.to_equirectangular(
                (args.height, args.width), preserve_dtype=False
            )
        )

    started = time.perf_counter()
    operation(views)
    first = time.perf_counter() - started
    samples = []
    for _ in range(args.repetitions):
        started = time.perf_counter()
        operation(views)
        samples.append(time.perf_counter() - started)
    return {
        "api": args.api,
        "route": args.route,
        "layout": args.layout,
        "view_count": len(views),
        "erp_shape_hw": [args.height, args.width],
        "face_shape_hw": [args.face_height, args.face_width],
        "first_call_seconds": first,
        "reused": _summary(samples),
        "peak_rss_bytes": _rss_bytes(),
        "panorai_origin": str(origin),
        "panorai_version": metadata.version("panorai"),
    }


def _run_fresh(args, route: str) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--api",
        args.api,
        "--route",
        route,
        "--layout",
        args.layout,
        "--count",
        str(args.count),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--face-height",
        str(args.face_height),
        "--face-width",
        str(args.face_width),
        "--hfov",
        str(args.hfov),
        "--vfov",
        str(args.vfov),
        "--repetitions",
        str(args.repetitions),
    ]
    if args.require_installed:
        command.append("--require-installed")
    environment = os.environ.copy()
    if not args.require_installed:
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(REPOSITORY_ROOT)
            if not existing
            else f"{REPOSITORY_ROOT}{os.pathsep}{existing}"
        )
    with tempfile.TemporaryDirectory(prefix="panorai-gnomonic-benchmark-") as cwd:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    return json.loads(completed.stdout)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--api", choices=("workflow", "legacy"), default="workflow")
    parser.add_argument("--route", choices=("selective", "full"), default="selective")
    parser.add_argument(
        "--layout",
        choices=("cube", "icosahedron", "fibonacci", "spiral"),
        default="fibonacci",
    )
    parser.add_argument("--count", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--face-height", type=int, default=256)
    parser.add_argument("--face-width", type=int, default=256)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--vfov", type=float, default=90.0)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--require-installed", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if (
        args.repetitions <= 0
        or min(args.height, args.width, args.face_height, args.face_width) <= 0
    ):
        raise SystemExit("shapes and repetitions must be positive")
    if args.worker:
        print(json.dumps(_worker(args), sort_keys=True))
        return 0
    full = _run_fresh(args, "full")
    selective = _run_fresh(args, "selective")
    report = {
        "schema": "panorai-gnomonic-faceset-benchmark/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "full": full,
        "selective": selective,
        "speedup_first": full["first_call_seconds"] / selective["first_call_seconds"],
        "speedup_reused_median": full["reused"]["median_seconds"]
        / selective["reused"]["median_seconds"],
        "peak_rss_reduction": 1.0
        - selective["peak_rss_bytes"] / full["peak_rss_bytes"],
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
