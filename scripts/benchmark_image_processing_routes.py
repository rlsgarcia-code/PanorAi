#!/usr/bin/env python3
"""Compare direct spherical filtering with project/filter/reconstruct.

Each route runs in a fresh process.  Timings include the complete operation
named by the route, while image loading and the rotation-quality probe are kept
outside the timed region.  The quality probe measures approximate equivariance
of an isotropic Gaussian under a fixed yaw rotation; it does not treat either
route as the other's numerical oracle.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = REPOSITORY_ROOT / "docs/_static/tutorials/nature-reserve-forest-erp.jpg"
ROUTES = ("spherical", "multiface")


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "raw_seconds": values,
        "median_seconds": statistics.median(values),
        "p95_seconds": _p95(values),
    }


def _load_image(path: Path, height: int, width: int):
    import numpy as np
    from PIL import Image

    with Image.open(path) as opened:
        resized = opened.convert("RGB").resize(
            (width, height), Image.Resampling.LANCZOS
        )
        return np.asarray(resized, dtype=np.float32) / np.float32(255.0)


def _yaw_rotation(degrees: float):
    import numpy as np

    angle = np.deg2rad(degrees)
    return np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float64,
    )


def _quality(reference, rotated_back) -> dict[str, float]:
    import numpy as np

    absolute = np.abs(reference.astype(np.float64) - rotated_back.astype(np.float64))
    return {
        "rotation_equivariance_mae": float(np.mean(absolute)),
        "rotation_equivariance_p95_abs": float(np.quantile(absolute, 0.95)),
        "rotation_equivariance_max_abs": float(np.max(absolute)),
    }


def _worker(args: argparse.Namespace) -> dict[str, Any]:
    import cv2
    import numpy as np
    import panorai
    from panorai.data import EquirectangularImage
    from panorai.image_processing import spherical_gaussian_blur, spherical_rotate

    origin = Path(panorai.__file__).resolve()
    if args.require_installed and origin.is_relative_to(REPOSITORY_ROOT):
        raise RuntimeError(f"panorai resolved inside the checkout: {origin}")

    image_path = Path(args.image).resolve()
    image = _load_image(image_path, args.height, args.width)

    def apply(source):
        if args.route == "spherical":
            return spherical_gaussian_blur(
                source,
                ksize=args.ksize,
                sigma=args.sigma,
                backend=args.backend,
            )
        panorama = EquirectangularImage(source)
        result = panorama.process_views(
            lambda face: cv2.GaussianBlur(
                face,
                (args.ksize, args.ksize),
                args.sigma,
                borderType=cv2.BORDER_REFLECT_101,
            ),
            layout=args.layout,
            size=args.face_size,
            fov=args.fov,
            blend="gaussian",
        )
        if not bool(np.all(result.support_mask)):
            raise RuntimeError("multiface route did not cover the complete ERP")
        return result.image

    started = time.perf_counter()
    first_output = apply(image)
    first_call = time.perf_counter() - started
    samples = []
    output = first_output
    for _ in range(args.repetitions):
        started = time.perf_counter()
        output = apply(image)
        samples.append(time.perf_counter() - started)

    rotation = _yaw_rotation(args.yaw_deg)
    rotated_input = spherical_rotate(image, rotation)
    rotated_output = apply(rotated_input)
    rotated_back = spherical_rotate(rotated_output, rotation.T)
    cycle_back = spherical_rotate(rotated_input, rotation.T)

    return {
        "route": args.route,
        "timed_operation": (
            "direct spherical Gaussian over the ERP"
            if args.route == "spherical"
            else "ERP to faces, planar OpenCV Gaussian, backprojection, Gaussian blending"
        ),
        "first_call_seconds": first_call,
        "reused": _summary(samples),
        "quality": _quality(output, rotated_back),
        "rotation_resampling_floor": _quality(image, cycle_back),
        "output": {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "finite": bool(np.isfinite(output).all()),
            "mean": float(np.mean(output, dtype=np.float64)),
            "std": float(np.std(output, dtype=np.float64)),
        },
        "panorai_origin": str(origin),
        "panorai_version": metadata.version("panorai"),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
    }


def _run_fresh(args: argparse.Namespace, route: str) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--route",
        route,
        "--image",
        str(Path(args.image).resolve()),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--ksize",
        str(args.ksize),
        "--sigma",
        str(args.sigma),
        "--backend",
        args.backend,
        "--layout",
        args.layout,
        "--face-size",
        str(args.face_size),
        "--fov",
        str(args.fov),
        "--yaw-deg",
        str(args.yaw_deg),
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
    with tempfile.TemporaryDirectory(prefix="panorai-processing-route-") as cwd:
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
    parser.add_argument("--route", choices=ROUTES, default="spherical")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--ksize", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=1.2)
    parser.add_argument(
        "--backend", choices=("auto", "native", "numpy"), default="auto"
    )
    parser.add_argument("--layout", choices=("cube", "icosahedron"), default="cube")
    parser.add_argument("--face-size", type=int, default=256)
    parser.add_argument("--fov", type=float, default=95.0)
    parser.add_argument("--yaw-deg", type=float, default=37.0)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--require-installed", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.worker:
        print(json.dumps(_worker(args), sort_keys=True))
        return 0
    image_path = Path(args.image).resolve()
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    routes = {route: _run_fresh(args, route) for route in ROUTES}
    report = {
        "schema": "panorai-image-processing-route-benchmark/v1",
        "interpretation": (
            "The routes implement similar Gaussian smoothing with matched local raster "
            "scales, not bit-identical operators. Rotation equivariance is a geometric "
            "quality probe; neither route is used as the other's oracle."
        ),
        "image": {"path": str(image_path), "sha256": image_sha256},
        "configuration": {
            "erp_shape_hw": [args.height, args.width],
            "kernel_size": args.ksize,
            "sigma_pixels": args.sigma,
            "spherical_backend": args.backend,
            "multiface_layout": args.layout,
            "face_shape_hw": [args.face_size, args.face_size],
            "fov_deg": args.fov,
            "yaw_probe_deg": args.yaw_deg,
            "repetitions": args.repetitions,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
        },
        "routes": routes,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
