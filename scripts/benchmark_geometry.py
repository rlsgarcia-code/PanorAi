#!/usr/bin/env python3
"""Reproducible installed-wheel benchmark for stable projection kernels.

The coordinator starts one fresh worker process per measurement.  This keeps
first-call timing and peak RSS attributable to a single library/operation and
prevents a checkout import from silently replacing the installed wheel.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
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
from typing import Any, Callable, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESOLUTIONS = ((512, 1024, 256), (1024, 2048, 512), (2048, 4096, 1024))
OPERATIONS = ("erp_to_gnomonic", "gnomonic_to_erp", "erp_to_cubemap", "cubemap_to_erp")
INTERPOLATIONS = ("nearest", "bilinear")
PINNED_COMPETITORS = {"py360convert": "1.0.4", "pyequilib": "0.6.0"}


@dataclass(frozen=True)
class Case:
    library: str
    backend: str
    batch: int
    operation: str
    interpolation: str
    erp_height: int
    erp_width: int
    face_size: int
    device: str = "cpu"

    @property
    def case_id(self) -> str:
        return (
            f"{self.library}-{self.backend}-{self.device}-b{self.batch}-"
            f"{self.operation}-{self.interpolation}-"
            f"{self.erp_height}x{self.erp_width}-f{self.face_size}"
        )


def nearest_rank_percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("at least one sample is required")
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def summarize_seconds(values: Iterable[float]) -> dict[str, Any]:
    raw = [float(value) for value in values]
    if not raw:
        raise ValueError("at least one sample is required")
    return {
        "count": len(raw),
        "raw": raw,
        "minimum": min(raw),
        "median": statistics.median(raw),
        "p95_nearest_rank": nearest_rank_percentile(raw, 0.95),
        "maximum": max(raw),
    }


def parse_resolution(value: str) -> tuple[int, int, int]:
    try:
        erp, face = value.lower().split(":", 1)
        height, width = erp.split("x", 1)
        result = (int(height), int(width), int(face))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("resolution must be HxW:FACE") from exc
    if any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("resolution values must be positive")
    return result


def build_cases(
    resolutions: Iterable[tuple[int, int, int]],
    libraries: Iterable[str],
    torch_device: str,
) -> list[Case]:
    """Return the explicit support matrix without inventing unsupported cases."""

    selected = set(libraries)
    cases: list[Case] = []
    for height, width, face in resolutions:
        for interpolation in INTERPOLATIONS:
            for operation in OPERATIONS:
                if "panorai" in selected:
                    cases.append(
                        Case(
                            "panorai",
                            "numpy",
                            1,
                            operation,
                            interpolation,
                            height,
                            width,
                            face,
                        )
                    )
                    for batch in (1, 8):
                        cases.append(
                            Case(
                                "panorai",
                                "torch",
                                batch,
                                operation,
                                interpolation,
                                height,
                                width,
                                face,
                                torch_device,
                            )
                        )
                if "py360convert" in selected and operation != "gnomonic_to_erp":
                    cases.append(
                        Case(
                            "py360convert",
                            "numpy",
                            1,
                            operation,
                            interpolation,
                            height,
                            width,
                            face,
                        )
                    )
                if "pyequilib" in selected:
                    for backend in ("numpy", "torch"):
                        for batch in (1, 8):
                            cases.append(
                                Case(
                                    "pyequilib",
                                    backend,
                                    batch,
                                    operation,
                                    interpolation,
                                    height,
                                    width,
                                    face,
                                    torch_device if backend == "torch" else "cpu",
                                )
                            )
    return cases


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _sync(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _sync(item)
        return
    if hasattr(value, "support_mask"):
        _sync(value.data)
        return
    module = type(value).__module__.split(".", 1)[0]
    if module != "torch":
        return
    import torch

    if value.device.type == "cuda":
        torch.cuda.synchronize(value.device)
    elif value.device.type == "mps":
        torch.mps.synchronize()


def _gradient_hwc(height: int, width: int):
    import numpy as np

    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    return np.stack(
        (
            np.broadcast_to(x, (height, width)),
            np.broadcast_to(y, (height, width)),
            (x + y) * 0.5,
        ),
        axis=-1,
    )


def _panorai_operation(case: Case) -> tuple[Callable[[], Any], tuple[int, ...]]:
    import numpy as np
    from panorai.geometry import (
        CUBE_FACE_ORDER,
        GnomonicSpec,
        cubemap_to_equirectangular,
        equirectangular_to_cubemap,
        equirectangular_to_gnomonic,
        gnomonic_to_equirectangular,
    )

    height, width, face = case.erp_height, case.erp_width, case.face_size
    spec = GnomonicSpec(
        center_lat_deg=11.0,
        center_lon_deg=-23.0,
        hfov_deg=90.0,
        vfov_deg=90.0,
        roll_deg=7.0,
        output_shape_hw=(face, face),
    )
    erp: Any = _gradient_hwc(height, width)
    perspective: Any = _gradient_hwc(face, face)
    cube: Any = {name: _gradient_hwc(face, face) for name in CUBE_FACE_ORDER}
    if case.backend == "torch":
        import torch

        erp = torch.from_numpy(np.moveaxis(erp, -1, 0)).to(case.device)
        perspective = torch.from_numpy(np.moveaxis(perspective, -1, 0)).to(case.device)
        cube = {
            name: torch.from_numpy(np.moveaxis(value, -1, 0)).to(case.device)
            for name, value in cube.items()
        }
        erp = erp.unsqueeze(0).repeat(case.batch, 1, 1, 1)
        perspective = perspective.unsqueeze(0).repeat(case.batch, 1, 1, 1)
        cube = {
            name: value.unsqueeze(0).repeat(case.batch, 1, 1, 1)
            for name, value in cube.items()
        }

    if case.operation == "erp_to_gnomonic":
        return (
            lambda: (
                equirectangular_to_gnomonic(
                    erp, spec, interpolation=case.interpolation
                ).data
            ),
            (case.batch, 3, face, face) if case.backend == "torch" else (face, face, 3),
        )
    if case.operation == "gnomonic_to_erp":
        return (
            lambda: gnomonic_to_equirectangular(
                perspective, spec, (height, width), interpolation=case.interpolation
            ),
            (case.batch, 3, height, width)
            if case.backend == "torch"
            else (height, width, 3),
        )
    if case.operation == "erp_to_cubemap":
        return (
            lambda: equirectangular_to_cubemap(
                erp, face, interpolation=case.interpolation
            ),
            (6,),
        )
    return (
        lambda: (
            cubemap_to_equirectangular(
                cube, (height, width), interpolation=case.interpolation
            ).data
        ),
        (case.batch, 3, height, width)
        if case.backend == "torch"
        else (height, width, 3),
    )


def _py360convert_operation(case: Case) -> tuple[Callable[[], Any], tuple[int, ...]]:
    import py360convert

    height, width, face = case.erp_height, case.erp_width, case.face_size
    erp = _gradient_hwc(height, width)
    cube = py360convert.e2c(
        erp, face_w=face, mode=case.interpolation, cube_format="dict"
    )
    if case.operation == "erp_to_gnomonic":
        return (
            lambda: py360convert.e2p(
                erp,
                (90.0, 90.0),
                -23.0,
                11.0,
                (face, face),
                in_rot_deg=7.0,
                mode=case.interpolation,
            ),
            (face, face, 3),
        )
    if case.operation == "erp_to_cubemap":
        return (
            lambda: py360convert.e2c(
                erp, face_w=face, mode=case.interpolation, cube_format="dict"
            ),
            (6,),
        )
    if case.operation == "cubemap_to_erp":
        return (
            lambda: py360convert.c2e(
                cube, height, width, mode=case.interpolation, cube_format="dict"
            ),
            (height, width, 3),
        )
    raise RuntimeError("py360convert does not expose perspective-to-ERP")


def _pyequilib_operation(case: Case) -> tuple[Callable[[], Any], tuple[int, ...]]:
    import equilib
    import numpy as np

    height, width, face = case.erp_height, case.erp_width, case.face_size
    rotation = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
    rotations: Any = [rotation.copy() for _ in range(case.batch)]
    erp: Any = np.moveaxis(_gradient_hwc(height, width), -1, 0)
    perspective: Any = np.moveaxis(_gradient_hwc(face, face), -1, 0)
    erp = np.repeat(erp[None], case.batch, axis=0)
    perspective = np.repeat(perspective[None], case.batch, axis=0)
    if case.backend == "torch":
        import torch

        erp = torch.from_numpy(erp).to(case.device)
        perspective = torch.from_numpy(perspective).to(case.device)
    cube = equilib.equi2cube(erp, rotations, face, "horizon", mode=case.interpolation)
    batch_prefix = (case.batch,)
    if case.operation == "erp_to_gnomonic":
        return (
            lambda: equilib.equi2pers(
                erp, rotations, face, face, 90.0, mode=case.interpolation
            ),
            batch_prefix + (3, face, face),
        )
    if case.operation == "gnomonic_to_erp":
        return (
            lambda: equilib.pers2equi(
                perspective, rotations, height, width, 90.0, mode=case.interpolation
            ),
            batch_prefix + (3, height, width),
        )
    if case.operation == "erp_to_cubemap":
        return (
            lambda: equilib.equi2cube(
                erp, rotations, face, "horizon", mode=case.interpolation
            ),
            batch_prefix + (3, face, 6 * face),
        )
    return (
        lambda: equilib.cube2equi(
            cube, "horizon", height, width, mode=case.interpolation
        ),
        (() if case.batch == 1 else batch_prefix) + (3, height, width),
    )


def _validate_output(
    case: Case, value: Any, expected_shape: tuple[int, ...]
) -> dict[str, Any]:
    import numpy as np

    if hasattr(value, "support_mask"):
        data = value.data
        actual_shape = tuple(data.shape)
        if actual_shape != expected_shape:
            raise AssertionError(
                f"unexpected shape {actual_shape}; expected {expected_shape}"
            )
        array = data.detach().cpu().numpy() if hasattr(data, "detach") else data
        support = value.support_mask
        support = (
            support.detach().cpu().numpy() if hasattr(support, "detach") else support
        )
        if array.ndim == support.ndim + 1:
            support = support[None] if case.backend == "torch" else support[..., None]
        elif array.ndim == support.ndim + 2:
            support = support[None, None]
        finite_on_support = bool(
            np.isfinite(array)[np.broadcast_to(support, array.shape)].all()
        )
        return {
            "shape": list(actual_shape),
            "support_shape": list(value.support_mask.shape),
            "finite_on_support": finite_on_support,
            "nonfinite_outside_support_allowed": True,
            "passed": finite_on_support,
        }
    if isinstance(value, dict):
        if len(value) != expected_shape[0]:
            raise AssertionError(f"unexpected face count: {len(value)}")
        arrays = [
            item.data if hasattr(item, "support_mask") else item
            for item in value.values()
        ]
        finite = all(
            bool(
                np.isfinite(
                    item.detach().cpu().numpy() if hasattr(item, "detach") else item
                ).all()
            )
            for item in arrays
        )
        return {"face_count": len(value), "finite": finite, "passed": finite}
    actual_shape = tuple(value.shape)
    if actual_shape != expected_shape:
        raise AssertionError(
            f"unexpected shape {actual_shape}; expected {expected_shape}"
        )
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else value
    finite = bool(np.isfinite(array).all())
    return {"shape": list(actual_shape), "finite": finite, "passed": finite}


def run_worker(case: Case, warmup: int, repetitions: int) -> dict[str, Any]:
    adapters = {
        "panorai": _panorai_operation,
        "py360convert": _py360convert_operation,
        "pyequilib": _pyequilib_operation,
    }
    rss_before = _rss_bytes()
    operation, expected_shape = adapters[case.library](case)
    started = time.perf_counter_ns()
    output = operation()
    _sync(output)
    cold_seconds = (time.perf_counter_ns() - started) / 1_000_000_000
    correctness = _validate_output(case, output, expected_shape)
    for _ in range(warmup):
        output = operation()
        _sync(output)
    raw = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        output = operation()
        _sync(output)
        raw.append((time.perf_counter_ns() - started) / 1_000_000_000)
    correctness = _validate_output(case, output, expected_shape)
    rss_peak = _rss_bytes()
    return {
        "case": asdict(case),
        "case_id": case.case_id,
        "cold_seconds": cold_seconds,
        "warm_seconds": summarize_seconds(raw),
        "rss_before_bytes": rss_before,
        "rss_peak_bytes": rss_peak,
        "rss_peak_delta_bytes": max(0, rss_peak - rss_before),
        "correctness_guard": correctness,
    }


def assert_installed_origin(source_root: Path) -> tuple[str, str]:
    import panorai

    origin = Path(panorai.__file__).resolve()
    try:
        origin.relative_to(source_root.resolve())
    except ValueError:
        pass
    else:
        raise RuntimeError(f"panorai resolved inside checkout: {origin}")
    return str(origin), metadata.version("panorai")


def _distribution_hash(distribution_name: str) -> str:
    distribution = metadata.distribution(distribution_name)
    digest = hashlib.sha256()
    for file in sorted(distribution.files or (), key=str):
        path = distribution.locate_file(file)
        if path.is_file() and (
            str(file).endswith(".py") or str(file).endswith((".so", ".pyd"))
        ):
            encoded = str(file).encode()
            digest.update(len(encoded).to_bytes(4, "big"))
            digest.update(encoded)
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _physical_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def _version_payload(libraries: set[str]) -> dict[str, str]:
    result = {"panorai": metadata.version("panorai")}
    if "py360convert" in libraries:
        result["py360convert"] = metadata.version("py360convert")
    if "pyequilib" in libraries:
        result["pyequilib"] = metadata.version("pyequilib")
    try:
        result["numpy"] = metadata.version("numpy")
    except metadata.PackageNotFoundError:
        pass
    try:
        result["torch"] = metadata.version("torch")
    except metadata.PackageNotFoundError:
        pass
    return result


def _run_case_subprocess(case: Case, args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-case",
        json.dumps(asdict(case), separators=(",", ":")),
        "--warmup",
        str(args.warmup),
        "--repetitions",
        str(args.repetitions),
    ]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONSAFEPATH"] = "1"
    completed = subprocess.run(
        command,
        cwd=tempfile.gettempdir(),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker failed for {case.case_id}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--resolution", type=parse_resolution, action="append")
    parser.add_argument(
        "--libraries",
        nargs="+",
        choices=("panorai", "py360convert", "pyequilib"),
        default=list(("panorai", "py360convert", "pyequilib")),
    )
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker-case", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.warmup < 0 or args.repetitions <= 0:
        raise SystemExit("warmup must be non-negative and repetitions positive")
    if args.worker_case:
        case = Case(**json.loads(args.worker_case))
        print(
            json.dumps(run_worker(case, args.warmup, args.repetitions), sort_keys=True)
        )
        return 0
    if args.output is None:
        raise SystemExit("--output is required")
    if args.output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing result: {args.output}")

    origin, panorai_version = assert_installed_origin(args.source_root)
    libraries = set(args.libraries)
    versions = _version_payload(libraries)
    for distribution, expected in PINNED_COMPETITORS.items():
        if distribution in libraries and versions.get(distribution) != expected:
            raise RuntimeError(
                f"{distribution} must be pinned to {expected}; found {versions.get(distribution)!r}"
            )
    resolutions = args.resolution or list(DEFAULT_RESOLUTIONS)
    cases = build_cases(resolutions, libraries, args.torch_device)
    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.case_id}", file=sys.stderr, flush=True)
        results.append(_run_case_subprocess(case, args))

    payload = {
        "schema": "panorai-geometry-benchmark/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "installed_artifact": {
            "origin": origin,
            "version": panorai_version,
            "distribution_sha256": _distribution_hash("panorai"),
            "outside_source_checkout": True,
        },
        "versions": versions,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "physical_memory_bytes": _physical_memory_bytes(),
            "torch_device": args.torch_device,
        },
        "configuration": {
            "resolutions": [list(item) for item in resolutions],
            "interpolations": list(INTERPOLATIONS),
            "operations": list(OPERATIONS),
            "warmup_after_cold": args.warmup,
            "warm_repetitions": args.repetitions,
            "fresh_process_per_case": True,
        },
        "comparability": {
            "panorai": {
                "class": "canonical",
                "notes": "geometry-v1 pixel-center lattice, frame, seams, support, and validity semantics",
            },
            "py360convert": {
                "class": "timing_only_different_conventions",
                "notes": "different frame/lattice and cubemap layout; no perspective-to-ERP public operation",
            },
            "pyequilib": {
                "class": "timing_only_different_conventions",
                "notes": "different rotation, frame/lattice, and cubemap layout conventions",
            },
            "numeric_equivalence_claimed": False,
        },
        "unsupported_matrix": [
            {
                "library": "panorai",
                "backend": "numpy",
                "batch": 8,
                "reason": "NumPy public image contract is HW/HWC",
            },
            {
                "library": "py360convert",
                "backend": "numpy",
                "batch": 8,
                "reason": "no public batch transform",
            },
            {
                "library": "py360convert",
                "operation": "gnomonic_to_erp",
                "reason": "no public perspective-to-ERP transform",
            },
            {
                "library": "py360convert",
                "backend": "torch",
                "reason": "NumPy-only library",
            },
        ],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(results)} cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
