#!/usr/bin/env python3
"""Reproduce the public 1024x2048 sphere-native Metric3D preview."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from panorai.experimental.deep_learning import (  # noqa: E402
    load_spherical_metric3d_convnext_large_v1,
)

SOURCE_URL = (
    "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/poly_haven_studio_2k.hdr"
)
SOURCE_SHA256 = "65ffbd4ffdce0914d61e5c28e078a14491a973cdbebbd82fc3c47b30153955b5"
OUTPUT_SHA256 = "5770255ad4c072f951a2b7201b407b4e0000ce5642da3f819af7f228a6868b64"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-hdr", type=Path, required=True)
    parser.add_argument("--metric3d-cache", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT
        / "docs/_static/tutorials/metric3d-large-spherical-1024x2048.jpg",
    )
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    return parser.parse_args()


def _tone_map(path: Path) -> np.ndarray:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise ValueError(f"unexpected Poly Haven Studio 2K SHA-256: {actual}")
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.shape != (1024, 2048, 3):
        raise ValueError("expected a 1024x2048 three-channel HDR panorama")
    rgb = cv2.cvtColor(bgr.astype(np.float32), cv2.COLOR_BGR2RGB)
    exposure = float(np.percentile(rgb, 99.5))
    mapped = np.clip(rgb / max(exposure, np.finfo(np.float32).eps), 0.0, 1.0)
    return np.rint(np.power(mapped, 1.0 / 2.2) * 255.0).astype(np.uint8)


def _phase_modulation(values: np.ndarray, rows: slice) -> float:
    profile = values[rows].mean(axis=0)
    phases = np.asarray([profile[index::8].mean() for index in range(8)])
    return float(phases.std() / max(abs(float(profile.mean())), 1e-12))


def _render(rgb: np.ndarray, depth: np.ndarray, output: Path) -> None:
    low, high = np.quantile(depth[np.isfinite(depth)], (0.02, 0.98))
    normalized = np.clip((depth - low) / max(high - low, 1e-8), 0.0, 1.0)
    colored = cv2.cvtColor(
        cv2.applyColorMap(
            np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO
        ),
        cv2.COLOR_BGR2RGB,
    )
    overlay = np.rint(0.18 * rgb + 0.82 * colored).astype(np.uint8)
    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((24, 24, 580, 76), radius=10, fill=(15, 23, 42))
    draw.text(
        (42, 42),
        "Metric3D Large sphere-native - 1024 x 2048",
        fill="white",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=92, optimize=True, progressive=True)


def main() -> int:
    args = _arguments()
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    import torch

    rgb = _tone_map(args.source_hdr)
    loaded = load_spherical_metric3d_convnext_large_v1(
        accept_upstream_terms=True,
        cache_dir=args.metric3d_cache,
        device="cpu",
        max_sampled_elements=30_000_000,
        timeout_seconds=1200,
    )
    values = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1)[None]

    def infer() -> np.ndarray:
        with torch.inference_mode():
            result = loaded.model(values)[0, 0].detach().cpu().numpy()
        if result.shape != (1024, 2048) or not np.isfinite(result).all():
            raise RuntimeError("Metric3D did not preserve the 1024x2048 ERP lattice")
        return np.asarray(result, dtype=np.float32)

    infer()
    durations: list[float] = []
    depth = None
    for _ in range(args.repetitions):
        started = perf_counter()
        depth = infer()
        durations.append((perf_counter() - started) * 1000.0)
    assert depth is not None
    _render(rgb, depth, args.output)
    metadata = {
        "schema": "panorai-metric3d-high-resolution/v1",
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "source_license": "CC0",
        "checkpoint_sha256": loaded.assets.checkpoint.spec.sha256,
        "input_shape_hw": [1024, 2048],
        "output_shape_hw": list(depth.shape),
        "output_range_m": [float(depth.min()), float(depth.max())],
        "output_mean_m": float(depth.mean()),
        "durations_ms": durations,
        "median_ms": float(np.median(durations)),
        "p95_ms": float(np.quantile(durations, 0.95)),
        "max_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "period_8_modulation": {
            "north": _phase_modulation(depth, slice(0, 128)),
            "middle": _phase_modulation(depth, slice(384, 640)),
            "south": _phase_modulation(depth, slice(896, 1024)),
        },
        "prediction_sha256": hashlib.sha256(
            np.ascontiguousarray(depth).tobytes()
        ).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    if (
        args.output.resolve()
        == (
            REPOSITORY_ROOT
            / "docs/_static/tutorials/metric3d-large-spherical-1024x2048.jpg"
        ).resolve()
        and metadata["output_sha256"] != OUTPUT_SHA256
    ):
        raise RuntimeError("tracked preview differs from its pinned output")
    if args.metadata is not None:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
