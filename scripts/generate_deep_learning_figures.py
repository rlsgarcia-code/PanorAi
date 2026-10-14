#!/usr/bin/env python3
"""Generate public CNN-depth figures from pinned external models and a CC0 ERP.

The model source and checkpoints remain in caller-owned caches. The generated
JPEGs contain only model outputs over the tracked Poly Haven derivative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from panorai import EquirectangularImage  # noqa: E402
from panorai.experimental.deep_learning import (  # noqa: E402
    load_da3metric_large,
    load_spherical_metric3d_convnext_tiny_v1,
)
from panorai.experimental.deep_learning.depth import (  # noqa: E402
    _patch_metric3d_v1_device_assumptions,
    _upstream_import_path,
)
from panorai.geometry import (  # noqa: E402
    CUBE_FACE_ORDER,
    cubemap_to_equirectangular,
    equirectangular_to_cubemap,
)

INPUT_SHA256 = "5333c2cecc55468fcbc64a252f4bc097fab064cae70bead4f76d6c2e6101c524"
MEAN = np.asarray((123.675, 116.28, 103.53), dtype=np.float32)
STD = np.asarray((58.395, 57.12, 57.375), dtype=np.float32)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=REPOSITORY_ROOT
        / "docs/_static/tutorials/nature-reserve-forest-erp.jpg",
    )
    parser.add_argument("--metric3d-cache", type=Path, required=True)
    parser.add_argument("--da3-cache", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "docs/_static/tutorials",
    )
    parser.add_argument("--repetitions", type=int, default=5)
    return parser.parse_args()


def _label(panel: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((12, 12, 222, 44), radius=8, fill=(15, 23, 42))
    draw.text((24, 22), text, fill=(255, 255, 255))
    return panel


def _depth_rgb(values: np.ndarray, low: float, high: float) -> np.ndarray:
    normalized = np.clip((values - low) / max(high - low, 1e-8), 0.0, 1.0)
    display = np.rint(normalized * 255.0).astype(np.uint8)
    return cv2.cvtColor(
        cv2.applyColorMap(display, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB
    )


def _timed(
    function: Callable[[], np.ndarray], repetitions: int
) -> tuple[np.ndarray, list[float]]:
    function()  # warm-up
    durations = []
    result = None
    for _ in range(repetitions):
        started = perf_counter()
        result = function()
        durations.append((perf_counter() - started) * 1000.0)
    assert result is not None
    return result, durations


def _load_planar_metric3d(assets: Any) -> Any:
    import torch

    source = Path(assets.source_root)
    with _upstream_import_path(source):
        model = torch.hub.load(
            str(source), "metric3d_convnext_tiny", source="local", pretrain=False
        )
    payload = torch.load(assets.checkpoint.path, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
    expected = ("depth_model.decoder.depth_expectation_anchor",)
    if incompatible.missing_keys or tuple(incompatible.unexpected_keys) not in (
        (),
        expected,
    ):
        raise RuntimeError(f"unexpected Metric3D state mismatch: {incompatible}")
    _patch_metric3d_v1_device_assumptions(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _tensor_input(rgb: np.ndarray) -> Any:
    import torch

    normalized = (rgb.astype(np.float32) - MEAN) / STD
    return torch.from_numpy(np.moveaxis(normalized, -1, 0))[None]


def _metric3d_multiface(model: Any, rgb: np.ndarray, face_size: int) -> np.ndarray:
    import torch

    faces = equirectangular_to_cubemap(
        rgb.astype(np.float32), face_size, interpolation="bilinear"
    )
    reference_height, reference_width = 544, 1216
    padded_width = int(
        math.ceil(face_size * reference_width / reference_height / 32.0) * 32
    )
    pad_total = padded_width - face_size
    pad_left = pad_total // 2
    coordinate = ((np.arange(face_size, dtype=np.float64) + 0.5) / face_size) * 2 - 1
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    radial_scale = np.sqrt(1.0 + xx * xx + yy * yy)
    radial_faces: dict[str, np.ndarray] = {}
    for face_name in CUBE_FACE_ORDER:
        face = np.asarray(faces[face_name].data)
        padded = cv2.copyMakeBorder(
            face,
            0,
            0,
            pad_left,
            pad_total - pad_left,
            cv2.BORDER_CONSTANT,
            value=tuple(float(item) for item in MEAN),
        )
        with torch.inference_mode():
            axial = model.inference({"input": _tensor_input(padded)})[0][
                0, 0, :, pad_left : pad_left + face_size
            ]
        axial_m = axial.detach().cpu().numpy() * ((face_size / 2.0) / 1000.0)
        radial_faces[face_name] = np.asarray(axial_m * radial_scale, dtype=np.float32)
    reconstructed = cubemap_to_equirectangular(
        radial_faces,
        rgb.shape[:2],
        interpolation="bilinear",
        invalid_policy="propagate",
    )
    return np.clip(np.asarray(reconstructed.data), 0.3, 150.0).astype(np.float32)


def _metric3d_benchmark(
    rgb: np.ndarray,
    cache: Path,
    repetitions: int,
) -> dict[str, Any]:
    import torch

    loaded = load_spherical_metric3d_convnext_tiny_v1(
        accept_upstream_terms=True,
        cache_dir=cache,
        device="cpu",
        max_sampled_elements=30_000_000,
        timeout_seconds=1200,
    )
    tensor = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1)[None]

    def native() -> np.ndarray:
        with torch.inference_mode():
            return loaded.model(tensor)[0, 0].detach().cpu().numpy()

    native_depth, native_ms = _timed(native, repetitions)
    planar = _load_planar_metric3d(loaded.assets)
    face_size = 192
    multiface_depth, multiface_ms = _timed(
        lambda: _metric3d_multiface(planar, rgb, face_size), repetitions
    )
    return {
        "input_shape_hw": list(rgb.shape[:2]),
        "multiface": {"layout": "cube", "face_size": face_size, "fov_deg": 90.0},
        "native_ms": native_ms,
        "multiface_ms": multiface_ms,
        "native_median_ms": float(np.median(native_ms)),
        "multiface_median_ms": float(np.median(multiface_ms)),
        "native_p95_ms": float(np.quantile(native_ms, 0.95)),
        "multiface_p95_ms": float(np.quantile(multiface_ms, 0.95)),
        "native_prediction_sha256": hashlib.sha256(
            np.ascontiguousarray(native_depth).tobytes()
        ).hexdigest(),
        "multiface_prediction_sha256": hashlib.sha256(
            np.ascontiguousarray(multiface_depth).tobytes()
        ).hexdigest(),
        "checkpoint_sha256": loaded.assets.checkpoint.spec.sha256,
        "source_sha256": loaded.assets.source.spec.sha256,
    }


def _da3_figure(rgb: np.ndarray, cache: Path, output: Path) -> dict[str, Any]:
    panorama = EquirectangularImage(rgb)
    views = panorama.views("cube", size=280, fov=100)
    loaded = load_da3metric_large(
        accept_upstream_terms=True,
        cache_dir=cache,
        device="cpu",
        timeout_seconds=1200,
    )
    started = perf_counter()
    predicted = views.map(
        loaded.tangent(views[0].spec), input="image", output="depth", units="m"
    )
    inference_ms = (perf_counter() - started) * 1000.0
    started = perf_counter()
    fused = predicted.reconstruct(modalities="depth", blend={"depth": "gaussian"})
    fusion_ms = (perf_counter() - started) * 1000.0
    radial = np.asarray(fused.depth)
    valid = fused.validity("depth")
    low = float(np.quantile(radial[valid], 0.02))
    high = float(np.quantile(radial[valid], 0.98))
    panels = (
        _label(Image.fromarray(rgb).resize((600, 300)), "Input ERP"),
        _label(
            Image.fromarray(_depth_rgb(radial, low, high)).resize((600, 300)),
            "Depth Anything: fused radial range",
        ),
    )
    figure = Image.new("RGB", (1200, 300), (15, 23, 42))
    figure.paste(panels[0], (0, 0))
    figure.paste(panels[1], (600, 0))
    figure.save(output, quality=91, optimize=True, progressive=True)
    return {
        "input_shape_hw": list(rgb.shape[:2]),
        "route": {"layout": "cube", "face_shape_hw": [280, 280], "fov_deg": 100.0},
        "inference_ms": inference_ms,
        "fusion_ms": fusion_ms,
        "parameter_count": loaded.parameter_count,
        "checkpoint_sha256": loaded.assets.checkpoint.spec.sha256,
        "source_sha256": loaded.assets.source.spec.sha256,
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def main() -> int:
    args = _arguments()
    if args.repetitions < 2:
        raise ValueError("repetitions must be at least two")
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != INPUT_SHA256:
        raise ValueError("input checksum does not match the documented CC0 ERP")
    rgb = np.asarray(
        Image.open(args.input)
        .convert("RGB")
        .resize((512, 256), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    da3_path = args.output_dir / "depth-anything-views.jpg"
    metadata = {
        "schema": "panorai-deep-learning-doc-figures/v1",
        "input_sha256": INPUT_SHA256,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "metric3d": _metric3d_benchmark(rgb, args.metric3d_cache, args.repetitions),
        "depth_anything_3": _da3_figure(rgb, args.da3_cache, da3_path),
    }
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
