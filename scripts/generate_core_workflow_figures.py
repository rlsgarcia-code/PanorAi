#!/usr/bin/env python3
"""Generate compact core-workflow figures from a checksum-pinned CC0 ERP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from panorai.core.numpy import SphericalGaussianBlur2d, SphericalResize  # noqa: E402


EXPECTED_INPUT_SHA256 = (
    "5333c2cecc55468fcbc64a252f4bc097fab064cae70bead4f76d6c2e6101c524"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=REPOSITORY_ROOT
        / "docs/_static/tutorials/nature-reserve-forest-erp.jpg",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "docs/_static/tutorials/spherical-dog-pyramid.jpg",
    )
    return parser.parse_args()


def _response_panel(response: np.ndarray, label: str) -> Image.Image:
    scale = max(float(np.percentile(np.abs(response), 99.5)), 1e-8)
    display = np.rint(np.clip(np.abs(response) / scale, 0.0, 1.0) * 255).astype(
        np.uint8
    )
    rgb = cv2.cvtColor(cv2.applyColorMap(display, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    panel = Image.fromarray(rgb).resize((300, 150), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((10, 10, 150, 38), radius=7, fill=(15, 23, 42))
    draw.text((20, 18), label, fill=(255, 255, 255))
    return panel


def main() -> int:
    args = _arguments()
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != EXPECTED_INPUT_SHA256:
        raise ValueError("input ERP checksum does not match the documented CC0 asset")

    rgb = np.asarray(Image.open(args.input).convert("RGB"))
    rgb = np.asarray(
        Image.fromarray(rgb).resize((512, 256), Image.Resampling.LANCZOS)
    )
    gray = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2GRAY)
    current = gray[None, None]
    small = SphericalGaussianBlur2d(sigma=1.0)
    large = SphericalGaussianBlur2d(sigma=2.0)
    resize = SphericalResize(scale_factor=0.5)

    responses: list[np.ndarray] = []
    shapes: list[list[int]] = []
    for _ in range(3):
        response = large(current) - small(current)
        responses.append(response[0, 0])
        shapes.append(list(response.shape[-2:]))
        current = resize(small(current))

    source = Image.fromarray(rgb).resize((300, 150), Image.Resampling.LANCZOS)
    source_draw = ImageDraw.Draw(source)
    source_draw.rounded_rectangle((10, 10, 150, 38), radius=7, fill=(15, 23, 42))
    source_draw.text((20, 18), "Input ERP", fill=(255, 255, 255))
    panels = [source]
    panels.extend(
        _response_panel(response, f"DoG octave {index}")
        for index, response in enumerate(responses)
    )
    figure = Image.new("RGB", (1200, 150), (15, 23, 42))
    for index, panel in enumerate(panels):
        figure.paste(panel, (index * 300, 0))
    figure.save(args.output, quality=91, optimize=True, progressive=True)

    print(
        json.dumps(
            {
                "input_sha256": EXPECTED_INPUT_SHA256,
                "working_shape_hw": [256, 512],
                "octave_shape_hw": shapes,
                "sigma_pair": [1.0, 2.0],
                "display": "absolute DoG normalized at per-octave 99.5 percentile",
                "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
