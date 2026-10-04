#!/usr/bin/env python3
"""Rebuild the public-data figures used by the vision tutorials.

The input is the CC0 Poly Haven ``nature_reserve_forest`` 1K HDRI. Download
and licensing metadata live beside the generated figures in
``docs/_static/tutorials/ATTRIBUTION.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

EXPECTED_SOURCE_SHA256 = (
    "6c943ddd683de2f3d9aaa62596961dfccdc9cf206adebfc198e70235ae5707cd"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_hdr", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def _load_tonemapped_rgb(path: Path) -> np.ndarray:
    if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_SOURCE_SHA256:
        raise ValueError("source HDRI checksum does not match the documented asset")
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("expected a three-channel HDR panorama")
    rgb = cv2.cvtColor(bgr.astype(np.float32), cv2.COLOR_BGR2RGB)
    exposure = float(np.percentile(rgb, 99.5))
    mapped = np.clip(rgb / max(exposure, np.finfo(np.float32).eps), 0.0, 1.0)
    mapped = np.power(mapped, 1.0 / 2.2)
    return np.round(mapped * 255.0).astype(np.uint8)


def _pipeline(preset: str) -> Any:
    from panorai.features import SphericalFeaturePipeline

    return SphericalFeaturePipeline.from_preset(
        preset,
        face_sampler="icosahedron",
        face_fov_deg=82.0,
        face_overlap_deg=8.0,
        face_shape_hw=(256, 256),
        edge_margin_px=8,
        max_features=700,
    )


def _color(index: int) -> tuple[int, int, int]:
    palette = (
        (255, 87, 51),
        (33, 150, 243),
        (76, 175, 80),
        (255, 193, 7),
        (156, 39, 176),
        (0, 188, 212),
    )
    return palette[index % len(palette)]


def _feature_panel(rgb: np.ndarray, preset: str) -> tuple[Image.Image, int]:
    feature_set = _pipeline(preset).extract(rgb, panorama_id=preset)
    canvas = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    order = np.argsort(feature_set.responses)[::-1][:450]
    face_to_index: dict[str, int] = {}
    for feature_index in order:
        xy = feature_set.source_erp_xy[feature_index]
        if not np.isfinite(xy).all():
            continue
        face_id = str(feature_set.face_ids[feature_index])
        color_index = face_to_index.setdefault(face_id, len(face_to_index))
        x, y = (float(xy[0]), float(xy[1]))
        radius = 2.4
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=_color(color_index),
            width=2,
        )
    draw.rounded_rectangle((12, 12, 190, 52), radius=8, fill=(15, 23, 42))
    draw.text((25, 22), preset.upper(), fill=(255, 255, 255))
    return canvas.resize((512, 256), Image.Resampling.LANCZOS), len(feature_set)


def _matching_panel(rgb_a: np.ndarray, rgb_b: np.ndarray) -> tuple[Image.Image, int]:
    pipeline = _pipeline("sift-flann")
    features_a = pipeline.extract(rgb_a, panorama_id="view-a")
    features_b = pipeline.extract(rgb_b, panorama_id="view-b")
    matches = pipeline.match(features_a, features_b)
    width, height = 768, 384
    top = Image.fromarray(rgb_a).resize((width, height), Image.Resampling.LANCZOS)
    bottom = Image.fromarray(rgb_b).resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height * 2 + 28), (15, 23, 42))
    canvas.paste(top, (0, 0))
    canvas.paste(bottom, (0, height + 28))
    draw = ImageDraw.Draw(canvas, "RGBA")
    valid_indices = np.flatnonzero(matches.valid)
    if valid_indices.size:
        ranked = valid_indices[np.argsort(matches.descriptor_distances[valid_indices])]
    else:
        ranked = valid_indices
    scale_x = width / rgb_a.shape[1]
    scale_y = height / rgb_a.shape[0]
    for line_index, match_index in enumerate(ranked[:70]):
        point_a = features_a.source_erp_xy[matches.feature_indices_a[match_index]]
        point_b = features_b.source_erp_xy[matches.feature_indices_b[match_index]]
        if not (np.isfinite(point_a).all() and np.isfinite(point_b).all()):
            continue
        a = (float(point_a[0] * scale_x), float(point_a[1] * scale_y))
        b = (
            float(point_b[0] * scale_x),
            float(point_b[1] * scale_y + height + 28),
        )
        color = (*_color(line_index), 145)
        draw.line((a, b), fill=color, width=1)
        draw.ellipse((a[0] - 2, a[1] - 2, a[0] + 2, a[1] + 2), fill=color)
        draw.ellipse((b[0] - 2, b[1] - 2, b[0] + 2, b[1] + 2), fill=color)
    draw.text(
        (12, height + 6),
        "SIFT + FLANN matches (best descriptor distances)",
        fill="white",
    )
    return canvas, len(matches)


def main() -> None:
    args = _arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rgb = _load_tonemapped_rgb(args.source_hdr)
    Image.fromarray(rgb).save(
        args.output_dir / "nature-reserve-forest-erp.jpg",
        quality=90,
        optimize=True,
        progressive=True,
    )

    panels: list[Image.Image] = []
    counts: dict[str, int] = {}
    for preset in ("sift-flann", "orb-hamming", "akaze-hamming"):
        panel, count = _feature_panel(rgb, preset)
        panels.append(panel)
        counts[preset] = count
    detector_comparison = Image.new("RGB", (1536, 256))
    for index, panel in enumerate(panels):
        detector_comparison.paste(panel, (512 * index, 0))
    detector_comparison.save(
        args.output_dir / "feature-detectors.jpg",
        quality=91,
        optimize=True,
        progressive=True,
    )

    shifted = np.roll(rgb, rgb.shape[1] // 18, axis=1)
    matching, match_count = _matching_panel(rgb, shifted)
    matching.save(
        args.output_dir / "feature-matches.jpg",
        quality=90,
        optimize=True,
        progressive=True,
    )
    metadata = {
        "source": "Poly Haven nature_reserve_forest 1K HDRI",
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "shape_hw": list(rgb.shape[:2]),
        "feature_counts": counts,
        "sift_flann_match_count": match_count,
        "comparison_transform": {
            "kind": "cyclic ERP longitude shift",
            "pixels": int(rgb.shape[1] // 18),
        },
    }
    (args.output_dir / "figure-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
