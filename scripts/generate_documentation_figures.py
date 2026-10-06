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


def _spherical_dog_figure(rgb: np.ndarray) -> tuple[Image.Image, dict[str, Any]]:
    from panorai.features import SphericalDoGSIFTConfig, SphericalDoGSIFTExtractor

    detector = SphericalDoGSIFTExtractor(
        SphericalDoGSIFTConfig(
            octaves=3,
            max_features=600,
            contrast_threshold=0.006,
        )
    )
    feature_set = detector.extract(rgb, panorama_id="spherical-dog-sift")
    canvas = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    order = np.argsort(feature_set.responses)[::-1][:450]
    for feature_index in order:
        feature = feature_set.features[int(feature_index)]
        x, y = (float(value) for value in feature.source_erp_xy)
        radius = min(7.0, 2.0 + feature.scale * 0.35)
        color = (*_color(feature.octave), 220)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=color,
            width=2,
        )
    draw.rounded_rectangle((14, 14, 495, 63), radius=9, fill=(15, 23, 42, 235))
    draw.text(
        (29, 28),
        "Spherical DoG detections; colour = octave",
        fill=(255, 255, 255, 255),
    )
    return canvas, {
        "feature_count": len(feature_set),
        "plotted_count": int(len(order)),
        "octaves": detector.config.octaves,
        "descriptor": "OpenCV SIFT on one tangent patch per keypoint",
    }


def _erp_panel(rgb: np.ndarray, label: str) -> Image.Image:
    panel = (
        Image.fromarray(rgb).convert("RGB").resize((768, 384), Image.Resampling.LANCZOS)
    )
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((14, 14, 330, 58), radius=9, fill=(15, 23, 42))
    draw.text((29, 27), label, fill=(255, 255, 255))
    return panel


def _spherical_processing_figure(rgb: np.ndarray) -> tuple[Image.Image, dict[str, Any]]:
    from panorai.image_processing import (
        spherical_canny,
        spherical_gaussian_blur,
        spherical_gradient,
    )

    floating = rgb.astype(np.float32) / 255.0
    gray = cv2.cvtColor(floating, cv2.COLOR_RGB2GRAY)
    gaussian_ksize = 9
    smoothed = spherical_gaussian_blur(floating, ksize=gaussian_ksize, sigma=2.0)
    gradient = spherical_gradient(gray, operator="scharr")
    gradient_scale = max(float(np.percentile(gradient.magnitude, 99.5)), 1e-6)
    gradient_u8 = np.rint(
        np.clip(gradient.magnitude / gradient_scale, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    gradient_rgb = cv2.cvtColor(
        cv2.applyColorMap(gradient_u8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB
    )
    threshold_low, threshold_high = 0.035, 0.085
    edges = spherical_canny(
        gray,
        threshold_low,
        threshold_high,
        gaussian_ksize=5,
        gaussian_sigma=1.2,
    )
    edge_rgb = np.repeat(edges[..., None], 3, axis=2)
    smooth_u8 = np.rint(np.clip(smoothed, 0.0, 1.0) * 255.0).astype(np.uint8)
    panels = (
        _erp_panel(rgb, "Original ERP (CC0)"),
        _erp_panel(smooth_u8, "Spherical Gaussian, 9x9"),
        _erp_panel(gradient_rgb, "Scharr tangent magnitude"),
        _erp_panel(edge_rgb, "Geodesic Canny"),
    )
    figure = Image.new("RGB", (1536, 768), (15, 23, 42))
    for index, panel in enumerate(panels):
        figure.paste(panel, ((index % 2) * 768, (index // 2) * 384))
    return figure, {
        "gaussian_ksize": gaussian_ksize,
        "gaussian_sigma": 2.0,
        "gradient_display_percentile": 99.5,
        "canny_threshold_low": threshold_low,
        "canny_threshold_high": threshold_high,
        "canny_edge_pixels": int(np.count_nonzero(edges)),
    }


def _line_chart(
    series: tuple[tuple[np.ndarray, tuple[int, int, int], str], ...],
    title: str,
    *,
    log_scale: bool = False,
) -> Image.Image:
    width, height = 768, 250
    image = Image.new("RGB", (width, height), (15, 23, 42))
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 64, 42, width - 24, height - 40
    draw.line((left, top, left, bottom, right, bottom), fill=(148, 163, 184), width=2)
    draw.text((24, 14), title, fill=(255, 255, 255))
    prepared = []
    maximum = 0.0
    for values, color, label in series:
        plotted = np.log1p(values) if log_scale else values.astype(np.float64)
        prepared.append((plotted, color, label))
        maximum = max(maximum, float(np.max(plotted)))
    maximum = max(maximum, np.finfo(np.float64).eps)
    for plotted, color, label in prepared:
        x = np.linspace(left, right, plotted.size)
        y = bottom - plotted / maximum * (bottom - top)
        points = [(float(px), float(py)) for px, py in zip(x, y, strict=True)]
        draw.line(points, fill=color, width=3)
    legend_x = left + 12
    for _, color, label in prepared:
        draw.line((legend_x, top + 13, legend_x + 24, top + 13), fill=color, width=3)
        draw.text((legend_x + 32, top + 5), label, fill=(226, 232, 240))
        legend_x += 180
    return image


def _histogram_equalization_figure(
    rgb: np.ndarray,
) -> tuple[Image.Image, dict[str, Any]]:
    from panorai.image_processing import spherical_equalize_histogram

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    equalized = spherical_equalize_histogram(gray)
    before = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    after = np.bincount(equalized.ravel(), minlength=256).astype(np.float64)
    latitude = np.pi / 2.0 - (np.arange(gray.shape[0]) + 0.5) * np.pi / gray.shape[0]
    area_weight = np.cos(latitude)

    top = Image.new("RGB", (1536, 384))
    top.paste(
        _erp_panel(np.repeat(gray[..., None], 3, axis=2), "Input luminance"), (0, 0)
    )
    top.paste(
        _erp_panel(
            np.repeat(equalized[..., None], 3, axis=2),
            "Solid-angle histogram equalization",
        ),
        (768, 0),
    )
    histogram = _line_chart(
        (
            (before, (56, 189, 248), "before"),
            (after, (251, 146, 60), "after"),
        ),
        "Pixel histogram (log scale)",
        log_scale=True,
    )
    weights = _line_chart(
        ((area_weight, (74, 222, 128), "cos(latitude)"),),
        "Relative solid-angle weight by ERP row",
    )
    figure = Image.new("RGB", (1536, 634), (15, 23, 42))
    figure.paste(top, (0, 0))
    figure.paste(histogram, (0, 384))
    figure.paste(weights, (768, 384))
    return figure, {
        "input_luminance_range": [int(gray.min()), int(gray.max())],
        "equalized_luminance_range": [int(equalized.min()), int(equalized.max())],
        "histogram_weight": "cos(latitude at ERP row center)",
    }


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
    spherical_dog, spherical_dog_parameters = _spherical_dog_figure(rgb)
    spherical_dog.save(
        args.output_dir / "spherical-dog-sift.jpg",
        quality=91,
        optimize=True,
        progressive=True,
    )
    processing, processing_parameters = _spherical_processing_figure(rgb)
    processing.save(
        args.output_dir / "spherical-image-processing.jpg",
        quality=91,
        optimize=True,
        progressive=True,
    )
    equalization, equalization_parameters = _histogram_equalization_figure(rgb)
    equalization.save(
        args.output_dir / "spherical-histogram-equalization.jpg",
        quality=91,
        optimize=True,
        progressive=True,
    )
    metadata = {
        "source": "Poly Haven nature_reserve_forest 1K HDRI",
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "shape_hw": list(rgb.shape[:2]),
        "feature_counts": counts,
        "sift_flann_match_count": match_count,
        "spherical_dog_sift": spherical_dog_parameters,
        "comparison_transform": {
            "kind": "cyclic ERP longitude shift",
            "pixels": int(rgb.shape[1] // 18),
        },
        "spherical_image_processing": processing_parameters,
        "spherical_histogram_equalization": equalization_parameters,
    }
    (args.output_dir / "figure-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
