#!/usr/bin/env python3
"""Build the application-first documentation gallery from pinned CC0 HDRIs.

The four source HDRIs remain in an external cache.  This script verifies their
exact bytes before decoding them and writes only the compact, metadata-free
documentation derivatives requested by the caller.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import BinaryIO, Callable
from urllib.request import urlopen

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


@dataclass(frozen=True, slots=True)
class SourceAsset:
    """One externally cached, checksum-pinned documentation source."""

    slug: str
    title: str
    filename: str
    url: str
    page_url: str
    authors: tuple[str, ...]
    size_bytes: int
    sha256: str


SOURCE_ASSETS = (
    SourceAsset(
        slug="boiler_room",
        title="Boiler Room",
        filename="boiler_room_2k.hdr",
        url=("https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/boiler_room_2k.hdr"),
        page_url="https://polyhaven.com/a/boiler_room",
        authors=("Oliksiy Yakovlyev",),
        size_bytes=6_707_438,
        sha256="07c353cb275f1e359cdfa1710acb77c5cd075ce7770859ca56641993263d2d19",
    ),
    SourceAsset(
        slug="boma",
        title="Boma",
        filename="boma_2k.hdr",
        url="https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/boma_2k.hdr",
        page_url="https://polyhaven.com/a/boma",
        authors=("Greg Zaal", "Jarod Guest"),
        size_bytes=6_954_266,
        sha256="9d2e230d5fdd89efd15b6081a3951c90c771687edd81bc1e08cbb66adb98dd8c",
    ),
    SourceAsset(
        slug="sundowner_overlook",
        title="Sundowner Overlook",
        filename="sundowner_overlook_2k.hdr",
        url=(
            "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/"
            "sundowner_overlook_2k.hdr"
        ),
        page_url="https://polyhaven.com/a/sundowner_overlook",
        authors=("Dario Barresi",),
        size_bytes=6_872_718,
        sha256="3cdcb8a8f066e02d5554ce2d6a2591c9a47e02959297f4b2649fd91a49014498",
    ),
    SourceAsset(
        slug="mosaic_tunnel",
        title="Mosaic Tunnel",
        filename="mosaic_tunnel_2k.hdr",
        url=(
            "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/mosaic_tunnel_2k.hdr"
        ),
        page_url="https://polyhaven.com/a/mosaic_tunnel",
        authors=("Greg Zaal",),
        size_bytes=6_625_466,
        sha256="47f9267f0620158ccdb7fa7bdf25fd3aa4cd9de3fb88a9bfea68b056c739d6d4",
    ),
)

CARD_SIZE = (1200, 600)
PINHOLE_WORKFLOW_SIZE = (1200, 300)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest without loading the complete file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(path: Path, asset: SourceAsset) -> Path:
    """Verify one cached source by size and SHA-256."""

    if not path.is_file():
        raise FileNotFoundError(path)
    received_size = path.stat().st_size
    if received_size != asset.size_bytes:
        raise ValueError(
            f"source size mismatch for {path}: expected {asset.size_bytes}, "
            f"received {received_size}"
        )
    received_sha256 = sha256_file(path)
    if received_sha256 != asset.sha256:
        raise ValueError(
            f"source SHA-256 mismatch for {path}: expected {asset.sha256}, "
            f"received {received_sha256}"
        )
    return path


def _default_opener(url: str) -> BinaryIO:
    return urlopen(url, timeout=90)  # noqa: S310 - URLs are fixed in SOURCE_ASSETS


def acquire_source(
    asset: SourceAsset,
    cache_dir: Path,
    *,
    offline: bool = False,
    opener: Callable[[str], BinaryIO] = _default_opener,
) -> Path:
    """Return a verified source, downloading a missing file atomically."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / asset.filename
    if target.exists():
        return verify_source(target, asset)
    if offline:
        raise FileNotFoundError(
            f"offline mode requires the verified cached source {target}"
        )
    partial = cache_dir / f".{asset.filename}.partial"
    try:
        with opener(asset.url) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        verify_source(partial, asset)
        partial.replace(target)
    finally:
        if partial.exists():
            partial.unlink()
    return verify_source(target, asset)


def _tone_map(path: Path, *, height: int = 512) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"expected a three-channel HDR panorama: {path}")
    rgb = cv2.cvtColor(bgr.astype(np.float32), cv2.COLOR_BGR2RGB)
    exposure = float(np.percentile(rgb, 99.5))
    mapped = np.clip(rgb / max(exposure, np.finfo(np.float32).eps), 0.0, 1.0)
    mapped = np.power(mapped, 1.0 / 2.2)
    uint8 = np.rint(mapped * 255.0).astype(np.uint8)
    return cv2.resize(uint8, (2 * height, height), interpolation=cv2.INTER_AREA)


def _resize(rgb: np.ndarray, size: tuple[int, int]) -> Image.Image:
    values = np.asarray(rgb)
    if values.dtype != np.uint8:
        values = np.rint(np.clip(values, 0.0, 255.0)).astype(np.uint8)
    return Image.fromarray(values).resize(size, Image.Resampling.LANCZOS)


def _save_jpeg(image: Image.Image, path: Path) -> None:
    image.convert("RGB").save(
        path,
        format="JPEG",
        quality=87,
        optimize=True,
        progressive=True,
        exif=b"",
    )


def _stacked_erp_pair(top: np.ndarray, bottom: np.ndarray) -> Image.Image:
    """Preserve ERP aspect ratio while presenting one clear visual comparison."""

    canvas = Image.new("RGB", CARD_SIZE, (15, 23, 42))
    canvas.paste(_resize(top, (1200, 300)), (0, 0))
    canvas.paste(_resize(bottom, (1200, 300)), (0, 300))
    return canvas


def _draw_wrapped_polyline(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    width: int,
    color: tuple[int, int, int],
) -> None:
    for start, end in zip(points[:-1], points[1:], strict=True):
        if abs(float(start[0] - end[0])) < width / 2:
            draw.line((*start, *end), fill=color, width=3)


def _projection_card(rgb: np.ndarray) -> tuple[Image.Image, dict[str, object]]:
    from panorai import EquirectangularImage
    from panorai.geometry import gnomonic_pixels_to_rays, rays_to_erp_pixels

    panorama = EquirectangularImage(rgb)
    faces = panorama.views("cube", size=256, fov=95.0)
    reconstructed = faces.reconstruct(blend="gaussian")
    reconstruction = np.rint(
        np.clip(np.asarray(reconstructed.image), 0.0, 255.0)
    ).astype(np.uint8)
    support = np.asarray(reconstructed.support_mask, dtype=bool)

    footprint = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(footprint)
    palette = (
        (34, 211, 238),
        (250, 204, 21),
        (244, 114, 182),
        (74, 222, 128),
        (251, 146, 60),
        (167, 139, 250),
    )
    edge_count = 100
    axis = np.linspace(-0.5, 255.5, edge_count)
    boundaries = (
        np.stack((axis, np.full_like(axis, -0.5)), axis=-1),
        np.stack((np.full_like(axis, 255.5), axis), axis=-1),
        np.stack((axis[::-1], np.full_like(axis, 255.5)), axis=-1),
        np.stack((np.full_like(axis, -0.5), axis[::-1]), axis=-1),
    )
    for index, face in enumerate(faces):
        for boundary in boundaries:
            rays = gnomonic_pixels_to_rays(boundary, face.spec).rays_xyz
            pixels = rays_to_erp_pixels(rays, rgb.shape[:2]).pixels_xy
            _draw_wrapped_polyline(draw, pixels, rgb.shape[1], palette[index])

    face_canvas = Image.new("RGB", (600, 300), (15, 23, 42))
    for index, face in enumerate(faces):
        cell = _resize(np.asarray(face.image), (200, 150))
        face_canvas.paste(cell, ((index % 3) * 200, (index // 3) * 150))

    difference = np.mean(
        np.abs(reconstruction.astype(np.float32) - rgb.astype(np.float32)), axis=2
    )
    scale = max(float(np.percentile(difference[support], 99.5)), 1e-6)
    error_u8 = np.rint(np.clip(difference / scale, 0.0, 1.0) * 255).astype(np.uint8)
    error = cv2.cvtColor(
        cv2.applyColorMap(error_u8, cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB
    )
    support_rgb = np.full_like(rgb, 104)
    support_rgb[support] = 226
    audit = np.concatenate(
        (error[:, : rgb.shape[1] // 2], support_rgb[:, rgb.shape[1] // 2 :]), axis=1
    )

    card = Image.new("RGB", CARD_SIZE, (15, 23, 42))
    card.paste(footprint.resize((600, 300), Image.Resampling.LANCZOS), (0, 0))
    card.paste(face_canvas, (600, 0))
    card.paste(_resize(reconstruction, (600, 300)), (0, 300))
    card.paste(_resize(audit, (600, 300)), (600, 300))
    return card, {
        "working_shape_hw": list(rgb.shape[:2]),
        "layout": "cube",
        "face_shape_hw": [256, 256],
        "fov_deg": 95.0,
        "blend": "gaussian",
        "support_fraction": float(support.mean()),
        "round_trip_mae": float(difference[support].mean()),
        "error_display_percentile_99_5": scale,
        "quadrants": [
            "source with exact face footprints",
            "six real projected faces",
            "reconstructed ERP",
            "round-trip error (left) and support (right)",
        ],
    }


def _pinhole_workflow_card(rgb: np.ndarray) -> tuple[Image.Image, dict[str, object]]:
    """Show one real ERP -> faces -> pinhole operation -> fusion workflow."""

    from panorai import EquirectangularImage

    panorama = EquirectangularImage(rgb)
    faces = panorama.views("icosahedron", size=160, fov=82.0)
    blurred_faces = faces.map(lambda face: cv2.GaussianBlur(face, (15, 15), 3.0))
    reconstructed = blurred_faces.reconstruct(blend="gaussian")
    fused = np.rint(np.clip(np.asarray(reconstructed.image), 0.0, 255.0)).astype(
        np.uint8
    )
    support = np.asarray(reconstructed.support_mask, dtype=bool)

    def face_mosaic(face_set) -> Image.Image:
        mosaic = Image.new("RGB", (285, 300), (15, 23, 42))
        for index, face in enumerate(face_set):
            tile = _resize(np.asarray(face.image), (71, 71))
            mosaic.paste(tile, ((index % 4) * 71, 43 + (index // 4) * 71))
        return mosaic

    card = Image.new("RGB", PINHOLE_WORKFLOW_SIZE, (15, 23, 42))
    card.paste(_resize(rgb, (285, 142)), (0, 79))
    card.paste(face_mosaic(faces), (305, 0))
    card.paste(face_mosaic(blurred_faces), (610, 0))
    card.paste(_resize(fused, (285, 142)), (915, 79))
    draw = ImageDraw.Draw(card)
    for start in (288, 593, 898):
        draw.line((start, 150, start + 12, 150), fill=(148, 163, 184), width=3)
        draw.polygon(
            ((start + 12, 144), (start + 17, 150), (start + 12, 156)),
            fill=(148, 163, 184),
        )

    return card, {
        "working_shape_hw": list(rgb.shape[:2]),
        "stages": [
            "source ERP",
            "12 real icosahedron-sampled gnomonic faces",
            "OpenCV Gaussian blur independently applied to every face",
            "Gaussian-blended ERP reconstruction",
        ],
        "layout": "icosahedron",
        "face_count": len(faces),
        "face_shape_hw": [160, 160],
        "fov_deg": 82.0,
        "pinhole_operation": {
            "library": "OpenCV",
            "name": "GaussianBlur",
            "kernel_size": [15, 15],
            "sigma": 3.0,
        },
        "blend": "gaussian",
        "support_fraction": float(support.mean()),
        "interpretation": (
            "real public-API output; panels are ordered left to right and "
            "contain no synthetic intermediate"
        ),
    }


def _processing_card(rgb: np.ndarray) -> tuple[Image.Image, dict[str, object]]:
    from panorai.image_processing import (
        spherical_canny,
        spherical_gaussian_blur,
    )

    floating = rgb.astype(np.float32) / 255.0
    gray = cv2.cvtColor(floating, cv2.COLOR_RGB2GRAY)
    smoothed = spherical_gaussian_blur(floating, ksize=9, sigma=2.0)
    edges = spherical_canny(
        gray,
        0.035,
        0.085,
        gaussian_ksize=5,
        gaussian_sigma=1.2,
        gradient_operator="scharr",
    )
    smooth_display = np.power(np.clip(smoothed, 0.0, 1.0), 0.60)
    smooth_u8 = np.rint(smooth_display * 255).astype(np.uint8)
    source_display = np.power(np.clip(floating, 0.0, 1.0), 0.60)
    edge_overlay = np.rint(source_display * 255.0 * 0.52).astype(np.uint8)
    display_edges = cv2.dilate(
        edges.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
    ).astype(bool)
    edge_overlay[display_edges] = np.array([34, 211, 238], dtype=np.uint8)
    return _stacked_erp_pair(smooth_u8, edge_overlay), {
        "working_shape_hw": list(rgb.shape[:2]),
        "panels": [
            "spherical Gaussian 9x9 sigma 2.0",
            "geodesic Canny over a dimmed source ERP",
        ],
        "canny_thresholds": [0.035, 0.085],
        "canny_edge_pixels": int(np.count_nonzero(edges)),
        "display": {
            "gamma": 0.60,
            "canny_visualization_dilation_px": 3,
            "note": "dilation is display-only; numeric edge counts use the undilated mask",
        },
    }


def _feature_pipeline():
    from panorai.features import SphericalFeaturePipeline

    return SphericalFeaturePipeline.from_preset(
        "sift-flann",
        face_sampler="icosahedron",
        face_fov_deg=82.0,
        face_overlap_deg=8.0,
        face_shape_hw=(256, 256),
        edge_margin_px=8,
        max_features=700,
    )


def _feature_card(rgb: np.ndarray) -> tuple[Image.Image, dict[str, object]]:
    shift_pixels = rgb.shape[1] // 3
    shifted = np.roll(rgb, shift_pixels, axis=1)
    pipeline = _feature_pipeline()
    features_a = pipeline.extract(rgb, panorama_id="mosaic-tunnel-a")
    features_b = pipeline.extract(shifted, panorama_id="mosaic-tunnel-yaw")
    matches = pipeline.match(features_a, features_b)

    canvas = Image.new("RGB", CARD_SIZE, (15, 23, 42))
    canvas.paste(_resize(rgb, (1200, 300)), (0, 0))
    canvas.paste(_resize(shifted, (1200, 300)), (0, 300))
    draw = ImageDraw.Draw(canvas, "RGBA")
    valid = np.flatnonzero(matches.valid)
    ranked = (
        valid[np.argsort(matches.descriptor_distances[valid])] if valid.size else valid
    )
    scale_x = 1200 / rgb.shape[1]
    scale_y = 300 / rgb.shape[0]
    palette = (
        (34, 211, 238, 150),
        (250, 204, 21, 150),
        (244, 114, 182, 150),
        (74, 222, 128, 150),
    )
    plotted = 0
    seam_matches = 0
    for order, match_index in enumerate(ranked[:90]):
        point_a = features_a.source_erp_xy[matches.feature_indices_a[match_index]]
        point_b = features_b.source_erp_xy[matches.feature_indices_b[match_index]]
        if not (np.isfinite(point_a).all() and np.isfinite(point_b).all()):
            continue
        a = (float(point_a[0] * scale_x), float(point_a[1] * scale_y))
        b = (
            float(point_b[0] * scale_x),
            float(point_b[1] * scale_y + 300),
        )
        color = palette[order % len(palette)]
        draw.line((a, b), fill=color, width=1)
        draw.ellipse((a[0] - 2, a[1] - 2, a[0] + 2, a[1] + 2), fill=color)
        draw.ellipse((b[0] - 2, b[1] - 2, b[0] + 2, b[1] + 2), fill=color)
        plotted += 1
        expected_b_x = (float(point_a[0]) + shift_pixels) % rgb.shape[1]
        circular_error = abs(float(point_b[0]) - expected_b_x)
        circular_error = min(circular_error, rgb.shape[1] - circular_error)
        if float(point_a[0]) + shift_pixels >= rgb.shape[1] and circular_error <= 5.0:
            seam_matches += 1
    return canvas, {
        "working_shape_hw": list(rgb.shape[:2]),
        "comparison_transform": {
            "kind": "cyclic ERP longitude shift",
            "pixels": shift_pixels,
            "degrees": 360.0 * shift_pixels / rgb.shape[1],
        },
        "features_a": len(features_a),
        "features_b": len(features_b),
        "valid_matches": int(valid.size),
        "plotted_matches": plotted,
        "plotted_matches_crossing_erp_seam": seam_matches,
    }


def _blue_yellow_red(values: np.ndarray) -> np.ndarray:
    normalized = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    low = normalized <= 0.5
    fraction = np.where(low, normalized * 2.0, (normalized - 0.5) * 2.0)
    blue = np.array([37.0, 99.0, 235.0])
    yellow = np.array([250.0, 204.0, 21.0])
    red = np.array([220.0, 38.0, 38.0])
    color = np.empty((*normalized.shape, 3), dtype=np.float64)
    color[low] = (1.0 - fraction[low, None]) * blue + fraction[low, None] * yellow
    color[~low] = (1.0 - fraction[~low, None]) * yellow + fraction[~low, None] * red
    return np.rint(color).astype(np.uint8)


def _seam_component(mask: np.ndarray, peak: tuple[int, int]) -> np.ndarray:
    """Return the 8-connected component at ``peak`` with ERP seam wrapping."""

    shift = mask.shape[1] // 2 - peak[1]
    shifted = np.roll(mask, shift, axis=1)
    shifted_peak = (peak[0], (peak[1] + shift) % mask.shape[1])
    labels, _ = ndimage.label(shifted, structure=np.ones((3, 3), dtype=np.uint8))
    label = int(labels[shifted_peak])
    selected = labels == label if label else np.zeros_like(mask, dtype=bool)
    return np.roll(selected, -shift, axis=1)


def _association_thumbnail(size: tuple[int, int]) -> Image.Image:
    """Draw the method relationship between two spherical region proposals."""

    width, height = size
    image = Image.new("RGB", size, (15, 23, 42))
    draw = ImageDraw.Draw(image, "RGBA")
    centres = (
        (int(width * 0.27), int(height * 0.64)),
        (int(width * 0.73), int(height * 0.64)),
    )
    radius = int(min(width, height) * 0.19)
    hypothesis = (int(width * 0.5), int(height * 0.20))
    for index, centre in enumerate(centres):
        draw.ellipse(
            (
                centre[0] - radius,
                centre[1] - radius,
                centre[0] + radius,
                centre[1] + radius,
            ),
            outline=(56, 189, 248, 255),
            width=4,
        )
        draw.ellipse(
            (
                centre[0] - radius,
                centre[1] - radius // 3,
                centre[0] + radius,
                centre[1] + radius // 3,
            ),
            outline=(148, 163, 184, 145),
            width=2,
        )
        draw.ellipse(
            (
                centre[0] - radius // 3,
                centre[1] - radius,
                centre[0] + radius // 3,
                centre[1] + radius,
            ),
            outline=(148, 163, 184, 120),
            width=2,
        )
        side = -1 if index == 0 else 1
        region_x = centre[0] + side * radius // 3
        region_y = centre[1] - radius // 3
        draw.ellipse(
            (
                region_x - radius // 3,
                region_y - radius // 4,
                region_x + radius // 3,
                region_y + radius // 4,
            ),
            fill=(250, 204, 21, 80),
            outline=(244, 114, 182, 255),
            width=3,
        )
        draw.polygon(
            (
                (centre[0], centre[1]),
                (region_x - radius // 2, region_y - radius // 2),
                (region_x + radius // 2, region_y - radius // 2),
                (region_x + radius // 2, region_y + radius // 2),
                (region_x - radius // 2, region_y + radius // 2),
            ),
            outline=(250, 204, 21, 210),
        )
        draw.line((*centre, *hypothesis), fill=(74, 222, 128, 225), width=4)
        draw.ellipse(
            (
                centre[0] - 5,
                centre[1] - 5,
                centre[0] + 5,
                centre[1] + 5,
            ),
            fill=(56, 189, 248, 255),
        )
    draw.line((*centres[0], *centres[1]), fill=(148, 163, 184, 130), width=3)
    draw.ellipse(
        (
            hypothesis[0] - 13,
            hypothesis[1] - 13,
            hypothesis[0] + 13,
            hypothesis[1] + 13,
        ),
        fill=(244, 114, 182, 255),
        outline=(255, 255, 255, 230),
        width=4,
    )
    return image


def _semantic_card(
    rgb: np.ndarray,
    *,
    device_name: str,
) -> tuple[Image.Image, dict[str, object]]:
    import torch

    from panorai.experimental.deep_learning import (
        ImageNetFCN,
        class_activation_map,
        load_pretrained_imagenet_model,
        port_module_with_report,
    )

    loaded = load_pretrained_imagenet_model("resnet18", progress=True)
    model = ImageNetFCN(loaded.model.eval(), "resnet18").eval()
    report = port_module_with_report(model)
    if report.remaining_planar_spatial_layers:
        raise RuntimeError("planar spatial layers remain in the semantic card model")
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    model = model.to(device)
    values = (
        torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None].to(device) / 255.0
    )
    transform = loaded.weights.transforms()
    mean = torch.tensor(transform.mean, device=device)[None, :, None, None]
    std = torch.tensor(transform.std, device=device)[None, :, None, None]
    values = (values - mean) / std
    with torch.inference_mode():
        dense = model.forward_dense(values)
        global_scores = model(values, spherical_average=True)[0]
        cam = class_activation_map(
            dense.logits,
            743,
            output_shape=rgb.shape[:2],
            positive_only=False,
        )
    raw = cam[0].float().cpu().numpy()
    support = np.ones(rgb.shape[:2], dtype=bool)
    positive = np.clip(raw, 0.0, None)
    valid = positive[support]
    low, high = float(valid.min()), float(valid.max())
    normalized = np.zeros_like(raw, dtype=np.float64)
    if high > low:
        normalized[support] = (positive[support] - low) / (high - low)
    relevance = float(torch.exp(global_scores[743] - global_scores.max()).cpu())
    probability = float(torch.softmax(global_scores, dim=0)[743].cpu())
    rank = int((global_scores > global_scores[743]).sum().cpu()) + 1

    heat = _blue_yellow_red(normalized)
    alpha = (0.18 + 0.55 * normalized)[..., None] * support[..., None]
    overlay = np.rint((1.0 - alpha) * rgb + alpha * heat).astype(np.uint8)
    peak = np.unravel_index(int(np.argmax(normalized)), normalized.shape)
    component = _seam_component(normalized >= 0.72, peak)
    if not component.any():
        component[peak] = True
    rows, columns = np.nonzero(component)
    row_center = float(np.average(rows, weights=normalized[component] + 1e-6))
    angles = 2.0 * np.pi * (columns + 0.5) / rgb.shape[1]
    weights = normalized[component] + 1e-6
    column_angle = math.atan2(
        float(np.sum(weights * np.sin(angles))),
        float(np.sum(weights * np.cos(angles))),
    )
    if column_angle < 0:
        column_angle += 2.0 * np.pi
    column_center = column_angle / (2.0 * np.pi) * rgb.shape[1] - 0.5
    latitude_deg = 90.0 - (row_center + 0.5) * 180.0 / rgb.shape[0]
    longitude_deg = (column_center + 0.5) * 360.0 / rgb.shape[1] - 180.0

    colorbar = _blue_yellow_red(np.linspace(0.0, 1.0, rgb.shape[1])[None])
    colorbar = np.repeat(colorbar, 18, axis=0)
    overlay[-18:] = colorbar

    card = Image.new("RGB", CARD_SIZE, (15, 23, 42))
    card.paste(_resize(overlay, (720, 360)), (0, 120))
    card.paste(_association_thumbnail((480, 600)), (720, 0))
    checkpoint = loaded.checkpoint.to_dict()
    checkpoint.pop("cache_path", None)
    checkpoint.pop("previously_cached", None)
    return card, {
        "working_shape_hw": list(rgb.shape[:2]),
        "architecture": "resnet18",
        "weights": str(loaded.weights),
        "checkpoint": checkpoint,
        "spherical_port": report.to_dict(),
        "class_index": 743,
        "class_name": "prison",
        "proxy_concept": "grid-like-structure",
        "global_rank": rank,
        "global_relevance": relevance,
        "classifier_probability": probability,
        "dense_shape_hw": list(dense.logits.shape[-2:]),
        "component_threshold": 0.72,
        "component_pixels": int(component.sum()),
        "proposal_center_lat_lon_deg": [latitude_deg, longitude_deg],
        "proposal_fov_deg": [70.0, 70.0],
        "interpretation": (
            "directional ImageNet evidence and a proposed spherical FOV; not a "
            "segmentation mask or calibrated probability"
        ),
        "panels": [
            "blue-yellow-red normalized CAM on the source ERP",
            "method diagram: two registered spherical FOV proposals associated with one shared hypothesis",
        ],
        "association_evidence": (
            "method diagram only; the card does not claim a measured second-view association"
        ),
    }


def _pose_range_card(stereo_path: Path) -> tuple[Image.Image, dict[str, object]]:
    """Pair spherical two-view geometry with the existing analytic range result."""

    card = Image.new("RGB", CARD_SIZE, (15, 23, 42))
    geometry = Image.new("RGB", (600, 600), (15, 23, 42))
    draw = ImageDraw.Draw(geometry, "RGBA")
    centres = ((155, 410), (445, 410))
    point = (315, 115)
    radius = 105
    for index, centre in enumerate(centres):
        color = (56, 189, 248, 255) if index == 0 else (244, 114, 182, 255)
        draw.ellipse(
            (
                centre[0] - radius,
                centre[1] - radius,
                centre[0] + radius,
                centre[1] + radius,
            ),
            outline=color,
            width=5,
        )
        draw.ellipse(
            (
                centre[0] - radius,
                centre[1] - radius // 3,
                centre[0] + radius,
                centre[1] + radius // 3,
            ),
            outline=(148, 163, 184, 130),
            width=2,
        )
        draw.ellipse(
            (
                centre[0] - radius // 3,
                centre[1] - radius,
                centre[0] + radius // 3,
                centre[1] + radius,
            ),
            outline=(148, 163, 184, 115),
            width=2,
        )
        draw.line((*centre, *point), fill=color, width=5)
        direction = np.asarray(point, dtype=np.float64) - np.asarray(centre)
        direction /= np.linalg.norm(direction)
        marker = np.asarray(point) - 28.0 * direction
        normal = np.array([-direction[1], direction[0]])
        draw.polygon(
            (
                point,
                tuple(marker + 10.0 * normal),
                tuple(marker - 10.0 * normal),
            ),
            fill=color,
        )
    draw.line((*centres[0], *centres[1]), fill=(250, 204, 21, 230), width=6)
    draw.polygon(((445, 410), (422, 398), (422, 422)), fill=(250, 204, 21, 255))
    draw.arc(
        (215, 330, 385, 500), start=205, end=330, fill=(74, 222, 128, 230), width=6
    )
    draw.ellipse(
        (299, 99, 331, 131),
        fill=(255, 255, 255, 255),
        outline=(250, 204, 21, 255),
        width=5,
    )
    card.paste(geometry, (0, 0))

    stereo = Image.open(stereo_path).convert("RGB")
    panel_width = stereo.width // 3
    panel_height = stereo.height // 2
    estimated = stereo.crop((2 * panel_width, 0, stereo.width, panel_height))
    error = stereo.crop((2 * panel_width, panel_height, stereo.width, stereo.height))
    card.paste(estimated.resize((600, 300), Image.Resampling.LANCZOS), (600, 0))
    card.paste(error.resize((600, 300), Image.Resampling.LANCZOS), (600, 300))
    return card, {
        "geometry_panel": (
            "method diagram: two registered spherical cameras, bearing rays, "
            "relative rotation, translation direction, and a shared 3D point"
        ),
        "range_panels": [
            "analytic-fixture estimated radial range",
            "analytic-fixture relative error",
        ],
        "source_asset": stereo_path.name,
        "interpretation": (
            "geometry is a method diagram; range and error are analytic fixture outputs"
        ),
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "docs/_static/tutorials",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        asset.slug: acquire_source(asset, args.cache_dir, offline=args.offline)
        for asset in SOURCE_ASSETS
    }
    sources = {slug: _tone_map(path, height=512) for slug, path in paths.items()}
    projection, projection_metadata = _projection_card(sources["sundowner_overlook"])
    pinhole_workflow, pinhole_workflow_metadata = _pinhole_workflow_card(
        sources["sundowner_overlook"]
    )
    processing, processing_metadata = _processing_card(sources["boma"])
    features, feature_metadata = _feature_card(sources["mosaic_tunnel"])
    semantics, semantic_metadata = _semantic_card(
        sources["boiler_room"], device_name=args.device
    )
    pose_range, pose_range_metadata = _pose_range_card(
        args.output_dir / "spherical-stereo-synthetic.png"
    )
    outputs = {
        "application-projection-sundowner.jpg": projection,
        "pinhole-workflow-sundowner.jpg": pinhole_workflow,
        "application-processing-boma.jpg": processing,
        "application-features-mosaic-tunnel.jpg": features,
        "application-pose-range.jpg": pose_range,
        "application-semantics-boiler-room.jpg": semantics,
    }
    for filename, image in outputs.items():
        _save_jpeg(image, args.output_dir / filename)
    derived = {
        filename: {
            "size_bytes": (args.output_dir / filename).stat().st_size,
            "sha256": sha256_file(args.output_dir / filename),
            "dimensions_wh": list(Image.open(args.output_dir / filename).size),
        }
        for filename in outputs
    }
    metadata = {
        "schema": "panorai-application-gallery/v1",
        "license": "CC0-1.0",
        "sources": [asdict(asset) for asset in SOURCE_ASSETS],
        "applications": {
            "projection": projection_metadata,
            "pinhole_workflow": pinhole_workflow_metadata,
            "sphere_native_processing": processing_metadata,
            "features_and_matching": feature_metadata,
            "semantic_localization": semantic_metadata,
            "pose_and_range": pose_range_metadata,
            "reconstruction_and_navigation": {
                "asset": "reconstruction-graph.svg",
                "evidence_type": "method diagram",
            },
        },
        "derived": derived,
        "distribution": (
            "source HDRIs and model checkpoints stay in external caches; only "
            "the listed documentation derivatives are tracked"
        ),
    }
    (args.output_dir / "application-gallery-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"derived": derived}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
