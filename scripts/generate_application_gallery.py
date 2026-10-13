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
        url=(
            "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/"
            "boiler_room_2k.hdr"
        ),
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
        slug="modern_buildings",
        title="Modern Buildings",
        filename="modern_buildings_2k.hdr",
        url=(
            "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/"
            "modern_buildings_2k.hdr"
        ),
        page_url="https://polyhaven.com/a/modern_buildings",
        authors=("Greg Zaal",),
        size_bytes=6_356_062,
        sha256="bdcd68558e138d5c3071dc0f565ad75b33953f589ff74a21d4ea5d8d9c15d87d",
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
)

CARD_SIZE = (1200, 600)
ATLAS_SIZE = (1600, 900)


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


def _erp_quadrants(images: tuple[np.ndarray, ...]) -> Image.Image:
    if len(images) != 4:
        raise ValueError("exactly four ERP panels are required")
    canvas = Image.new("RGB", CARD_SIZE, (15, 23, 42))
    for index, rgb in enumerate(images):
        canvas.paste(_resize(rgb, (600, 300)), ((index % 2) * 600, index // 2 * 300))
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
    audit = np.concatenate((error[:, : rgb.shape[1] // 2], support_rgb[:, rgb.shape[1] // 2 :]), axis=1)

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


def _processing_card(rgb: np.ndarray) -> tuple[Image.Image, dict[str, object]]:
    from panorai.image_processing import (
        spherical_canny,
        spherical_gaussian_blur,
        spherical_gradient,
    )

    floating = rgb.astype(np.float32) / 255.0
    gray = cv2.cvtColor(floating, cv2.COLOR_RGB2GRAY)
    smoothed = spherical_gaussian_blur(floating, ksize=9, sigma=2.0)
    gradient = spherical_gradient(gray, operator="scharr")
    gradient_scale = max(float(np.percentile(gradient.magnitude, 99.5)), 1e-6)
    gradient_u8 = np.rint(
        np.clip(gradient.magnitude / gradient_scale, 0.0, 1.0) * 255
    ).astype(np.uint8)
    gradient_rgb = cv2.cvtColor(
        cv2.applyColorMap(gradient_u8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB
    )
    edges = spherical_canny(
        gray,
        0.035,
        0.085,
        gaussian_ksize=5,
        gaussian_sigma=1.2,
        gradient_operator="scharr",
    )
    smooth_u8 = np.rint(np.clip(smoothed, 0.0, 1.0) * 255).astype(np.uint8)
    edge_rgb = np.repeat(edges[..., None], 3, axis=2)
    return _erp_quadrants((rgb, smooth_u8, gradient_rgb, edge_rgb)), {
        "working_shape_hw": list(rgb.shape[:2]),
        "quadrants": [
            "source ERP",
            "spherical Gaussian 9x9 sigma 2.0",
            "tangent Scharr magnitude",
            "geodesic Canny",
        ],
        "gradient_display_percentile_99_5": gradient_scale,
        "canny_thresholds": [0.035, 0.085],
        "canny_edge_pixels": int(np.count_nonzero(edges)),
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
    shift_pixels = rgb.shape[1] // 18
    shifted = np.roll(rgb, shift_pixels, axis=1)
    pipeline = _feature_pipeline()
    features_a = pipeline.extract(rgb, panorama_id="modern-buildings-a")
    features_b = pipeline.extract(shifted, panorama_id="modern-buildings-yaw")
    matches = pipeline.match(features_a, features_b)

    canvas = Image.new("RGB", CARD_SIZE, (15, 23, 42))
    canvas.paste(_resize(rgb, (1200, 300)), (0, 0))
    canvas.paste(_resize(shifted, (1200, 300)), (0, 300))
    draw = ImageDraw.Draw(canvas, "RGBA")
    valid = np.flatnonzero(matches.valid)
    ranked = (
        valid[np.argsort(matches.descriptor_distances[valid])]
        if valid.size
        else valid
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
        seam_margin = 0.08 * rgb.shape[1]
        if point_a[0] < seam_margin or point_a[0] > rgb.shape[1] - seam_margin:
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
        "plotted_source_seam_matches": seam_matches,
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
    color[~low] = (
        (1.0 - fraction[~low, None]) * yellow + fraction[~low, None] * red
    )
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


def _semantic_card(
    rgb: np.ndarray,
    *,
    device_name: str,
) -> tuple[Image.Image, dict[str, object]]:
    import torch

    from panorai.experimental.deep_learning import (
        ImageNetFCN,
        load_pretrained_imagenet_model,
        port_module_with_report,
        regrid_dense_lattice_to_erp,
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
        torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None].to(device)
        / 255.0
    )
    transform = loaded.weights.transforms()
    mean = torch.tensor(transform.mean, device=device)[None, :, None, None]
    std = torch.tensor(transform.std, device=device)[None, :, None, None]
    values = (values - mean) / std
    with torch.inference_mode():
        dense = model.forward_dense(values, spherical_average=True)
        global_scores = model(values, spherical_average=True)[0]
        regridded = regrid_dense_lattice_to_erp(
            dense.logits[:, 743:744], rgb.shape[:2], model.lattice_geometry
        )
    raw = regridded.values[0, 0].float().cpu().numpy()
    support = regridded.support.cpu().numpy().astype(bool)
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

    component_view = rgb.copy()
    component_view[~component] = np.rint(0.28 * component_view[~component]).astype(
        np.uint8
    )
    proposal = Image.fromarray(rgb).convert("RGB")
    proposal_draw = ImageDraw.Draw(proposal, "RGBA")
    radius_x = 70.0 / 360.0 * rgb.shape[1]
    radius_y = 70.0 / 180.0 * rgb.shape[0]
    for offset in (-rgb.shape[1], 0, rgb.shape[1]):
        proposal_draw.ellipse(
            (
                column_center - radius_x + offset,
                row_center - radius_y,
                column_center + radius_x + offset,
                row_center + radius_y,
            ),
            outline=(250, 204, 21, 255),
            width=5,
        )
    proposal_rgb = np.asarray(proposal)
    colorbar = _blue_yellow_red(np.linspace(0.0, 1.0, rgb.shape[1])[None])
    colorbar = np.repeat(colorbar, 18, axis=0)
    overlay[-18:] = colorbar

    card = _erp_quadrants((rgb, overlay, component_view, proposal_rgb))
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
        "quadrants": [
            "source ERP",
            "blue-yellow-red normalized CAM",
            "peak-connected spherical CAM component",
            "70 degree proposed FOV around component mean",
        ],
    }


def _method_thumbnail(size: tuple[int, int]) -> Image.Image:
    image = Image.new("RGB", size, (15, 23, 42))
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = size
    centres = ((int(width * 0.23), int(height * 0.55)), (int(width * 0.77), int(height * 0.55)))
    point = (int(width * 0.5), int(height * 0.30))
    for centre in centres:
        radius = int(min(width, height) * 0.20)
        draw.ellipse(
            (centre[0] - radius, centre[1] - radius, centre[0] + radius, centre[1] + radius),
            outline=(56, 189, 248, 255),
            width=max(2, width // 220),
        )
        draw.line((*centre, *point), fill=(250, 204, 21, 220), width=max(2, width // 260))
    draw.ellipse(
        (point[0] - 8, point[1] - 8, point[0] + 8, point[1] + 8),
        fill=(244, 114, 182, 255),
    )
    draw.line((*centres[0], *centres[1]), fill=(148, 163, 184, 170), width=2)
    return image


def _atlas(
    cards: tuple[Image.Image, Image.Image, Image.Image, Image.Image],
    stereo_path: Path,
) -> Image.Image:
    stereo = Image.open(stereo_path).convert("RGB")
    method = _method_thumbnail(CARD_SIZE)
    panels = (*cards, stereo, method)
    atlas = Image.new("RGB", ATLAS_SIZE, (15, 23, 42))
    cell = (520, 260)
    x_positions = (20, 540, 1060)
    y_positions = (180, 460)
    for index, panel in enumerate(panels):
        fitted = panel.resize(cell, Image.Resampling.LANCZOS)
        atlas.paste(fitted, (x_positions[index % 3], y_positions[index // 3]))
    return atlas


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
    sources = {
        slug: _tone_map(path, height=512) for slug, path in paths.items()
    }
    projection, projection_metadata = _projection_card(
        sources["sundowner_overlook"]
    )
    processing, processing_metadata = _processing_card(sources["boma"])
    features, feature_metadata = _feature_card(sources["modern_buildings"])
    semantics, semantic_metadata = _semantic_card(
        sources["boiler_room"], device_name=args.device
    )
    outputs = {
        "application-projection-sundowner.jpg": projection,
        "application-processing-boma.jpg": processing,
        "application-features-modern-buildings.jpg": features,
        "application-semantics-boiler-room.jpg": semantics,
    }
    for filename, image in outputs.items():
        _save_jpeg(image, args.output_dir / filename)
    overview = _atlas(
        (projection, processing, features, semantics),
        args.output_dir / "spherical-stereo-synthetic.png",
    )
    _save_jpeg(overview, args.output_dir / "applications-overview.jpg")
    derived = {
        filename: {
            "size_bytes": (args.output_dir / filename).stat().st_size,
            "sha256": sha256_file(args.output_dir / filename),
            "dimensions_wh": list(Image.open(args.output_dir / filename).size),
        }
        for filename in (*outputs, "applications-overview.jpg")
    }
    metadata = {
        "schema": "panorai-application-gallery/v1",
        "license": "CC0-1.0",
        "sources": [asdict(asset) for asset in SOURCE_ASSETS],
        "applications": {
            "projection": projection_metadata,
            "sphere_native_processing": processing_metadata,
            "features_and_matching": feature_metadata,
            "semantic_localization": semantic_metadata,
            "pose_and_range": {
                "asset": "spherical-stereo-synthetic.png",
                "evidence_type": "analytic fixture",
            },
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
