#!/usr/bin/env python3
"""Generate the real-texture spherical-stereo documentation panel."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np

from panorai.geometry import erp_pixels_to_rays, rays_to_erp_pixels
from panorai.stereo import (
    SphericalStereoOptions,
    estimate_spherical_range,
    render_spherical_stereo_result,
)

SOURCE_HDR_SHA256 = "dfc8505761018d644997a803f332339328af2b53e82023fb3bec37cefdf43b83"


def _tone_map_source(source_hdr: Path, texture_path: Path) -> None:
    if hashlib.sha256(source_hdr.read_bytes()).hexdigest() != SOURCE_HDR_SHA256:
        raise ValueError("source HDRI checksum does not match Poly Haven Studio")
    bgr = cv2.imread(str(source_hdr), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("expected a three-channel HDR panorama")
    rgb = cv2.cvtColor(bgr.astype(np.float32), cv2.COLOR_BGR2RGB)
    exposure = float(np.percentile(rgb, 99.5))
    mapped = np.clip(rgb / max(exposure, np.finfo(np.float32).eps), 0.0, 1.0)
    mapped = np.power(mapped, 1.0 / 2.2)
    texture = np.rint(mapped * 255.0).astype(np.uint8)
    texture_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(texture_path),
        cv2.cvtColor(texture, cv2.COLOR_RGB2BGR),
        (cv2.IMWRITE_JPEG_QUALITY, 92),
    ):
        raise RuntimeError(f"failed to write {texture_path}")


def _render_sphere(
    center: np.ndarray,
    rays: np.ndarray,
    texture: np.ndarray,
    radius: float = 4.0,
    sphere_center: tuple[float, float, float] = (0.6, -0.25, 0.35),
) -> tuple[np.ndarray, np.ndarray]:
    offset = center - np.asarray(sphere_center, dtype=np.float32)
    projection = np.sum(rays * offset, axis=-1)
    radial_range = -projection + np.sqrt(
        projection * projection + radius * radius - float(offset @ offset)
    )
    points = center + radial_range[..., None] * rays
    texture_rays = (points - np.asarray(sphere_center, dtype=np.float32)) / radius
    texture_pixels = rays_to_erp_pixels(texture_rays, texture.shape[:2]).pixels_xy
    padded = np.concatenate((texture[:, -1:], texture, texture[:, :1]), axis=1)
    image = cv2.remap(
        padded,
        np.asarray(texture_pixels[..., 0] + 1.0, dtype=np.float32),
        np.asarray(texture_pixels[..., 1], dtype=np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return image.astype(np.float32), radial_range.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/_static/tutorials/spherical-stereo-synthetic.png"),
    )
    parser.add_argument(
        "--texture",
        type=Path,
        default=Path("docs/_static/tutorials/poly-haven-studio-erp.jpg"),
    )
    parser.add_argument(
        "--source-hdr",
        type=Path,
        help="optional downloaded Poly Haven Studio 1K HDR used to rebuild texture",
    )
    args = parser.parse_args()

    if args.source_hdr is not None:
        _tone_map_source(args.source_hdr, args.texture)
    texture_bgr = cv2.imread(str(args.texture), cv2.IMREAD_COLOR)
    if texture_bgr is None:
        raise RuntimeError(f"failed to read {args.texture}")
    texture = cv2.cvtColor(texture_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    shape_hw = (192, 384)
    y, x = np.indices(shape_hw, dtype=np.float32)
    rays = erp_pixels_to_rays(np.stack((x, y), axis=-1), shape_hw)
    center_b_in_a = np.asarray((0.35, 0.03, 0.12), dtype=np.float32)
    first, reference_range = _render_sphere(np.zeros(3), rays, texture)
    second, _ = _render_sphere(center_b_in_a, rays, texture)
    result = estimate_spherical_range(
        first,
        second,
        np.eye(3),
        -center_b_in_a,
        options=SphericalStereoOptions(
            min_range=2.5,
            max_range=5.5,
            num_hypotheses=128,
            window_size=5,
            pole_margin_fraction=0.05,
            min_texture_std=0.003,
            min_confidence=0.0,
            max_matching_cost=0.9,
            consistency_relative_tolerance=0.08,
            consistency_absolute_tolerance=0.08,
        ),
    )
    panel = render_spherical_stereo_result(
        first,
        result,
        target_erp=second,
        reference_range=reference_range,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write {args.output}")
    print(args.output)


if __name__ == "__main__":
    main()
