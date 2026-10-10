#!/usr/bin/env python3
"""Export the centres of two observed Gaussian surfaces as a fused RGB PLY."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.two_view_gaussian_splatting.p74 import (  # noqa: E402
    load_native_angular_rgb,
)

TARGET_ID = "P-74+MD-05_concluido_326+G046"
SOURCE_ID = "P-74+MD-05_concluido_326+G047"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backproject the centres of the G046/G047 Gaussian surfaces, "
            "transform them into the G046 frame, fuse duplicate voxels, and "
            "write a binary RGB PLY."
        )
    )
    parser.add_argument("--p74-root", required=True, type=Path)
    parser.add_argument("--surface-run", required=True, type=Path)
    parser.add_argument("--refined-pose", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--voxel-size-m", type=float, default=0.01)
    parser.add_argument("--min-range-m", type=float, default=0.3)
    parser.add_argument("--max-range-m", type=float, default=15.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not np.isfinite(args.voxel_size_m) or args.voxel_size_m <= 0.0:
        raise SystemExit("voxel-size-m must be positive and finite")
    if args.max_range_m <= args.min_range_m:
        raise SystemExit("max-range-m must exceed min-range-m")

    target_radial = np.load(args.surface_run / "stereo-surface-radial-m.npy")
    target_valid = np.load(args.surface_run / "stereo-surface-valid.npy")
    source_radial = np.load(args.surface_run / "source-stereo-surface-radial-m.npy")
    source_valid = np.load(args.surface_run / "source-stereo-surface-valid.npy")
    shape = tuple(target_radial.shape)
    if (
        target_valid.shape != shape
        or source_radial.shape != shape
        or source_valid.shape != shape
    ):
        raise SystemExit("all surface arrays must share one HW shape")

    images = args.p74_root / "images"
    target_rgb, target_support = load_native_angular_rgb(
        images / f"{TARGET_ID}_rgb.png", shape
    )
    source_rgb, source_support = load_native_angular_rgb(
        images / f"{SOURCE_ID}_rgb.png", shape
    )
    pose = json.loads(args.refined_pose.read_text(encoding="utf-8"))
    rotation_source_from_target = np.asarray(
        pose["rotation_source_from_target"], dtype=np.float64
    )
    translation_source_from_target = np.asarray(
        pose["translation_source_from_target_m"], dtype=np.float64
    )

    target_mask = _range_mask(
        target_radial,
        np.asarray(target_valid, dtype=bool) & target_support,
        args.min_range_m,
        args.max_range_m,
    )
    source_mask = _range_mask(
        source_radial,
        np.asarray(source_valid, dtype=bool) & source_support,
        args.min_range_m,
        args.max_range_m,
    )
    target_points = backproject_erp_centres(target_radial, target_mask)
    source_points = backproject_erp_centres(source_radial, source_mask)
    source_points_target = transform_source_points_to_target(
        source_points,
        rotation_source_from_target,
        translation_source_from_target,
    )
    points = np.concatenate((target_points, source_points_target), axis=0)
    colors = np.concatenate((target_rgb[target_mask], source_rgb[source_mask]), axis=0)
    view_bits = np.concatenate(
        (
            np.ones(target_points.shape[0], dtype=np.uint8),
            np.full(source_points.shape[0], 2, dtype=np.uint8),
        )
    )
    fused_points, fused_colors, observations, fused_view_bits = fuse_voxels(
        points,
        colors,
        view_bits,
        voxel_size_m=args.voxel_size_m,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_binary_rgb_ply(
        args.output,
        fused_points,
        fused_colors,
        observations,
        fused_view_bits,
    )
    if args.preview is not None:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        write_point_cloud_preview(args.preview, fused_points, fused_colors)
    metadata = {
        "schema": "panorai-two-view-gaussian-centres/v1",
        "coordinate_frame": TARGET_ID,
        "source_views": [TARGET_ID, SOURCE_ID],
        "point_semantics": "voxel-averaged Gaussian centres; no synthetic samples",
        "shape_hw": list(shape),
        "voxel_size_m": args.voxel_size_m,
        "range_m": [args.min_range_m, args.max_range_m],
        "input": {
            "target_points": int(target_points.shape[0]),
            "source_points": int(source_points.shape[0]),
            "total_points": int(points.shape[0]),
        },
        "output": {
            "points": int(fused_points.shape[0]),
            "observed_by_both_views": int(np.count_nonzero(fused_view_bits == 3)),
            "bounds_min_m": fused_points.min(axis=0).astype(float).tolist(),
            "bounds_max_m": fused_points.max(axis=0).astype(float).tolist(),
            "ply_path": str(args.output),
            "ply_sha256": _sha256(args.output),
            "ply_size_bytes": args.output.stat().st_size,
            "preview_path": str(args.preview) if args.preview is not None else None,
        },
        "inputs": {
            "surface_run": str(args.surface_run),
            "refined_pose": str(args.refined_pose),
            "refined_pose_sha256": _sha256(args.refined_pose),
        },
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return 0


def backproject_erp_centres(radial_range_m: Any, valid_hw: Any) -> np.ndarray:
    """Backproject valid ERP pixel-centre ranges into camera-frame XYZ."""

    radial = np.asarray(radial_range_m, dtype=np.float32)
    valid = np.asarray(valid_hw, dtype=bool)
    if radial.ndim != 2 or valid.shape != radial.shape:
        raise ValueError("radial_range_m and valid_hw must share shape HW")
    height, width = radial.shape
    rows, columns = np.nonzero(valid)
    latitude = np.pi / 2.0 - (rows.astype(np.float64) + 0.5) / height * np.pi
    longitude = (columns.astype(np.float64) + 0.5) / width * (2.0 * np.pi) - np.pi
    cos_latitude = np.cos(latitude)
    rays = np.column_stack(
        (
            cos_latitude * np.sin(longitude),
            np.sin(latitude),
            cos_latitude * np.cos(longitude),
        )
    )
    return (rays * radial[valid][:, None]).astype(np.float32)


def transform_source_points_to_target(
    points_source: Any,
    rotation_source_from_target: Any,
    translation_source_from_target: Any,
) -> np.ndarray:
    """Apply the inverse of X_source = R X_target + t to row-vector points."""

    points = np.asarray(points_source, dtype=np.float64)
    rotation = np.asarray(rotation_source_from_target, dtype=np.float64)
    translation = np.asarray(translation_source_from_target, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_source must have shape (N, 3)")
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("pose must have shapes R=(3,3), t=(3,)")
    return ((points - translation[None]) @ rotation).astype(np.float32)


def fuse_voxels(
    points_xyz: Any,
    colors_rgb: Any,
    view_bits: Any,
    *,
    voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average Gaussian centres and RGB values that occupy the same voxel."""

    points = np.asarray(points_xyz, dtype=np.float32)
    colors = np.asarray(colors_rgb, dtype=np.uint8)
    bits = np.asarray(view_bits, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3)")
    if colors.shape != points.shape or bits.shape != (points.shape[0],):
        raise ValueError("colors/view_bits must match points")
    if points.shape[0] == 0:
        raise ValueError("at least one point is required")
    keys = np.floor(points.astype(np.float64) / voxel_size_m).astype(np.int32)
    _, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    voxel_count = counts.size
    fused_points = np.empty((voxel_count, 3), dtype=np.float32)
    fused_colors = np.empty((voxel_count, 3), dtype=np.uint8)
    for axis in range(3):
        coordinate_sum = np.bincount(
            inverse, weights=points[:, axis], minlength=voxel_count
        )
        fused_points[:, axis] = coordinate_sum / counts
        color_sum = np.bincount(inverse, weights=colors[:, axis], minlength=voxel_count)
        fused_colors[:, axis] = np.clip(np.rint(color_sum / counts), 0, 255).astype(
            np.uint8
        )
    fused_bits = np.zeros(voxel_count, dtype=np.uint8)
    np.bitwise_or.at(fused_bits, inverse, bits)
    observations = np.minimum(counts, np.iinfo(np.uint16).max).astype(np.uint16)
    return fused_points, fused_colors, observations, fused_bits


def write_binary_rgb_ply(
    path: Path,
    points_xyz: Any,
    colors_rgb: Any,
    observations: Any,
    view_bits: Any,
) -> None:
    """Write a little-endian PLY with RGB and fusion provenance fields."""

    points = np.asarray(points_xyz, dtype=np.float32)
    colors = np.asarray(colors_rgb, dtype=np.uint8)
    counts = np.asarray(observations, dtype=np.uint16)
    bits = np.asarray(view_bits, dtype=np.uint8)
    vertex = np.empty(
        points.shape[0],
        dtype=np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
                ("observations", "<u2"),
                ("view_mask", "u1"),
            ]
        ),
    )
    vertex["x"], vertex["y"], vertex["z"] = points.T
    vertex["red"], vertex["green"], vertex["blue"] = colors.T
    vertex["observations"] = counts
    vertex["view_mask"] = bits
    header = "\n".join(
        (
            "ply",
            "format binary_little_endian 1.0",
            "comment PanorAi two-view Gaussian centres in G046 frame",
            f"element vertex {points.shape[0]}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property ushort observations",
            "property uchar view_mask",
            "end_header",
            "",
        )
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        vertex.tofile(stream)


def write_point_cloud_preview(
    path: Path,
    points_xyz: Any,
    colors_rgb: Any,
    *,
    max_points: int = 180_000,
) -> None:
    """Write deterministic perspective/top/side previews of a point cloud."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = np.asarray(points_xyz, dtype=np.float32)
    colors = np.asarray(colors_rgb, dtype=np.uint8)
    stride = max(1, int(np.ceil(points.shape[0] / max_points)))
    sample = points[::stride]
    sample_colors = colors[::stride].astype(np.float32) / 255.0
    robust_min = np.quantile(points, 0.005, axis=0)
    robust_max = np.quantile(points, 0.995, axis=0)
    centre = 0.5 * (robust_min + robust_max)
    half_span = 0.55 * float(np.max(robust_max - robust_min))
    figure = plt.figure(figsize=(18, 6), facecolor="#050607")
    views = (
        ("Perspective", 18.0, -62.0),
        ("Top", 90.0, -90.0),
        ("Front", 0.0, -90.0),
    )
    for index, (title, elevation, azimuth) in enumerate(views, start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        axis.set_facecolor("#050607")
        axis.scatter(
            sample[:, 0],
            sample[:, 2],
            sample[:, 1],
            c=sample_colors,
            s=0.32,
            linewidths=0.0,
            depthshade=False,
        )
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_xlim(centre[0] - half_span, centre[0] + half_span)
        axis.set_ylim(centre[2] - half_span, centre[2] + half_span)
        axis.set_zlim(centre[1] - half_span, centre[1] + half_span)
        axis.set_title(title, color="white", pad=8)
        axis.set_axis_off()
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.94, wspace=0.01)
    figure.savefig(path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def _range_mask(
    radial: np.ndarray, valid: np.ndarray, minimum: float, maximum: float
) -> np.ndarray:
    return valid & np.isfinite(radial) & (radial >= minimum) & (radial <= maximum)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
