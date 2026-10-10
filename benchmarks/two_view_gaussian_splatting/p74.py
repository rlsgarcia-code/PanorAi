"""P74 input adapters local to the Gaussian benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

P74_POLAR_LIMIT_RAD = math.radians(150.0)
P74_FROM_PANORAI = np.asarray(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class RegisteredPose:
    rotation_source_from_target: np.ndarray
    translation_source_from_target_m: np.ndarray
    convention: str = "X_source = R_source_from_target @ X_target + t"


def relative_pose_from_scene_transforms(
    rotation_scene_from_target_p74: Any,
    translation_scene_from_target_p74_m: Any,
    rotation_scene_from_source_p74: Any,
    translation_scene_from_source_p74_m: Any,
) -> RegisteredPose:
    """Convert registered P74 scan poses into the canonical PanorAi frame."""

    rotation_scene_target = np.asarray(rotation_scene_from_target_p74, dtype=np.float64)
    translation_scene_target = np.asarray(
        translation_scene_from_target_p74_m, dtype=np.float64
    ).reshape(3)
    rotation_scene_source = np.asarray(rotation_scene_from_source_p74, dtype=np.float64)
    translation_scene_source = np.asarray(
        translation_scene_from_source_p74_m, dtype=np.float64
    ).reshape(3)
    for name, rotation in (
        ("target", rotation_scene_target),
        ("source", rotation_scene_source),
    ):
        if rotation.shape != (3, 3):
            raise ValueError(f"{name} rotation must be 3x3")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9):
            raise ValueError(f"{name} rotation must be orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-9):
            raise ValueError(f"{name} rotation must have determinant +1")
    adapter = P74_FROM_PANORAI
    rotation = adapter.T @ rotation_scene_source.T @ rotation_scene_target @ adapter
    translation = (
        adapter.T
        @ rotation_scene_source.T
        @ (translation_scene_target - translation_scene_source)
    )
    return RegisteredPose(
        rotation_source_from_target=rotation,
        translation_source_from_target_m=translation,
    )


def load_registered_pose(target_npz: Path, source_npz: Path) -> RegisteredPose:
    """Read the two registered-pose members without loading organized XYZ."""

    with np.load(target_npz) as target, np.load(source_npz) as source:
        target_translation_key = (
            "translation" if "translation" in target.files else "translation_vector"
        )
        source_translation_key = (
            "translation" if "translation" in source.files else "translation_vector"
        )
        return relative_pose_from_scene_transforms(
            target["rotation_matrix"],
            target[target_translation_key],
            source["rotation_matrix"],
            source[source_translation_key],
        )


def load_native_angular_rgb(
    rgb_path: Path,
    output_shape_hw: tuple[int, int],
    *,
    row_chunk: int = 64,
    antialias_samples: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Regrid P74 native-polar RGB onto a canonical 2:1 ERP lattice.

    ``antialias_samples`` is the number of stratified samples per output-pixel
    axis.  Values greater than one integrate the native image footprint before
    reducing resolution instead of point-sampling it with bilinear filtering.
    """

    if row_chunk < 1:
        raise ValueError("row_chunk must be positive")
    if antialias_samples < 1:
        raise ValueError("antialias_samples must be positive")
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read {rgb_path}")
    source = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    source_height, source_width = source.shape[:2]
    period = source_width - 1
    height, width = output_shape_hw
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    support = np.empty((height, width), dtype=bool)
    shadow = np.asarray([124, 116, 104], dtype=np.uint8)
    subpixel_offsets = (
        np.arange(antialias_samples, dtype=np.float64) + 0.5
    ) / antialias_samples - 0.5
    for start in range(0, height, row_chunk):
        stop = min(height, start + row_chunk)
        rows = np.arange(start, stop, dtype=np.float64)
        center_polar = (rows + 0.5) / height * math.pi
        row_support = center_polar <= P74_POLAR_LIMIT_RAD + 1e-12
        accumulated = np.zeros((stop - start, width, 3), dtype=np.float32)
        for row_offset in subpixel_offsets:
            polar = (rows + 0.5 + row_offset) / height * math.pi
            sample_support = polar <= P74_POLAR_LIMIT_RAD + 1e-12
            map_y_row = polar / P74_POLAR_LIMIT_RAD * (source_height - 1)
            for column_offset in subpixel_offsets:
                longitude = (
                    np.arange(width, dtype=np.float64) + 0.5 + column_offset
                ) / width * (2.0 * math.pi) - math.pi
                map_x_row = np.mod(-longitude / (2.0 * math.pi) * period, period)
                map_x = np.broadcast_to(map_x_row[None], (stop - start, width)).astype(
                    np.float32
                )
                map_y = np.broadcast_to(map_y_row[:, None], map_x.shape).astype(
                    np.float32
                )
                sample = cv2.remap(
                    source,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_WRAP,
                )
                sample[~sample_support] = shadow
                accumulated += sample.astype(np.float32)
        accumulated /= float(antialias_samples * antialias_samples)
        chunk = np.clip(np.rint(accumulated), 0, 255).astype(np.uint8)
        chunk[~row_support] = shadow
        rgb[start:stop] = chunk
        support[start:stop] = row_support[:, None]
    return rgb, support


def load_native_registered_radial(
    npz_path: Path,
    output_shape_hw: tuple[int, int],
    *,
    antialias_samples: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Regrid an observed P74 XYZ surface into canonical ERP radial range.

    This is an evaluation/oracle adapter.  It reads the registered scan's
    organized XYZ image but returns range in that camera's local frame; the
    registered pose is still applied separately by the renderer.  No missing
    sample is interpolated outside the native polar support.
    """

    if antialias_samples < 1:
        raise ValueError("antialias_samples must be positive")
    with np.load(npz_path) as archive:
        xyz = np.asarray(archive["xyz_image"])
        if xyz.ndim != 3 or xyz.shape[2] != 3:
            raise ValueError("xyz_image must have shape (H, W, 3)")
        source_height, source_width = xyz.shape[:2]
        period = source_width - 1
        height, width = output_shape_hw
        accumulated = np.zeros((height, width), dtype=np.float64)
        valid_count = np.zeros((height, width), dtype=np.int16)
        subpixel_offsets = (
            np.arange(antialias_samples, dtype=np.float64) + 0.5
        ) / antialias_samples - 0.5
        for row_offset in subpixel_offsets:
            polar = (
                (np.arange(height, dtype=np.float64) + 0.5 + row_offset)
                / height
                * math.pi
            )
            sample_support = polar <= P74_POLAR_LIMIT_RAD + 1e-12
            map_y_row = polar / P74_POLAR_LIMIT_RAD * (source_height - 1)
            for column_offset in subpixel_offsets:
                longitude = (
                    np.arange(width, dtype=np.float64) + 0.5 + column_offset
                ) / width * (2.0 * math.pi) - math.pi
                map_x_row = np.mod(-longitude / (2.0 * math.pi) * period, period)
                map_x = np.broadcast_to(map_x_row[None], (height, width)).astype(
                    np.float32
                )
                map_y = np.broadcast_to(map_y_row[:, None], map_x.shape).astype(
                    np.float32
                )
                sampled_xyz = cv2.remap(
                    xyz,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_WRAP,
                )
                sampled_range = np.linalg.norm(sampled_xyz, axis=2)
                sample_valid = (
                    sample_support[:, None]
                    & np.isfinite(sampled_range)
                    & (sampled_range > 0.01)
                )
                accumulated[sample_valid] += sampled_range[sample_valid]
                valid_count[sample_valid] += 1
    valid = valid_count > 0
    radial = np.full(output_shape_hw, np.nan, dtype=np.float32)
    radial[valid] = (accumulated[valid] / valid_count[valid]).astype(np.float32)
    return radial, valid


__all__ = [
    "P74_FROM_PANORAI",
    "P74_POLAR_LIMIT_RAD",
    "RegisteredPose",
    "load_native_angular_rgb",
    "load_native_registered_radial",
    "load_registered_pose",
    "relative_pose_from_scene_transforms",
]
