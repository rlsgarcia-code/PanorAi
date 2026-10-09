"""P74 adapters kept outside the reusable refinement mathematics."""

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

    R_scene_target = np.asarray(rotation_scene_from_target_p74, dtype=np.float64)
    t_scene_target = np.asarray(
        translation_scene_from_target_p74_m, dtype=np.float64
    ).reshape(3)
    R_scene_source = np.asarray(rotation_scene_from_source_p74, dtype=np.float64)
    t_scene_source = np.asarray(
        translation_scene_from_source_p74_m, dtype=np.float64
    ).reshape(3)
    for name, rotation in (
        ("target", R_scene_target),
        ("source", R_scene_source),
    ):
        if rotation.shape != (3, 3):
            raise ValueError(f"{name} rotation must be 3x3")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9):
            raise ValueError(f"{name} rotation must be orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-9):
            raise ValueError(f"{name} rotation must have determinant +1")
    adapter = P74_FROM_PANORAI
    rotation = adapter.T @ R_scene_source.T @ R_scene_target @ adapter
    translation = adapter.T @ R_scene_source.T @ (t_scene_target - t_scene_source)
    return RegisteredPose(
        rotation_source_from_target=rotation,
        translation_source_from_target_m=translation,
    )


def load_registered_pose(target_npz: Path, source_npz: Path) -> RegisteredPose:
    """Read the two small registered-pose members without loading organized XYZ."""

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
) -> tuple[np.ndarray, np.ndarray]:
    """Regrid P74 native-polar RGB without angular minification.

    P74 stores polar angles 0--150 degrees with an endpoint-inclusive
    longitude period.  ``output_shape_hw`` is the previously audited native
    angular-density ERP lattice, not an arbitrary model input resize.
    """

    if row_chunk < 1:
        raise ValueError("row_chunk must be positive")
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read {rgb_path}")
    source = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    source_height, source_width = source.shape[:2]
    period = source_width - 1
    height, width = output_shape_hw
    longitude = (np.arange(width, dtype=np.float64) + 0.5) / width * (
        2.0 * math.pi
    ) - math.pi
    map_x_row = np.mod(-longitude / (2.0 * math.pi) * period, period)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    support = np.empty((height, width), dtype=bool)
    shadow = np.asarray([124, 116, 104], dtype=np.uint8)
    for start in range(0, height, row_chunk):
        stop = min(height, start + row_chunk)
        rows = np.arange(start, stop, dtype=np.float64)
        polar = (rows + 0.5) / height * math.pi
        row_support = polar <= P74_POLAR_LIMIT_RAD + 1e-12
        map_x = np.broadcast_to(map_x_row[None], (stop - start, width)).astype(
            np.float32
        )
        map_y = np.broadcast_to(
            (polar / P74_POLAR_LIMIT_RAD * (source_height - 1))[:, None],
            map_x.shape,
        ).astype(np.float32)
        chunk = cv2.remap(
            source,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        chunk[~row_support] = shadow
        rgb[start:stop] = chunk
        support[start:stop] = row_support[:, None]
    return rgb, support


__all__ = [
    "P74_FROM_PANORAI",
    "P74_POLAR_LIMIT_RAD",
    "RegisteredPose",
    "load_native_angular_rgb",
    "load_registered_pose",
    "relative_pose_from_scene_transforms",
]
