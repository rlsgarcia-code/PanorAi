"""P74 adapters and metric helpers for the multi-scene landmark-BA benchmark.

The module deliberately keeps dataset conventions outside ``panorai``.  It
never infers validity from RGB or a zero depth sentinel, and it samples the
registered organized cloud only during evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

P74_SHADOW_ANGLE_DEG = 30.0
P74_POLAR_LIMIT_RAD = math.radians(150.0)
P74_FROM_PANORAI = np.asarray(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class RegisteredPose:
    rotation_source_from_target: np.ndarray
    translation_source_from_target_m: np.ndarray

    @property
    def source_center_in_target_m(self) -> np.ndarray:
        return (
            -self.rotation_source_from_target.T @ self.translation_source_from_target_m
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_bearings_to_native_pixels(
    bearings_p74: Any,
    shape_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Map P74-frame unit bearings to its endpoint-inclusive polar raster."""

    rays = np.asarray(bearings_p74, dtype=np.float64)
    if rays.ndim != 2 or rays.shape[1] != 3:
        raise ValueError("bearings_p74 must have shape (N, 3)")
    norms = np.linalg.norm(rays, axis=1)
    finite = np.all(np.isfinite(rays), axis=1) & (norms > 0.0)
    unit = np.zeros_like(rays)
    unit[finite] = rays[finite] / norms[finite, None]
    longitude = np.arctan2(-unit[:, 1], unit[:, 0])
    latitude = np.arcsin(np.clip(unit[:, 2], -1.0, 1.0))
    polar = math.pi / 2.0 - latitude
    height, width = shape_hw
    pixels = np.column_stack(
        (
            np.mod(longitude / (2.0 * math.pi) * (width - 1), width - 1),
            polar / P74_POLAR_LIMIT_RAD * (height - 1),
        )
    )
    support = finite & (polar >= -1e-12) & (polar <= P74_POLAR_LIMIT_RAD + 1e-12)
    return pixels, support


def canonicalize_p74_rgb_native(rgb_path: Path) -> tuple[Any, np.ndarray]:
    """Return a full-detail canonical ERP container without angular resizing."""

    from panorai.data import EquirectangularImage
    from panorai.geometry import erp_pixels_to_rays

    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read {rgb_path}")
    source = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    observed_height, width = source.shape[:2]
    materialized_height = int(
        round(observed_height / (1.0 - P74_SHADOW_ANGLE_DEG / 180.0))
    )
    observed_rgb = np.empty_like(source)
    columns = np.arange(width, dtype=np.float64)
    for row_start in range(0, observed_height, 64):
        row_stop = min(observed_height, row_start + 64)
        xx, yy = np.meshgrid(
            columns,
            np.arange(row_start, row_stop, dtype=np.float64),
        )
        canonical_pixels = np.column_stack((xx.ravel(), yy.ravel()))
        canonical_rays = erp_pixels_to_rays(
            canonical_pixels,
            (materialized_height, width),
        )
        native_pixels, support = scan_bearings_to_native_pixels(
            (P74_FROM_PANORAI @ canonical_rays.T).T,
            source.shape[:2],
        )
        if not support.all():
            raise RuntimeError("observed P74 rows escaped native polar support")
        observed_rgb[row_start:row_stop] = cv2.remap(
            source,
            native_pixels[:, 0].reshape(row_stop - row_start, width).astype(np.float32),
            native_pixels[:, 1].reshape(row_stop - row_start, width).astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
    panorama = EquirectangularImage(
        observed_rgb,
        shadow_angle=P74_SHADOW_ANGLE_DEG,
        shadow_padded=False,
    )
    panorama.preprocess()
    validity = np.asarray(panorama.validity("image"), dtype=bool)
    validity &= np.asarray(panorama.support_mask, dtype=bool)
    return panorama, validity


def load_registered_pose(target_npz: Path, source_npz: Path) -> RegisteredPose:
    """Load registered scanner transforms and convert them to PanorAi axes."""

    with np.load(target_npz) as target, np.load(source_npz) as source:
        target_t = target[
            "translation" if "translation" in target.files else "translation_vector"
        ]
        source_t = source[
            "translation" if "translation" in source.files else "translation_vector"
        ]
        rotation_scene_target = np.asarray(target["rotation_matrix"], dtype=np.float64)
        rotation_scene_source = np.asarray(source["rotation_matrix"], dtype=np.float64)
    adapter = P74_FROM_PANORAI
    rotation = adapter.T @ rotation_scene_source.T @ rotation_scene_target @ adapter
    translation = (
        adapter.T
        @ rotation_scene_source.T
        @ (np.asarray(target_t).reshape(3) - np.asarray(source_t).reshape(3))
    )
    return RegisteredPose(rotation, translation)


def sample_registered_radial_range(
    target_npz: Path,
    target_bearings_panorai: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample evaluation-only radial range from the organized target cloud."""

    rays = np.asarray(target_bearings_panorai, dtype=np.float64)
    with np.load(target_npz, mmap_mode="r") as data:
        xyz = np.asarray(data["xyz_image"])
        pixels, support = scan_bearings_to_native_pixels(
            (P74_FROM_PANORAI @ rays.T).T,
            xyz.shape[:2],
        )
        columns = np.mod(
            np.rint(pixels[:, 0]).astype(np.int64),
            xyz.shape[1] - 1,
        )
        rows = np.clip(
            np.rint(pixels[:, 1]).astype(np.int64),
            0,
            xyz.shape[0] - 1,
        )
        points = np.asarray(xyz[rows, columns], dtype=np.float64)
    radial = np.linalg.norm(points, axis=1)
    valid = (
        support
        & np.all(np.isfinite(points), axis=1)
        & np.isfinite(radial)
        & (radial > 1e-3)
    )
    radial[~valid] = np.nan
    return radial, valid


def sample_erp_nearest(array: Any, bearings: Any) -> np.ndarray:
    """Sample a canonical ERP array at unit bearings with periodic longitude."""

    from panorai.geometry import rays_to_erp_pixels

    values = np.asarray(array)
    pixels = rays_to_erp_pixels(
        np.asarray(bearings, dtype=np.float64),
        values.shape[:2],
    ).pixels_xy
    columns = np.mod(np.rint(pixels[:, 0]).astype(np.int64), values.shape[1])
    rows = np.clip(
        np.rint(pixels[:, 1]).astype(np.int64),
        0,
        values.shape[0] - 1,
    )
    return values[rows, columns]


def radial_metrics(
    prediction: Any, truth: Any, validity: Any
) -> dict[str, float | int]:
    """Return metric and scale-invariant sparse radial-range diagnostics."""

    pred = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    selected = (
        np.asarray(validity, dtype=bool)
        & np.isfinite(pred)
        & np.isfinite(target)
        & (pred > 0.0)
        & (target > 0.0)
    )
    if not selected.any():
        return {"count": 0}
    pred = pred[selected]
    target = target[selected]
    log_error = np.log(pred) - np.log(target)
    scale = float(np.exp(-np.median(log_error)))
    aligned = pred * scale
    ratio = np.maximum(pred / target, target / pred)
    return {
        "count": int(pred.size),
        "abs_rel": float(np.mean(np.abs(pred - target) / target)),
        "rmse_m": float(np.sqrt(np.mean((pred - target) ** 2))),
        "delta_1": float(np.mean(ratio < 1.25)),
        "log_rmse": float(np.sqrt(np.mean(log_error**2))),
        "si_log_rmse": float(np.sqrt(np.mean((log_error - log_error.mean()) ** 2))),
        "median_scale": scale,
        "scale_aligned_abs_rel": float(np.mean(np.abs(aligned - target) / target)),
    }


def rotation_error_deg(observed: Any, reference: Any) -> float:
    delta = np.asarray(observed) @ np.asarray(reference).T
    cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def vector_angle_deg(first: Any, second: Any) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    cosine = float(a @ b) / float(np.linalg.norm(a) * np.linalg.norm(b))
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


__all__ = [
    "P74_FROM_PANORAI",
    "RegisteredPose",
    "canonicalize_p74_rgb_native",
    "load_registered_pose",
    "radial_metrics",
    "rotation_error_deg",
    "sample_erp_nearest",
    "sample_registered_radial_range",
    "scan_bearings_to_native_pixels",
    "sha256",
    "vector_angle_deg",
]
