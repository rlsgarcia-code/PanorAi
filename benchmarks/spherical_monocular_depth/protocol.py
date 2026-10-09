"""Auditable helpers for the P74 spherical monocular-depth experiment.

This is benchmark code, not a PanorAi public API.  Model code, checkpoints,
P74 data, and generated predictions intentionally remain external.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import struct
from typing import Any
import zipfile

import numpy as np


SCHEMA = "panorai-p74-spherical-monocular-depth-cnn/v1"
P74_ADAPTER = "eq-native-polar-0-150-endpoint-inclusive/v1"
P74_POLAR_LIMIT_RAD = math.radians(150.0)
P74_FROM_PANORAI = np.asarray(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
FROZEN_SAMPLE = (
    "P-74+MD-04_concluido_408+W_121",
    "P-74+MD-05_concluido_326+G046",
    "P-74+MD-08_missing_files+M-014",
)
MODEL_DEPTH_RANGE_M = (0.3, 150.0)
MODEL_LATTICE_MULTIPLE = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mmap_npy_member(path: Path, member: str) -> np.memmap:
    """Memory-map one uncompressed NPY member of a P74 NPZ."""

    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{path}:{member} is compressed; direct mmap is unsafe")
        header_offset = info.header_offset
    with path.open("rb") as stream:
        stream.seek(header_offset)
        header = stream.read(30)
        name_length = struct.unpack_from("<H", header, 26)[0]
        extra_length = struct.unpack_from("<H", header, 28)[0]
        stream.seek(header_offset + 30 + name_length + extra_length)
        version = np.lib.format.read_magic(stream)
        reader = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        shape, fortran_order, dtype = reader(stream)
        offset = stream.tell()
    return np.memmap(
        path,
        mode="r",
        dtype=dtype,
        offset=offset,
        shape=shape,
        order="F" if fortran_order else "C",
    )


@dataclass(frozen=True)
class P74Frame:
    panorama_id: str
    family: str
    rgb: np.ndarray
    radial_range: np.ndarray
    validity: np.ndarray
    source_shape_hw: tuple[int, int]


def load_p74_frame(
    npz_path: Path,
    rgb_path: Path,
    output_shape_hw: tuple[int, int],
    *,
    row_chunk: int = 64,
) -> P74Frame:
    """Sample native P74 RGB/XYZ onto a canonical pixel-centre ERP grid."""

    import cv2

    xyz = mmap_npy_member(npz_path, "xyz_image.npy")
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read {rgb_path}")
    source_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if xyz.shape != (*source_rgb.shape[:2], 3):
        raise ValueError("P74 RGB and organized XYZ shapes disagree")

    if row_chunk < 1:
        raise ValueError("row_chunk must be positive")
    height, width = output_shape_hw
    source_height, source_width = source_rgb.shape[:2]
    period = source_width - 1
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    radial = np.empty((height, width), dtype=np.float32)
    validity = np.empty((height, width), dtype=bool)
    # The audited P74 basis turns canonical ERP longitude into native
    # longitude with the opposite sign.  Computing this separably avoids the
    # multi-gigabyte full-resolution pixels/rays/native temporary arrays.
    canonical_longitude = (np.arange(width, dtype=np.float64) + 0.5) / width * (
        2.0 * math.pi
    ) - math.pi
    map_x_row = np.mod(-canonical_longitude / (2.0 * math.pi) * period, period)
    for row_start in range(0, height, row_chunk):
        row_stop = min(height, row_start + row_chunk)
        output_rows = np.arange(row_start, row_stop, dtype=np.float64)
        polar = (output_rows + 0.5) / height * math.pi
        support_rows = polar <= P74_POLAR_LIMIT_RAD + 1e-12
        map_x = np.broadcast_to(map_x_row[None], (row_stop - row_start, width)).astype(
            np.float32
        )
        map_y = np.broadcast_to(
            (polar / P74_POLAR_LIMIT_RAD * (source_height - 1))[:, None],
            map_x.shape,
        ).astype(np.float32)
        rgb_chunk = cv2.remap(
            source_rgb,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        columns = np.mod(np.rint(map_x).astype(np.int64), period)
        rows = np.clip(np.rint(map_y).astype(np.int64), 0, source_height - 1)
        sampled_xyz = np.asarray(xyz[rows, columns], dtype=np.float64)
        chunk_radial = np.linalg.norm(sampled_xyz, axis=-1)
        data_valid = np.all(np.isfinite(sampled_xyz), axis=-1) & (chunk_radial > 1e-8)
        chunk_validity = support_rows[:, None] & data_valid
        chunk_radial = chunk_radial.astype(np.float32)
        chunk_radial[~chunk_validity] = np.nan
        rgb_chunk[~support_rows, :] = np.asarray([124, 116, 104], dtype=np.uint8)
        rgb[row_start:row_stop] = rgb_chunk
        radial[row_start:row_stop] = chunk_radial
        validity[row_start:row_stop] = chunk_validity
    parts = npz_path.stem.split("+")
    return P74Frame(
        panorama_id=npz_path.stem,
        family="+".join(parts[1:-1]),
        rgb=rgb,
        radial_range=radial,
        validity=validity,
        source_shape_hw=(source_height, source_width),
    )


def native_angular_erp_shape(source_shape_hw: tuple[int, int]) -> tuple[int, int]:
    """Return a 2:1 ERP that does not minify either native angular axis.

    P74 materializes only 0--150 degrees of polar support.  The equivalent
    full-sphere height is inferred independently from vertical and horizontal
    sampling. The denser estimate is rounded upward to the model lattice
    multiple; width is then exactly twice the height.
    """

    source_height, source_width = source_shape_hw
    if source_height < 2 or source_width < 2:
        raise ValueError("source shape must contain at least two samples per axis")
    vertical_full_height = (source_height - 1) * math.pi / P74_POLAR_LIMIT_RAD
    horizontal_full_height = (source_width - 1) / 2.0
    # Round upward from the denser native axis.  The canonical reprojection is
    # therefore never a minification in either angular direction.
    full_height = max(vertical_full_height, horizontal_full_height)
    height = (
        int(math.ceil(full_height / MODEL_LATTICE_MULTIPLE)) * MODEL_LATTICE_MULTIPLE
    )
    return height, 2 * height


def native_angular_cube_face_size(erp_shape_hw: tuple[int, int]) -> int:
    """Match cube-face centre angular sampling without minifying the ERP."""

    height, width = erp_shape_hw
    if width != 2 * height:
        raise ValueError("canonical ERP must have a 2:1 shape")
    face_size = width / math.pi
    return int(math.ceil(face_size / MODEL_LATTICE_MULTIPLE)) * MODEL_LATTICE_MULTIPLE


def axial_to_radial(axial: np.ndarray) -> np.ndarray:
    """Convert 90-degree pinhole axial depth to radial range."""

    height, width = axial.shape
    x = ((np.arange(width, dtype=np.float64) + 0.5) / width) * 2.0 - 1.0
    y = ((np.arange(height, dtype=np.float64) + 0.5) / height) * 2.0 - 1.0
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return (axial * np.sqrt(1.0 + xx * xx + yy * yy)).astype(np.float32)


def area_weights(shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    rows = np.arange(height, dtype=np.float64)
    latitude = math.pi / 2.0 - (rows + 0.5) / height * math.pi
    return np.broadcast_to(np.cos(latitude)[:, None], (height, width)).copy()


def depth_metrics(
    prediction: np.ndarray, target: np.ndarray, validity: np.ndarray
) -> dict[str, float | int]:
    """Solid-angle-weighted metric-depth scores on an explicit mask."""

    valid = (
        np.asarray(validity, dtype=bool)
        & np.isfinite(prediction)
        & np.isfinite(target)
        & (prediction > 0.0)
        & (target > 0.0)
    )
    if not valid.any():
        raise ValueError("no valid positive prediction/target samples")
    weight = area_weights(target.shape)[valid]
    weight /= weight.sum()
    pred = prediction[valid].astype(np.float64)
    truth = target[valid].astype(np.float64)
    error = pred - truth
    ratio = np.maximum(pred / truth, truth / pred)
    log_error = np.log(pred) - np.log(truth)
    return {
        "valid_pixels": int(valid.sum()),
        "weighted_support": float(area_weights(target.shape)[valid].sum()),
        "abs_rel": float(np.sum(weight * np.abs(error) / truth)),
        "sq_rel": float(np.sum(weight * error * error / truth)),
        "rmse_m": float(np.sqrt(np.sum(weight * error * error))),
        "rmse_log": float(np.sqrt(np.sum(weight * log_error * log_error))),
        "silog": float(
            np.sqrt(
                max(
                    0.0, np.sum(weight * log_error**2) - np.sum(weight * log_error) ** 2
                )
            )
            * 100.0
        ),
        "delta_1": float(np.sum(weight * (ratio < 1.25))),
        "delta_2": float(np.sum(weight * (ratio < 1.25**2))),
        "delta_3": float(np.sum(weight * (ratio < 1.25**3))),
        "prediction_median_m": float(np.median(pred)),
        "prediction_min_m": float(np.min(pred)),
        "prediction_max_m": float(np.max(pred)),
        "prediction_at_0_3m_fraction": float(np.sum(weight * (pred <= 0.300001))),
        "prediction_at_150m_fraction": float(np.sum(weight * (pred >= 149.999))),
        "target_median_m": float(np.median(truth)),
    }


def scale_invariant_structure_metrics(
    prediction: np.ndarray, target: np.ndarray, validity: np.ndarray
) -> dict[str, float | int]:
    """Compare radial 3D structure after eliminating one global scale.

    Corresponding ERP pixels share a unit ray, so the optimally aligned 3D
    point-to-point residual is exactly the aligned radial residual.  Dividing
    by target RMS range makes the result dimensionless and invariant to a
    multiplicative rescaling of either prediction or scene units.
    """

    valid = (
        np.asarray(validity, dtype=bool)
        & np.isfinite(prediction)
        & np.isfinite(target)
        & (prediction > 0.0)
        & (target > 0.0)
    )
    if not valid.any():
        raise ValueError("no valid positive prediction/target samples")
    weights = area_weights(target.shape)[valid]
    weights /= weights.sum()
    pred = prediction[valid].astype(np.float64)
    truth = target[valid].astype(np.float64)
    denominator = float(np.sum(weights * pred * pred))
    if denominator <= 0.0:
        raise ValueError("prediction has zero weighted energy")
    optimal_scale = float(np.sum(weights * pred * truth) / denominator)
    aligned_error = optimal_scale * pred - truth
    target_energy = float(np.sum(weights * truth * truth))
    aligned_relative_3d_rmse = math.sqrt(
        float(np.sum(weights * aligned_error * aligned_error)) / target_energy
    )

    log_pred = np.log(pred)
    log_truth = np.log(truth)
    log_pred_centered = log_pred - np.sum(weights * log_pred)
    log_truth_centered = log_truth - np.sum(weights * log_truth)
    covariance = float(np.sum(weights * log_pred_centered * log_truth_centered))
    variance_product = float(
        np.sum(weights * log_pred_centered**2) * np.sum(weights * log_truth_centered**2)
    )
    log_correlation = (
        covariance / math.sqrt(variance_product) if variance_product > 0.0 else 0.0
    )
    centered_log_error = (log_pred - log_truth) - np.sum(
        weights * (log_pred - log_truth)
    )
    return {
        "valid_pixels": int(valid.sum()),
        "optimal_prediction_scale": optimal_scale,
        "scale_aligned_relative_3d_rmse": aligned_relative_3d_rmse,
        "scale_invariant_log_rmse": float(
            np.sqrt(np.sum(weights * centered_log_error**2))
        ),
        "log_depth_correlation": log_correlation,
    }


def _erp_rays_for_indices(
    rows: np.ndarray, columns: np.ndarray, shape_hw: tuple[int, int]
) -> np.ndarray:
    height, width = shape_hw
    longitude = ((columns.astype(np.float64) + 0.5) / width) * (2.0 * math.pi) - math.pi
    latitude = (math.pi / 2.0) - ((rows.astype(np.float64) + 0.5) / height) * math.pi
    sin_lon = np.sin(longitude)[None]
    cos_lon = np.cos(longitude)[None]
    sin_lat = np.sin(latitude)[:, None]
    cos_lat = np.cos(latitude)[:, None]
    return np.stack(
        (
            np.broadcast_to(sin_lon * cos_lat, (rows.size, columns.size)),
            np.broadcast_to(sin_lat, (rows.size, columns.size)),
            np.broadcast_to(cos_lon * cos_lat, (rows.size, columns.size)),
        ),
        axis=-1,
    )


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(sorted_values[np.searchsorted(cumulative, quantile * cumulative[-1])])


def normal_structure_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    validity: np.ndarray,
    *,
    stride_px: int = 4,
    target_max_log_step: float = 0.10,
    row_chunk: int = 32,
) -> dict[str, float | int]:
    """Measure scale-invariant local 3D surface-normal agreement on an ERP."""

    if prediction.shape != target.shape or target.shape != validity.shape:
        raise ValueError("prediction, target, and validity shapes must agree")
    if stride_px < 1 or row_chunk < 1:
        raise ValueError("stride_px and row_chunk must be positive")
    height, width = target.shape
    center_rows = np.arange(stride_px, height - stride_px, stride_px)
    columns = np.arange(0, width, stride_px)
    if center_rows.size == 0 or columns.size == 0:
        raise ValueError("input is too small for the requested normal stride")
    left_columns = np.mod(columns - stride_px, width)
    right_columns = np.mod(columns + stride_px, width)
    angles_parts: list[np.ndarray] = []
    weights_parts: list[np.ndarray] = []
    candidate_count = int(center_rows.size * columns.size)

    for start in range(0, center_rows.size, row_chunk):
        rows = center_rows[start : start + row_chunk]
        up_rows = rows - stride_px
        down_rows = rows + stride_px
        index_sets = (
            np.ix_(rows, columns),
            np.ix_(rows, left_columns),
            np.ix_(rows, right_columns),
            np.ix_(up_rows, columns),
            np.ix_(down_rows, columns),
        )
        joint = np.ones((rows.size, columns.size), dtype=bool)
        for index in index_sets:
            joint &= validity[index]
            joint &= np.isfinite(prediction[index]) & (prediction[index] > 0.0)
            joint &= np.isfinite(target[index]) & (target[index] > 0.0)

        target_center = target[index_sets[0]]
        for index in index_sets[1:]:
            joint &= (
                np.abs(np.log(target[index]) - np.log(target_center))
                <= target_max_log_step
            )
        if not joint.any():
            continue

        rays_center = _erp_rays_for_indices(rows, columns, target.shape)
        rays_left = _erp_rays_for_indices(rows, left_columns, target.shape)
        rays_right = _erp_rays_for_indices(rows, right_columns, target.shape)
        rays_up = _erp_rays_for_indices(up_rows, columns, target.shape)
        rays_down = _erp_rays_for_indices(down_rows, columns, target.shape)

        def normals(values: np.ndarray) -> np.ndarray:
            points_center = values[index_sets[0]][..., None] * rays_center
            dx = (
                values[index_sets[2]][..., None] * rays_right
                - values[index_sets[1]][..., None] * rays_left
            )
            dy = (
                values[index_sets[4]][..., None] * rays_down
                - values[index_sets[3]][..., None] * rays_up
            )
            result = np.cross(dx, dy)
            norm = np.linalg.norm(result, axis=-1, keepdims=True)
            result = result / np.maximum(norm, 1e-15)
            away = np.sum(result * points_center, axis=-1) > 0.0
            result[away] *= -1.0
            return result

        pred_normal = normals(prediction)
        truth_normal = normals(target)
        finite = (
            np.all(np.isfinite(pred_normal), axis=-1)
            & np.all(np.isfinite(truth_normal), axis=-1)
            & joint
        )
        if not finite.any():
            continue
        dot = np.sum(pred_normal * truth_normal, axis=-1)
        angles = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))[finite]
        latitude = (math.pi / 2.0) - ((rows + 0.5) / height) * math.pi
        weights = np.broadcast_to(np.cos(latitude)[:, None], joint.shape)[finite]
        angles_parts.append(angles)
        weights_parts.append(weights)

    if not angles_parts:
        return {
            "normal_valid_samples": 0,
            "normal_candidate_samples": candidate_count,
            "normal_support_fraction": 0.0,
            "normal_mean_deg": float("nan"),
            "normal_median_deg": float("nan"),
            "normal_p90_deg": float("nan"),
            "normal_within_11_25_deg": 0.0,
            "normal_within_22_5_deg": 0.0,
            "normal_within_30_deg": 0.0,
            "normal_stride_px": stride_px,
            "target_max_log_step": target_max_log_step,
        }
    angles = np.concatenate(angles_parts)
    weights = np.concatenate(weights_parts)
    weights /= weights.sum()
    return {
        "normal_valid_samples": int(angles.size),
        "normal_candidate_samples": candidate_count,
        "normal_support_fraction": float(angles.size / candidate_count),
        "normal_mean_deg": float(np.sum(weights * angles)),
        "normal_median_deg": _weighted_quantile(angles, weights, 0.5),
        "normal_p90_deg": _weighted_quantile(angles, weights, 0.9),
        "normal_within_11_25_deg": float(np.sum(weights * (angles < 11.25))),
        "normal_within_22_5_deg": float(np.sum(weights * (angles < 22.5))),
        "normal_within_30_deg": float(np.sum(weights * (angles < 30.0))),
        "normal_stride_px": stride_px,
        "target_max_log_step": target_max_log_step,
    }


def write_binary_ply(
    path: Path,
    radial_range: np.ndarray,
    rgb: np.ndarray,
    validity: np.ndarray,
    *,
    max_points: int = 2_000_000,
) -> dict[str, int | str]:
    """Write a deterministic view cloud in the canonical panorama frame."""

    if radial_range.shape != validity.shape or rgb.shape != (*validity.shape, 3):
        raise ValueError("radial range, RGB, and validity shapes disagree")
    if max_points < 1:
        raise ValueError("max_points must be positive")
    valid = (
        np.asarray(validity, dtype=bool)
        & np.isfinite(radial_range)
        & (radial_range > 0.0)
    )
    valid_count = int(valid.sum())
    if valid_count == 0:
        raise ValueError("no valid points to export")
    spatial_stride = max(1, int(math.ceil(math.sqrt(valid_count / max_points))))
    rows, columns = np.nonzero(valid[::spatial_stride, ::spatial_stride])
    rows *= spatial_stride
    columns *= spatial_stride
    if rows.size > max_points:
        selection_stride = int(math.ceil(rows.size / max_points))
        rows = rows[::selection_stride]
        columns = columns[::selection_stride]
    # The selected rows/columns need not form a dense Cartesian product after
    # the final point cap, so use the direct pixel-centre formula here.
    longitude = ((columns.astype(np.float64) + 0.5) / radial_range.shape[1]) * (
        2.0 * math.pi
    ) - math.pi
    latitude = (math.pi / 2.0) - (
        (rows.astype(np.float64) + 0.5) / radial_range.shape[0]
    ) * math.pi
    unit = np.column_stack(
        (
            np.sin(longitude) * np.cos(latitude),
            np.sin(latitude),
            np.cos(longitude) * np.cos(latitude),
        )
    )
    points = unit * radial_range[rows, columns, None]
    colors = rgb[rows, columns]
    vertices = np.empty(
        rows.size,
        dtype=np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        ),
    )
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment PanorAi canonical frame: +X right, +Y up, +Z forward\n"
        "comment radial range in metres; deterministic ERP grid decimation\n"
        f"element vertex {vertices.size}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)
    return {
        "path": str(path),
        "valid_dense_points": valid_count,
        "written_points": int(vertices.size),
        "spatial_stride_px": spatial_stride,
    }


def seam_score(prediction: np.ndarray, validity: np.ndarray) -> dict[str, float | int]:
    rows = validity[:, 0] & validity[:, -1]
    rows &= np.isfinite(prediction[:, 0]) & np.isfinite(prediction[:, -1])
    if not rows.any():
        return {"rows": 0, "mae_m": float("nan")}
    jump = np.abs(prediction[rows, 0] - prediction[rows, -1])
    return {"rows": int(rows.sum()), "mae_m": float(np.mean(jump))}


def band_metrics(
    prediction: np.ndarray, target: np.ndarray, validity: np.ndarray
) -> dict[str, dict[str, float | int] | None]:
    height = target.shape[0]
    latitude = 90.0 - (np.arange(height) + 0.5) / height * 180.0
    bands = {
        "north_75_90": latitude >= 75.0,
        "equator_pm15": np.abs(latitude) <= 15.0,
        "p74_south_boundary_m60_m45": (latitude >= -60.0) & (latitude < -45.0),
    }
    result: dict[str, dict[str, float | int] | None] = {}
    for name, selected_rows in bands.items():
        mask = validity & selected_rows[:, None]
        result[name] = depth_metrics(prediction, target, mask) if mask.any() else None
    return result


@dataclass(frozen=True)
class PortRecord:
    path: str
    source_type: str
    target_type: str
    weight_shape: tuple[int, ...] | None
    parameter_identity_preserved: bool | None


class SphericalConvTranspose2d:  # constructed lazily to keep Torch optional here
    pass


def port_metric3d_spatial_layers(
    model: Any, *, max_sampled_elements: int | None = None
) -> dict[str, Any]:
    """Port every learned spatial convolution while preserving Parameters."""

    import torch
    from torch import nn
    from torch.nn import functional as torch_functional
    from panorai.image_processing.torch import (
        SphericalConv2d,
        _sample_tangent_neighbourhood,
    )

    if max_sampled_elements is not None and max_sampled_elements < 1:
        raise ValueError("max_sampled_elements must be positive")

    def output_size(
        size: int,
        *,
        kernel: int,
        stride: int,
        padding: int,
        dilation: int,
    ) -> int:
        result = math.floor(
            (size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
        )
        if result < 1:
            raise ValueError("input is too small for the output lattice")
        return result

    def tangent_grid_rows(
        input_shape: tuple[int, int],
        output_shape: tuple[int, int],
        kernel_size: tuple[int, int],
        dilation: tuple[int, int],
        row_start: int,
        row_stop: int,
        *,
        device: Any,
        dtype: Any,
    ) -> Any:
        input_height, input_width = input_shape
        output_height, output_width = output_shape
        kernel_height, kernel_width = kernel_size
        y = torch.arange(row_start, row_stop, device=device, dtype=dtype)[:, None]
        x = torch.arange(output_width, device=device, dtype=dtype)[None, :]
        longitude = ((x + 0.5) / output_width) * (2.0 * math.pi) - math.pi
        latitude = (math.pi / 2.0) - ((y + 0.5) / output_height) * math.pi
        longitude = longitude.expand(row_stop - row_start, output_width)
        latitude = latitude.expand(row_stop - row_start, output_width)
        sin_lon = torch.sin(longitude)
        cos_lon = torch.cos(longitude)
        sin_lat = torch.sin(latitude)
        cos_lat = torch.cos(latitude)
        rays = torch.stack((sin_lon * cos_lat, sin_lat, cos_lon * cos_lat), dim=-1)
        east = torch.stack((cos_lon, torch.zeros_like(cos_lon), -sin_lon), dim=-1)
        north = torch.stack((-sin_lon * sin_lat, cos_lat, -cos_lon * sin_lat), dim=-1)
        step_east = 2.0 * math.pi / input_width
        step_north = math.pi / input_height
        kernel_y = torch.arange(kernel_height, device=device, dtype=dtype)
        kernel_x = torch.arange(kernel_width, device=device, dtype=dtype)
        north_offset = (
            -(kernel_y - (kernel_height - 1.0) / 2.0) * dilation[0] * step_north
        )
        east_offset = (kernel_x - (kernel_width - 1.0) / 2.0) * dilation[1] * step_east
        north_offset, east_offset = torch.meshgrid(
            north_offset, east_offset, indexing="ij"
        )
        north_offset = north_offset.reshape(-1, 1, 1)
        east_offset = east_offset.reshape(-1, 1, 1)
        radius = torch.hypot(east_offset, north_offset)
        sinc = torch.where(
            radius == 0.0, torch.ones_like(radius), torch.sinc(radius / math.pi)
        )
        tangent = (
            east_offset[..., None] * east[None] + north_offset[..., None] * north[None]
        )
        sample_rays = (
            torch.cos(radius)[..., None] * rays[None] + sinc[..., None] * tangent
        )
        sample_rays = torch_functional.normalize(sample_rays, dim=-1)
        sample_lon = torch.atan2(sample_rays[..., 0], sample_rays[..., 2])
        sample_lat = torch.asin(sample_rays[..., 1].clamp(-1.0, 1.0))
        sample_x = (sample_lon + math.pi) / (2.0 * math.pi) * input_width - 0.5
        sample_y = (math.pi / 2.0 - sample_lat) / math.pi * input_height - 0.5
        normalized_x = 2.0 * (sample_x + 1.0) / (input_width + 1.0) - 1.0
        normalized_y = (
            torch.zeros_like(sample_y)
            if input_height == 1
            else 2.0 * sample_y / (input_height - 1.0) - 1.0
        )
        return torch.stack((normalized_x, normalized_y), dim=-1).reshape(
            -1, output_width, 2
        )

    def sampled_rows(
        values: Any,
        wrapped: Any,
        output_shape: tuple[int, int],
        kernel_size: tuple[int, int],
        dilation: tuple[int, int],
        row_start: int,
        row_stop: int,
    ) -> Any:
        grid = tangent_grid_rows(
            tuple(values.shape[-2:]),
            output_shape,
            kernel_size,
            dilation,
            row_start,
            row_stop,
            device=values.device,
            dtype=values.dtype,
        )
        sampled = torch_functional.grid_sample(
            wrapped,
            grid[None].expand(values.shape[0], -1, -1, -1),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.reshape(
            values.shape[0],
            values.shape[1],
            kernel_size[0] * kernel_size[1],
            row_stop - row_start,
            output_shape[1],
        )

    def rows_per_chunk(
        values: Any, kernel_size: tuple[int, int], output_width: int
    ) -> int:
        assert max_sampled_elements is not None
        elements_per_row = (
            values.shape[0]
            * values.shape[1]
            * kernel_size[0]
            * kernel_size[1]
            * output_width
        )
        return max(1, max_sampled_elements // elements_per_row)

    class _ChunkedSphericalConv2d(nn.Module):
        def __init__(self, source: nn.Conv2d) -> None:
            super().__init__()
            self.in_channels = source.in_channels
            self.out_channels = source.out_channels
            self.kernel_size = tuple(source.kernel_size)
            self.stride = tuple(source.stride)
            self.padding = tuple(source.padding)
            self.dilation = tuple(source.dilation)
            self.groups = source.groups
            self.weight = source.weight
            self.bias = source.bias

        def forward(self, values: Any) -> Any:
            if self.kernel_size == (1, 1) and self.stride == (1, 1):
                return torch_functional.conv2d(
                    values, self.weight, self.bias, groups=self.groups
                )
            output_shape = (
                output_size(
                    values.shape[-2],
                    kernel=self.kernel_size[0],
                    stride=self.stride[0],
                    padding=self.padding[0],
                    dilation=self.dilation[0],
                ),
                output_size(
                    values.shape[-1],
                    kernel=self.kernel_size[1],
                    stride=self.stride[1],
                    padding=self.padding[1],
                    dilation=self.dilation[1],
                ),
            )
            chunk_rows = rows_per_chunk(values, self.kernel_size, output_shape[1])
            sample_count = self.kernel_size[0] * self.kernel_size[1]
            input_per_group = self.in_channels // self.groups
            output_per_group = self.out_channels // self.groups
            grouped_weights = self.weight.reshape(
                self.groups, output_per_group, input_per_group, sample_count
            )
            wrapped = torch.cat((values[..., -1:], values, values[..., :1]), dim=-1)
            parts = []
            for row_start in range(0, output_shape[0], chunk_rows):
                row_stop = min(output_shape[0], row_start + chunk_rows)
                sampled = sampled_rows(
                    values,
                    wrapped,
                    output_shape,
                    self.kernel_size,
                    self.dilation,
                    row_start,
                    row_stop,
                )
                grouped = sampled.reshape(
                    values.shape[0],
                    self.groups,
                    input_per_group,
                    sample_count,
                    row_stop - row_start,
                    output_shape[1],
                )
                part = torch.einsum(
                    "ngckhw,gock->ngohw", grouped, grouped_weights
                ).reshape(
                    values.shape[0],
                    self.out_channels,
                    row_stop - row_start,
                    output_shape[1],
                )
                if self.bias is not None:
                    part = part + self.bias[None, :, None, None]
                parts.append(part)
            return torch.cat(parts, dim=-2)

    class _SphericalConvTranspose2d(nn.Module):
        def __init__(self, source: nn.ConvTranspose2d) -> None:
            super().__init__()
            if source.groups != 1:
                raise ValueError("benchmark spherical transpose supports groups=1 only")
            self.weight = source.weight
            self.bias = source.bias
            self.kernel_size = tuple(source.kernel_size)
            self.stride = tuple(source.stride)
            self.padding = tuple(source.padding)
            self.output_padding = tuple(source.output_padding)
            self.dilation = tuple(source.dilation)

        def forward(self, values: Any) -> Any:
            height = (
                (values.shape[-2] - 1) * self.stride[0] + 1 + self.output_padding[0]
            )
            width = (values.shape[-1] - 1) * self.stride[1] + 1 + self.output_padding[1]
            expanded = values.new_zeros(values.shape[0], values.shape[1], height, width)
            expanded[..., :: self.stride[0], :: self.stride[1]] = values
            effective_padding = (
                self.dilation[0] * (self.kernel_size[0] - 1) - self.padding[0],
                self.dilation[1] * (self.kernel_size[1] - 1) - self.padding[1],
            )
            flipped = torch.flip(self.weight, dims=(-2, -1)).flatten(2)
            if max_sampled_elements is None:
                sampled, _ = _sample_tangent_neighbourhood(
                    expanded,
                    kernel_size=self.kernel_size,
                    stride=(1, 1),
                    padding=effective_padding,
                    dilation=self.dilation,
                )
                result = torch.einsum("nckhw,cok->nohw", sampled, flipped)
                if self.bias is not None:
                    result = result + self.bias[None, :, None, None]
                return result
            output_shape = (
                output_size(
                    expanded.shape[-2],
                    kernel=self.kernel_size[0],
                    stride=1,
                    padding=effective_padding[0],
                    dilation=self.dilation[0],
                ),
                output_size(
                    expanded.shape[-1],
                    kernel=self.kernel_size[1],
                    stride=1,
                    padding=effective_padding[1],
                    dilation=self.dilation[1],
                ),
            )
            chunk_rows = rows_per_chunk(expanded, self.kernel_size, output_shape[1])
            wrapped = torch.cat(
                (expanded[..., -1:], expanded, expanded[..., :1]), dim=-1
            )
            parts = []
            for row_start in range(0, output_shape[0], chunk_rows):
                row_stop = min(output_shape[0], row_start + chunk_rows)
                sampled = sampled_rows(
                    expanded,
                    wrapped,
                    output_shape,
                    self.kernel_size,
                    self.dilation,
                    row_start,
                    row_stop,
                )
                part = torch.einsum("nckhw,cok->nohw", sampled, flipped)
                if self.bias is not None:
                    part = part + self.bias[None, :, None, None]
                parts.append(part)
            return torch.cat(parts, dim=-2)

    records: list[PortRecord] = []
    collapsed_reflection_pads: list[str] = []

    def collapse(module: nn.Module, prefix: str = "") -> None:
        children = list(module.named_children())
        for index, (name, child) in enumerate(children):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.ReflectionPad2d):
                if index + 1 >= len(children) or not isinstance(
                    children[index + 1][1], nn.Conv2d
                ):
                    raise RuntimeError(
                        f"unsupported standalone ReflectionPad2d at {path}"
                    )
                padding = child.padding
                if isinstance(padding, int):
                    pad = (padding, padding)
                else:
                    left, right, top, bottom = padding
                    if left != right or top != bottom:
                        raise RuntimeError(f"asymmetric reflection padding at {path}")
                    pad = (top, left)
                next_conv = children[index + 1][1]
                next_conv.padding = pad
                setattr(module, name, nn.Identity())
                collapsed_reflection_pads.append(path)
            else:
                collapse(child, path)

    def replace(module: nn.Module, prefix: str = "") -> None:
        for name, child in tuple(module.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.ConvTranspose2d):
                replacement = _SphericalConvTranspose2d(child)
                records.append(
                    PortRecord(
                        path,
                        "ConvTranspose2d",
                        "SphericalConvTranspose2d",
                        tuple(child.weight.shape),
                        replacement.weight is child.weight
                        and replacement.bias is child.bias,
                    )
                )
                setattr(module, name, replacement)
            elif isinstance(child, nn.Conv2d):
                replacement = (
                    SphericalConv2d(child)
                    if max_sampled_elements is None
                    else _ChunkedSphericalConv2d(child)
                )
                records.append(
                    PortRecord(
                        path,
                        "Conv2d",
                        type(replacement).__name__,
                        tuple(child.weight.shape),
                        replacement.weight is child.weight
                        and replacement.bias is child.bias,
                    )
                )
                setattr(module, name, replacement)
            else:
                replace(child, path)

    collapse(model)
    replace(model)
    remaining = [
        name
        for name, child in model.named_modules()
        if isinstance(child, (nn.Conv2d, nn.ConvTranspose2d, nn.ReflectionPad2d))
    ]
    if remaining:
        raise RuntimeError(f"unported spatial operators: {remaining}")
    learned = [record for record in records if record.weight_shape is not None]
    if not all(record.parameter_identity_preserved for record in learned):
        raise RuntimeError("a ported learned layer copied/replaced its Parameter")
    return {
        "ported_layer_count": len(records),
        "conv2d_count": sum(record.source_type == "Conv2d" for record in records),
        "conv_transpose2d_count": sum(
            record.source_type == "ConvTranspose2d" for record in records
        ),
        "collapsed_reflection_pads": collapsed_reflection_pads,
        "remaining_planar_learned_spatial_layers": remaining,
        "all_parameter_identities_preserved": True,
        "layers": [asdict(record) for record in records],
        "nonlearned_resampling_policy": "nearest ERP resampling; same spherical lattice",
        "max_sampled_elements_per_chunk": max_sampled_elements,
    }


def patch_metric3d_device_assumptions(model: Any) -> None:
    """Patch two upstream v1 CUDA literals without changing learned state."""

    import types
    import torch

    decoder = model.depth_model.decoder

    def get_bins(self: Any, bins_num: int) -> Any:
        parameter = next(self.parameters())
        return torch.exp(
            torch.linspace(
                math.log(self.min_val),
                math.log(self.max_val),
                bins_num,
                device=parameter.device,
                dtype=parameter.dtype,
            )
        )

    def create_mesh_grid(
        self: Any,
        height: int,
        width: int,
        batch: int,
        device: Any = None,
        set_buffer: bool = True,
    ) -> Any:
        del device, set_buffer
        parameter = next(self.parameters())
        y, x = torch.meshgrid(
            torch.arange(height, device=parameter.device, dtype=parameter.dtype),
            torch.arange(width, device=parameter.device, dtype=parameter.dtype),
            indexing="ij",
        )
        return torch.stack((x, y))[None].repeat(batch, 1, 1, 1)

    decoder.get_bins = types.MethodType(get_bins, decoder)
    decoder.create_mesh_grid = types.MethodType(create_mesh_grid, decoder)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
