"""Project Gaussian centres back to ERP and refine a monocular radial prior.

This module closes the visible-surface loop used by the two-view benchmark:
monocular depth seeds Gaussians, the two posed views filter/fuse them, and the
resulting Gaussian centres become confidence-weighted depth anchors again.
It deliberately operates on observed surfaces only and never fills invalid
monocular support with invented geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PLY_VERTEX_DTYPE = np.dtype(
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
)


@dataclass(frozen=True, slots=True)
class GaussianDepthFeedbackOptions:
    """Conservative settings for Gaussian-to-depth feedback."""

    gaussian_sigma_px: float = 0.85
    gaussian_radius_px: int = 2
    occlusion_tolerance_m: float = 0.08
    photometric_sigma: float = 0.16
    target_only_confidence: float = 0.04
    source_only_confidence: float = 0.55
    two_view_confidence: float = 0.95
    two_view_agreement_log_sigma: float = 0.12
    maximum_anchor_log_residual: float = 0.45
    maximum_log_correction: float = 0.60
    cloud_data_weight: float = 8.0
    landmark_data_weight: float = 48.0
    prior_residual_weight: float = 0.025
    smoothness_weight: float = 1.0
    color_sigma: float = 0.10
    iterations: int = 64
    min_range_m: float = 0.3
    max_range_m: float = 15.0

    def __post_init__(self) -> None:
        positive = (
            "gaussian_sigma_px",
            "occlusion_tolerance_m",
            "photometric_sigma",
            "two_view_agreement_log_sigma",
            "maximum_anchor_log_residual",
            "maximum_log_correction",
            "cloud_data_weight",
            "landmark_data_weight",
            "prior_residual_weight",
            "smoothness_weight",
            "color_sigma",
            "min_range_m",
            "max_range_m",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "target_only_confidence",
            "source_only_confidence",
            "two_view_confidence",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.gaussian_radius_px < 0:
            raise ValueError("gaussian_radius_px must be non-negative")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if self.max_range_m <= self.min_range_m:
            raise ValueError("max_range_m must exceed min_range_m")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GaussianDepthAnchors:
    """Rasterized visible-surface observations on an ERP solve lattice."""

    radial_m: np.ndarray
    confidence: np.ndarray
    view_mask: np.ndarray
    point_count: np.ndarray


@dataclass(slots=True)
class GaussianDepthFeedbackResult:
    """Native refined depth plus solve-lattice diagnostics."""

    refined_radial_m: np.ndarray
    refined_confidence: np.ndarray
    correction_log_solve: np.ndarray
    anchors: GaussianDepthAnchors
    accepted_anchor: np.ndarray
    hard_landmark_anchor: np.ndarray
    report: dict[str, Any]


def read_gaussian_centres_ply(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the exact binary PLY schema written by the Gaussian exporter."""

    ply_path = Path(path)
    with ply_path.open("rb") as stream:
        header_lines: list[str] = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("PLY ended before end_header")
            try:
                decoded = line.decode("ascii").rstrip("\r\n")
            except UnicodeDecodeError as error:
                raise ValueError("PLY header must be ASCII") from error
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        if header_lines[:2] != ["ply", "format binary_little_endian 1.0"]:
            raise ValueError("expected a binary little-endian PLY")
        vertex_lines = [
            line for line in header_lines if line.startswith("element vertex ")
        ]
        if len(vertex_lines) != 1:
            raise ValueError("PLY must declare exactly one vertex element")
        vertex_count = int(vertex_lines[0].split()[-1])
        expected_properties = [
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property ushort observations",
            "property uchar view_mask",
        ]
        properties = [line for line in header_lines if line.startswith("property ")]
        if properties != expected_properties:
            raise ValueError(
                "PLY vertex properties do not match the Gaussian-centre schema"
            )
        payload = stream.read()
    expected_bytes = vertex_count * PLY_VERTEX_DTYPE.itemsize
    if len(payload) != expected_bytes:
        raise ValueError(
            f"PLY payload has {len(payload)} bytes; expected {expected_bytes}"
        )
    vertices = np.frombuffer(payload, dtype=PLY_VERTEX_DTYPE, count=vertex_count)
    points = np.column_stack((vertices["x"], vertices["y"], vertices["z"]))
    colors = np.column_stack((vertices["red"], vertices["green"], vertices["blue"]))
    return (
        points.astype(np.float32, copy=False),
        colors.astype(np.uint8, copy=False),
        vertices["observations"].copy(),
        vertices["view_mask"].copy(),
    )


def rasterize_gaussian_depth_anchors(
    points_target_xyz: Any,
    colors_rgb: Any,
    observations: Any,
    view_mask: Any,
    target_rgb_hwc: Any,
    *,
    options: GaussianDepthFeedbackOptions | None = None,
) -> GaussianDepthAnchors:
    """Splat Gaussian centres into a visibility-aware spherical depth map.

    The nearest radial surface is selected first.  Gaussian contributions
    behind it are rejected, target/source evidence is accumulated separately,
    and confidence is highest only when both view bits support the visible
    surface.  Longitude wraps; latitude never wraps.
    """

    settings = options or GaussianDepthFeedbackOptions()
    points = np.asarray(points_target_xyz, dtype=np.float32)
    colors = np.asarray(colors_rgb, dtype=np.uint8)
    counts = np.asarray(observations, dtype=np.uint16)
    bits = np.asarray(view_mask, dtype=np.uint8)
    rgb_u8 = np.asarray(target_rgb_hwc)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_target_xyz must have shape (N, 3)")
    if colors.shape != points.shape:
        raise ValueError("colors_rgb must have shape (N, 3)")
    if counts.shape != (points.shape[0],) or bits.shape != (points.shape[0],):
        raise ValueError("observations/view_mask must have shape (N,)")
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError("target_rgb_hwc must have shape (H, W, 3)")
    if not np.issubdtype(rgb_u8.dtype, np.integer):
        raise ValueError("target_rgb_hwc must use an integer RGB dtype")
    if np.any((bits < 1) | (bits > 3)):
        raise ValueError("view_mask values must be 1, 2, or 3")

    height, width = rgb_u8.shape[:2]
    radial = np.linalg.norm(points.astype(np.float64), axis=1)
    safe = np.maximum(radial, 1e-12)
    longitude = np.arctan2(points[:, 0], points[:, 2])
    latitude = np.arcsin(np.clip(points[:, 1] / safe, -1.0, 1.0))
    u = (longitude + np.pi) / (2.0 * np.pi) * width - 0.5
    v = (np.pi / 2.0 - latitude) / np.pi * height - 0.5
    finite = (
        np.isfinite(u)
        & np.isfinite(v)
        & np.isfinite(radial)
        & (radial >= settings.min_range_m)
        & (radial <= settings.max_range_m)
    )
    u = u[finite]
    v = v[finite]
    radial = radial[finite]
    colors = colors[finite]
    counts = counts[finite]
    bits = bits[finite]
    if radial.size == 0:
        shape = (height, width)
        return GaussianDepthAnchors(
            radial_m=np.full(shape, np.nan, dtype=np.float32),
            confidence=np.zeros(shape, dtype=np.float32),
            view_mask=np.zeros(shape, dtype=np.uint8),
            point_count=np.zeros(shape, dtype=np.uint32),
        )

    base_x = np.floor(u).astype(np.int64)
    base_y = np.floor(v).astype(np.int64)
    pixel_count = height * width
    nearest = np.full(pixel_count, np.inf, dtype=np.float64)
    offsets = range(-settings.gaussian_radius_px, settings.gaussian_radius_px + 1)
    for dy in offsets:
        iy = base_y + dy
        vertical = (iy >= 0) & (iy < height)
        if not np.any(vertical):
            continue
        for dx in offsets:
            ix = np.remainder(base_x + dx, width)
            index = iy[vertical] * width + ix[vertical]
            np.minimum.at(nearest, index, radial[vertical])

    target_weight = np.zeros(pixel_count, dtype=np.float64)
    source_weight = np.zeros(pixel_count, dtype=np.float64)
    target_log_sum = np.zeros(pixel_count, dtype=np.float64)
    source_log_sum = np.zeros(pixel_count, dtype=np.float64)
    point_count = np.zeros(pixel_count, dtype=np.uint32)
    output_bits = np.zeros(pixel_count, dtype=np.uint8)
    normalized_colors = colors.astype(np.float32) / 255.0
    normalized_target = rgb_u8.astype(np.float32) / float(np.iinfo(rgb_u8.dtype).max)
    observation_factor = 0.5 + 0.5 * np.minimum(
        1.0,
        np.log1p(counts.astype(np.float64)) / math.log(5.0),
    )
    log_radial = np.log(radial)
    inv_two_sigma_sq = 0.5 / (settings.gaussian_sigma_px**2)

    for dy in offsets:
        iy_all = base_y + dy
        vertical = (iy_all >= 0) & (iy_all < height)
        if not np.any(vertical):
            continue
        selected = np.nonzero(vertical)[0]
        selected_y = iy_all[selected]
        for dx in offsets:
            ix_unwrapped = base_x[selected] + dx
            ix = np.remainder(ix_unwrapped, width)
            index = selected_y * width + ix
            visible = radial[selected] <= (
                nearest[index] + settings.occlusion_tolerance_m
            )
            if not np.any(visible):
                continue
            chosen = selected[visible]
            iy = selected_y[visible]
            ix = ix[visible]
            index = index[visible]
            delta_u = u[chosen] - ix_unwrapped[visible]
            delta_v = v[chosen] - iy.astype(np.float64)
            spatial = np.exp(
                -(delta_u * delta_u + delta_v * delta_v) * inv_two_sigma_sq
            )
            color_l1 = np.mean(
                np.abs(normalized_colors[chosen] - normalized_target[iy, ix]), axis=1
            )
            photometric = np.exp(-color_l1 / settings.photometric_sigma)
            weight = spatial * photometric * observation_factor[chosen]
            chosen_bits = bits[chosen]
            target_scale = np.where(
                chosen_bits == 3, 1.0, settings.target_only_confidence
            )
            source_scale = np.where(
                chosen_bits == 3, 1.0, settings.source_only_confidence
            )
            target_valid = (chosen_bits & 1) != 0
            source_valid = (chosen_bits & 2) != 0
            target_contribution = weight * target_scale * target_valid
            source_contribution = weight * source_scale * source_valid
            np.add.at(target_weight, index, target_contribution)
            np.add.at(source_weight, index, source_contribution)
            np.add.at(
                target_log_sum,
                index,
                target_contribution * log_radial[chosen],
            )
            np.add.at(
                source_log_sum,
                index,
                source_contribution * log_radial[chosen],
            )
            np.add.at(point_count, index, 1)
            np.bitwise_or.at(output_bits, index, chosen_bits)

    target_valid = target_weight > 1e-10
    source_valid = source_weight > 1e-10
    target_log = np.zeros(pixel_count, dtype=np.float64)
    source_log = np.zeros(pixel_count, dtype=np.float64)
    target_log[target_valid] = (
        target_log_sum[target_valid] / target_weight[target_valid]
    )
    source_log[source_valid] = (
        source_log_sum[source_valid] / source_weight[source_valid]
    )
    both = target_valid & source_valid
    only_target = target_valid & ~source_valid
    only_source = source_valid & ~target_valid
    total_weight = target_weight + source_weight
    combined_log = np.zeros(pixel_count, dtype=np.float64)
    valid = total_weight > 1e-10
    combined_log[valid] = (
        target_log_sum[valid] + source_log_sum[valid]
    ) / total_weight[valid]
    density = 1.0 - np.exp(-total_weight)
    confidence = np.zeros(pixel_count, dtype=np.float64)
    confidence[only_target] = settings.target_only_confidence * density[only_target]
    confidence[only_source] = settings.source_only_confidence * density[only_source]
    agreement = np.exp(
        -np.abs(target_log[both] - source_log[both])
        / settings.two_view_agreement_log_sigma
    )
    confidence[both] = settings.two_view_confidence * density[both] * agreement
    anchor_radial = np.full(pixel_count, np.nan, dtype=np.float32)
    anchor_radial[valid] = np.exp(combined_log[valid]).astype(np.float32)
    output_bits[only_target] = 1
    output_bits[only_source] = 2
    output_bits[both] = 3
    return GaussianDepthAnchors(
        radial_m=anchor_radial.reshape(height, width),
        confidence=np.clip(confidence, 0.0, 1.0)
        .astype(np.float32)
        .reshape(height, width),
        view_mask=output_bits.reshape(height, width),
        point_count=point_count.reshape(height, width),
    )


def refine_depth_from_gaussian_anchors(
    prior_radial_m: Any,
    target_rgb_solve_hwc: Any,
    anchors: GaussianDepthAnchors,
    *,
    landmark_points_target: Any | None = None,
    options: GaussianDepthFeedbackOptions | None = None,
) -> GaussianDepthFeedbackResult:
    """Solve and upsample an edge-aware log-depth correction field."""

    settings = options or GaussianDepthFeedbackOptions()
    prior = np.asarray(prior_radial_m, dtype=np.float32)
    rgb = np.asarray(target_rgb_solve_hwc)
    if prior.ndim != 2:
        raise ValueError("prior_radial_m must have shape (H, W)")
    solve_shape = anchors.radial_m.shape
    if rgb.shape != (*solve_shape, 3):
        raise ValueError("target_rgb_solve_hwc must match the anchor shape")
    for name, value in (
        ("confidence", anchors.confidence),
        ("view_mask", anchors.view_mask),
        ("point_count", anchors.point_count),
    ):
        if value.shape != solve_shape:
            raise ValueError(f"anchor {name} must share one HW shape")

    prior_grid, prior_grid_valid = _resize_valid_continuous(prior, solve_shape)
    anchor_radial = np.asarray(anchors.radial_m, dtype=np.float32).copy()
    anchor_confidence = np.asarray(anchors.confidence, dtype=np.float32).copy()
    hard_landmark = np.zeros(solve_shape, dtype=bool)
    landmark_count = 0
    if landmark_points_target is not None:
        landmark_count = _merge_landmark_anchors(
            anchor_radial,
            anchor_confidence,
            hard_landmark,
            landmark_points_target,
            settings,
        )

    anchor_valid = (
        prior_grid_valid
        & np.isfinite(anchor_radial)
        & (anchor_radial >= settings.min_range_m)
        & (anchor_radial <= settings.max_range_m)
        & (anchor_confidence > 0.0)
    )
    residual = np.zeros(solve_shape, dtype=np.float32)
    residual[anchor_valid] = np.log(
        anchor_radial[anchor_valid] / prior_grid[anchor_valid]
    ).astype(np.float32)
    accepted = anchor_valid & (
        (np.abs(residual) <= settings.maximum_anchor_log_residual) | hard_landmark
    )
    residual = np.clip(
        residual, -settings.maximum_log_correction, settings.maximum_log_correction
    )
    data_weight = np.zeros(solve_shape, dtype=np.float32)
    data_weight[accepted] = settings.cloud_data_weight * anchor_confidence[accepted]
    data_weight[hard_landmark & accepted] += settings.landmark_data_weight
    correction, confidence = _solve_edge_aware_residual(
        residual,
        data_weight,
        anchor_confidence,
        rgb,
        settings,
    )

    native_height, native_width = prior.shape
    correction_native = cv2.resize(
        correction,
        (native_width, native_height),
        interpolation=cv2.INTER_LINEAR,
    )
    confidence_native = cv2.resize(
        confidence,
        (native_width, native_height),
        interpolation=cv2.INTER_LINEAR,
    )
    correction_native = np.clip(
        correction_native,
        -settings.maximum_log_correction,
        settings.maximum_log_correction,
    )
    native_valid = (
        np.isfinite(prior)
        & (prior >= settings.min_range_m)
        & (prior <= settings.max_range_m)
    )
    refined = np.full(prior.shape, np.nan, dtype=np.float32)
    refined[native_valid] = np.clip(
        prior[native_valid] * np.exp(correction_native[native_valid]),
        settings.min_range_m,
        settings.max_range_m,
    ).astype(np.float32)
    confidence_native = np.where(native_valid, confidence_native, 0.0).astype(
        np.float32
    )
    if landmark_points_target is not None:
        _enforce_native_landmark_pixels(
            refined,
            confidence_native,
            landmark_points_target,
            settings,
        )

    report = {
        "options": settings.to_dict(),
        "native_shape_hw": list(prior.shape),
        "solve_shape_hw": list(solve_shape),
        "cloud_anchor_pixels": int(np.count_nonzero(anchors.confidence > 0.0)),
        "two_view_anchor_pixels": int(np.count_nonzero(anchors.view_mask == 3)),
        "accepted_anchor_pixels": int(np.count_nonzero(accepted)),
        "rejected_anchor_pixels": int(np.count_nonzero(anchor_valid & ~accepted)),
        "hard_landmark_anchor_pixels": int(np.count_nonzero(hard_landmark)),
        "landmarks_input": landmark_count,
        "native_valid_pixels": int(np.count_nonzero(native_valid)),
        "native_confident_pixels": int(np.count_nonzero(confidence_native >= 0.10)),
        "native_changed_pixels_1pct": int(
            np.count_nonzero(
                native_valid & (np.abs(correction_native) >= math.log(1.01))
            )
        ),
        "correction_log_median": float(np.median(correction_native[native_valid])),
        "correction_log_p01": float(np.quantile(correction_native[native_valid], 0.01)),
        "correction_log_p99": float(np.quantile(correction_native[native_valid], 0.99)),
    }
    return GaussianDepthFeedbackResult(
        refined_radial_m=refined,
        refined_confidence=confidence_native,
        correction_log_solve=correction,
        anchors=anchors,
        accepted_anchor=accepted,
        hard_landmark_anchor=hard_landmark,
        report=report,
    )


def _merge_landmark_anchors(
    radial: np.ndarray,
    confidence: np.ndarray,
    hard: np.ndarray,
    landmark_points_target: Any,
    settings: GaussianDepthFeedbackOptions,
) -> int:
    points = np.asarray(landmark_points_target, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("landmark_points_target must have shape (N, 3)")
    ranges = np.linalg.norm(points, axis=1)
    valid = (
        np.all(np.isfinite(points), axis=1)
        & np.isfinite(ranges)
        & (ranges >= settings.min_range_m)
        & (ranges <= settings.max_range_m)
    )
    points = points[valid]
    ranges = ranges[valid]
    if ranges.size == 0:
        return 0
    rows, columns = _project_nearest_pixels(points, radial.shape)
    # Duplicate landmarks in one pixel are combined in log range.
    sums = np.zeros(radial.size, dtype=np.float64)
    counts = np.zeros(radial.size, dtype=np.uint16)
    flat = rows * radial.shape[1] + columns
    np.add.at(sums, flat, np.log(ranges))
    np.add.at(counts, flat, 1)
    selected = counts > 0
    radial.flat[selected] = np.exp(sums[selected] / counts[selected]).astype(np.float32)
    confidence.flat[selected] = 1.0
    hard.flat[selected] = True
    return int(ranges.size)


def _enforce_native_landmark_pixels(
    refined: np.ndarray,
    confidence: np.ndarray,
    landmark_points_target: Any,
    settings: GaussianDepthFeedbackOptions,
) -> None:
    points = np.asarray(landmark_points_target, dtype=np.float64)
    ranges = np.linalg.norm(points, axis=1)
    valid = (
        np.all(np.isfinite(points), axis=1)
        & np.isfinite(ranges)
        & (ranges >= settings.min_range_m)
        & (ranges <= settings.max_range_m)
    )
    rows, columns = _project_nearest_pixels(points[valid], refined.shape)
    refined[rows, columns] = ranges[valid].astype(np.float32)
    confidence[rows, columns] = 1.0


def _solve_edge_aware_residual(
    anchor_residual: np.ndarray,
    data_weight: np.ndarray,
    anchor_confidence: np.ndarray,
    rgb_hwc: np.ndarray,
    settings: GaussianDepthFeedbackOptions,
) -> tuple[np.ndarray, np.ndarray]:
    colors = rgb_hwc.astype(np.float32) / float(np.iinfo(rgb_hwc.dtype).max)
    left_color = np.roll(colors, 1, axis=1)
    right_color = np.roll(colors, -1, axis=1)
    up_color = np.empty_like(colors)
    down_color = np.empty_like(colors)
    up_color[0] = colors[0]
    up_color[1:] = colors[:-1]
    down_color[-1] = colors[-1]
    down_color[:-1] = colors[1:]
    left_weight = np.exp(
        -np.mean(np.abs(colors - left_color), axis=2) / settings.color_sigma
    ).astype(np.float32)
    right_weight = np.exp(
        -np.mean(np.abs(colors - right_color), axis=2) / settings.color_sigma
    ).astype(np.float32)
    up_weight = np.exp(
        -np.mean(np.abs(colors - up_color), axis=2) / settings.color_sigma
    ).astype(np.float32)
    down_weight = np.exp(
        -np.mean(np.abs(colors - down_color), axis=2) / settings.color_sigma
    ).astype(np.float32)
    up_weight[0] = 0.0
    down_weight[-1] = 0.0
    smooth_denominator = settings.smoothness_weight * (
        left_weight + right_weight + up_weight + down_weight
    )
    denominator = settings.prior_residual_weight + data_weight + smooth_denominator
    correction = np.zeros(anchor_residual.shape, dtype=np.float32)
    propagated_confidence = np.zeros(anchor_residual.shape, dtype=np.float32)
    for _ in range(settings.iterations):
        left = np.roll(correction, 1, axis=1)
        right = np.roll(correction, -1, axis=1)
        up = np.empty_like(correction)
        down = np.empty_like(correction)
        up[0] = 0.0
        up[1:] = correction[:-1]
        down[-1] = 0.0
        down[:-1] = correction[1:]
        correction = (
            data_weight * anchor_residual
            + settings.smoothness_weight
            * (
                left_weight * left
                + right_weight * right
                + up_weight * up
                + down_weight * down
            )
        ) / denominator
        correction = np.clip(
            correction,
            -settings.maximum_log_correction,
            settings.maximum_log_correction,
        )

        confidence_left = np.roll(propagated_confidence, 1, axis=1)
        confidence_right = np.roll(propagated_confidence, -1, axis=1)
        confidence_up = np.empty_like(propagated_confidence)
        confidence_down = np.empty_like(propagated_confidence)
        confidence_up[0] = 0.0
        confidence_up[1:] = propagated_confidence[:-1]
        confidence_down[-1] = 0.0
        confidence_down[:-1] = propagated_confidence[1:]
        propagated_confidence = (
            data_weight * anchor_confidence
            + settings.smoothness_weight
            * (
                left_weight * confidence_left
                + right_weight * confidence_right
                + up_weight * confidence_up
                + down_weight * confidence_down
            )
        ) / denominator
        propagated_confidence = np.clip(propagated_confidence, 0.0, 1.0)
    return correction.astype(np.float32), propagated_confidence.astype(np.float32)


def _resize_valid_continuous(
    values_hw: np.ndarray, output_shape_hw: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values_hw) & (values_hw > 0.0)
    width = output_shape_hw[1]
    height = output_shape_hw[0]
    interpolation = (
        cv2.INTER_AREA
        if height <= values_hw.shape[0] and width <= values_hw.shape[1]
        else cv2.INTER_LINEAR
    )
    numerator = cv2.resize(
        np.where(valid, values_hw, 0.0).astype(np.float32),
        (width, height),
        interpolation=interpolation,
    )
    support = cv2.resize(
        valid.astype(np.float32),
        (width, height),
        interpolation=interpolation,
    )
    resized = numerator / np.maximum(support, 1e-6)
    return resized.astype(np.float32), support >= 0.99


def _project_nearest_pixels(
    points_xyz: np.ndarray, shape_hw: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    ranges = np.linalg.norm(points_xyz, axis=1)
    bearings = points_xyz / np.maximum(ranges[:, None], 1e-12)
    height, width = shape_hw
    longitude = np.arctan2(bearings[:, 0], bearings[:, 2])
    latitude = np.arcsin(np.clip(bearings[:, 1], -1.0, 1.0))
    columns = np.mod(
        np.rint((longitude + np.pi) / (2.0 * np.pi) * width - 0.5).astype(np.int64),
        width,
    )
    rows = np.clip(
        np.rint((np.pi / 2.0 - latitude) / np.pi * height - 0.5).astype(np.int64),
        0,
        height - 1,
    )
    return rows, columns


__all__ = [
    "GaussianDepthAnchors",
    "GaussianDepthFeedbackOptions",
    "GaussianDepthFeedbackResult",
    "PLY_VERTEX_DTYPE",
    "rasterize_gaussian_depth_anchors",
    "read_gaussian_centres_ply",
    "refine_depth_from_gaussian_anchors",
]
