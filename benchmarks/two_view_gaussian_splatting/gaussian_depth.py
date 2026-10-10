"""Small differentiable spherical Gaussian renderer for a two-view test.

This is deliberately a bounded geometry experiment, not a replacement for a
CUDA 3DGS renderer.  One Gaussian is initialized per supported target pixel.
The compatibility route optimizes one smooth log-range field.  The opt-in full
hierarchy unlocks strongly regularized covariance, opacity, and view-independent
color residuals one factor at a time.  Camera poses, sparse landmarks, Gaussian
count, and tangential mean coordinates always remain fixed.  The restriction
makes the experiment useful for answering whether Gaussian splatting can
densify an already constrained visible surface without quietly hallucinating
unobserved geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class GaussianDepthStage:
    """One bounded level of coarse-to-fine Gaussian geometry refinement."""

    correction_shape_hw: tuple[int, int]
    iterations: int
    learning_rate: float
    optimize_radial: bool = True
    surface_aligned: bool = False
    optimize_covariance: bool = False
    covariance_learning_rate: float = 0.01
    optimize_opacity: bool = False
    opacity_learning_rate: float = 0.004
    optimize_color: bool = False
    color_learning_rate: float = 0.002

    def __post_init__(self) -> None:
        height, width = self.correction_shape_hw
        if height < 2 or width < 4:
            raise ValueError("stage correction_shape_hw must be at least (2, 4)")
        if self.iterations < 0:
            raise ValueError("stage iterations must be nonnegative")
        for name in (
            "learning_rate",
            "covariance_learning_rate",
            "opacity_learning_rate",
            "color_learning_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"stage {name} must be positive and finite")
        if self.optimize_covariance and not self.surface_aligned:
            raise ValueError(
                "optimize_covariance requires a surface-aligned covariance basis"
            )
        if not any(
            (
                self.optimize_radial,
                self.optimize_covariance,
                self.optimize_opacity,
                self.optimize_color,
            )
        ):
            raise ValueError("a stage must optimize at least one Gaussian factor")


@dataclass(frozen=True, slots=True)
class GaussianDepthOptions:
    """Frozen optimization settings for the feasibility experiment."""

    correction_shape_hw: tuple[int, int] = (16, 32)
    iterations: int = 40
    learning_rate: float = 0.035
    gaussian_sigma_px: float = 0.85
    gaussian_radius_px: int = 1
    opacity: float = 0.92
    occlusion_tau_m: float = 0.08
    minimum_training_coverage: float = 0.08
    photometric_weight: float = 1.0
    prior_weight: float = 0.04
    smoothness_weight: float = 0.12
    landmark_weight: float = 2.0
    landmark_huber_delta_log: float = 0.06
    max_abs_log_correction: float = math.log(1.5)
    min_range_m: float = 0.3
    max_range_m: float = 15.0
    stages: tuple[GaussianDepthStage, ...] | None = None
    stage_residual_weight: float = 0.08
    covariance_prior_weight: float = 0.20
    covariance_smoothness_weight: float = 0.16
    max_abs_log_covariance_scale: float = math.log(1.25)
    opacity_prior_weight: float = 0.30
    opacity_smoothness_weight: float = 0.16
    max_abs_opacity_logit_delta: float = 1.0
    color_prior_weight: float = 0.45
    color_smoothness_weight: float = 0.20
    max_abs_color_delta: float = 0.08

    def __post_init__(self) -> None:
        gh, gw = self.correction_shape_hw
        if gh < 2 or gw < 4:
            raise ValueError("correction_shape_hw must be at least (2, 4)")
        if self.iterations < 0:
            raise ValueError("iterations must be nonnegative")
        for name in (
            "learning_rate",
            "gaussian_sigma_px",
            "opacity",
            "occlusion_tau_m",
            "photometric_weight",
            "landmark_weight",
            "landmark_huber_delta_log",
            "max_abs_log_correction",
            "min_range_m",
            "max_range_m",
            "covariance_prior_weight",
            "covariance_smoothness_weight",
            "max_abs_log_covariance_scale",
            "opacity_prior_weight",
            "opacity_smoothness_weight",
            "max_abs_opacity_logit_delta",
            "color_prior_weight",
            "color_smoothness_weight",
            "max_abs_color_delta",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.gaussian_radius_px < 1:
            raise ValueError("gaussian_radius_px must be at least one")
        if not 0.0 < self.opacity <= 1.0:
            raise ValueError("opacity must be in (0, 1]")
        if not 0.0 <= self.minimum_training_coverage <= 1.0:
            raise ValueError("minimum_training_coverage must be in [0, 1]")
        if self.max_range_m <= self.min_range_m:
            raise ValueError("max_range_m must exceed min_range_m")
        if (
            not math.isfinite(self.stage_residual_weight)
            or self.stage_residual_weight < 0
        ):
            raise ValueError("stage_residual_weight must be finite and nonnegative")
        if self.stages is not None:
            if not self.stages:
                raise ValueError("stages must be None or contain at least one stage")
            previous = (0, 0)
            for stage in self.stages:
                shape = stage.correction_shape_hw
                if shape[0] < previous[0] or shape[1] < previous[1]:
                    raise ValueError("stage correction shapes must be nondecreasing")
                if not stage.optimize_radial and shape != previous:
                    raise ValueError(
                        "a non-radial stage must preserve the preceding shape"
                    )
                previous = shape

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolved_stages(self) -> tuple[GaussianDepthStage, ...]:
        """Return explicit stages while preserving the legacy single-grid route."""

        if self.stages is not None:
            return self.stages
        return (
            GaussianDepthStage(
                correction_shape_hw=self.correction_shape_hw,
                iterations=self.iterations,
                learning_rate=self.learning_rate,
            ),
        )


@dataclass(frozen=True, slots=True)
class GaussianDepthResult:
    """Optimized visible-surface state returned by the hierarchical route."""

    radial_m: np.ndarray
    stage_radial_m: tuple[np.ndarray, ...]
    covariance_log_scales_hw2: np.ndarray | None
    opacity_hw: np.ndarray | None
    color_hwc: np.ndarray | None
    surface_aligned: bool
    report: dict[str, Any]


def recommended_gaussian_depth_stages(
    iterations: tuple[int, int, int, int] = (20, 20, 20, 20),
) -> tuple[GaussianDepthStage, ...]:
    """Return the radial hierarchy followed by covariance-only refinement."""

    if len(iterations) != 4:
        raise ValueError("the recommended hierarchy requires four iteration counts")
    return (
        GaussianDepthStage((16, 32), int(iterations[0]), 0.035),
        GaussianDepthStage((32, 64), int(iterations[1]), 0.025),
        GaussianDepthStage(
            (64, 128),
            int(iterations[2]),
            0.015,
            surface_aligned=True,
        ),
        GaussianDepthStage(
            (64, 128),
            int(iterations[3]),
            0.015,
            optimize_radial=False,
            surface_aligned=True,
            optimize_covariance=True,
            covariance_learning_rate=0.006,
        ),
    )


def recommended_full_factor_gaussian_depth_stages(
    iterations: tuple[int, int, int, int, int, int] = (20, 20, 20, 20, 12, 8),
) -> tuple[GaussianDepthStage, ...]:
    """Unlock means, covariance, opacity, then DC color in conservative order."""

    if len(iterations) != 6:
        raise ValueError("the full factor hierarchy requires six iteration counts")
    radial_and_covariance = recommended_gaussian_depth_stages(iterations[:4])
    shape = radial_and_covariance[-1].correction_shape_hw
    return radial_and_covariance + (
        GaussianDepthStage(
            shape,
            int(iterations[4]),
            0.015,
            optimize_radial=False,
            surface_aligned=True,
            optimize_opacity=True,
            opacity_learning_rate=0.003,
        ),
        GaussianDepthStage(
            shape,
            int(iterations[5]),
            0.015,
            optimize_radial=False,
            surface_aligned=True,
            optimize_color=True,
            color_learning_rate=0.0015,
        ),
    )


def erp_rays(
    shape_hw: tuple[int, int], *, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Return PanorAi-frame unit rays at ERP pixel centers."""

    height, width = shape_hw
    rows = torch.arange(height, dtype=torch.float32, device=device)
    columns = torch.arange(width, dtype=torch.float32, device=device)
    latitude = math.pi / 2.0 - (rows + 0.5) / height * math.pi
    longitude = (columns + 0.5) / width * (2.0 * math.pi) - math.pi
    lat, lon = torch.meshgrid(latitude, longitude, indexing="ij")
    cos_lat = torch.cos(lat)
    return torch.stack(
        (cos_lat * torch.sin(lon), torch.sin(lat), cos_lat * torch.cos(lon)),
        dim=-1,
    )


def project_erp(
    points_camera: torch.Tensor, shape_hw: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project camera-frame 3D points to ERP coordinates and radial range."""

    if points_camera.ndim != 2 or points_camera.shape[1] != 3:
        raise ValueError("points_camera must have shape (N, 3)")
    height, width = shape_hw
    radial = torch.linalg.vector_norm(points_camera, dim=1)
    safe = torch.clamp(radial, min=torch.finfo(points_camera.dtype).eps)
    longitude = torch.atan2(points_camera[:, 0], points_camera[:, 2])
    latitude = torch.asin(torch.clamp(points_camera[:, 1] / safe, -1.0, 1.0))
    u = (longitude + math.pi) / (2.0 * math.pi) * width - 0.5
    v = (math.pi / 2.0 - latitude) / math.pi * height - 0.5
    return u, v, radial


def align_depth_scale_from_landmarks(
    prior_range_hw: Any,
    landmark_points_target: Any,
    *,
    min_range_m: float = 0.3,
    max_range_m: float = 15.0,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Robustly align one relative/biased range map to BA landmark ranges.

    A single positive scale is fitted in log range with a median estimator.
    The model intentionally avoids an unconstrained affine inverse-depth fit:
    with sparse landmarks that fit can reverse depth ordering while reducing
    residuals, which is unacceptable for geometric initialization.
    """

    prior = np.asarray(prior_range_hw, dtype=np.float64)
    points = np.asarray(landmark_points_target, dtype=np.float64)
    if prior.ndim != 2:
        raise ValueError("prior_range_hw must have shape (H, W)")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("landmark_points_target must have shape (N, 3)")
    landmark_range = np.linalg.norm(points, axis=1)
    bearings = points / np.maximum(landmark_range[:, None], 1e-12)
    sampled = _sample_erp_numpy(prior, bearings)
    valid = (
        np.isfinite(sampled)
        & np.isfinite(landmark_range)
        & (sampled > 0.0)
        & (landmark_range >= min_range_m)
        & (landmark_range <= max_range_m)
    )
    if int(valid.sum()) < 3:
        raise ValueError("at least three valid landmark/prior pairs are required")
    log_offset = np.log(landmark_range[valid]) - np.log(sampled[valid])
    log_scale = float(np.median(log_offset))
    scale = float(math.exp(log_scale))
    aligned = np.clip(prior * scale, min_range_m, max_range_m)
    before = np.abs(np.log(sampled[valid] / landmark_range[valid]))
    after_values = np.clip(sampled[valid] * scale, min_range_m, max_range_m)
    after = np.abs(np.log(after_values / landmark_range[valid]))
    return aligned.astype(np.float32), {
        "landmark_count": int(valid.sum()),
        "scale": scale,
        "median_abs_log_before": float(np.median(before)),
        "median_abs_log_after": float(np.median(after)),
        "p90_abs_log_after": float(np.quantile(after, 0.9)),
    }


def resize_periodic_field(
    field_hw_or_hwc: Any, output_shape_hw: tuple[int, int]
) -> np.ndarray:
    """Bilinearly resize an ERP field with longitude-periodic boundaries."""

    field = np.asarray(field_hw_or_hwc, dtype=np.float32)
    if field.ndim not in (2, 3):
        raise ValueError("field must have shape (H, W) or (H, W, C)")
    input_height, input_width = field.shape[:2]
    output_height, output_width = output_shape_hw
    if min(input_height, input_width, output_height, output_width) < 1:
        raise ValueError("input and output shapes must be nonempty")
    squeezed = field.ndim == 2
    values = field[..., None] if squeezed else field

    source_y = (
        np.arange(output_height, dtype=np.float64) + 0.5
    ) * input_height / output_height - 0.5
    source_x = (
        np.arange(output_width, dtype=np.float64) + 0.5
    ) * input_width / output_width - 0.5
    y0_raw = np.floor(source_y).astype(np.int64)
    x0_raw = np.floor(source_x).astype(np.int64)
    y_weight = np.clip(source_y - y0_raw, 0.0, 1.0).astype(np.float32)
    x_weight = (source_x - x0_raw).astype(np.float32)
    y0 = np.clip(y0_raw, 0, input_height - 1)
    y1 = np.clip(y0_raw + 1, 0, input_height - 1)
    x0 = np.mod(x0_raw, input_width)
    x1 = np.mod(x0_raw + 1, input_width)

    top = (
        values[y0[:, None], x0[None]] * (1.0 - x_weight)[None, :, None]
        + values[y0[:, None], x1[None]] * x_weight[None, :, None]
    )
    bottom = (
        values[y1[:, None], x0[None]] * (1.0 - x_weight)[None, :, None]
        + values[y1[:, None], x1[None]] * x_weight[None, :, None]
    )
    resized = (
        top * (1.0 - y_weight)[:, None, None] + bottom * y_weight[:, None, None]
    ).astype(np.float32)
    return resized[..., 0] if squeezed else resized


def transfer_log_range_correction(
    aligned_range_optimization_hw: Any,
    optimized_range_optimization_hw: Any,
    aligned_range_output_hw: Any,
    *,
    min_range_m: float = 0.3,
    max_range_m: float = 15.0,
    max_abs_log_correction: float = math.log(1.5),
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the learned low-resolution log-range correction at another size.

    The dense output starts from the independently resized high-resolution
    monocular prior.  Only the bounded multiplicative correction learned by
    the Gaussian optimizer is transferred; low-resolution depth samples are
    never enlarged into the final product.
    """

    aligned_optimization = np.asarray(aligned_range_optimization_hw, dtype=np.float32)
    optimized_optimization = np.asarray(
        optimized_range_optimization_hw, dtype=np.float32
    )
    aligned_output = np.asarray(aligned_range_output_hw, dtype=np.float32)
    if aligned_optimization.shape != optimized_optimization.shape:
        raise ValueError("optimization aligned and optimized ranges must agree")
    if aligned_optimization.ndim != 2 or aligned_output.ndim != 2:
        raise ValueError("range inputs must have shape (H, W)")
    valid = (
        np.isfinite(aligned_optimization)
        & np.isfinite(optimized_optimization)
        & (aligned_optimization > 0.0)
        & (optimized_optimization > 0.0)
    )
    correction = np.zeros_like(aligned_optimization, dtype=np.float32)
    correction[valid] = np.log(
        optimized_optimization[valid] / aligned_optimization[valid]
    )
    correction = np.clip(correction, -max_abs_log_correction, max_abs_log_correction)
    transferred = resize_periodic_field(correction, aligned_output.shape)
    output = aligned_output * np.exp(transferred)
    output = np.where(
        np.isfinite(aligned_output) & (aligned_output > 0.0),
        np.clip(output, min_range_m, max_range_m),
        np.nan,
    )
    return output.astype(np.float32), transferred.astype(np.float32)


def render_spherical_gaussians(
    target_rgb_hwc: torch.Tensor,
    target_range_hw: torch.Tensor,
    target_valid_hw: torch.Tensor,
    rotation_view_from_target: torch.Tensor,
    translation_view_from_target: torch.Tensor,
    *,
    extra_points_target: torch.Tensor | None = None,
    extra_colors: torch.Tensor | None = None,
    sigma_px: float = 0.85,
    radius_px: int = 1,
    opacity: float = 0.92,
    opacity_hw: torch.Tensor | None = None,
    occlusion_tau_m: float = 0.08,
    surface_aligned: bool = False,
    surface_depth_edge_log: float = 0.16,
    covariance_log_scales_hw2: torch.Tensor | None = None,
    projected_covariance_mode: str = "legacy",
    point_chunk_size: int | None = None,
    compositing_mode: str = "normalized",
    alpha_depth_bins: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render supported target Gaussians into a posed ERP view.

    ``normalized`` preserves the legacy soft nearest-depth accumulation.
    ``alpha`` performs front-to-back compositing: exact per-pixel sorting on
    the materialized route and bounded depth-bin compositing on the chunked
    high-resolution route.  ``opacity_hw`` supplies confidence/learned alpha
    without changing Gaussian means or colors.
    """

    if target_rgb_hwc.ndim != 3 or target_rgb_hwc.shape[2] != 3:
        raise ValueError("target_rgb_hwc must have shape (H, W, 3)")
    height, width = target_range_hw.shape
    if target_rgb_hwc.shape[:2] != (height, width):
        raise ValueError("RGB and range shapes must agree")
    if target_valid_hw.shape != (height, width):
        raise ValueError("target_valid_hw must match target range")
    if opacity_hw is not None and opacity_hw.shape != (height, width):
        raise ValueError("opacity_hw must match target range")
    if covariance_log_scales_hw2 is not None:
        if not surface_aligned:
            raise ValueError("covariance_log_scales_hw2 requires surface_aligned=True")
        if covariance_log_scales_hw2.shape != (height, width, 2):
            raise ValueError("covariance_log_scales_hw2 must have shape (H, W, 2)")
    if projected_covariance_mode not in ("legacy", "jacobian"):
        raise ValueError("projected_covariance_mode must be 'legacy' or 'jacobian'")
    if point_chunk_size is not None and point_chunk_size < 1:
        raise ValueError("point_chunk_size must be positive when provided")
    if compositing_mode not in ("normalized", "alpha"):
        raise ValueError("compositing_mode must be 'normalized' or 'alpha'")
    if alpha_depth_bins < 2:
        raise ValueError("alpha_depth_bins must be at least two")
    rays = erp_rays((height, width), device=target_range_hw.device)
    valid = target_valid_hw & torch.isfinite(target_range_hw) & (target_range_hw > 0.0)
    selected_rays = rays[valid]
    selected_range = target_range_hw[valid]
    colors = target_rgb_hwc[valid]
    if opacity_hw is None:
        opacities = torch.full_like(selected_range, float(opacity))
    else:
        opacities = torch.clamp(opacity_hw[valid], 0.0, 1.0) * float(opacity)
    points_target = selected_rays * selected_range[:, None]
    inverse_covariance: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    if surface_aligned:
        inverse_covariance = _projected_surface_inverse_covariance(
            target_range_hw,
            valid,
            rotation_view_from_target,
            translation_view_from_target,
            lowpass_sigma_px=sigma_px,
            maximum_sigma_px=max(
                float(sigma_px),
                float(radius_px)
                / (2.0 if projected_covariance_mode == "jacobian" else 3.0),
            ),
            depth_edge_log=surface_depth_edge_log,
            covariance_log_scales_hw2=covariance_log_scales_hw2,
            centered_jacobian=projected_covariance_mode == "jacobian",
        )
    if extra_points_target is not None or extra_colors is not None:
        if extra_points_target is None or extra_colors is None:
            raise ValueError(
                "extra_points_target and extra_colors must be provided together"
            )
        if extra_points_target.ndim != 2 or extra_points_target.shape[1] != 3:
            raise ValueError("extra_points_target must have shape (M, 3)")
        if extra_colors.shape != extra_points_target.shape:
            raise ValueError("extra_colors must have shape (M, 3)")
        points_target = torch.cat(
            (points_target, extra_points_target.to(points_target)), dim=0
        )
        colors = torch.cat((colors, extra_colors.to(colors)), dim=0)
        opacities = torch.cat(
            (
                opacities,
                torch.full(
                    (extra_points_target.shape[0],),
                    float(opacity),
                    dtype=opacities.dtype,
                    device=opacities.device,
                ),
            )
        )
        if inverse_covariance is not None:
            extra_count = extra_points_target.shape[0]
            isotropic_inverse = 1.0 / float(sigma_px * sigma_px)
            inverse_covariance = (
                torch.cat(
                    (
                        inverse_covariance[0],
                        torch.full(
                            (extra_count,),
                            isotropic_inverse,
                            dtype=target_range_hw.dtype,
                            device=target_range_hw.device,
                        ),
                    )
                ),
                torch.cat(
                    (
                        inverse_covariance[1],
                        torch.zeros(
                            (extra_count,),
                            dtype=target_range_hw.dtype,
                            device=target_range_hw.device,
                        ),
                    )
                ),
                torch.cat(
                    (
                        inverse_covariance[2],
                        torch.full(
                            (extra_count,),
                            isotropic_inverse,
                            dtype=target_range_hw.dtype,
                            device=target_range_hw.device,
                        ),
                    )
                ),
            )
    points_view = (
        points_target @ rotation_view_from_target.T + translation_view_from_target[None]
    )
    u, v, radial = project_erp(points_view, (height, width))
    finite = (
        torch.isfinite(u) & torch.isfinite(v) & torch.isfinite(radial) & (radial > 0.0)
    )
    u = u[finite]
    v = v[finite]
    radial = radial[finite]
    colors = colors[finite]
    opacities = opacities[finite]
    if inverse_covariance is not None:
        inverse_covariance = tuple(
            component[finite] for component in inverse_covariance
        )
    if u.numel() == 0:
        zeros_rgb = torch.zeros_like(target_rgb_hwc)
        zeros = torch.zeros_like(target_range_hw)
        return zeros_rgb, zeros, torch.full_like(target_range_hw, float("nan"))

    if point_chunk_size is not None:
        return _rasterize_projected_gaussians_chunked(
            u,
            v,
            radial,
            colors,
            opacities,
            inverse_covariance,
            shape_hw=(height, width),
            sigma_px=sigma_px,
            radius_px=radius_px,
            occlusion_tau_m=occlusion_tau_m,
            point_chunk_size=point_chunk_size,
            compositing_mode=compositing_mode,
            alpha_depth_bins=alpha_depth_bins,
        )

    base_u = torch.floor(u).to(torch.int64)
    base_v = torch.floor(v).to(torch.int64)
    pixel_indices: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    depths: list[torch.Tensor] = []
    repeated_colors: list[torch.Tensor] = []
    inv_two_sigma_sq = 0.5 / float(sigma_px * sigma_px)
    for dy in range(-radius_px, radius_px + 1):
        iy = base_v + dy
        vertical = (iy >= 0) & (iy < height)
        if not torch.any(vertical):
            continue
        for dx in range(-radius_px, radius_px + 1):
            ix_unwrapped = base_u + dx
            delta_u = u - ix_unwrapped.to(u.dtype)
            delta_v = v - iy.to(v.dtype)
            if inverse_covariance is None:
                exponent = (delta_u * delta_u + delta_v * delta_v) * inv_two_sigma_sq
            else:
                inv00, inv01, inv11 = inverse_covariance
                exponent = 0.5 * (
                    inv00 * delta_u * delta_u
                    + 2.0 * inv01 * delta_u * delta_v
                    + inv11 * delta_v * delta_v
                )
            gaussian_weight = opacities * torch.exp(-exponent)
            contribution_valid = vertical & (gaussian_weight > 1e-5)
            if not torch.any(contribution_valid):
                continue
            ix = torch.remainder(ix_unwrapped[contribution_valid], width)
            iy_valid = iy[contribution_valid]
            pixel_indices.append(iy_valid * width + ix)
            weights.append(gaussian_weight[contribution_valid])
            depths.append(radial[contribution_valid])
            repeated_colors.append(colors[contribution_valid])

    flat_index = torch.cat(pixel_indices)
    weight = torch.cat(weights)
    depth = torch.cat(depths)
    contribution_rgb = torch.cat(repeated_colors)
    pixel_count = height * width
    if compositing_mode == "alpha":
        return _compose_alpha_sorted(
            flat_index,
            weight,
            depth,
            contribution_rgb,
            shape_hw=(height, width),
        )
    nearest = torch.full(
        (pixel_count,), float("inf"), dtype=depth.dtype, device=depth.device
    )
    nearest.scatter_reduce_(
        0, flat_index, depth.detach(), reduce="amin", include_self=True
    )
    relative_depth = torch.clamp(depth - nearest[flat_index], min=0.0)
    visibility = torch.exp(-relative_depth / float(occlusion_tau_m))
    weight = weight * visibility

    weight_sum = torch.zeros((pixel_count,), dtype=weight.dtype, device=weight.device)
    weight_sum.scatter_add_(0, flat_index, weight)
    rgb_sum = torch.zeros((pixel_count, 3), dtype=weight.dtype, device=weight.device)
    rgb_sum.scatter_add_(
        0, flat_index[:, None].expand(-1, 3), weight[:, None] * contribution_rgb
    )
    depth_sum = torch.zeros((pixel_count,), dtype=weight.dtype, device=weight.device)
    depth_sum.scatter_add_(0, flat_index, weight * depth)
    safe_weight = torch.clamp(weight_sum, min=1e-8)
    rendered_rgb = (rgb_sum / safe_weight[:, None]).reshape(height, width, 3)
    rendered_depth = (depth_sum / safe_weight).reshape(height, width)
    rendered_depth = torch.where(
        weight_sum.reshape(height, width) > 0.0,
        rendered_depth,
        torch.full_like(rendered_depth, float("nan")),
    )
    coverage = (1.0 - torch.exp(-weight_sum)).reshape(height, width)
    return rendered_rgb, coverage, rendered_depth


def _iter_projected_gaussian_contributions(
    u: torch.Tensor,
    v: torch.Tensor,
    radial: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    inverse_covariance: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    *,
    shape_hw: tuple[int, int],
    sigma_px: float,
    radius_px: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield one bounded projected contribution block at a time."""

    height, width = shape_hw
    base_u = torch.floor(u).to(torch.int64)
    base_v = torch.floor(v).to(torch.int64)
    inv_two_sigma_sq = 0.5 / float(sigma_px * sigma_px)
    for dy in range(-radius_px, radius_px + 1):
        iy = base_v + dy
        vertical = (iy >= 0) & (iy < height)
        if not torch.any(vertical):
            continue
        for dx in range(-radius_px, radius_px + 1):
            ix_unwrapped = base_u + dx
            delta_u = u - ix_unwrapped.to(u.dtype)
            delta_v = v - iy.to(v.dtype)
            if inverse_covariance is None:
                exponent = (delta_u * delta_u + delta_v * delta_v) * inv_two_sigma_sq
            else:
                inv00, inv01, inv11 = inverse_covariance
                exponent = 0.5 * (
                    inv00 * delta_u * delta_u
                    + 2.0 * inv01 * delta_u * delta_v
                    + inv11 * delta_v * delta_v
                )
            gaussian_weight = opacities * torch.exp(-exponent)
            contribution_valid = vertical & (gaussian_weight > 1e-5)
            if not torch.any(contribution_valid):
                continue
            ix = torch.remainder(ix_unwrapped[contribution_valid], width)
            iy_valid = iy[contribution_valid]
            yield (
                iy_valid * width + ix,
                gaussian_weight[contribution_valid],
                radial[contribution_valid],
                colors[contribution_valid],
            )


def _rasterize_projected_gaussians_chunked(
    u: torch.Tensor,
    v: torch.Tensor,
    radial: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    inverse_covariance: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    *,
    shape_hw: tuple[int, int],
    sigma_px: float,
    radius_px: int,
    occlusion_tau_m: float,
    point_chunk_size: int,
    compositing_mode: str,
    alpha_depth_bins: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rasterize in two passes without materializing every splat contribution."""

    height, width = shape_hw
    pixel_count = height * width
    nearest = torch.full(
        (pixel_count,), float("inf"), dtype=radial.dtype, device=radial.device
    )
    point_count = int(u.numel())
    for start in range(0, point_count, point_chunk_size):
        stop = min(point_count, start + point_chunk_size)
        chunk_covariance = (
            tuple(component[start:stop] for component in inverse_covariance)
            if inverse_covariance is not None
            else None
        )
        for flat_index, _, depth, _ in _iter_projected_gaussian_contributions(
            u[start:stop],
            v[start:stop],
            radial[start:stop],
            colors[start:stop],
            opacities[start:stop],
            chunk_covariance,
            shape_hw=shape_hw,
            sigma_px=sigma_px,
            radius_px=radius_px,
        ):
            nearest.scatter_reduce_(
                0, flat_index, depth.detach(), reduce="amin", include_self=True
            )

    if compositing_mode == "alpha":
        return _rasterize_alpha_depth_bins(
            u,
            v,
            radial,
            colors,
            opacities,
            inverse_covariance,
            nearest,
            shape_hw=shape_hw,
            sigma_px=sigma_px,
            radius_px=radius_px,
            depth_bin_width_m=occlusion_tau_m,
            point_chunk_size=point_chunk_size,
            depth_bins=alpha_depth_bins,
        )

    weight_sum = torch.zeros((pixel_count,), dtype=radial.dtype, device=radial.device)
    rgb_sum = torch.zeros((pixel_count, 3), dtype=radial.dtype, device=radial.device)
    depth_sum = torch.zeros((pixel_count,), dtype=radial.dtype, device=radial.device)
    for start in range(0, point_count, point_chunk_size):
        stop = min(point_count, start + point_chunk_size)
        chunk_covariance = (
            tuple(component[start:stop] for component in inverse_covariance)
            if inverse_covariance is not None
            else None
        )
        for (
            flat_index,
            weight,
            depth,
            contribution_rgb,
        ) in _iter_projected_gaussian_contributions(
            u[start:stop],
            v[start:stop],
            radial[start:stop],
            colors[start:stop],
            opacities[start:stop],
            chunk_covariance,
            shape_hw=shape_hw,
            sigma_px=sigma_px,
            radius_px=radius_px,
        ):
            relative_depth = torch.clamp(depth - nearest[flat_index], min=0.0)
            visible_weight = weight * torch.exp(
                -relative_depth / float(occlusion_tau_m)
            )
            weight_sum.scatter_add_(0, flat_index, visible_weight)
            rgb_sum.scatter_add_(
                0,
                flat_index[:, None].expand(-1, 3),
                visible_weight[:, None] * contribution_rgb,
            )
            depth_sum.scatter_add_(0, flat_index, visible_weight * depth)

    safe_weight = torch.clamp(weight_sum, min=1e-8)
    rendered_rgb = (rgb_sum / safe_weight[:, None]).reshape(height, width, 3)
    rendered_depth = (depth_sum / safe_weight).reshape(height, width)
    rendered_depth = torch.where(
        weight_sum.reshape(height, width) > 0.0,
        rendered_depth,
        torch.full_like(rendered_depth, float("nan")),
    )
    coverage = (1.0 - torch.exp(-weight_sum)).reshape(height, width)
    return rendered_rgb, coverage, rendered_depth


def _compose_alpha_sorted(
    flat_index: torch.Tensor,
    alpha: torch.Tensor,
    depth: torch.Tensor,
    colors: torch.Tensor,
    *,
    shape_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exactly alpha-compose materialized contributions front to back."""

    height, width = shape_hw
    pixel_count = height * width
    # Stable pixel sorting after a stable depth sort preserves ascending depth
    # inside each pixel segment without encoding floats into integer keys.
    depth_order = torch.argsort(depth.detach(), stable=True)
    pixel_order = torch.argsort(flat_index[depth_order], stable=True)
    order = depth_order[pixel_order]
    sorted_index = flat_index[order]
    sorted_alpha = torch.clamp(alpha[order], 0.0, 0.999)
    sorted_depth = depth[order]
    sorted_colors = colors[order]

    log_remaining = torch.log1p(-sorted_alpha)
    cumulative_log = torch.cumsum(log_remaining, dim=0)
    element_index = torch.arange(
        sorted_index.numel(), dtype=torch.int64, device=sorted_index.device
    )
    segment_start = torch.ones_like(sorted_index, dtype=torch.bool)
    segment_start[1:] = sorted_index[1:] != sorted_index[:-1]
    start_index = torch.where(
        segment_start, element_index, torch.zeros_like(element_index)
    )
    current_start = torch.cummax(start_index, dim=0).values
    prefix_position = torch.clamp(current_start - 1, min=0)
    prefix_log = torch.where(
        current_start > 0,
        cumulative_log[prefix_position],
        torch.zeros_like(cumulative_log),
    )
    log_transmittance_before = cumulative_log - log_remaining - prefix_log
    contribution = torch.exp(log_transmittance_before) * sorted_alpha

    coverage_flat = torch.zeros((pixel_count,), dtype=alpha.dtype, device=alpha.device)
    coverage_flat.scatter_add_(0, sorted_index, contribution)
    rgb_flat = torch.zeros((pixel_count, 3), dtype=alpha.dtype, device=alpha.device)
    rgb_flat.scatter_add_(
        0,
        sorted_index[:, None].expand(-1, 3),
        contribution[:, None] * sorted_colors,
    )
    depth_flat = torch.zeros((pixel_count,), dtype=alpha.dtype, device=alpha.device)
    depth_flat.scatter_add_(0, sorted_index, contribution * sorted_depth)
    safe_coverage = torch.clamp(coverage_flat, min=1e-8)
    rendered_depth = (depth_flat / safe_coverage).reshape(height, width)
    rendered_depth = torch.where(
        coverage_flat.reshape(height, width) > 0.0,
        rendered_depth,
        torch.full_like(rendered_depth, float("nan")),
    )
    rendered_rgb = (rgb_flat / safe_coverage[:, None]).reshape(height, width, 3)
    return (
        rendered_rgb,
        coverage_flat.reshape(height, width),
        rendered_depth,
    )


def _rasterize_alpha_depth_bins(
    u: torch.Tensor,
    v: torch.Tensor,
    radial: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    inverse_covariance: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    nearest: torch.Tensor,
    *,
    shape_hw: tuple[int, int],
    sigma_px: float,
    radius_px: int,
    depth_bin_width_m: float,
    point_chunk_size: int,
    depth_bins: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bounded-memory alpha compositing using front-to-back depth slabs."""

    height, width = shape_hw
    pixel_count = height * width
    bin_pixel_count = depth_bins * pixel_count
    alpha_sum = torch.zeros(
        (bin_pixel_count,), dtype=radial.dtype, device=radial.device
    )
    log_remaining_sum = torch.zeros_like(alpha_sum)
    depth_alpha_sum = torch.zeros_like(alpha_sum)
    rgb_alpha_sum = torch.zeros(
        (bin_pixel_count, 3), dtype=radial.dtype, device=radial.device
    )
    point_count = int(u.numel())
    bin_width = max(float(depth_bin_width_m), 1e-4)
    for start in range(0, point_count, point_chunk_size):
        stop = min(point_count, start + point_chunk_size)
        chunk_covariance = (
            tuple(component[start:stop] for component in inverse_covariance)
            if inverse_covariance is not None
            else None
        )
        for (
            flat_index,
            alpha,
            depth,
            contribution_rgb,
        ) in _iter_projected_gaussian_contributions(
            u[start:stop],
            v[start:stop],
            radial[start:stop],
            colors[start:stop],
            opacities[start:stop],
            chunk_covariance,
            shape_hw=shape_hw,
            sigma_px=sigma_px,
            radius_px=radius_px,
        ):
            relative_depth = torch.clamp(depth - nearest[flat_index], min=0.0)
            depth_bin = torch.clamp(
                torch.floor(relative_depth / bin_width).to(torch.int64),
                min=0,
                max=depth_bins - 1,
            )
            bin_index = depth_bin * pixel_count + flat_index
            bounded_alpha = torch.clamp(alpha, 0.0, 0.999)
            alpha_sum.scatter_add_(0, bin_index, bounded_alpha)
            log_remaining_sum.scatter_add_(0, bin_index, torch.log1p(-bounded_alpha))
            depth_alpha_sum.scatter_add_(0, bin_index, bounded_alpha * depth)
            rgb_alpha_sum.scatter_add_(
                0,
                bin_index[:, None].expand(-1, 3),
                bounded_alpha[:, None] * contribution_rgb,
            )

    alpha_sum = alpha_sum.reshape(depth_bins, pixel_count)
    bin_opacity = 1.0 - torch.exp(log_remaining_sum.reshape(depth_bins, pixel_count))
    safe_alpha = torch.clamp(alpha_sum, min=1e-8)
    bin_rgb = rgb_alpha_sum.reshape(depth_bins, pixel_count, 3) / safe_alpha[..., None]
    bin_depth = depth_alpha_sum.reshape(depth_bins, pixel_count) / safe_alpha
    transmittance = torch.ones((pixel_count,), dtype=radial.dtype, device=radial.device)
    rgb_flat = torch.zeros((pixel_count, 3), dtype=radial.dtype, device=radial.device)
    depth_flat = torch.zeros_like(transmittance)
    coverage_flat = torch.zeros_like(transmittance)
    for depth_bin in range(depth_bins):
        contribution = transmittance * bin_opacity[depth_bin]
        rgb_flat = rgb_flat + contribution[:, None] * bin_rgb[depth_bin]
        depth_flat = depth_flat + contribution * bin_depth[depth_bin]
        coverage_flat = coverage_flat + contribution
        transmittance = transmittance * (1.0 - bin_opacity[depth_bin])
    safe_coverage = torch.clamp(coverage_flat, min=1e-8)
    rendered_depth = (depth_flat / safe_coverage).reshape(height, width)
    rendered_depth = torch.where(
        coverage_flat.reshape(height, width) > 0.0,
        rendered_depth,
        torch.full_like(rendered_depth, float("nan")),
    )
    rendered_rgb = (rgb_flat / safe_coverage[:, None]).reshape(height, width, 3)
    return (
        rendered_rgb,
        coverage_flat.reshape(height, width),
        rendered_depth,
    )


def _centered_projected_surface_tangents(
    u: torch.Tensor,
    v: torch.Tensor,
    safe_range: torch.Tensor,
    valid: torch.Tensor,
    depth_edge_log: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return centered ERP derivatives with one-sided edge-safe fallbacks."""

    width = u.shape[1]
    left_valid = torch.roll(valid, shifts=1, dims=1)
    right_valid = torch.roll(valid, shifts=-1, dims=1)
    left_range = torch.roll(safe_range, shifts=1, dims=1)
    right_range = torch.roll(safe_range, shifts=-1, dims=1)
    left_consistent = left_valid & (
        torch.abs(torch.log(left_range) - torch.log(safe_range)) <= depth_edge_log
    )
    right_consistent = right_valid & (
        torch.abs(torch.log(right_range) - torch.log(safe_range)) <= depth_edge_log
    )
    left_u = torch.roll(u, shifts=1, dims=1)
    left_v = torch.roll(v, shifts=1, dims=1)
    right_u = torch.roll(u, shifts=-1, dims=1)
    right_v = torch.roll(v, shifts=-1, dims=1)
    right_du = torch.remainder(right_u - u + width / 2.0, width) - width / 2.0
    left_du = torch.remainder(u - left_u + width / 2.0, width) - width / 2.0
    right_dv = right_v - v
    left_dv = v - left_v
    both_x = left_consistent & right_consistent
    tangent_x_u = torch.where(
        both_x,
        0.5 * (left_du + right_du),
        torch.where(
            right_consistent,
            right_du,
            torch.where(left_consistent, left_du, torch.ones_like(u)),
        ),
    )
    tangent_x_v = torch.where(
        both_x,
        0.5 * (left_dv + right_dv),
        torch.where(
            right_consistent,
            right_dv,
            torch.where(left_consistent, left_dv, torch.zeros_like(v)),
        ),
    )

    up_valid = torch.zeros_like(valid)
    up_valid[1:] = valid[:-1]
    down_valid = torch.zeros_like(valid)
    down_valid[:-1] = valid[1:]
    up_range = torch.ones_like(safe_range)
    up_range[1:] = safe_range[:-1]
    down_range = torch.ones_like(safe_range)
    down_range[:-1] = safe_range[1:]
    up_consistent = up_valid & (
        torch.abs(torch.log(up_range) - torch.log(safe_range)) <= depth_edge_log
    )
    down_consistent = down_valid & (
        torch.abs(torch.log(down_range) - torch.log(safe_range)) <= depth_edge_log
    )
    up_u = torch.zeros_like(u)
    up_v = torch.zeros_like(v)
    up_u[1:] = u[:-1]
    up_v[1:] = v[:-1]
    down_u = torch.zeros_like(u)
    down_v = torch.zeros_like(v)
    down_u[:-1] = u[1:]
    down_v[:-1] = v[1:]
    down_du = torch.remainder(down_u - u + width / 2.0, width) - width / 2.0
    up_du = torch.remainder(u - up_u + width / 2.0, width) - width / 2.0
    down_dv = down_v - v
    up_dv = v - up_v
    both_y = up_consistent & down_consistent
    tangent_y_u = torch.where(
        both_y,
        0.5 * (up_du + down_du),
        torch.where(
            down_consistent,
            down_du,
            torch.where(up_consistent, up_du, torch.zeros_like(u)),
        ),
    )
    tangent_y_v = torch.where(
        both_y,
        0.5 * (up_dv + down_dv),
        torch.where(
            down_consistent,
            down_dv,
            torch.where(up_consistent, up_dv, torch.ones_like(v)),
        ),
    )
    return tangent_x_u, tangent_x_v, tangent_y_u, tangent_y_v


def _projected_surface_inverse_covariance(
    target_range_hw: torch.Tensor,
    valid_hw: torch.Tensor,
    rotation_view_from_target: torch.Tensor,
    translation_view_from_target: torch.Tensor,
    *,
    lowpass_sigma_px: float,
    maximum_sigma_px: float,
    depth_edge_log: float,
    covariance_log_scales_hw2: torch.Tensor | None,
    centered_jacobian: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate an EWA ellipse from a discrete projected-surface Jacobian."""

    height, width = target_range_hw.shape
    rays = erp_rays((height, width), device=target_range_hw.device)
    safe_range = torch.where(
        valid_hw, target_range_hw, torch.ones_like(target_range_hw)
    )
    points_target = rays * safe_range[..., None]
    points_view = (
        points_target.reshape(-1, 3) @ rotation_view_from_target.T
        + translation_view_from_target[None]
    )
    u, v, _ = project_erp(points_view, (height, width))
    u = u.reshape(height, width)
    v = v.reshape(height, width)

    if not centered_jacobian:
        right_valid = torch.roll(valid_hw, shifts=-1, dims=1)
        right_range = torch.roll(safe_range, shifts=-1, dims=1)
        right_consistent = right_valid & (
            torch.abs(torch.log(right_range) - torch.log(safe_range)) <= depth_edge_log
        )
        right_u = torch.roll(u, shifts=-1, dims=1)
        right_v = torch.roll(v, shifts=-1, dims=1)
        tangent_x_u = torch.remainder(right_u - u + width / 2.0, width) - width / 2.0
        tangent_x_v = right_v - v
        tangent_x_u = torch.where(right_consistent, tangent_x_u, torch.ones_like(u))
        tangent_x_v = torch.where(right_consistent, tangent_x_v, torch.zeros_like(v))

        down_valid = torch.zeros_like(valid_hw)
        down_valid[:-1] = valid_hw[1:]
        down_range = torch.ones_like(safe_range)
        down_range[:-1] = safe_range[1:]
        down_consistent = down_valid & (
            torch.abs(torch.log(down_range) - torch.log(safe_range)) <= depth_edge_log
        )
        down_u = torch.zeros_like(u)
        down_v = torch.zeros_like(v)
        down_u[:-1] = u[1:]
        down_v[:-1] = v[1:]
        tangent_y_u = torch.remainder(down_u - u + width / 2.0, width) - width / 2.0
        tangent_y_v = down_v - v
        tangent_y_u = torch.where(down_consistent, tangent_y_u, torch.zeros_like(u))
        tangent_y_v = torch.where(down_consistent, tangent_y_v, torch.ones_like(v))
    else:
        tangent_x_u, tangent_x_v, tangent_y_u, tangent_y_v = (
            _centered_projected_surface_tangents(
                u, v, safe_range, valid_hw, depth_edge_log
            )
        )

    footprint_scale = 0.65
    lowpass_variance = float(lowpass_sigma_px * lowpass_sigma_px)
    cov00 = (
        footprint_scale
        * footprint_scale
        * (tangent_x_u * tangent_x_u + tangent_y_u * tangent_y_u)
        + lowpass_variance
    )
    cov01 = (
        footprint_scale
        * footprint_scale
        * (tangent_x_u * tangent_x_v + tangent_y_u * tangent_y_v)
    )
    cov11 = (
        footprint_scale
        * footprint_scale
        * (tangent_x_v * tangent_x_v + tangent_y_v * tangent_y_v)
        + lowpass_variance
    )

    half_trace = 0.5 * (cov00 + cov11)
    root = torch.sqrt(torch.clamp(0.25 * (cov00 - cov11) ** 2 + cov01 * cov01, min=0.0))
    lambda_major = torch.clamp(
        half_trace + root,
        min=lowpass_variance,
        max=float(maximum_sigma_px * maximum_sigma_px),
    )
    lambda_minor = torch.clamp(
        half_trace - root,
        min=lowpass_variance,
        max=float(maximum_sigma_px * maximum_sigma_px),
    )
    if covariance_log_scales_hw2 is not None:
        # The tangent-derived eigenvectors remain fixed.  Only their two
        # principal standard deviations move, so the learned covariance cannot
        # rotate arbitrarily to explain image residuals.
        scale_variance = torch.exp(2.0 * covariance_log_scales_hw2)
        minimum_variance = 0.5 * lowpass_variance
        maximum_variance = max(float((maximum_sigma_px + 1.0) ** 2), lowpass_variance)
        lambda_major = torch.clamp(
            lambda_major * scale_variance[..., 0],
            min=minimum_variance,
            max=maximum_variance,
        )
        lambda_minor = torch.clamp(
            lambda_minor * scale_variance[..., 1],
            min=minimum_variance,
            max=maximum_variance,
        )
    angle = 0.5 * torch.atan2(2.0 * cov01, cov00 - cov11)
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    inverse_major = 1.0 / lambda_major
    inverse_minor = 1.0 / lambda_minor
    inv00 = cosine * cosine * inverse_major + sine * sine * inverse_minor
    inv11 = sine * sine * inverse_major + cosine * cosine * inverse_minor
    inv01 = cosine * sine * (inverse_major - inverse_minor)
    return inv00[valid_hw], inv01[valid_hw], inv11[valid_hw]


def optimize_gaussian_depth(
    target_rgb_hwc: Any,
    aligned_range_hw: Any,
    target_valid_hw: Any,
    source_rgb_hwc: Any,
    source_valid_hw: Any,
    rotation_source_from_target: Any,
    translation_source_from_target: Any,
    landmark_points_target: Any,
    *,
    options: GaussianDepthOptions | None = None,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compatibility wrapper returning radial range and a serializable report."""

    result = optimize_hierarchical_gaussian_depth(
        target_rgb_hwc,
        aligned_range_hw,
        target_valid_hw,
        source_rgb_hwc,
        source_valid_hw,
        rotation_source_from_target,
        translation_source_from_target,
        landmark_points_target,
        options=options,
        device=device,
    )
    return result.radial_m, result.report


def optimize_hierarchical_gaussian_depth(
    target_rgb_hwc: Any,
    aligned_range_hw: Any,
    target_valid_hw: Any,
    source_rgb_hwc: Any,
    source_valid_hw: Any,
    rotation_source_from_target: Any,
    translation_source_from_target: Any,
    landmark_points_target: Any,
    *,
    options: GaussianDepthOptions | None = None,
    device: torch.device | str = "cpu",
) -> GaussianDepthResult:
    """Refine means, covariance, opacity, and color in explicit stages.

    The stage residual is initialized at zero and regularized toward the
    upsampled preceding level.  The optional covariance field has two channels:
    log standard-deviation scales along the tangent-derived major and minor
    image-space axes.  Optional opacity is a bounded logit delta and optional
    color is a bounded view-independent (DC) residual.  Pose, sparse landmarks,
    Gaussian count, and tangential mean coordinates are never optimizer
    parameters.  Higher-order spherical harmonics are intentionally excluded:
    two views do not constrain their angular degrees of freedom.
    """

    settings = options or GaussianDepthOptions()
    compute_device = torch.device(device)
    target_rgb = _float_rgb_tensor(target_rgb_hwc, device=compute_device)
    source_rgb = _float_rgb_tensor(source_rgb_hwc, device=compute_device)
    aligned = torch.as_tensor(
        np.asarray(aligned_range_hw), dtype=torch.float32, device=compute_device
    )
    target_valid = torch.as_tensor(
        np.asarray(target_valid_hw), dtype=torch.bool, device=compute_device
    )
    source_valid = torch.as_tensor(
        np.asarray(source_valid_hw), dtype=torch.bool, device=compute_device
    )
    if target_rgb.shape != source_rgb.shape:
        raise ValueError("target and source RGB images must have the same shape")
    if aligned.shape != target_rgb.shape[:2]:
        raise ValueError("aligned range must match RGB spatial shape")
    rotation = torch.as_tensor(
        np.asarray(rotation_source_from_target),
        dtype=torch.float32,
        device=compute_device,
    )
    translation = torch.as_tensor(
        np.asarray(translation_source_from_target),
        dtype=torch.float32,
        device=compute_device,
    ).reshape(3)
    if rotation.shape != (3, 3):
        raise ValueError("rotation_source_from_target must be 3x3")

    landmarks = np.asarray(landmark_points_target, dtype=np.float64)
    landmark_ranges = np.linalg.norm(landmarks, axis=1)
    landmark_bearings = landmarks / np.maximum(landmark_ranges[:, None], 1e-12)
    landmark_pixels = _erp_pixels_numpy(landmark_bearings, tuple(aligned.shape))
    landmark_x = torch.as_tensor(
        np.rint(landmark_pixels[:, 0]).astype(np.int64), device=compute_device
    )
    landmark_y = torch.as_tensor(
        np.rint(landmark_pixels[:, 1]).astype(np.int64), device=compute_device
    )
    landmark_x = torch.remainder(landmark_x, aligned.shape[1])
    landmark_y = torch.clamp(landmark_y, 0, aligned.shape[0] - 1)
    landmark_ranges_t = torch.as_tensor(
        landmark_ranges, dtype=torch.float32, device=compute_device
    )
    landmark_valid = (
        torch.isfinite(landmark_ranges_t)
        & (landmark_ranges_t >= settings.min_range_m)
        & (landmark_ranges_t <= settings.max_range_m)
    )
    landmark_colors = target_rgb[landmark_y, landmark_x]
    landmark_points_t = torch.as_tensor(
        landmarks, dtype=torch.float32, device=compute_device
    )

    with torch.no_grad():
        initial_rgb, initial_coverage, _ = render_spherical_gaussians(
            target_rgb,
            aligned,
            target_valid,
            rotation,
            translation,
            extra_points_target=landmark_points_t,
            extra_colors=landmark_colors,
            sigma_px=settings.gaussian_sigma_px,
            radius_px=settings.gaussian_radius_px,
            opacity=settings.opacity,
            occlusion_tau_m=settings.occlusion_tau_m,
        )
        fixed_training_weight = (
            source_valid.to(torch.float32)
            * (initial_coverage >= settings.minimum_training_coverage).to(torch.float32)
            * initial_coverage
        ).detach()
        initial_l1 = _weighted_l1(initial_rgb, source_rgb, fixed_training_weight)

    stages = settings.resolved_stages()
    history: list[dict[str, float | int]] = []
    stage_reports: list[dict[str, Any]] = []
    stage_radial_outputs: list[np.ndarray] = []
    previous_correction_grid: torch.Tensor | None = None
    previous_covariance_grid: torch.Tensor | None = None
    previous_opacity_grid: torch.Tensor | None = None
    previous_color_grid: torch.Tensor | None = None
    global_iteration = 0
    for stage_index, stage in enumerate(stages):
        shape = stage.correction_shape_hw
        if previous_correction_grid is None:
            base_correction_grid = torch.zeros(
                (1, 1, *shape), dtype=torch.float32, device=compute_device
            )
        else:
            base_correction_grid = F.interpolate(
                previous_correction_grid,
                size=shape,
                mode="bilinear",
                align_corners=False,
            ).detach()
        correction_residual = torch.zeros_like(
            base_correction_grid, requires_grad=stage.optimize_radial
        )
        parameter_groups: list[dict[str, Any]] = []
        if stage.optimize_radial:
            parameter_groups.append(
                {"params": [correction_residual], "lr": stage.learning_rate}
            )
        covariance_residual: torch.Tensor | None = None
        if stage.surface_aligned:
            if previous_covariance_grid is None:
                base_covariance_grid = torch.zeros(
                    (1, 2, *shape), dtype=torch.float32, device=compute_device
                )
            else:
                base_covariance_grid = F.interpolate(
                    previous_covariance_grid,
                    size=shape,
                    mode="bilinear",
                    align_corners=False,
                ).detach()
            if stage.optimize_covariance:
                covariance_residual = torch.zeros_like(
                    base_covariance_grid, requires_grad=True
                )
                parameter_groups.append(
                    {
                        "params": [covariance_residual],
                        "lr": stage.covariance_learning_rate,
                    }
                )
        else:
            base_covariance_grid = None

        opacity_residual: torch.Tensor | None = None
        if previous_opacity_grid is not None or stage.optimize_opacity:
            if previous_opacity_grid is None:
                base_opacity_grid = torch.zeros(
                    (1, 1, *shape), dtype=torch.float32, device=compute_device
                )
            else:
                base_opacity_grid = F.interpolate(
                    previous_opacity_grid,
                    size=shape,
                    mode="bilinear",
                    align_corners=False,
                ).detach()
            if stage.optimize_opacity:
                opacity_residual = torch.zeros_like(
                    base_opacity_grid, requires_grad=True
                )
                parameter_groups.append(
                    {
                        "params": [opacity_residual],
                        "lr": stage.opacity_learning_rate,
                    }
                )
        else:
            base_opacity_grid = None

        color_residual: torch.Tensor | None = None
        if previous_color_grid is not None or stage.optimize_color:
            if previous_color_grid is None:
                base_color_grid = torch.zeros(
                    (1, 3, *shape), dtype=torch.float32, device=compute_device
                )
            else:
                base_color_grid = F.interpolate(
                    previous_color_grid,
                    size=shape,
                    mode="bilinear",
                    align_corners=False,
                ).detach()
            if stage.optimize_color:
                color_residual = torch.zeros_like(base_color_grid, requires_grad=True)
                parameter_groups.append(
                    {"params": [color_residual], "lr": stage.color_learning_rate}
                )
        else:
            base_color_grid = None

        optimizer = torch.optim.Adam(parameter_groups, eps=1e-6)
        stage_history: list[dict[str, float | int]] = []
        for step in range(stage.iterations):
            optimizer.zero_grad(set_to_none=True)
            correction_grid = torch.clamp(
                base_correction_grid + correction_residual,
                -settings.max_abs_log_correction,
                settings.max_abs_log_correction,
            )
            correction = F.interpolate(
                correction_grid,
                size=tuple(aligned.shape),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            current_range = torch.clamp(
                aligned * torch.exp(correction),
                settings.min_range_m,
                settings.max_range_m,
            )
            covariance_grid = _current_covariance_grid(
                base_covariance_grid,
                covariance_residual,
                settings.max_abs_log_covariance_scale,
            )
            covariance_hw2 = _expand_covariance_grid(
                covariance_grid, tuple(aligned.shape)
            )
            opacity_grid = _current_bounded_grid(
                base_opacity_grid,
                opacity_residual,
                settings.max_abs_opacity_logit_delta,
            )
            opacity_hw = _expand_opacity_grid(
                opacity_grid, tuple(aligned.shape), settings.opacity
            )
            color_grid = _current_bounded_grid(
                base_color_grid,
                color_residual,
                settings.max_abs_color_delta,
            )
            current_rgb = _expand_color_grid(
                color_grid, target_rgb, tuple(aligned.shape)
            )
            rendered, coverage, _ = render_spherical_gaussians(
                current_rgb,
                current_range,
                target_valid,
                rotation,
                translation,
                extra_points_target=landmark_points_t,
                extra_colors=landmark_colors,
                sigma_px=settings.gaussian_sigma_px,
                radius_px=settings.gaussian_radius_px,
                opacity=(1.0 if opacity_hw is not None else settings.opacity),
                opacity_hw=opacity_hw,
                occlusion_tau_m=settings.occlusion_tau_m,
                surface_aligned=stage.surface_aligned,
                covariance_log_scales_hw2=covariance_hw2,
            )
            photometric = _weighted_l1(rendered, source_rgb, fixed_training_weight)
            prior_loss = torch.mean(correction_grid * correction_grid)
            smoothness = _periodic_grid_smoothness(correction_grid)
            stage_residual_loss = torch.mean(correction_residual * correction_residual)
            landmark_prediction = current_range[landmark_y, landmark_x]
            log_error = torch.log(
                torch.clamp(landmark_prediction, min=1e-6)
            ) - torch.log(torch.clamp(landmark_ranges_t, min=1e-6))
            if torch.any(landmark_valid):
                landmark_loss = F.huber_loss(
                    log_error[landmark_valid],
                    torch.zeros_like(log_error[landmark_valid]),
                    delta=settings.landmark_huber_delta_log,
                )
            else:
                landmark_loss = torch.zeros(
                    (), dtype=torch.float32, device=compute_device
                )
            if covariance_grid is None:
                covariance_prior = torch.zeros(
                    (), dtype=torch.float32, device=compute_device
                )
                covariance_smoothness = torch.zeros_like(covariance_prior)
            else:
                covariance_prior = torch.mean(covariance_grid * covariance_grid)
                covariance_smoothness = _periodic_grid_smoothness(covariance_grid)
            if opacity_grid is None:
                opacity_prior = torch.zeros(
                    (), dtype=torch.float32, device=compute_device
                )
                opacity_smoothness = torch.zeros_like(opacity_prior)
            else:
                opacity_prior = torch.mean(opacity_grid * opacity_grid)
                opacity_smoothness = _periodic_grid_smoothness(opacity_grid)
            if color_grid is None:
                color_prior = torch.zeros(
                    (), dtype=torch.float32, device=compute_device
                )
                color_smoothness = torch.zeros_like(color_prior)
            else:
                color_prior = torch.mean(color_grid * color_grid)
                color_smoothness = _periodic_grid_smoothness(color_grid)
            loss = (
                settings.photometric_weight * photometric
                + settings.prior_weight * prior_loss
                + settings.smoothness_weight * smoothness
                + settings.landmark_weight * landmark_loss
                + (
                    settings.stage_residual_weight * stage_residual_loss
                    if stage_index > 0 and stage.optimize_radial
                    else 0.0
                )
                + settings.covariance_prior_weight * covariance_prior
                + settings.covariance_smoothness_weight * covariance_smoothness
                + settings.opacity_prior_weight * opacity_prior
                + settings.opacity_smoothness_weight * opacity_smoothness
                + settings.color_prior_weight * color_prior
                + settings.color_smoothness_weight * color_smoothness
            )
            loss.backward()
            trainable: list[torch.Tensor] = []
            if stage.optimize_radial:
                trainable.append(correction_residual)
            if covariance_residual is not None:
                trainable.append(covariance_residual)
            if opacity_residual is not None:
                trainable.append(opacity_residual)
            if color_residual is not None:
                trainable.append(color_residual)
            nonfinite_gradient_elements = 0
            for parameter in trainable:
                gradient = parameter.grad
                if gradient is None:
                    raise RuntimeError(
                        "a staged Gaussian parameter received no gradient"
                    )
                nonfinite_gradient_elements += int(
                    torch.count_nonzero(~torch.isfinite(gradient)).item()
                )
                torch.nan_to_num_(gradient, nan=0.0, posinf=1.0, neginf=-1.0)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            with torch.no_grad():
                total_correction = torch.nan_to_num(
                    base_correction_grid + correction_residual,
                    nan=0.0,
                    posinf=settings.max_abs_log_correction,
                    neginf=-settings.max_abs_log_correction,
                ).clamp(
                    -settings.max_abs_log_correction,
                    settings.max_abs_log_correction,
                )
                correction_residual.copy_(total_correction - base_correction_grid)
                if covariance_residual is not None:
                    total_covariance = torch.nan_to_num(
                        base_covariance_grid + covariance_residual,
                        nan=0.0,
                        posinf=settings.max_abs_log_covariance_scale,
                        neginf=-settings.max_abs_log_covariance_scale,
                    ).clamp(
                        -settings.max_abs_log_covariance_scale,
                        settings.max_abs_log_covariance_scale,
                    )
                    covariance_residual.copy_(total_covariance - base_covariance_grid)
                if opacity_residual is not None:
                    total_opacity = torch.nan_to_num(
                        base_opacity_grid + opacity_residual,
                        nan=0.0,
                        posinf=settings.max_abs_opacity_logit_delta,
                        neginf=-settings.max_abs_opacity_logit_delta,
                    ).clamp(
                        -settings.max_abs_opacity_logit_delta,
                        settings.max_abs_opacity_logit_delta,
                    )
                    opacity_residual.copy_(total_opacity - base_opacity_grid)
                if color_residual is not None:
                    total_color = torch.nan_to_num(
                        base_color_grid + color_residual,
                        nan=0.0,
                        posinf=settings.max_abs_color_delta,
                        neginf=-settings.max_abs_color_delta,
                    ).clamp(
                        -settings.max_abs_color_delta,
                        settings.max_abs_color_delta,
                    )
                    color_residual.copy_(total_color - base_color_grid)
            global_iteration += 1
            entry: dict[str, float | int] = {
                "iteration": global_iteration,
                "stage": stage_index + 1,
                "stage_iteration": step + 1,
                "loss": float(loss.detach()),
                "photometric_l1": float(photometric.detach()),
                "prior": float(prior_loss.detach()),
                "smoothness": float(smoothness.detach()),
                "stage_residual": float(stage_residual_loss.detach()),
                "landmark_huber": float(landmark_loss.detach()),
                "covariance_prior": float(covariance_prior.detach()),
                "covariance_smoothness": float(covariance_smoothness.detach()),
                "opacity_prior": float(opacity_prior.detach()),
                "opacity_smoothness": float(opacity_smoothness.detach()),
                "color_prior": float(color_prior.detach()),
                "color_smoothness": float(color_smoothness.detach()),
                "nonfinite_gradient_elements": nonfinite_gradient_elements,
                "coverage_fraction": float(
                    (coverage >= settings.minimum_training_coverage)
                    .to(torch.float32)
                    .mean()
                    .detach()
                ),
            }
            history.append(entry)
            stage_history.append(entry)

        previous_correction_grid = torch.clamp(
            base_correction_grid + correction_residual.detach(),
            -settings.max_abs_log_correction,
            settings.max_abs_log_correction,
        )
        if base_covariance_grid is not None:
            covariance_delta = (
                covariance_residual.detach()
                if covariance_residual is not None
                else torch.zeros_like(base_covariance_grid)
            )
            previous_covariance_grid = torch.clamp(
                base_covariance_grid + covariance_delta,
                -settings.max_abs_log_covariance_scale,
                settings.max_abs_log_covariance_scale,
            )
        if base_opacity_grid is not None:
            opacity_delta = (
                opacity_residual.detach()
                if opacity_residual is not None
                else torch.zeros_like(base_opacity_grid)
            )
            previous_opacity_grid = torch.clamp(
                base_opacity_grid + opacity_delta,
                -settings.max_abs_opacity_logit_delta,
                settings.max_abs_opacity_logit_delta,
            )
        if base_color_grid is not None:
            color_delta = (
                color_residual.detach()
                if color_residual is not None
                else torch.zeros_like(base_color_grid)
            )
            previous_color_grid = torch.clamp(
                base_color_grid + color_delta,
                -settings.max_abs_color_delta,
                settings.max_abs_color_delta,
            )
        with torch.no_grad():
            stage_correction = F.interpolate(
                previous_correction_grid,
                size=tuple(aligned.shape),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            stage_range = torch.clamp(
                aligned * torch.exp(stage_correction),
                settings.min_range_m,
                settings.max_range_m,
            )
            stage_radial_outputs.append(stage_range.cpu().numpy().astype(np.float32))
        stage_reports.append(
            {
                "stage": stage_index + 1,
                "correction_shape_hw": list(shape),
                "iterations": stage.iterations,
                "surface_aligned": stage.surface_aligned,
                "optimize_radial": stage.optimize_radial,
                "optimize_covariance": stage.optimize_covariance,
                "optimize_opacity": stage.optimize_opacity,
                "optimize_color": stage.optimize_color,
                "radial_parameter_count": (
                    int(math.prod(shape)) if stage.optimize_radial else 0
                ),
                "covariance_parameter_count": (
                    2 * int(math.prod(shape)) if stage.optimize_covariance else 0
                ),
                "opacity_parameter_count": (
                    int(math.prod(shape)) if stage.optimize_opacity else 0
                ),
                "color_parameter_count": (
                    3 * int(math.prod(shape)) if stage.optimize_color else 0
                ),
                "initial_loss": (
                    float(stage_history[0]["loss"]) if stage_history else None
                ),
                "final_loss": (
                    float(stage_history[-1]["loss"]) if stage_history else None
                ),
            }
        )

    with torch.no_grad():
        if previous_correction_grid is None:
            raise RuntimeError("at least one optimization stage is required")
        final_correction = F.interpolate(
            previous_correction_grid,
            size=tuple(aligned.shape),
            mode="bilinear",
            align_corners=False,
        )[0, 0].clamp(
            -settings.max_abs_log_correction,
            settings.max_abs_log_correction,
        )
        final_range = torch.clamp(
            aligned * torch.exp(final_correction),
            settings.min_range_m,
            settings.max_range_m,
        )
        final_stage = stages[-1]
        final_covariance_hw2 = _expand_covariance_grid(
            previous_covariance_grid if final_stage.surface_aligned else None,
            tuple(aligned.shape),
        )
        final_opacity_hw = _expand_opacity_grid(
            previous_opacity_grid, tuple(aligned.shape), settings.opacity
        )
        final_color_hwc = _expand_color_grid(
            previous_color_grid, target_rgb, tuple(aligned.shape)
        )
        final_rgb, final_coverage, _ = render_spherical_gaussians(
            final_color_hwc,
            final_range,
            target_valid,
            rotation,
            translation,
            extra_points_target=landmark_points_t,
            extra_colors=landmark_colors,
            sigma_px=settings.gaussian_sigma_px,
            radius_px=settings.gaussian_radius_px,
            opacity=(1.0 if final_opacity_hw is not None else settings.opacity),
            opacity_hw=final_opacity_hw,
            occlusion_tau_m=settings.occlusion_tau_m,
            surface_aligned=final_stage.surface_aligned,
            covariance_log_scales_hw2=final_covariance_hw2,
        )
        final_l1 = _weighted_l1(final_rgb, source_rgb, fixed_training_weight)
    report: dict[str, Any] = {
        "device": str(compute_device),
        "options": settings.to_dict(),
        "history": history,
        "stages": stage_reports,
        "optimized_factors": {
            "radial_means": True,
            "surface_aligned_covariance_scales": any(
                stage.optimize_covariance for stage in stages
            ),
            "tangential_means": False,
            "rgb_dc_residual": any(stage.optimize_color for stage in stages),
            "higher_order_spherical_harmonics": False,
            "opacity": any(stage.optimize_opacity for stage in stages),
            "pose": False,
            "split_or_prune": False,
        },
        "training_pixels": int(torch.count_nonzero(fixed_training_weight).item()),
        "initial_source_l1": float(initial_l1),
        "final_source_l1": float(final_l1),
        "initial_coverage_fraction": float(
            (initial_coverage >= settings.minimum_training_coverage)
            .to(torch.float32)
            .mean()
        ),
        "final_coverage_fraction": float(
            (final_coverage >= settings.minimum_training_coverage)
            .to(torch.float32)
            .mean()
        ),
        "median_abs_log_correction": float(torch.median(torch.abs(final_correction))),
        "maximum_abs_log_correction": float(torch.max(torch.abs(final_correction))),
    }
    if final_covariance_hw2 is not None:
        report["median_abs_log_covariance_scale"] = float(
            torch.median(torch.abs(final_covariance_hw2))
        )
        report["maximum_abs_log_covariance_scale"] = float(
            torch.max(torch.abs(final_covariance_hw2))
        )
        covariance_numpy = final_covariance_hw2.cpu().numpy().astype(np.float32)
    else:
        covariance_numpy = None
    opacity_numpy = (
        final_opacity_hw.cpu().numpy().astype(np.float32)
        if final_opacity_hw is not None
        else None
    )
    color_numpy = (
        final_color_hwc.cpu().numpy().astype(np.float32)
        if previous_color_grid is not None
        else None
    )
    return GaussianDepthResult(
        radial_m=final_range.cpu().numpy().astype(np.float32),
        stage_radial_m=tuple(stage_radial_outputs),
        covariance_log_scales_hw2=covariance_numpy,
        opacity_hw=opacity_numpy,
        color_hwc=color_numpy,
        surface_aligned=final_stage.surface_aligned,
        report=report,
    )


def _current_covariance_grid(
    base: torch.Tensor | None,
    residual: torch.Tensor | None,
    maximum_abs_log_scale: float,
) -> torch.Tensor | None:
    if base is None:
        return None
    if residual is None:
        return base
    return torch.clamp(
        base + residual,
        -maximum_abs_log_scale,
        maximum_abs_log_scale,
    )


def _current_bounded_grid(
    base: torch.Tensor | None,
    residual: torch.Tensor | None,
    maximum_absolute_value: float,
) -> torch.Tensor | None:
    if base is None:
        return None
    if residual is None:
        return base
    return torch.clamp(
        base + residual,
        -maximum_absolute_value,
        maximum_absolute_value,
    )


def _expand_opacity_grid(
    logit_delta_grid: torch.Tensor | None,
    output_shape_hw: tuple[int, int],
    base_opacity: float,
) -> torch.Tensor | None:
    if logit_delta_grid is None:
        return None
    expanded = F.interpolate(
        logit_delta_grid,
        size=output_shape_hw,
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    bounded_base = min(max(float(base_opacity), 1e-5), 1.0 - 1e-5)
    base_logit = math.log(bounded_base / (1.0 - bounded_base))
    return torch.sigmoid(expanded + base_logit)


def _expand_color_grid(
    color_delta_grid: torch.Tensor | None,
    base_color_hwc: torch.Tensor,
    output_shape_hw: tuple[int, int],
) -> torch.Tensor:
    if color_delta_grid is None:
        return base_color_hwc
    expanded = F.interpolate(
        color_delta_grid,
        size=output_shape_hw,
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)
    return torch.clamp(base_color_hwc + expanded, 0.0, 1.0)


def _expand_covariance_grid(
    grid: torch.Tensor | None, shape_hw: tuple[int, int]
) -> torch.Tensor | None:
    if grid is None:
        return None
    return F.interpolate(
        grid,
        size=shape_hw,
        mode="bilinear",
        align_corners=False,
    )[
        0
    ].permute(1, 2, 0)


def _periodic_grid_smoothness(grid: torch.Tensor) -> torch.Tensor:
    horizontal = grid[:, :, :, 1:] - grid[:, :, :, :-1]
    seam = grid[:, :, :, :1] - grid[:, :, :, -1:]
    vertical = grid[:, :, 1:, :] - grid[:, :, :-1, :]
    return (
        torch.mean(horizontal * horizontal)
        + torch.mean(seam * seam)
        + torch.mean(vertical * vertical)
    )


def _float_rgb_tensor(
    image: Any, *, device: torch.device | str = "cpu"
) -> torch.Tensor:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("RGB input must have shape (H, W, 3)")
    if np.issubdtype(array.dtype, np.integer):
        result = array.astype(np.float32) / float(np.iinfo(array.dtype).max)
    else:
        result = array.astype(np.float32)
    return torch.as_tensor(np.clip(result, 0.0, 1.0), device=device)


def _weighted_l1(
    prediction: torch.Tensor, target: torch.Tensor, weight_hw: torch.Tensor
) -> torch.Tensor:
    total_weight = torch.clamp(torch.sum(weight_hw), min=1e-8)
    return (
        torch.sum(torch.mean(torch.abs(prediction - target), dim=2) * weight_hw)
        / total_weight
    )


def _erp_pixels_numpy(bearings: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    normalized = bearings / np.maximum(
        np.linalg.norm(bearings, axis=1, keepdims=True), 1e-12
    )
    longitude = np.arctan2(normalized[:, 0], normalized[:, 2])
    latitude = np.arcsin(np.clip(normalized[:, 1], -1.0, 1.0))
    x = (longitude + math.pi) / (2.0 * math.pi) * width - 0.5
    y = (math.pi / 2.0 - latitude) / math.pi * height - 0.5
    return np.column_stack((x, y))


def _sample_erp_numpy(image: np.ndarray, bearings: np.ndarray) -> np.ndarray:
    pixels = _erp_pixels_numpy(bearings, image.shape)
    x = pixels[:, 0]
    y = pixels[:, 1]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = x - x0
    wy = y - y0
    height, width = image.shape
    valid = (y0 >= 0) & (y1 < height)
    x0 %= width
    x1 %= width
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)
    values = (
        image[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + image[y0, x1] * wx * (1.0 - wy)
        + image[y1, x0] * (1.0 - wx) * wy
        + image[y1, x1] * wx * wy
    )
    values[~valid] = np.nan
    return values
