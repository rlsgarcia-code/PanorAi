"""Small differentiable spherical Gaussian renderer for a two-view test.

This is deliberately a geometry-only experiment, not a replacement for the
CUDA 3DGS renderer.  One isotropic Gaussian is initialized per supported
target pixel.  RGB, opacity and camera poses remain fixed; only a smooth
log-range correction field is optimized from the second image and sparse BA
landmarks.  The restriction makes the experiment useful for answering whether
Gaussian splatting can densify an already constrained visible surface without
quietly hallucinating unobserved geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    occlusion_tau_m: float = 0.08,
    surface_aligned: bool = False,
    surface_depth_edge_log: float = 0.16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render supported target Gaussians into a posed ERP view.

    The renderer uses normalized elliptical-free Gaussian accumulation and a
    detached nearest-depth visibility gate.  It is differentiable with respect
    to the Gaussian means inside each raster cell and runs on plain PyTorch CPU
    or Apple MPS.
    """

    if target_rgb_hwc.ndim != 3 or target_rgb_hwc.shape[2] != 3:
        raise ValueError("target_rgb_hwc must have shape (H, W, 3)")
    height, width = target_range_hw.shape
    if target_rgb_hwc.shape[:2] != (height, width):
        raise ValueError("RGB and range shapes must agree")
    if target_valid_hw.shape != (height, width):
        raise ValueError("target_valid_hw must match target range")
    rays = erp_rays((height, width), device=target_range_hw.device)
    valid = target_valid_hw & torch.isfinite(target_range_hw) & (target_range_hw > 0.0)
    selected_rays = rays[valid]
    selected_range = target_range_hw[valid]
    colors = target_rgb_hwc[valid]
    points_target = selected_rays * selected_range[:, None]
    inverse_covariance: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    if surface_aligned:
        inverse_covariance = _projected_surface_inverse_covariance(
            target_range_hw,
            valid,
            rotation_view_from_target,
            translation_view_from_target,
            lowpass_sigma_px=sigma_px,
            maximum_sigma_px=max(float(sigma_px), float(radius_px) / 3.0),
            depth_edge_log=surface_depth_edge_log,
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
    if inverse_covariance is not None:
        inverse_covariance = tuple(
            component[finite] for component in inverse_covariance
        )
    if u.numel() == 0:
        zeros_rgb = torch.zeros_like(target_rgb_hwc)
        zeros = torch.zeros_like(target_range_hw)
        return zeros_rgb, zeros, torch.full_like(target_range_hw, float("nan"))

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
            gaussian_weight = float(opacity) * torch.exp(-exponent)
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


def _projected_surface_inverse_covariance(
    target_range_hw: torch.Tensor,
    valid_hw: torch.Tensor,
    rotation_view_from_target: torch.Tensor,
    translation_view_from_target: torch.Tensor,
    *,
    lowpass_sigma_px: float,
    maximum_sigma_px: float,
    depth_edge_log: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate an EWA ellipse from projected right/down surface tangents."""

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
    """Optimize only Gaussian radial positions against the second image."""

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

    correction_grid = torch.zeros(
        (1, 1, *settings.correction_shape_hw),
        dtype=torch.float32,
        device=compute_device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([correction_grid], lr=settings.learning_rate, eps=1e-6)
    history: list[dict[str, float | int]] = []
    for step in range(settings.iterations):
        optimizer.zero_grad(set_to_none=True)
        correction = F.interpolate(
            correction_grid,
            size=tuple(aligned.shape),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        correction = torch.clamp(
            correction,
            -settings.max_abs_log_correction,
            settings.max_abs_log_correction,
        )
        current_range = torch.clamp(
            aligned * torch.exp(correction),
            settings.min_range_m,
            settings.max_range_m,
        )
        rendered, coverage, _ = render_spherical_gaussians(
            target_rgb,
            current_range,
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
        photometric = _weighted_l1(rendered, source_rgb, fixed_training_weight)
        prior_loss = torch.mean(correction_grid * correction_grid)
        horizontal = correction_grid[:, :, :, 1:] - correction_grid[:, :, :, :-1]
        seam = correction_grid[:, :, :, :1] - correction_grid[:, :, :, -1:]
        vertical = correction_grid[:, :, 1:, :] - correction_grid[:, :, :-1, :]
        smoothness = (
            torch.mean(horizontal * horizontal)
            + torch.mean(seam * seam)
            + torch.mean(vertical * vertical)
        )
        landmark_prediction = current_range[landmark_y, landmark_x]
        log_error = torch.log(torch.clamp(landmark_prediction, min=1e-6)) - torch.log(
            torch.clamp(landmark_ranges_t, min=1e-6)
        )
        if torch.any(landmark_valid):
            landmark_loss = F.huber_loss(
                log_error[landmark_valid],
                torch.zeros_like(log_error[landmark_valid]),
                delta=settings.landmark_huber_delta_log,
            )
        else:
            landmark_loss = torch.zeros((), dtype=torch.float32, device=compute_device)
        loss = (
            settings.photometric_weight * photometric
            + settings.prior_weight * prior_loss
            + settings.smoothness_weight * smoothness
            + settings.landmark_weight * landmark_loss
        )
        loss.backward()
        gradient = correction_grid.grad
        if gradient is None:
            raise RuntimeError("correction grid did not receive a gradient")
        nonfinite_gradient_elements = int(
            torch.count_nonzero(~torch.isfinite(gradient)).item()
        )
        if nonfinite_gradient_elements:
            torch.nan_to_num_(gradient, nan=0.0, posinf=1.0, neginf=-1.0)
        torch.nn.utils.clip_grad_norm_([correction_grid], max_norm=1.0)
        optimizer.step()
        with torch.no_grad():
            torch.nan_to_num_(
                correction_grid,
                nan=0.0,
                posinf=settings.max_abs_log_correction,
                neginf=-settings.max_abs_log_correction,
            )
            correction_grid.clamp_(
                -settings.max_abs_log_correction,
                settings.max_abs_log_correction,
            )
        history.append(
            {
                "iteration": step + 1,
                "loss": float(loss.detach()),
                "photometric_l1": float(photometric.detach()),
                "prior": float(prior_loss.detach()),
                "smoothness": float(smoothness.detach()),
                "landmark_huber": float(landmark_loss.detach()),
                "nonfinite_gradient_elements": nonfinite_gradient_elements,
                "coverage_fraction": float(
                    (coverage >= settings.minimum_training_coverage)
                    .to(torch.float32)
                    .mean()
                    .detach()
                ),
            }
        )

    with torch.no_grad():
        final_correction = F.interpolate(
            correction_grid,
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
        final_rgb, final_coverage, _ = render_spherical_gaussians(
            target_rgb,
            final_range,
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
        final_l1 = _weighted_l1(final_rgb, source_rgb, fixed_training_weight)
    report: dict[str, Any] = {
        "device": str(compute_device),
        "options": settings.to_dict(),
        "history": history,
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
    return final_range.cpu().numpy().astype(np.float32), report


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
