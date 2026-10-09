"""Differentiable resize-free multiview refinement on a canonical ERP.

This module is benchmark-local research code.  It deliberately does not add a
training pipeline or a model implementation to the PanorAi package.  The
refiner consumes a model-agnostic radial-range prior and registered central
ERP source views, then optimizes one sparse log-range residual per native
target pixel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import numpy as np


INTERFACE = "panorai-spherical-multiview-depth-refinement/v1-experimental"


@dataclass(frozen=True, slots=True)
class DepthPrior:
    """Backbone-neutral radial-range prior on a native ERP lattice."""

    radial_range_m: Any
    validity_mask: Any
    confidence: Any | None = None
    provenance: str = "unspecified"


@dataclass(frozen=True, slots=True)
class SourceView:
    """Canonical ERP source and metric pose from target to source.

    The pose convention is ``X_source = R_source_from_target @ X_target +
    t_source_from_target``.  Translation and depth must use metres together.
    """

    rgb: Any
    rotation_source_from_target: Any
    translation_source_from_target_m: Any
    validity_mask: Any | None = None
    view_id: str = "source"


@dataclass(frozen=True, slots=True)
class RefinementOptions:
    """Controls for native-grid sparse differentiable optimization."""

    min_range_m: float = 0.3
    max_range_m: float = 15.0
    epochs: int = 2
    row_batch: int = 16
    learning_rate: float = 0.03
    prior_weight: float = 0.10
    pairwise_weight: float = 0.05
    pairwise_edge_gamma: float = 12.0
    photometric_clip: float = 0.35
    min_texture_std: float = 0.01
    feature_window: int = 7
    max_abs_log_residual: float = 4.0
    initialization_hypotheses: int = 9
    initialization_log_radius: float = math.log(2.0)
    initialization_temperature: float = 0.02
    initialization_prior_weight: float = 0.20
    minimum_consistent_views: int = 2
    seed: int = 28031

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_range_m) or self.min_range_m <= 0.0:
            raise ValueError("min_range_m must be positive and finite")
        if not math.isfinite(self.max_range_m) or self.max_range_m <= self.min_range_m:
            raise ValueError("max_range_m must exceed min_range_m")
        for name in (
            "epochs",
            "row_batch",
            "feature_window",
            "initialization_hypotheses",
            "minimum_consistent_views",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.feature_window % 2 == 0:
            raise ValueError("feature_window must be odd")
        if (
            self.initialization_hypotheses < 3
            or self.initialization_hypotheses % 2 == 0
        ):
            raise ValueError("initialization_hypotheses must be odd and at least 3")
        for name in (
            "learning_rate",
            "prior_weight",
            "pairwise_weight",
            "pairwise_edge_gamma",
            "photometric_clip",
            "max_abs_log_residual",
            "initialization_log_radius",
            "initialization_temperature",
            "initialization_prior_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative and finite")
        if (
            self.learning_rate == 0.0
            or self.photometric_clip == 0.0
            or self.initialization_log_radius == 0.0
            or self.initialization_temperature == 0.0
        ):
            raise ValueError(
                "learning_rate, photometric_clip, initialization_log_radius, "
                "and initialization_temperature must be positive"
            )
        if not 0.0 <= self.min_texture_std <= 1.0:
            raise ValueError("min_texture_std must lie in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RefinementResult:
    """Native-resolution refined radial range and audit diagnostics."""

    radial_range_m: Any
    validity_mask: Any
    log_residual: Any
    history: tuple[dict[str, float | int], ...]
    options: RefinementOptions
    prior_provenance: str
    source_view_ids: tuple[str, ...]
    interface: str = INTERFACE


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional benchmark runtime
        raise ImportError("Torch is required for multiview depth refinement") from error
    return torch


def _as_rgb_tensor(value: Any, *, device: Any, name: str) -> Any:
    torch = _torch()
    tensor = torch.as_tensor(value, device=device)
    if tensor.ndim == 3 and tensor.shape[-1] == 3:
        tensor = tensor.permute(2, 0, 1)
    elif tensor.ndim == 4 and tensor.shape[0] == 1 and tensor.shape[1] == 3:
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"{name} must use HWC RGB, CHW RGB, or 1CHW RGB")
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.float32) / float(torch.iinfo(tensor.dtype).max)
    else:
        tensor = tensor.to(torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")
    minimum = float(tensor.amin().item())
    maximum = float(tensor.amax().item())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(f"floating {name} must lie in [0, 1]")
    return tensor.contiguous()


def _as_hw_tensor(
    value: Any,
    *,
    device: Any,
    dtype: Any,
    shape_hw: tuple[int, int],
    name: str,
) -> Any:
    torch = _torch()
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.shape != shape_hw:
        raise ValueError(f"{name} must have shape {shape_hw}")
    return tensor.contiguous()


def _validate_pose(rotation: Any, translation: Any, *, device: Any) -> tuple[Any, Any]:
    torch = _torch()
    R = torch.as_tensor(rotation, dtype=torch.float32, device=device)
    t = torch.as_tensor(translation, dtype=torch.float32, device=device)
    if R.shape != (3, 3) or t.shape != (3,):
        raise ValueError("rotation must be 3x3 and translation must have shape (3,)")
    if not torch.isfinite(R).all() or not torch.isfinite(t).all():
        raise ValueError("pose must be finite")
    identity = torch.eye(3, dtype=R.dtype, device=R.device)
    if not torch.allclose(R.T @ R, identity, rtol=0.0, atol=1e-5):
        raise ValueError("rotation must be orthonormal")
    if not torch.isclose(torch.linalg.det(R), R.new_tensor(1.0), rtol=0.0, atol=1e-5):
        raise ValueError("rotation must have determinant +1")
    if float(torch.linalg.vector_norm(t).item()) <= 1e-8:
        raise ValueError("translation must have non-zero metric scale")
    return R, t


def _circular_local_statistics(luma: Any, window: int) -> tuple[Any, Any]:
    torch = _torch()
    functional = torch.nn.functional
    radius = window // 2
    horizontal = torch.cat((luma[..., -radius:], luma, luma[..., :radius]), dim=-1)
    padded = functional.pad(horizontal, (0, 0, radius, radius), mode="replicate")
    mean = functional.avg_pool2d(padded, window, stride=1)
    square_mean = functional.avg_pool2d(padded * padded, window, stride=1)
    std = torch.sqrt(torch.clamp(square_mean - mean * mean, min=0.0))
    return mean, std


def photometric_features(rgb: Any, *, window: int = 7) -> tuple[Any, Any]:
    """Return locally normalized luminance/gradient features and texture."""

    torch = _torch()
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError("rgb must have CHW layout")
    luma = (0.2126 * rgb[0:1] + 0.7152 * rgb[1:2] + 0.0722 * rgb[2:3])[None]
    mean, std = _circular_local_statistics(luma, window)
    normalized = torch.clamp((luma - mean) / torch.clamp(std, min=0.02), -3.0, 3.0)
    east = 0.5 * (torch.roll(luma, -1, dims=-1) - torch.roll(luma, 1, dims=-1))
    north = torch.empty_like(luma)
    north[..., 1:-1, :] = 0.5 * (luma[..., :-2, :] - luma[..., 2:, :])
    north[..., 0, :] = luma[..., 0, :] - luma[..., 1, :]
    north[..., -1, :] = luma[..., -2, :] - luma[..., -1, :]
    features = torch.cat(
        (
            normalized / 3.0,
            torch.clamp(east / 0.10, -2.0, 2.0) / 2.0,
            torch.clamp(north / 0.10, -2.0, 2.0) / 2.0,
        ),
        dim=1,
    )[0]
    return features.contiguous(), std[0, 0].contiguous()


def rays_for_pixels(rows: Any, columns: Any, shape_hw: tuple[int, int]) -> Any:
    """Canonical pixel-centre ERP rays without materializing a full lattice."""

    torch = _torch()
    height, width = shape_hw
    longitude = (columns.to(torch.float32) + 0.5) / width * (2.0 * math.pi) - math.pi
    latitude = math.pi / 2.0 - (rows.to(torch.float32) + 0.5) / height * math.pi
    cos_latitude = torch.cos(latitude)
    return torch.stack(
        (
            torch.sin(longitude) * cos_latitude,
            torch.sin(latitude),
            torch.cos(longitude) * cos_latitude,
        ),
        dim=-1,
    )


def project_points_to_grid(points: Any, shape_hw: tuple[int, int]) -> Any:
    """Project 3D points to an align_corners=False canonical ERP grid."""

    torch = _torch()
    norm = torch.linalg.vector_norm(points, dim=-1)
    unit_y = torch.clamp(points[..., 1] / torch.clamp(norm, min=1e-12), -1.0, 1.0)
    longitude = torch.atan2(points[..., 0], points[..., 2])
    latitude = torch.asin(unit_y)
    grid_x = longitude / math.pi
    grid_y = -2.0 * latitude / math.pi
    return torch.stack((grid_x, grid_y), dim=-1)


def _seam_safe_sample(source_chw: Any, grid: Any) -> Any:
    """Bilinearly sample a canonical ERP with explicit horizontal wrapping."""

    torch = _torch()
    functional = torch.nn.functional
    _, height, width = source_chw.shape
    padded = torch.cat((source_chw[..., -1:], source_chw, source_chw[..., :1]), dim=-1)
    # grid_x is longitude/pi. Convert to original pixel centre, shift one
    # padded column, then normalize for align_corners=False on W+2.
    original_x = (grid[..., 0] + 1.0) * 0.5 * width - 0.5
    padded_x = original_x + 1.0
    padded_grid_x = 2.0 * (padded_x + 0.5) / (width + 2) - 1.0
    padded_grid = torch.stack((padded_grid_x, grid[..., 1]), dim=-1)
    sample_grid = padded_grid.reshape(1, -1, 1, 2)
    sampled = functional.grid_sample(
        padded[None],
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled[0, :, :, 0].T


def warp_source(
    source_chw: Any,
    rows: Any,
    columns: Any,
    radial_range_m: Any,
    rotation_source_from_target: Any,
    translation_source_from_target_m: Any,
    shape_hw: tuple[int, int],
) -> Any:
    """Differentiably warp one source ERP to selected target pixels."""

    rays = rays_for_pixels(rows, columns, shape_hw)
    points_target = rays * radial_range_m[:, None]
    points_source = (
        points_target @ rotation_source_from_target.T + translation_source_from_target_m
    )
    grid = project_points_to_grid(points_source, shape_hw)
    return _seam_safe_sample(source_chw, grid)


def _feature_cost(target: Any, source: Any, clip: float) -> Any:
    torch = _torch()
    intensity = torch.clamp(torch.abs(target[:, 0] - source[:, 0]), max=clip)
    gradient = torch.clamp(
        0.5
        * (
            torch.abs(target[:, 1] - source[:, 1])
            + torch.abs(target[:, 2] - source[:, 2])
        ),
        max=clip,
    )
    return 0.75 * intensity + 0.25 * gradient


def _weighted_mean(values: Any, weights: Any) -> Any:
    torch = _torch()
    return torch.sum(values * weights) / torch.clamp(torch.sum(weights), min=1e-12)


def _batch_indices(
    start: int, stop: int, width: int, *, device: Any
) -> tuple[Any, Any, Any]:
    torch = _torch()
    rows = torch.arange(start, stop, device=device, dtype=torch.long)
    columns = torch.arange(width, device=device, dtype=torch.long)
    grid_rows, grid_columns = torch.meshgrid(rows, columns, indexing="ij")
    linear = grid_rows * width + grid_columns
    return linear.reshape(-1), grid_rows.reshape(-1), grid_columns.reshape(-1)


class DifferentiableSphericalDepthRefiner:
    """Optimize a sparse native-pixel log-depth residual from multiple ERPs."""

    interface = INTERFACE
    stability = "experimental"
    depth_semantics = "radial_range_m"

    def __init__(self, options: RefinementOptions | None = None) -> None:
        self.options = options or RefinementOptions()

    def refine(
        self,
        prior: DepthPrior,
        target_rgb: Any,
        sources: Iterable[SourceView],
        *,
        device: str | Any = "cpu",
        progress: Any | None = None,
    ) -> RefinementResult:
        torch = _torch()
        target = _as_rgb_tensor(target_rgb, device=device, name="target_rgb")
        shape_hw = (int(target.shape[1]), int(target.shape[2]))
        height, width = shape_hw
        radial = _as_hw_tensor(
            prior.radial_range_m,
            device=device,
            dtype=torch.float32,
            shape_hw=shape_hw,
            name="prior.radial_range_m",
        )
        validity = _as_hw_tensor(
            prior.validity_mask,
            device=device,
            dtype=torch.bool,
            shape_hw=shape_hw,
            name="prior.validity_mask",
        )
        validity = validity & torch.isfinite(radial) & (radial > 0.0)
        radial = torch.clamp(
            torch.nan_to_num(radial, nan=self.options.min_range_m),
            self.options.min_range_m,
            self.options.max_range_m,
        )
        if prior.confidence is None:
            prior_confidence = torch.ones(shape_hw, device=device, dtype=torch.float32)
        else:
            prior_confidence = _as_hw_tensor(
                prior.confidence,
                device=device,
                dtype=torch.float32,
                shape_hw=shape_hw,
                name="prior.confidence",
            )
            if not torch.isfinite(prior_confidence).all():
                raise ValueError("prior.confidence must be finite")
            prior_confidence = torch.clamp(prior_confidence, 0.0, 1.0)

        target_features, target_texture = photometric_features(
            target, window=self.options.feature_window
        )
        target_luma = 0.2126 * target[0] + 0.7152 * target[1] + 0.0722 * target[2]
        prepared_sources: list[tuple[Any, Any, Any, str]] = []
        for source in sources:
            rgb = _as_rgb_tensor(
                source.rgb, device=device, name=f"{source.view_id}.rgb"
            )
            if tuple(rgb.shape[1:]) != shape_hw:
                raise ValueError("all source ERPs must share the target native shape")
            R, t = _validate_pose(
                source.rotation_source_from_target,
                source.translation_source_from_target_m,
                device=device,
            )
            features, _ = photometric_features(rgb, window=self.options.feature_window)
            if source.validity_mask is None:
                source_validity = torch.ones(
                    shape_hw, device=device, dtype=torch.float32
                )
            else:
                source_validity = _as_hw_tensor(
                    source.validity_mask,
                    device=device,
                    dtype=torch.float32,
                    shape_hw=shape_hw,
                    name=f"{source.view_id}.validity_mask",
                )
            packed = torch.cat((features, source_validity[None]), dim=0).contiguous()
            prepared_sources.append((packed, R, t, source.view_id))
        if not prepared_sources:
            raise ValueError("at least one source view is required")
        if self.options.minimum_consistent_views > len(prepared_sources):
            raise ValueError(
                "minimum_consistent_views cannot exceed the source-view count"
            )

        residual = torch.nn.Embedding(height * width, 1, sparse=True, device=device)
        with torch.no_grad():
            residual.weight.zero_()
        initialization_offsets = torch.linspace(
            -self.options.initialization_log_radius,
            self.options.initialization_log_radius,
            self.options.initialization_hypotheses,
            device=device,
            dtype=torch.float32,
        )
        initialization_cost_sum = 0.0
        initialization_weight_sum = 0.0
        with torch.no_grad():
            for start in range(0, height, self.options.row_batch):
                stop = min(height, start + self.options.row_batch)
                linear, rows, columns = _batch_indices(
                    start, stop, width, device=device
                )
                selected = validity[rows, columns]
                if not bool(selected.any()):
                    continue
                linear = linear[selected]
                rows = rows[selected]
                columns = columns[selected]
                count = int(rows.numel())
                prior_depth = radial[rows, columns]
                hypotheses = torch.clamp(
                    prior_depth[None, :] * torch.exp(initialization_offsets[:, None]),
                    self.options.min_range_m,
                    self.options.max_range_m,
                )
                repeated_rows = rows.repeat(self.options.initialization_hypotheses)
                repeated_columns = columns.repeat(
                    self.options.initialization_hypotheses
                )
                flat_depth = hypotheses.reshape(-1)
                target_batch = target_features[:, rows, columns].T
                target_repeated = target_batch.repeat(
                    self.options.initialization_hypotheses, 1
                )
                view_costs = []
                view_supports = []
                for packed, R, t, _ in prepared_sources:
                    sampled = warp_source(
                        packed,
                        repeated_rows,
                        repeated_columns,
                        flat_depth,
                        R,
                        t,
                        shape_hw,
                    )
                    view_costs.append(
                        _feature_cost(
                            target_repeated,
                            sampled[:, :3],
                            self.options.photometric_clip,
                        ).reshape(self.options.initialization_hypotheses, count)
                    )
                    view_supports.append(
                        (sampled[:, 3] >= 0.999).reshape(
                            self.options.initialization_hypotheses, count
                        )
                    )
                costs = torch.stack(view_costs, dim=0)
                supports = torch.stack(view_supports, dim=0)
                costs = torch.where(
                    supports,
                    costs,
                    costs.new_full((), self.options.photometric_clip),
                )
                support_count = torch.sum(supports, dim=0)
                consensus_cost = torch.sum(
                    torch.where(supports, costs, torch.zeros_like(costs)), dim=0
                ) / torch.clamp(support_count, min=1)
                consensus_cost = torch.where(
                    support_count >= self.options.minimum_consistent_views,
                    consensus_cost,
                    consensus_cost.new_full((), self.options.photometric_clip),
                )
                consensus_cost = (
                    consensus_cost
                    + self.options.initialization_prior_weight
                    * torch.abs(initialization_offsets[:, None])
                )
                probability = torch.softmax(
                    -consensus_cost / self.options.initialization_temperature,
                    dim=0,
                )
                initial = torch.sum(
                    probability * initialization_offsets[:, None], dim=0
                )
                usable = torch.any(
                    support_count >= self.options.minimum_consistent_views,
                    dim=0,
                ) & (target_texture[rows, columns] >= self.options.min_texture_std)
                initial = torch.where(usable, initial, torch.zeros_like(initial))
                residual.weight[linear, 0] = initial
                solid_angle = torch.clamp(
                    torch.cos(
                        math.pi / 2.0
                        - (rows.to(torch.float32) + 0.5) / height * math.pi
                    ),
                    min=1e-4,
                )
                selected_cost = torch.sum(probability * consensus_cost, dim=0)
                initialization_cost_sum += float(
                    torch.sum(selected_cost * solid_angle).item()
                )
                initialization_weight_sum += float(torch.sum(solid_angle).item())
        optimizer = torch.optim.SparseAdam(
            residual.parameters(), lr=self.options.learning_rate
        )
        generator = np.random.default_rng(self.options.seed)
        starts = list(range(0, height, self.options.row_batch))
        history: list[dict[str, float | int]] = [
            {
                "epoch": 0,
                "initialization_expected_cost": initialization_cost_sum
                / max(initialization_weight_sum, 1e-12),
                "valid_pixels": int(validity.sum().item()),
            }
        ]

        for epoch in range(self.options.epochs):
            order = generator.permutation(len(starts))
            totals = {
                "photometric": 0.0,
                "prior": 0.0,
                "pairwise": 0.0,
                "objective": 0.0,
                "photometric_pixels": 0,
                "valid_pixels": 0,
            }
            for block_index in order:
                start = starts[int(block_index)]
                stop = min(height, start + self.options.row_batch)
                linear, rows, columns = _batch_indices(
                    start, stop, width, device=device
                )
                center_valid = validity[rows, columns]
                if not bool(center_valid.any()):
                    continue
                linear = linear[center_valid]
                rows = rows[center_valid]
                columns = columns[center_valid]
                raw = residual(linear)[:, 0]
                bounded_raw = torch.clamp(
                    raw,
                    -self.options.max_abs_log_residual,
                    self.options.max_abs_log_residual,
                )
                prior_depth = radial[rows, columns]
                depth = torch.clamp(
                    prior_depth * torch.exp(bounded_raw),
                    self.options.min_range_m,
                    self.options.max_range_m,
                )

                target_batch = target_features[:, rows, columns].T
                view_costs = []
                view_support = []
                for packed, R, t, _ in prepared_sources:
                    sampled = warp_source(
                        packed,
                        rows,
                        columns,
                        depth,
                        R,
                        t,
                        shape_hw,
                    )
                    view_costs.append(
                        _feature_cost(
                            target_batch,
                            sampled[:, :3],
                            self.options.photometric_clip,
                        )
                    )
                    view_support.append(sampled[:, 3] >= 0.999)
                costs = torch.stack(view_costs, dim=0)
                supports = torch.stack(view_support, dim=0)
                costs = torch.where(
                    supports,
                    costs,
                    costs.new_full((), self.options.photometric_clip),
                )
                support_count = torch.sum(supports, dim=0)
                consensus_cost = torch.sum(
                    torch.where(supports, costs, torch.zeros_like(costs)), dim=0
                ) / torch.clamp(support_count, min=1)
                supported = support_count >= self.options.minimum_consistent_views
                textured = target_texture[rows, columns] >= self.options.min_texture_std
                photo_valid = supported & textured
                solid_angle = torch.clamp(
                    torch.cos(
                        math.pi / 2.0
                        - (rows.to(torch.float32) + 0.5) / height * math.pi
                    ),
                    min=1e-4,
                )
                photo_weight = solid_angle * photo_valid.to(torch.float32)
                photo_loss = _weighted_mean(consensus_cost, photo_weight)

                confidence = prior_confidence[rows, columns]
                prior_loss = _weighted_mean(
                    torch.abs(bounded_raw),
                    solid_angle * (0.25 + 0.75 * confidence),
                )

                east_columns = torch.remainder(columns + 1, width)
                south_rows = torch.clamp(rows + 1, max=height - 1)
                east_linear = rows * width + east_columns
                south_linear = south_rows * width + columns
                east_raw = torch.clamp(
                    residual(east_linear)[:, 0],
                    -self.options.max_abs_log_residual,
                    self.options.max_abs_log_residual,
                )
                south_raw = torch.clamp(
                    residual(south_linear)[:, 0],
                    -self.options.max_abs_log_residual,
                    self.options.max_abs_log_residual,
                )
                center_luma = target_luma[rows, columns]
                east_edge = torch.abs(center_luma - target_luma[rows, east_columns])
                south_edge = torch.abs(center_luma - target_luma[south_rows, columns])
                east_weight = torch.exp(-self.options.pairwise_edge_gamma * east_edge)
                south_weight = (
                    solid_angle
                    * torch.exp(-self.options.pairwise_edge_gamma * south_edge)
                    * (south_rows != rows).to(torch.float32)
                )
                pairwise_loss = 0.5 * (
                    _weighted_mean(torch.abs(bounded_raw - east_raw), east_weight)
                    + _weighted_mean(torch.abs(bounded_raw - south_raw), south_weight)
                )

                objective = (
                    photo_loss
                    + self.options.prior_weight * prior_loss
                    + self.options.pairwise_weight * pairwise_loss
                )
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                optimizer.step()

                count = int(rows.numel())
                photo_count = int(photo_valid.sum().item())
                totals["photometric"] += float(photo_loss.detach().item()) * photo_count
                totals["prior"] += float(prior_loss.detach().item()) * count
                totals["pairwise"] += float(pairwise_loss.detach().item()) * count
                totals["objective"] += float(objective.detach().item()) * count
                totals["photometric_pixels"] += photo_count
                totals["valid_pixels"] += count

            with torch.no_grad():
                residual.weight.clamp_(
                    -self.options.max_abs_log_residual,
                    self.options.max_abs_log_residual,
                )
            record: dict[str, float | int] = {
                "epoch": epoch + 1,
                "photometric_loss": totals["photometric"]
                / max(1, totals["photometric_pixels"]),
                "prior_loss": totals["prior"] / max(1, totals["valid_pixels"]),
                "pairwise_loss": totals["pairwise"] / max(1, totals["valid_pixels"]),
                "objective": totals["objective"] / max(1, totals["valid_pixels"]),
                "photometric_pixels": totals["photometric_pixels"],
                "valid_pixels": totals["valid_pixels"],
            }
            history.append(record)
            if progress is not None:
                progress(record)

        with torch.no_grad():
            log_residual = residual.weight[:, 0].reshape(shape_hw)
            refined = torch.clamp(
                radial * torch.exp(log_residual),
                self.options.min_range_m,
                self.options.max_range_m,
            )
            refined = torch.where(
                validity, refined, torch.full_like(refined, torch.nan)
            )
        return RefinementResult(
            radial_range_m=refined.detach(),
            validity_mask=validity.detach(),
            log_residual=log_residual.detach(),
            history=tuple(history),
            options=self.options,
            prior_provenance=prior.provenance,
            source_view_ids=tuple(item[3] for item in prepared_sources),
        )


def reprojection_score(
    radial_range_m: Any,
    validity_mask: Any,
    target_rgb: Any,
    source: SourceView,
    *,
    row_batch: int = 16,
    feature_window: int = 7,
    min_texture_std: float = 0.01,
    photometric_clip: float = 0.35,
    device: str | Any = "cpu",
) -> dict[str, float | int]:
    """Evaluate one frozen map against a source without modifying the map."""

    torch = _torch()
    target = _as_rgb_tensor(target_rgb, device=device, name="target_rgb")
    shape_hw = (int(target.shape[1]), int(target.shape[2]))
    height, width = shape_hw
    radial = _as_hw_tensor(
        radial_range_m,
        device=device,
        dtype=torch.float32,
        shape_hw=shape_hw,
        name="radial_range_m",
    )
    validity = _as_hw_tensor(
        validity_mask,
        device=device,
        dtype=torch.bool,
        shape_hw=shape_hw,
        name="validity_mask",
    )
    source_rgb = _as_rgb_tensor(source.rgb, device=device, name=f"{source.view_id}.rgb")
    if tuple(source_rgb.shape[1:]) != shape_hw:
        raise ValueError("source ERP must share target native shape")
    R, t = _validate_pose(
        source.rotation_source_from_target,
        source.translation_source_from_target_m,
        device=device,
    )
    target_features, texture = photometric_features(target, window=feature_window)
    source_features, _ = photometric_features(source_rgb, window=feature_window)
    if source.validity_mask is None:
        source_validity = torch.ones(shape_hw, device=device, dtype=torch.float32)
    else:
        source_validity = _as_hw_tensor(
            source.validity_mask,
            device=device,
            dtype=torch.float32,
            shape_hw=shape_hw,
            name=f"{source.view_id}.validity_mask",
        )
    packed = torch.cat((source_features, source_validity[None]), dim=0)
    cost_sum = 0.0
    weight_sum = 0.0
    accepted = 0
    with torch.no_grad():
        for start in range(0, height, row_batch):
            stop = min(height, start + row_batch)
            _, rows, columns = _batch_indices(start, stop, width, device=device)
            selected = (
                validity[rows, columns]
                & torch.isfinite(radial[rows, columns])
                & (texture[rows, columns] >= min_texture_std)
            )
            if not bool(selected.any()):
                continue
            rows = rows[selected]
            columns = columns[selected]
            depth = radial[rows, columns]
            warped = warp_source(packed, rows, columns, depth, R, t, shape_hw)
            supported = warped[:, 3] >= 0.999
            if not bool(supported.any()):
                continue
            rows = rows[supported]
            columns = columns[supported]
            warped = warped[supported, :3]
            cost = _feature_cost(
                target_features[:, rows, columns].T,
                warped,
                photometric_clip,
            )
            weights = torch.clamp(
                torch.cos(
                    math.pi / 2.0 - (rows.to(torch.float32) + 0.5) / height * math.pi
                ),
                min=1e-4,
            )
            cost_sum += float(torch.sum(cost * weights).item())
            weight_sum += float(torch.sum(weights).item())
            accepted += int(rows.numel())
    return {
        "view_id": source.view_id,
        "valid_pixels": accepted,
        "weighted_mean_feature_cost": cost_sum / max(weight_sum, 1e-12),
        "weighted_support": weight_sum,
    }


__all__ = [
    "INTERFACE",
    "DepthPrior",
    "DifferentiableSphericalDepthRefiner",
    "RefinementOptions",
    "RefinementResult",
    "SourceView",
    "photometric_features",
    "project_points_to_grid",
    "rays_for_pixels",
    "reprojection_score",
    "warp_source",
]
