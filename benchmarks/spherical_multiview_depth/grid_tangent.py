"""Semi-dense spherical grid matching with transported tangent patches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import numpy as np

from .refinement import (
    _as_hw_tensor,
    _as_rgb_tensor,
    _seam_safe_sample,
    _torch,
    _validate_pose,
    project_points_to_grid,
    rays_for_pixels,
)
from .tangent_seeds import (
    PropagatedDepthResult,
    TangentDepthSeedOptions,
    TangentDepthSeedResult,
    propagate_tangent_depth_seeds,
)


INTERFACE = "panorai-experimental-grid-tangent-depth/v1"


@dataclass(frozen=True, slots=True)
class GridTangentOptions:
    """Configuration for native-grid local inverse-range patch search."""

    stride_px: int = 32
    patch_samples: int = 7
    support_radius_deg: float = 1.0
    hypotheses: int = 17
    range_factor: float = 1.5
    batch_size: int = 256
    min_range_m: float = 0.3
    max_range_m: float = 15.0
    minimum_patch_support: float = 0.90
    minimum_patch_std: float = 0.02
    maximum_zncc_cost: float = 0.40
    minimum_cost_margin: float = 0.0001
    maximum_source_log_disagreement: float = 0.10
    minimum_propagation_weight: float = 0.05

    def __post_init__(self) -> None:
        for name in ("stride_px", "patch_samples", "hypotheses", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.patch_samples < 3 or self.patch_samples % 2 == 0:
            raise ValueError("patch_samples must be odd and at least 3")
        if self.hypotheses < 3 or self.hypotheses % 2 == 0:
            raise ValueError("hypotheses must be odd and at least 3")
        if not math.isfinite(self.range_factor) or self.range_factor <= 1.0:
            raise ValueError("range_factor must exceed 1")
        if not math.isfinite(self.min_range_m) or self.min_range_m <= 0.0:
            raise ValueError("min_range_m must be positive")
        if not math.isfinite(self.max_range_m) or self.max_range_m <= self.min_range_m:
            raise ValueError("max_range_m must exceed min_range_m")
        for name in (
            "support_radius_deg",
            "minimum_patch_std",
            "maximum_zncc_cost",
            "minimum_cost_margin",
            "maximum_source_log_disagreement",
            "minimum_propagation_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0.0 < self.minimum_patch_support <= 1.0:
            raise ValueError("minimum_patch_support must lie in (0, 1]")
        if self.support_radius_deg >= 45.0:
            raise ValueError("support_radius_deg must be below 45 degrees")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GridTangentProposal:
    """One source's depth proposal aligned with a shared target grid."""

    rows: np.ndarray
    columns: np.ndarray
    prior_range_m: np.ndarray
    proposed_range_m: np.ndarray
    accepted: np.ndarray
    confidence: np.ndarray
    best_cost: np.ndarray
    cost_margin: np.ndarray
    target_patch_std: np.ndarray
    best_hypothesis_index: np.ndarray
    source_view_id: str
    options: GridTangentOptions
    interface: str = INTERFACE

    def __post_init__(self) -> None:
        count = np.asarray(self.rows).size
        for name in (
            "rows",
            "columns",
            "prior_range_m",
            "proposed_range_m",
            "accepted",
            "confidence",
            "best_cost",
            "cost_margin",
            "target_patch_std",
            "best_hypothesis_index",
        ):
            if np.asarray(getattr(self, name)).shape != (count,):
                raise ValueError(f"{name} must have shape (N,)")

    def describe(self) -> dict[str, Any]:
        accepted = np.asarray(self.accepted, dtype=bool)
        return {
            "interface": self.interface,
            "source_view_id": self.source_view_id,
            "grid_points": int(accepted.size),
            "accepted_points": int(accepted.sum()),
            "accepted_fraction": float(accepted.mean()) if accepted.size else 0.0,
            "median_best_cost": _median(self.best_cost[accepted]),
            "median_cost_margin": _median(self.cost_margin[accepted]),
            "median_confidence": _median(self.confidence[accepted]),
            "options": self.options.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class FusedGridTangentProposal:
    """Union and strict-consensus proposals on the common grid."""

    rows: np.ndarray
    columns: np.ndarray
    prior_range_m: np.ndarray
    union_range_m: np.ndarray
    consensus_range_m: np.ndarray
    union_accepted: np.ndarray
    consensus_accepted: np.ndarray
    union_confidence: np.ndarray
    consensus_confidence: np.ndarray
    accepted_source_count: np.ndarray
    source_log_disagreement: np.ndarray
    source_view_ids: tuple[str, ...]
    options: GridTangentOptions
    interface: str = INTERFACE


def regular_spherical_grid(
    shape_hw: tuple[int, int], stride_px: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic pixel-centre grid without resizing the ERP."""

    height, width = shape_hw
    if height < 1 or width < 2:
        raise ValueError("shape_hw must be a positive ERP shape")
    if isinstance(stride_px, bool) or not isinstance(stride_px, int) or stride_px < 1:
        raise ValueError("stride_px must be a positive integer")
    offset = stride_px // 2
    rows = np.arange(offset, height, stride_px, dtype=np.int64)
    columns = np.arange(offset, width, stride_px, dtype=np.int64)
    row_grid, column_grid = np.meshgrid(rows, columns, indexing="ij")
    return row_grid.ravel(), column_grid.ravel()


def infer_grid_tangent_proposal(
    seed_range_m: Any,
    target_rgb: Any,
    source_rgb: Any,
    target_validity: Any,
    source_validity: Any,
    rotation_source_from_target: Any,
    translation_source_from_target_m: Any,
    *,
    rows: Any | None = None,
    columns: Any | None = None,
    options: GridTangentOptions | None = None,
    source_view_id: str = "source",
    device: str | Any = "cpu",
    progress: Any | None = None,
) -> GridTangentProposal:
    """Search depth hypotheses with a pose-transported tangent ZNCC patch."""

    settings = options or GridTangentOptions()
    torch = _torch()
    target = _as_rgb_tensor(target_rgb, device=device, name="target_rgb")
    source = _as_rgb_tensor(source_rgb, device=device, name="source_rgb")
    shape_hw = (int(target.shape[1]), int(target.shape[2]))
    if tuple(source.shape[1:]) != shape_hw:
        raise ValueError("target and source RGB must share the native ERP shape")
    seed = _as_hw_tensor(
        seed_range_m,
        device=device,
        dtype=torch.float32,
        shape_hw=shape_hw,
        name="seed_range_m",
    )
    target_mask = _as_hw_tensor(
        target_validity,
        device=device,
        dtype=torch.bool,
        shape_hw=shape_hw,
        name="target_validity",
    )
    source_mask = _as_hw_tensor(
        source_validity,
        device=device,
        dtype=torch.bool,
        shape_hw=shape_hw,
        name="source_validity",
    )
    rotation, translation = _validate_pose(
        rotation_source_from_target,
        translation_source_from_target_m,
        device=device,
    )
    if rows is None or columns is None:
        if rows is not None or columns is not None:
            raise ValueError("rows and columns must be supplied together")
        row_array, column_array = regular_spherical_grid(shape_hw, settings.stride_px)
    else:
        row_array = np.asarray(rows, dtype=np.int64)
        column_array = np.asarray(columns, dtype=np.int64)
        if row_array.shape != column_array.shape or row_array.ndim != 1:
            raise ValueError("rows and columns must have matching shape (N,)")
        if (
            (row_array < 0).any()
            or (row_array >= shape_hw[0]).any()
            or (column_array < 0).any()
            or (column_array >= shape_hw[1]).any()
        ):
            raise ValueError("grid coordinates must lie inside the ERP")
    row_tensor = torch.as_tensor(row_array, dtype=torch.long, device=device)
    column_tensor = torch.as_tensor(column_array, dtype=torch.long, device=device)
    count = int(row_tensor.numel())
    prior = seed[row_tensor, column_tensor]
    center_valid = (
        target_mask[row_tensor, column_tensor]
        & torch.isfinite(prior)
        & (prior >= settings.min_range_m)
        & (prior <= settings.max_range_m)
    )
    target_luma = 0.2126 * target[0:1] + 0.7152 * target[1:2] + 0.0722 * target[2:3]
    source_luma = 0.2126 * source[0:1] + 0.7152 * source[1:2] + 0.0722 * source[2:3]
    target_packed = torch.cat((target_luma, target_mask[None].to(target.dtype)), dim=0)
    source_packed = torch.cat((source_luma, source_mask[None].to(source.dtype)), dim=0)
    offsets = torch.linspace(
        -math.log(settings.range_factor),
        math.log(settings.range_factor),
        settings.hypotheses,
        dtype=prior.dtype,
        device=device,
    )
    tangent_coordinate = torch.linspace(
        -math.tan(math.radians(settings.support_radius_deg)),
        math.tan(math.radians(settings.support_radius_deg)),
        settings.patch_samples,
        dtype=prior.dtype,
        device=device,
    )
    tangent_y, tangent_x = torch.meshgrid(
        tangent_coordinate, tangent_coordinate, indexing="ij"
    )
    patch_offsets = torch.stack((tangent_x.ravel(), tangent_y.ravel()), dim=1)

    proposed = torch.full((count,), torch.nan, device=device)
    accepted = torch.zeros(count, dtype=torch.bool, device=device)
    confidence = torch.zeros(count, device=device)
    best_cost = torch.full((count,), torch.inf, device=device)
    cost_margin = torch.zeros(count, device=device)
    target_std = torch.zeros(count, device=device)
    best_index = torch.full((count,), -1, dtype=torch.long, device=device)

    with torch.inference_mode():
        for start in range(0, count, settings.batch_size):
            stop = min(count, start + settings.batch_size)
            batch = _grid_tangent_batch(
                prior=prior[start:stop],
                rows=row_tensor[start:stop],
                columns=column_tensor[start:stop],
                center_valid=center_valid[start:stop],
                target_packed=target_packed,
                source_packed=source_packed,
                rotation=rotation,
                translation=translation,
                offsets=offsets,
                patch_offsets=patch_offsets,
                shape_hw=shape_hw,
                settings=settings,
            )
            proposed[start:stop] = batch["proposed_range_m"]
            accepted[start:stop] = batch["accepted"]
            confidence[start:stop] = batch["confidence"]
            best_cost[start:stop] = batch["best_cost"]
            cost_margin[start:stop] = batch["cost_margin"]
            target_std[start:stop] = batch["target_patch_std"]
            best_index[start:stop] = batch["best_hypothesis_index"]
            if progress is not None and (
                stop == count or stop % max(settings.batch_size * 16, 1) == 0
            ):
                progress(
                    {
                        "source_view_id": source_view_id,
                        "processed": stop,
                        "total": count,
                        "accepted": int(accepted[:stop].sum().item()),
                    }
                )
    return GridTangentProposal(
        rows=row_array,
        columns=column_array,
        prior_range_m=prior.cpu().numpy(),
        proposed_range_m=proposed.cpu().numpy(),
        accepted=accepted.cpu().numpy(),
        confidence=confidence.cpu().numpy(),
        best_cost=best_cost.cpu().numpy(),
        cost_margin=cost_margin.cpu().numpy(),
        target_patch_std=target_std.cpu().numpy(),
        best_hypothesis_index=best_index.cpu().numpy(),
        source_view_id=source_view_id,
        options=settings,
    )


def _grid_tangent_batch(
    *,
    prior: Any,
    rows: Any,
    columns: Any,
    center_valid: Any,
    target_packed: Any,
    source_packed: Any,
    rotation: Any,
    translation: Any,
    offsets: Any,
    patch_offsets: Any,
    shape_hw: tuple[int, int],
    settings: GridTangentOptions,
) -> dict[str, Any]:
    torch = _torch()
    count = int(prior.numel())
    centers = rays_for_pixels(rows, columns, shape_hw)
    up = centers.new_tensor((0.0, 1.0, 0.0)).expand_as(centers)
    east = torch.linalg.cross(up, centers, dim=1)
    near_pole = torch.linalg.vector_norm(east, dim=1) < 1e-6
    fallback = centers.new_tensor((0.0, 0.0, 1.0)).expand_as(centers)
    east = torch.where(
        near_pole[:, None], torch.linalg.cross(fallback, centers, dim=1), east
    )
    east = east / torch.clamp(torch.linalg.vector_norm(east, dim=1)[:, None], min=1e-8)
    north = torch.linalg.cross(centers, east, dim=1)
    north = north / torch.clamp(
        torch.linalg.vector_norm(north, dim=1)[:, None], min=1e-8
    )
    patch_rays = (
        centers[:, None, :]
        + patch_offsets[None, :, :1] * east[:, None, :]
        + patch_offsets[None, :, 1:] * north[:, None, :]
    )
    patch_rays = patch_rays / torch.clamp(
        torch.linalg.vector_norm(patch_rays, dim=2)[:, :, None], min=1e-8
    )
    target_grid = project_points_to_grid(patch_rays, shape_hw)
    target_sample = _seam_safe_sample(target_packed, target_grid.reshape(-1, 2))
    patch_count = int(patch_offsets.shape[0])
    target_sample = target_sample.reshape(count, patch_count, 2)
    target_values = target_sample[:, :, 0]
    target_support = target_sample[:, :, 1] >= 0.999

    inverse = torch.reciprocal(torch.clamp(prior, min=1e-8))[None] * torch.exp(
        offsets[:, None]
    )
    hypotheses = torch.clamp(
        torch.reciprocal(inverse), settings.min_range_m, settings.max_range_m
    )
    points_target = patch_rays[None] * hypotheses[:, :, None, None]
    points_source = points_target @ rotation.T + translation
    source_grid = project_points_to_grid(points_source, shape_hw)
    source_sample = _seam_safe_sample(source_packed, source_grid.reshape(-1, 2))
    source_sample = source_sample.reshape(settings.hypotheses, count, patch_count, 2)
    source_values = source_sample[:, :, :, 0]
    support = target_support[None] & (source_sample[:, :, :, 1] >= 0.999)
    support_count = support.sum(dim=2)
    safe_count = torch.clamp(support_count, min=1)
    target_masked = torch.where(support, target_values[None], 0.0)
    source_masked = torch.where(support, source_values, 0.0)
    target_mean = target_masked.sum(dim=2) / safe_count
    source_mean = source_masked.sum(dim=2) / safe_count
    target_centered = torch.where(
        support, target_values[None] - target_mean[:, :, None], 0.0
    )
    source_centered = torch.where(support, source_values - source_mean[:, :, None], 0.0)
    target_std = torch.sqrt(
        torch.sum(target_centered * target_centered, dim=2) / safe_count
    )
    source_std = torch.sqrt(
        torch.sum(source_centered * source_centered, dim=2) / safe_count
    )
    covariance = torch.sum(target_centered * source_centered, dim=2) / safe_count
    valid = (
        center_valid[None]
        & (support_count >= math.ceil(settings.minimum_patch_support * patch_count))
        & (target_std >= settings.minimum_patch_std)
        & (source_std >= settings.minimum_patch_std)
    )
    correlation = covariance / torch.clamp(target_std * source_std, min=1e-8)
    cost = torch.where(
        valid,
        1.0 - torch.clamp(correlation, -1.0, 1.0),
        correlation.new_full((), torch.inf),
    )
    best_values, best_indices = torch.min(cost, dim=0)
    best_two = torch.topk(cost, k=2, dim=0, largest=False).values
    margin = best_two[1] - best_two[0]
    interior = (best_indices > 0) & (best_indices < settings.hypotheses - 1)
    index = torch.arange(count, device=prior.device)
    left_index = torch.clamp(best_indices - 1, 0, settings.hypotheses - 1)
    right_index = torch.clamp(best_indices + 1, 0, settings.hypotheses - 1)
    left = cost[left_index, index]
    center = cost[best_indices, index]
    right = cost[right_index, index]
    denominator = left - 2.0 * center + right
    substep = torch.where(
        interior & torch.isfinite(denominator) & (torch.abs(denominator) > 1e-8),
        0.5 * (left - right) / denominator,
        torch.zeros_like(center),
    )
    substep = torch.clamp(substep, -1.0, 1.0)
    step = offsets[1] - offsets[0]
    refined_offset = offsets[best_indices] + substep * step
    proposed = torch.clamp(
        prior / torch.exp(refined_offset), settings.min_range_m, settings.max_range_m
    )
    accepted = (
        interior
        & torch.isfinite(best_values)
        & torch.isfinite(margin)
        & (best_values <= settings.maximum_zncc_cost)
        & (margin >= settings.minimum_cost_margin)
    )
    texture = target_std[best_indices, index]
    cost_confidence = torch.clamp(
        1.0 - best_values / settings.maximum_zncc_cost, 0.0, 1.0
    )
    margin_confidence = torch.clamp(
        margin / max(4.0 * settings.minimum_cost_margin, 1e-8), 0.0, 1.0
    )
    confidence = torch.where(
        accepted, torch.sqrt(cost_confidence * margin_confidence), 0.0
    )
    return {
        "proposed_range_m": proposed,
        "accepted": accepted,
        "confidence": confidence,
        "best_cost": best_values,
        "cost_margin": margin,
        "target_patch_std": texture,
        "best_hypothesis_index": best_indices,
    }


def fuse_grid_tangent_proposals(
    proposals: Iterable[GridTangentProposal],
) -> FusedGridTangentProposal:
    """Fuse single-source proposals and expose strict all-source consensus."""

    items = tuple(proposals)
    if not items:
        raise ValueError("at least one grid proposal is required")
    first = items[0]
    for item in items[1:]:
        if item.options != first.options:
            raise ValueError("all proposals must share options")
        if not np.array_equal(item.rows, first.rows) or not np.array_equal(
            item.columns, first.columns
        ):
            raise ValueError("all proposals must share the same target grid")
        if not np.allclose(item.prior_range_m, first.prior_range_m, equal_nan=True):
            raise ValueError("all proposals must share the same prior samples")
    accepted = np.stack([item.accepted for item in items])
    confidence = np.stack([item.confidence for item in items])
    log_range = np.log(
        np.clip(
            np.stack([item.proposed_range_m for item in items]),
            first.options.min_range_m,
            first.options.max_range_m,
        )
    )
    weights = np.where(accepted, confidence, 0.0)
    weight_sum = weights.sum(axis=0)
    count = accepted.sum(axis=0)
    fused_log = np.divide(
        np.sum(weights * log_range, axis=0),
        weight_sum,
        out=np.full(weight_sum.shape, np.nan),
        where=weight_sum > 0.0,
    )
    minimum = np.min(np.where(accepted, log_range, np.inf), axis=0)
    maximum = np.max(np.where(accepted, log_range, -np.inf), axis=0)
    disagreement = np.where(count >= 2, maximum - minimum, 0.0)
    compatible = disagreement <= first.options.maximum_source_log_disagreement
    union_accepted = (count >= 1) & compatible
    consensus_accepted = (count == len(items)) & compatible
    union_range = np.exp(fused_log)
    consensus_range = union_range.copy()
    union_range[~union_accepted] = np.nan
    consensus_range[~consensus_accepted] = np.nan
    mean_confidence = np.divide(
        weight_sum,
        np.maximum(count, 1),
        out=np.zeros_like(weight_sum),
        where=count > 0,
    )
    return FusedGridTangentProposal(
        rows=first.rows.copy(),
        columns=first.columns.copy(),
        prior_range_m=first.prior_range_m.copy(),
        union_range_m=union_range,
        consensus_range_m=consensus_range,
        union_accepted=union_accepted,
        consensus_accepted=consensus_accepted,
        union_confidence=np.where(union_accepted, mean_confidence, 0.0),
        consensus_confidence=np.where(consensus_accepted, mean_confidence, 0.0),
        accepted_source_count=count.astype(np.uint8),
        source_log_disagreement=disagreement,
        source_view_ids=tuple(item.source_view_id for item in items),
        options=first.options,
    )


def propagate_fused_grid(
    seed_range_m: Any,
    target_rgb: Any,
    fused: FusedGridTangentProposal,
    *,
    mode: str = "union",
) -> PropagatedDepthResult:
    """Use VAL-030's bounded spherical propagation on accepted grid seeds."""

    if mode not in {"union", "consensus"}:
        raise ValueError("mode must be 'union' or 'consensus'")
    torch = _torch()
    seed = np.asarray(seed_range_m, dtype=np.float32)
    rows = torch.as_tensor(fused.rows, dtype=torch.long)
    columns = torch.as_tensor(fused.columns, dtype=torch.long)
    bearings = rays_for_pixels(rows, columns, seed.shape).cpu().numpy()
    accepted = fused.union_accepted if mode == "union" else fused.consensus_accepted
    solved = fused.union_range_m if mode == "union" else fused.consensus_range_m
    confidence = (
        fused.union_confidence if mode == "union" else fused.consensus_confidence
    )
    propagation_options = TangentDepthSeedOptions(
        min_range_m=fused.options.min_range_m,
        max_range_m=fused.options.max_range_m,
        descriptor_radius_sigmas=6.0,
        minimum_propagation_radius_deg=fused.options.support_radius_deg,
        maximum_propagation_radius_deg=fused.options.support_radius_deg,
        minimum_propagation_weight=fused.options.minimum_propagation_weight,
    )
    pixels = np.stack((fused.columns, fused.rows), axis=1).astype(np.float64)
    sparse = TangentDepthSeedResult(
        target_bearings=bearings,
        source_bearings=bearings,
        target_pixels_xy=pixels,
        prior_range_m=fused.prior_range_m,
        solved_range_m=solved,
        source_range_m=solved,
        accepted=accepted,
        reprojection_error_deg=np.full(accepted.size, np.nan),
        parallax_deg=np.full(accepted.size, np.nan),
        ray_miss_m=np.full(accepted.size, np.nan),
        best_hypothesis_index=np.full(accepted.size, -1, dtype=np.int64),
        confidence=confidence,
        target_scale_deg=np.full(accepted.size, fused.options.support_radius_deg / 6.0),
        descriptor_distance=np.full(accepted.size, np.nan),
        options=propagation_options,
        descriptor_adapter=(
            f"tangent-zncc-grid:{fused.options.patch_samples}x"
            f"{fused.options.patch_samples}:radius={fused.options.support_radius_deg}deg"
        ),
        pose_source="registered-metric",
    )
    return propagate_tangent_depth_seeds(
        seed, target_rgb, (sparse,), options=propagation_options
    )


def _median(value: Any) -> float | None:
    array = np.asarray(value)
    finite = array[np.isfinite(array)]
    return None if not finite.size else float(np.median(finite))


__all__ = [
    "FusedGridTangentProposal",
    "GridTangentOptions",
    "GridTangentProposal",
    "INTERFACE",
    "fuse_grid_tangent_proposals",
    "infer_grid_tangent_proposal",
    "propagate_fused_grid",
    "regular_spherical_grid",
]
