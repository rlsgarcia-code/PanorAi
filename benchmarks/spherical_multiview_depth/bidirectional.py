"""Bidirectional spherical local cost volume around a radial-depth seed."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

from .refinement import (
    INTERFACE,
    DepthPrior,
    SourceView,
    _as_hw_tensor,
    _as_rgb_tensor,
    _batch_indices,
    _feature_cost,
    _seam_safe_sample,
    _torch,
    _validate_pose,
    photometric_features,
    project_points_to_grid,
    rays_for_pixels,
    warp_source,
)


BIDIRECTIONAL_INTERFACE = f"{INTERFACE}/bidirectional-cost-volume-v1"


@dataclass(frozen=True, slots=True)
class BidirectionalCostVolumeOptions:
    """Controls for native-resolution reciprocal spherical matching."""

    min_range_m: float = 0.3
    max_range_m: float = 15.0
    hypotheses: int = 17
    inverse_log_radius: float = math.log(1.5)
    temperature: float = 0.02
    row_batch: int = 16
    feature_window: int = 7
    descriptor_radius: int = 0
    photometric_clip: float = 0.35
    min_texture_std: float = 0.01
    minimum_confidence: float = 0.02
    maximum_cycle_px: float = 2.0
    maximum_cycle_log_range: float = 0.08
    maximum_fusion_log_disagreement: float = 0.12

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_range_m) or self.min_range_m <= 0.0:
            raise ValueError("min_range_m must be positive and finite")
        if not math.isfinite(self.max_range_m) or self.max_range_m <= self.min_range_m:
            raise ValueError("max_range_m must exceed min_range_m")
        for name in ("hypotheses", "row_batch", "feature_window"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hypotheses < 3 or self.hypotheses % 2 == 0:
            raise ValueError("hypotheses must be odd and at least 3")
        if self.feature_window % 2 == 0:
            raise ValueError("feature_window must be odd")
        if (
            isinstance(self.descriptor_radius, bool)
            or not isinstance(self.descriptor_radius, int)
            or self.descriptor_radius < 0
        ):
            raise ValueError("descriptor_radius must be a non-negative integer")
        for name in (
            "inverse_log_radius",
            "temperature",
            "photometric_clip",
            "maximum_cycle_px",
            "maximum_cycle_log_range",
            "maximum_fusion_log_disagreement",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0.0 <= self.min_texture_std <= 1.0:
            raise ValueError("min_texture_std must lie in [0, 1]")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must lie in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BidirectionalCostVolumeResult:
    """Forward and reciprocal fused radial-range maps on the seed lattice."""

    forward_range_m: Any
    reciprocal_range_m: Any
    validity_mask: Any
    forward_confidence: Any
    reciprocal_confidence: Any
    reciprocal_accepted_count: Any
    source_summaries: tuple[dict[str, float | int | str], ...]
    options: BidirectionalCostVolumeOptions
    prior_provenance: str
    source_view_ids: tuple[str, ...]
    interface: str = BIDIRECTIONAL_INTERFACE


def _distribution_statistics(cost: Any, support: Any, temperature: float) -> tuple:
    """Return soft probabilities, expected-index confidence and support."""

    torch = _torch()
    penalized = torch.where(support, cost, cost.new_full((), 1.0))
    probability = torch.softmax(-penalized / temperature, dim=0)
    supported = torch.any(support, dim=0)
    entropy = -torch.sum(
        probability * torch.log(torch.clamp(probability, min=1e-12)), dim=0
    ) / math.log(cost.shape[0])
    best_two = torch.topk(penalized, k=2, dim=0, largest=False).values
    margin = torch.clamp(best_two[1] - best_two[0], min=0.0)
    margin_confidence = 1.0 - torch.exp(-margin / temperature)
    confidence = torch.clamp((1.0 - entropy) * margin_confidence, 0.0, 1.0)
    confidence = torch.where(supported, confidence, torch.zeros_like(confidence))
    return probability, confidence, supported


def matching_features(rgb: Any, *, window: int, radius: int) -> tuple[Any, Any]:
    """Build a seam-safe locally normalized patch descriptor on the native ERP."""

    torch = _torch()
    functional = torch.nn.functional
    base, texture = photometric_features(rgb, window=window)
    normalized_luma = base[0:1]
    if radius == 0:
        return base, texture
    vertically_padded = functional.pad(
        normalized_luma, (0, 0, radius, radius), mode="replicate"
    )
    height = normalized_luma.shape[1]
    patch_channels = []
    for delta_y in range(-radius, radius + 1):
        vertical = vertically_padded[:, radius + delta_y : radius + delta_y + height, :]
        for delta_x in range(-radius, radius + 1):
            patch_channels.append(torch.roll(vertical, -delta_x, dims=-1))
    descriptor = torch.cat((*patch_channels, base[1:]), dim=0)
    return descriptor.contiguous(), texture


def _descriptor_cost(target: Any, source: Any, clip: float) -> Any:
    torch = _torch()
    if target.shape[1] == 3:
        return _feature_cost(target, source, clip)
    return torch.mean(torch.clamp(torch.abs(target - source), max=clip), dim=1)


def bidirectional_cost_volume_batch(
    *,
    seed_range_m: Any,
    rows: Any,
    columns: Any,
    target_features_chw: Any,
    source_features_chw: Any,
    source_validity_hw: Any,
    target_validity_hw: Any,
    rotation_source_from_target: Any,
    translation_source_from_target_m: Any,
    shape_hw: tuple[int, int],
    options: BidirectionalCostVolumeOptions,
) -> dict[str, Any]:
    """Evaluate differentiable B->A and A->B local volumes for one stripe."""

    torch = _torch()
    count = int(seed_range_m.numel())
    offsets = torch.linspace(
        -options.inverse_log_radius,
        options.inverse_log_radius,
        options.hypotheses,
        dtype=seed_range_m.dtype,
        device=seed_range_m.device,
    )
    inverse_seed = torch.reciprocal(seed_range_m)
    inverse_hypotheses = inverse_seed[None] * torch.exp(offsets[:, None])
    hypotheses = torch.clamp(
        torch.reciprocal(inverse_hypotheses),
        options.min_range_m,
        options.max_range_m,
    )
    repeated_rows = rows.repeat(options.hypotheses)
    repeated_columns = columns.repeat(options.hypotheses)
    target_anchor = target_features_chw[:, rows, columns].T
    target_repeated = target_anchor.repeat(options.hypotheses, 1)
    source_packed = torch.cat(
        (source_features_chw, source_validity_hw[None].to(source_features_chw.dtype)),
        dim=0,
    )
    forward_sample = warp_source(
        source_packed,
        repeated_rows,
        repeated_columns,
        hypotheses.reshape(-1),
        rotation_source_from_target,
        translation_source_from_target_m,
        shape_hw,
    )
    forward_cost = _descriptor_cost(
        target_repeated,
        forward_sample[:, :-1],
        options.photometric_clip,
    ).reshape(options.hypotheses, count)
    forward_support = (forward_sample[:, -1] >= 0.999).reshape(
        options.hypotheses, count
    )
    forward_probability, forward_confidence, forward_supported = (
        _distribution_statistics(forward_cost, forward_support, options.temperature)
    )
    forward_log_range = torch.sum(
        forward_probability * torch.log(torch.clamp(hypotheses, min=1e-8)), dim=0
    )
    forward_range = torch.exp(forward_log_range)

    target_rays = rays_for_pixels(rows, columns, shape_hw)
    forward_points_target = target_rays * forward_range[:, None]
    forward_points_source = (
        forward_points_target @ rotation_source_from_target.T
        + translation_source_from_target_m
    )
    source_anchor_range = torch.linalg.vector_norm(forward_points_source, dim=-1)
    source_anchor_ray = forward_points_source / torch.clamp(
        source_anchor_range[:, None], min=1e-8
    )
    source_anchor_grid = project_points_to_grid(forward_points_source, shape_hw)
    source_anchor_feature = _seam_safe_sample(source_features_chw, source_anchor_grid)
    source_anchor_validity = _seam_safe_sample(
        source_validity_hw[None].to(source_features_chw.dtype), source_anchor_grid
    )[:, 0]

    reverse_inverse_seed = torch.reciprocal(source_anchor_range)
    reverse_inverse_hypotheses = reverse_inverse_seed[None] * torch.exp(
        offsets[:, None]
    )
    reverse_hypotheses = torch.clamp(
        torch.reciprocal(reverse_inverse_hypotheses),
        options.min_range_m,
        options.max_range_m,
    )
    reverse_points_source = source_anchor_ray[None] * reverse_hypotheses[..., None]
    reverse_points_target = (
        reverse_points_source - translation_source_from_target_m
    ) @ rotation_source_from_target
    reverse_grid = project_points_to_grid(reverse_points_target, shape_hw)
    target_packed = torch.cat(
        (target_features_chw, target_validity_hw[None].to(target_features_chw.dtype)),
        dim=0,
    )
    reverse_sample = _seam_safe_sample(target_packed, reverse_grid.reshape(-1, 2))
    source_anchor_repeated = source_anchor_feature.repeat(options.hypotheses, 1)
    reverse_cost = _descriptor_cost(
        source_anchor_repeated,
        reverse_sample[:, :-1],
        options.photometric_clip,
    ).reshape(options.hypotheses, count)
    reverse_support = (reverse_sample[:, -1] >= 0.999).reshape(
        options.hypotheses, count
    ) & (source_anchor_validity[None] >= 0.999)
    reverse_probability, reverse_confidence, reverse_supported = (
        _distribution_statistics(reverse_cost, reverse_support, options.temperature)
    )
    reverse_log_range = torch.sum(
        reverse_probability * torch.log(torch.clamp(reverse_hypotheses, min=1e-8)),
        dim=0,
    )
    reverse_range = torch.exp(reverse_log_range)
    cycle_points_source = source_anchor_ray * reverse_range[:, None]
    cycle_points_target = (
        cycle_points_source - translation_source_from_target_m
    ) @ rotation_source_from_target
    cycle_range_target = torch.linalg.vector_norm(cycle_points_target, dim=-1)
    cycle_ray_target = cycle_points_target / torch.clamp(
        cycle_range_target[:, None], min=1e-8
    )
    cycle_angle = torch.acos(
        torch.clamp(torch.sum(cycle_ray_target * target_rays, dim=-1), -1.0, 1.0)
    )
    cycle_px = cycle_angle * (shape_hw[1] / (2.0 * math.pi))
    cycle_log_range = torch.abs(
        torch.log(torch.clamp(cycle_range_target, min=1e-8))
        - torch.log(torch.clamp(forward_range, min=1e-8))
    )
    confidence = torch.sqrt(forward_confidence * reverse_confidence)
    reciprocal = (
        forward_supported
        & reverse_supported
        & (confidence >= options.minimum_confidence)
        & (cycle_px <= options.maximum_cycle_px)
        & (cycle_log_range <= options.maximum_cycle_log_range)
    )
    return {
        "forward_range_m": forward_range,
        "forward_confidence": forward_confidence,
        "forward_supported": forward_supported,
        "reverse_confidence": reverse_confidence,
        "confidence": confidence,
        "reciprocal": reciprocal,
        "cycle_px": cycle_px,
        "cycle_log_range": cycle_log_range,
        "forward_cost_min": torch.min(
            torch.where(forward_support, forward_cost, forward_cost.new_full((), 1.0)),
            dim=0,
        ).values,
    }


class BidirectionalSphericalCostVolume:
    """Fuse reciprocal local depth proposals from fixed registered source views."""

    interface = BIDIRECTIONAL_INTERFACE
    stability = "experimental"
    depth_semantics = "radial_range_m"

    def __init__(self, options: BidirectionalCostVolumeOptions | None = None) -> None:
        self.options = options or BidirectionalCostVolumeOptions()

    def infer(
        self,
        prior: DepthPrior,
        target_rgb: Any,
        sources: Iterable[SourceView],
        *,
        device: str | Any = "cpu",
        progress: Any | None = None,
    ) -> BidirectionalCostVolumeResult:
        torch = _torch()
        target = _as_rgb_tensor(target_rgb, device=device, name="target_rgb")
        shape_hw = (int(target.shape[1]), int(target.shape[2]))
        height, width = shape_hw
        seed = _as_hw_tensor(
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
        validity = validity & torch.isfinite(seed) & (seed > 0.0)
        seed = torch.clamp(
            torch.nan_to_num(seed, nan=self.options.min_range_m),
            self.options.min_range_m,
            self.options.max_range_m,
        )
        target_features, target_texture = matching_features(
            target,
            window=self.options.feature_window,
            radius=self.options.descriptor_radius,
        )
        prepared_sources = []
        for source in sources:
            source_rgb = _as_rgb_tensor(
                source.rgb, device=device, name=f"{source.view_id}.rgb"
            )
            if tuple(source_rgb.shape[1:]) != shape_hw:
                raise ValueError("all source ERPs must share the target native shape")
            R, t = _validate_pose(
                source.rotation_source_from_target,
                source.translation_source_from_target_m,
                device=device,
            )
            source_features, _ = matching_features(
                source_rgb,
                window=self.options.feature_window,
                radius=self.options.descriptor_radius,
            )
            source_validity = (
                torch.ones(shape_hw, device=device, dtype=torch.bool)
                if source.validity_mask is None
                else _as_hw_tensor(
                    source.validity_mask,
                    device=device,
                    dtype=torch.bool,
                    shape_hw=shape_hw,
                    name=f"{source.view_id}.validity_mask",
                )
            )
            prepared_sources.append(
                (source_features, source_validity, R, t, source.view_id)
            )
        if not prepared_sources:
            raise ValueError("at least one source view is required")

        forward_log_sum = torch.zeros(shape_hw, device=device)
        forward_weight = torch.zeros(shape_hw, device=device)
        reciprocal_log_sum = torch.zeros(shape_hw, device=device)
        reciprocal_weight = torch.zeros(shape_hw, device=device)
        reciprocal_count = torch.zeros(shape_hw, device=device, dtype=torch.uint8)
        reciprocal_min_log = torch.full(shape_hw, torch.inf, device=device)
        reciprocal_max_log = torch.full(shape_hw, -torch.inf, device=device)
        summaries: list[dict[str, float | int | str]] = []

        with torch.inference_mode():
            for source_features, source_validity, R, t, view_id in prepared_sources:
                totals = {
                    "candidate_pixels": 0,
                    "forward_supported_pixels": 0,
                    "reciprocal_pixels": 0,
                    "confidence_sum": 0.0,
                    "cycle_px_sum": 0.0,
                    "forward_min_cost_sum": 0.0,
                }
                for start in range(0, height, self.options.row_batch):
                    stop = min(height, start + self.options.row_batch)
                    linear, rows, columns = _batch_indices(
                        start, stop, width, device=device
                    )
                    selected = validity[rows, columns] & (
                        target_texture[rows, columns] >= self.options.min_texture_std
                    )
                    if not bool(selected.any()):
                        continue
                    linear = linear[selected]
                    rows = rows[selected]
                    columns = columns[selected]
                    batch = bidirectional_cost_volume_batch(
                        seed_range_m=seed[rows, columns],
                        rows=rows,
                        columns=columns,
                        target_features_chw=target_features,
                        source_features_chw=source_features,
                        source_validity_hw=source_validity,
                        target_validity_hw=validity,
                        rotation_source_from_target=R,
                        translation_source_from_target_m=t,
                        shape_hw=shape_hw,
                        options=self.options,
                    )
                    forward_ok = batch["forward_supported"] & (
                        batch["forward_confidence"] >= self.options.minimum_confidence
                    )
                    forward_confidence = torch.where(
                        forward_ok,
                        batch["forward_confidence"],
                        torch.zeros_like(batch["forward_confidence"]),
                    )
                    forward_log = torch.log(
                        torch.clamp(batch["forward_range_m"], min=1e-8)
                    )
                    forward_log_sum.view(-1)[linear] += forward_confidence * forward_log
                    forward_weight.view(-1)[linear] += forward_confidence

                    reciprocal = batch["reciprocal"]
                    reciprocal_confidence = torch.where(
                        reciprocal,
                        batch["confidence"],
                        torch.zeros_like(batch["confidence"]),
                    )
                    reciprocal_log_sum.view(-1)[linear] += (
                        reciprocal_confidence * forward_log
                    )
                    reciprocal_weight.view(-1)[linear] += reciprocal_confidence
                    reciprocal_count.view(-1)[linear] += reciprocal.to(torch.uint8)
                    reciprocal_min_log.view(-1)[linear] = torch.minimum(
                        reciprocal_min_log.view(-1)[linear],
                        torch.where(
                            reciprocal,
                            forward_log,
                            forward_log.new_full((), torch.inf),
                        ),
                    )
                    reciprocal_max_log.view(-1)[linear] = torch.maximum(
                        reciprocal_max_log.view(-1)[linear],
                        torch.where(
                            reciprocal,
                            forward_log,
                            forward_log.new_full((), -torch.inf),
                        ),
                    )
                    totals["candidate_pixels"] += int(rows.numel())
                    totals["forward_supported_pixels"] += int(forward_ok.sum().item())
                    totals["reciprocal_pixels"] += int(reciprocal.sum().item())
                    totals["confidence_sum"] += float(
                        batch["confidence"][reciprocal].sum().item()
                    )
                    totals["cycle_px_sum"] += float(
                        batch["cycle_px"][reciprocal].sum().item()
                    )
                    totals["forward_min_cost_sum"] += float(
                        batch["forward_cost_min"][forward_ok].sum().item()
                    )
                summary: dict[str, float | int | str] = {
                    "view_id": view_id,
                    **totals,
                    "reciprocal_fraction": totals["reciprocal_pixels"]
                    / max(1, totals["candidate_pixels"]),
                    "reciprocal_mean_confidence": totals["confidence_sum"]
                    / max(1, totals["reciprocal_pixels"]),
                    "reciprocal_mean_cycle_px": totals["cycle_px_sum"]
                    / max(1, totals["reciprocal_pixels"]),
                    "forward_mean_min_cost": totals["forward_min_cost_sum"]
                    / max(1, totals["forward_supported_pixels"]),
                }
                summaries.append(summary)
                if progress is not None:
                    progress(summary)

        seed_log = torch.log(torch.clamp(seed, min=1e-8))
        forward_log = torch.where(
            forward_weight > 0.0,
            forward_log_sum / torch.clamp(forward_weight, min=1e-12),
            seed_log,
        )
        forward = torch.exp(forward_log)
        reciprocal_log = torch.where(
            reciprocal_weight > 0.0,
            reciprocal_log_sum / torch.clamp(reciprocal_weight, min=1e-12),
            seed_log,
        )
        disagreement = reciprocal_max_log - reciprocal_min_log
        reciprocal_usable = (reciprocal_weight > 0.0) & (
            (reciprocal_count == 1)
            | (disagreement <= self.options.maximum_fusion_log_disagreement)
        )
        reciprocal = torch.where(reciprocal_usable, torch.exp(reciprocal_log), seed)
        forward = torch.clamp(
            forward, self.options.min_range_m, self.options.max_range_m
        )
        reciprocal = torch.clamp(
            reciprocal, self.options.min_range_m, self.options.max_range_m
        )
        nan = torch.full_like(seed, torch.nan)
        forward = torch.where(validity, forward, nan)
        reciprocal = torch.where(validity, reciprocal, nan)
        return BidirectionalCostVolumeResult(
            forward_range_m=forward.detach(),
            reciprocal_range_m=reciprocal.detach(),
            validity_mask=validity.detach(),
            forward_confidence=torch.clamp(forward_weight, 0.0, 1.0).detach(),
            reciprocal_confidence=torch.clamp(reciprocal_weight, 0.0, 1.0).detach(),
            reciprocal_accepted_count=reciprocal_count.detach(),
            source_summaries=tuple(summaries),
            options=self.options,
            prior_provenance=prior.provenance,
            source_view_ids=tuple(item[4] for item in prepared_sources),
        )


__all__ = [
    "BIDIRECTIONAL_INTERFACE",
    "BidirectionalCostVolumeOptions",
    "BidirectionalCostVolumeResult",
    "BidirectionalSphericalCostVolume",
    "bidirectional_cost_volume_batch",
]
