"""Latent-conditioned convex fusion for tangent-view DA3 depth predictions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


INTERFACE = "panorai-da3-latent-conditioned-depth-fusion/v1-experimental"
LOCAL_INPUT_NAMES = (
    "signed_log_depth_disagreement",
    "absolute_log_depth_disagreement",
    "normalized_depth_gradient",
    "normalized_face_x",
    "normalized_face_y",
    "incidence_radius_squared",
    "overlap_log_depth_dispersion",
    "signed_token_gate",
)
TOKEN_LATENT_DIM = 8


class LatentConditionedDepthFusion:
    """Factory namespace for the fixed FiLM-conditioned fusion scorer."""

    @staticmethod
    def create(
        *,
        maximum_logit_correction: float = 2.0,
        local_mean: Any | None = None,
        local_std: Any | None = None,
        latent_mean: Any | None = None,
        latent_std: Any | None = None,
    ) -> Any:
        import torch
        from torch import nn

        if maximum_logit_correction <= 0.0:
            raise ValueError("maximum_logit_correction must be positive")

        class _Fusion(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.maximum_logit_correction = float(maximum_logit_correction)
                local_center = torch.zeros(len(LOCAL_INPUT_NAMES), dtype=torch.float32)
                local_scale = torch.ones(len(LOCAL_INPUT_NAMES), dtype=torch.float32)
                latent_center = torch.zeros(TOKEN_LATENT_DIM, dtype=torch.float32)
                latent_scale = torch.ones(TOKEN_LATENT_DIM, dtype=torch.float32)
                if local_mean is not None:
                    local_center.copy_(torch.as_tensor(local_mean, dtype=torch.float32))
                if local_std is not None:
                    local_scale.copy_(torch.as_tensor(local_std, dtype=torch.float32))
                if latent_mean is not None:
                    latent_center.copy_(
                        torch.as_tensor(latent_mean, dtype=torch.float32)
                    )
                if latent_std is not None:
                    latent_scale.copy_(torch.as_tensor(latent_std, dtype=torch.float32))
                self.register_buffer("local_mean", local_center)
                self.register_buffer("local_std", local_scale.clamp_min(1e-6))
                self.register_buffer("latent_mean", latent_center)
                self.register_buffer("latent_std", latent_scale.clamp_min(1e-6))
                self.local_encoder = nn.Linear(len(LOCAL_INPUT_NAMES), 16)
                self.conditioner = nn.Linear(TOKEN_LATENT_DIM, 32)
                self.output = nn.Linear(16, 1)
                nn.init.zeros_(self.output.weight)
                nn.init.zeros_(self.output.bias)

            def contribution_logits(
                self,
                local_features: Any,
                token_latent: Any,
                token_gate: Any,
                gaussian_weight: Any,
                validity: Any,
            ) -> Any:
                if local_features.shape[:-1] != token_latent.shape[:-1]:
                    raise ValueError("local features and token latent shapes disagree")
                if local_features.shape[-1] != len(LOCAL_INPUT_NAMES):
                    raise ValueError("unexpected local fusion feature count")
                if token_latent.shape[-1] != TOKEN_LATENT_DIM:
                    raise ValueError("unexpected token latent width")
                if token_gate.shape != local_features.shape[:-1]:
                    raise ValueError("token gate shape disagrees with contributions")
                if gaussian_weight.shape != token_gate.shape:
                    raise ValueError(
                        "Gaussian weight shape disagrees with contributions"
                    )
                if validity.shape != token_gate.shape:
                    raise ValueError("validity shape disagrees with contributions")

                normalized_local = (local_features - self.local_mean) / self.local_std
                normalized_latent = (token_latent - self.latent_mean) / self.latent_std
                encoded = self.local_encoder(normalized_local)
                gamma, beta = self.conditioner(normalized_latent).chunk(2, dim=-1)
                calibrated = torch.nn.functional.gelu(
                    (1.0 + 0.1 * torch.tanh(gamma)) * encoded + beta
                )
                correction = self.maximum_logit_correction * torch.tanh(
                    self.output(calibrated).squeeze(-1)
                )
                logits = (
                    torch.log(gaussian_weight.clamp_min(1e-12))
                    + token_gate.clamp(0.0, 1.0) * correction
                )
                return logits.masked_fill(~validity, -torch.inf)

            def forward(
                self,
                depths: Any,
                local_features: Any,
                token_latent: Any,
                token_gate: Any,
                gaussian_weight: Any,
                validity: Any,
            ) -> tuple[Any, Any]:
                logits = self.contribution_logits(
                    local_features,
                    token_latent,
                    token_gate,
                    gaussian_weight,
                    validity,
                )
                supported = validity.any(dim=-1)
                safe_logits = torch.where(supported[..., None], logits, 0.0)
                weights = torch.softmax(safe_logits, dim=-1)
                weights = torch.where(validity, weights, 0.0)
                denominator = weights.sum(dim=-1).clamp_min(1e-12)
                weights = weights / denominator[..., None]
                fused = torch.sum(weights * depths, dim=-1)
                fused = torch.where(
                    supported, fused, torch.full_like(fused, float("nan"))
                )
                return fused, weights

        return _Fusion()


def fusion_parameter_count(fusion: Any) -> int:
    return int(sum(parameter.numel() for parameter in fusion.parameters()))


def save_fusion(
    path: Path, fusion: Any, *, metadata: dict[str, Any]
) -> tuple[Path, Path]:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        name: value.detach().cpu() for name, value in fusion.state_dict().items()
    }
    save_file(tensors, str(path))
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path, metadata_path


def load_fusion(path: Path, *, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    from safetensors.torch import load_file

    tensors = load_file(str(path), device="cpu")
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fusion = LatentConditionedDepthFusion.create(
        maximum_logit_correction=metadata["maximum_logit_correction"],
        local_mean=tensors["local_mean"],
        local_std=tensors["local_std"],
        latent_mean=tensors["latent_mean"],
        latent_std=tensors["latent_std"],
    )
    fusion.load_state_dict(tensors, strict=True)
    return fusion.to(device).eval(), metadata


@dataclass(frozen=True, slots=True)
class RayFusionBundle:
    """Padded variable-cardinality contributions for selected ERP rays."""

    flat_indices: np.ndarray
    depths: np.ndarray
    local_features: np.ndarray
    token_latents: np.ndarray
    token_gate_strength: np.ndarray
    gaussian_weights: np.ndarray
    validity: np.ndarray
    source_indices: np.ndarray


def _sample_map(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    import cv2

    source = np.asarray(values, dtype=np.float32)
    sample_x = np.asarray(x, dtype=np.float32).reshape(-1)
    sample_y = np.asarray(y, dtype=np.float32).reshape(-1)
    # OpenCV requires both destination dimensions to fit signed int16 even
    # though remap coordinates are float32.  Preserve identical sampling while
    # splitting large same-face ERP batches below that implementation limit.
    parts = []
    for start in range(0, sample_x.size, 30_000):
        stop = min(sample_x.size, start + 30_000)
        sampled = cv2.remap(
            source,
            sample_x[start:stop].reshape(-1, 1),
            sample_y[start:stop].reshape(-1, 1),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        parts.append(sampled.reshape((stop - start, *values.shape[2:])))
    return np.concatenate(parts, axis=0)


def build_aligned_ray_bundle(
    plan: Any,
    tangent_depths: np.ndarray,
    token_gates: np.ndarray,
    token_latents: np.ndarray,
    flat_indices: np.ndarray,
    *,
    maximum_contributions: int = 6,
    tangent_log_gradients: np.ndarray | None = None,
) -> RayFusionBundle:
    """Sample depth, gate, and latent from the exact same panorama rays."""

    from panorai.geometry import (
        GnomonicSpec,
        erp_pixels_to_rays,
        rays_to_gnomonic_pixels,
    )

    depths = np.asarray(tangent_depths, dtype=np.float32)
    gates = np.asarray(token_gates, dtype=np.float32)
    latents = np.asarray(token_latents, dtype=np.float32)
    gradients = (
        np.asarray(tangent_log_gradients, dtype=np.float32)
        if tangent_log_gradients is not None
        else None
    )
    indices = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
    view_count, full_height, full_width = depths.shape
    feature_height, feature_width = gates.shape[1:]
    if depths.shape[0] != len(plan.centers_lat_lon_deg):
        raise ValueError("tangent depth count disagrees with the tangent plan")
    if gates.shape[0] != view_count:
        raise ValueError("token gate count disagrees with tangent depths")
    if latents.shape != (view_count, feature_height, feature_width, TOKEN_LATENT_DIM):
        raise ValueError("token latent shape disagrees with token gates")
    if (full_height, full_width) != tuple(plan.view_shape_hw):
        raise ValueError("tangent depth shape disagrees with the tangent plan")
    if gradients is not None and gradients.shape != depths.shape:
        raise ValueError("tangent log-gradient shape disagrees with tangent depths")
    if maximum_contributions < 1:
        raise ValueError("maximum_contributions must be positive")
    pixel_count = plan.erp_shape_hw[0] * plan.erp_shape_hw[1]
    if bool(((indices < 0) | (indices >= pixel_count)).any()):
        raise ValueError("flat ERP indices are out of bounds")

    pixels = np.stack(
        (indices % plan.erp_shape_hw[1], indices // plan.erp_shape_hw[1]), axis=-1
    ).astype(np.float64)
    rays = erp_pixels_to_rays(pixels, plan.erp_shape_hw)
    sample_count = indices.size
    projected_x = np.empty((view_count, sample_count), dtype=np.float32)
    projected_y = np.empty_like(projected_x)
    geometric_validity = np.zeros((view_count, sample_count), dtype=bool)
    gaussian = np.zeros((view_count, sample_count), dtype=np.float32)
    for view_index, (latitude, longitude) in enumerate(plan.centers_lat_lon_deg):
        spec = GnomonicSpec(
            center_lat_deg=latitude,
            center_lon_deg=longitude,
            hfov_deg=plan.hfov_deg,
            vfov_deg=plan.vfov_deg,
            output_shape_hw=plan.view_shape_hw,
        )
        projected = rays_to_gnomonic_pixels(rays, spec)
        x = np.asarray(projected.pixels_xy[:, 0], dtype=np.float32)
        y = np.asarray(projected.pixels_xy[:, 1], dtype=np.float32)
        valid = np.asarray(projected.valid, dtype=bool)
        projected_x[view_index] = x
        projected_y[view_index] = y
        geometric_validity[view_index] = valid
        plane_x = (2.0 * (x.astype(np.float64) + 0.5) / full_width - 1.0) * np.tan(
            np.deg2rad(plan.hfov_deg) / 2.0
        )
        plane_y = (2.0 * (y.astype(np.float64) + 0.5) / full_height - 1.0) * np.tan(
            np.deg2rad(plan.vfov_deg) / 2.0
        )
        center_score = 1.0 / np.sqrt(1.0 + plane_x * plane_x + plane_y * plane_y)
        gaussian[view_index] = np.where(
            valid, np.exp(6.0 * (center_score - 1.0)), 0.0
        ).astype(np.float32)

    contribution_count = min(maximum_contributions, view_count)
    ray_weights = gaussian.T
    selected = np.argpartition(-ray_weights, kth=contribution_count - 1, axis=1)[
        :, :contribution_count
    ]
    selected_weights = np.take_along_axis(ray_weights, selected, axis=1)
    order = np.argsort(-selected_weights, axis=1)
    selected = np.take_along_axis(selected, order, axis=1).astype(np.int16)
    selected_weights = np.take_along_axis(selected_weights, order, axis=1)
    valid = selected_weights > 0.0

    output_depths = np.zeros((sample_count, contribution_count), dtype=np.float32)
    output_gates = np.zeros_like(output_depths)
    output_latents = np.zeros(
        (sample_count, contribution_count, TOKEN_LATENT_DIM), dtype=np.float32
    )
    output_gradients = np.zeros_like(output_depths)
    output_x = np.zeros_like(output_depths)
    output_y = np.zeros_like(output_depths)
    output_incidence = np.zeros_like(output_depths)
    for view_index in range(view_count):
        rows, slots = np.nonzero((selected == view_index) & valid)
        if rows.size == 0:
            continue
        x = projected_x[view_index, rows]
        y = projected_y[view_index, rows]
        sampled_depth = _sample_map(depths[view_index], x, y)
        feature_x = (x + 0.5) * feature_width / full_width - 0.5
        feature_y = (y + 0.5) * feature_height / full_height - 0.5
        sampled_gate = _sample_map(gates[view_index], feature_x, feature_y)
        sampled_latent = _sample_map(latents[view_index], feature_x, feature_y)
        if gradients is None:
            log_depth = np.log(np.maximum(depths[view_index], 1e-6))
            grad_y, grad_x = np.gradient(log_depth)
            gradient = np.hypot(grad_x, grad_y).astype(np.float32)
        else:
            gradient = gradients[view_index]
        sampled_gradient = _sample_map(gradient, x, y)
        output_depths[rows, slots] = sampled_depth
        output_gates[rows, slots] = sampled_gate
        output_latents[rows, slots] = sampled_latent
        output_gradients[rows, slots] = sampled_gradient
        normalized_x = 2.0 * (x + 0.5) / full_width - 1.0
        normalized_y = 2.0 * (y + 0.5) / full_height - 1.0
        output_x[rows, slots] = normalized_x
        output_y[rows, slots] = normalized_y
        output_incidence[rows, slots] = normalized_x**2 + normalized_y**2
    valid &= np.isfinite(output_depths) & (output_depths > 0.0)
    log_values = np.where(valid, np.log(np.maximum(output_depths, 1e-6)), np.nan)
    supported = valid.any(axis=1)
    median = np.zeros(sample_count, dtype=np.float32)
    dispersion = np.zeros(sample_count, dtype=np.float32)
    median[supported] = np.nanmedian(log_values[supported], axis=1)
    dispersion[supported] = np.nanstd(log_values[supported], axis=1)
    disagreement = log_values - median[:, None]
    local_features = np.stack(
        (
            np.nan_to_num(disagreement),
            np.nan_to_num(np.abs(disagreement)),
            output_gradients,
            output_x,
            output_y,
            output_incidence,
            np.broadcast_to(dispersion[:, None], output_depths.shape),
            output_gates / 0.1,
        ),
        axis=-1,
    ).astype(np.float32)
    selected_weights = np.where(valid, selected_weights, 0.0).astype(np.float32)
    gate_strength = np.clip(np.abs(output_gates) / 0.1, 0.0, 1.0).astype(np.float32)
    return RayFusionBundle(
        flat_indices=indices,
        depths=output_depths,
        local_features=local_features,
        token_latents=output_latents,
        token_gate_strength=gate_strength,
        gaussian_weights=selected_weights,
        validity=valid,
        source_indices=selected,
    )


def infer_gated_learned_fusion_erp(
    model: Any,
    rgb: np.ndarray,
    source_support: np.ndarray,
    plan: Any,
    *,
    scratch_dir: Path,
    gate_checkpoint: Path,
    fusion_checkpoint: Path,
    device: str = "cpu",
    maximum_sources_per_target: int = 6,
    position_transport: bool = True,
    row_chunk: int = 32,
    fusion_batch_size: int = 8192,
    keep_feature_stores: bool = False,
    reuse_gated_stores: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run gated DA3 and replace Gaussian output fusion with learned convex fusion."""

    import torch

    from benchmarks.spherical_monocular_depth.da3_spherical_attention import (
        infer_sparse_attention_erp,
    )

    if row_chunk < 1 or fusion_batch_size < 1:
        raise ValueError("row_chunk and fusion_batch_size must be positive")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    gaussian_path = scratch_dir / "gated-gaussian-radial-m.npy"
    gaussian_validity_path = scratch_dir / "gated-gaussian-validity.npy"
    if reuse_gated_stores:
        required = {
            "tangent_depth_path": scratch_dir / "da3-tangent-depths-f32.npy",
            "gate_map_path": scratch_dir / "da3-token-gates-f32.dat",
            "latent_map_path": scratch_dir / "da3-token-latents-f32.dat",
        }
        for path in (*required.values(), gaussian_path, gaussian_validity_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        gaussian = np.load(gaussian_path, mmap_mode="r")
        gaussian_validity = np.load(gaussian_validity_path, mmap_mode="r")
        gated_report = {
            "stores": {name: str(path) for name, path in required.items()},
            "learned_parameters_added": 257,
            "resumed_after_cv2_sampling_batch_fix": True,
            "gate_checkpoint": str(gate_checkpoint),
        }
    else:
        gaussian, gaussian_validity, gated_report = infer_sparse_attention_erp(
            model,
            rgb,
            source_support,
            plan,
            scratch_dir=scratch_dir,
            device=device,
            alpha=1.0,
            maximum_sources_per_target=maximum_sources_per_target,
            position_transport=position_transport,
            keep_feature_stores=True,
            gate_checkpoint=gate_checkpoint,
        )
        np.save(gaussian_path, gaussian)
        np.save(gaussian_validity_path, gaussian_validity)
    stores = gated_report["stores"]
    tangent_depths = np.load(stores["tangent_depth_path"], mmap_mode="r")
    feature_shape = (plan.view_shape_hw[0] // 14, plan.view_shape_hw[1] // 14)
    token_gates = np.memmap(
        stores["gate_map_path"],
        dtype="float32",
        mode="r",
        shape=(len(plan.centers_lat_lon_deg), *feature_shape),
    )
    token_latents = np.memmap(
        stores["latent_map_path"],
        dtype="float32",
        mode="r",
        shape=(len(plan.centers_lat_lon_deg), *feature_shape, TOKEN_LATENT_DIM),
    )
    fusion, fusion_metadata = load_fusion(fusion_checkpoint, device=device)
    tangent_log_gradients = np.empty_like(tangent_depths, dtype=np.float32)
    for view_index in range(tangent_depths.shape[0]):
        log_depth = np.log(np.maximum(tangent_depths[view_index], 1e-6))
        grad_y, grad_x = np.gradient(log_depth)
        tangent_log_gradients[view_index] = np.hypot(grad_x, grad_y)
    height, width = plan.erp_shape_hw
    learned = np.full((height, width), np.nan, dtype=np.float32)
    learned_validity = np.zeros((height, width), dtype=bool)
    contribution_histogram = {
        str(value): 0 for value in range(1, maximum_sources_per_target + 1)
    }
    with torch.inference_mode():
        for start_row in range(0, height, row_chunk):
            stop_row = min(height, start_row + row_chunk)
            indices = np.arange(start_row * width, stop_row * width, dtype=np.int64)
            bundle = build_aligned_ray_bundle(
                plan,
                tangent_depths,
                token_gates,
                token_latents,
                indices,
                maximum_contributions=maximum_sources_per_target,
                tangent_log_gradients=tangent_log_gradients,
            )
            values = np.full(indices.size, np.nan, dtype=np.float32)
            valid = bundle.validity.any(axis=1) & source_support[
                start_row:stop_row
            ].reshape(-1)
            valid_indices = np.flatnonzero(valid)
            for start in range(0, valid_indices.size, fusion_batch_size):
                selected = valid_indices[start : start + fusion_batch_size]
                fused, _ = fusion(
                    torch.from_numpy(bundle.depths[selected]).to(device),
                    torch.from_numpy(bundle.local_features[selected]).to(device),
                    torch.from_numpy(bundle.token_latents[selected]).to(device),
                    torch.from_numpy(bundle.token_gate_strength[selected]).to(device),
                    torch.from_numpy(bundle.gaussian_weights[selected]).to(device),
                    torch.from_numpy(bundle.validity[selected]).to(device),
                )
                values[selected] = fused.detach().float().cpu().numpy()
            learned[start_row:stop_row] = values.reshape(stop_row - start_row, width)
            learned_validity[start_row:stop_row] = valid.reshape(
                stop_row - start_row, width
            )
            counts = bundle.validity[valid].sum(axis=1)
            for value in range(1, maximum_sources_per_target + 1):
                contribution_histogram[str(value)] += int((counts == value).sum())
    report = {
        "interface": INTERFACE,
        "gated_inference": gated_report,
        "gate_checkpoint": str(gate_checkpoint),
        "fusion_checkpoint": str(fusion_checkpoint),
        "fusion_metadata": fusion_metadata,
        "maximum_sources_per_target": maximum_sources_per_target,
        "variable_cardinality": True,
        "permutation_equivariant": True,
        "same_ray_alignment": "depth, signed gate, and 8D latent share exact gnomonic back-projection coordinates",
        "contribution_histogram": contribution_histogram,
        "control_gated_gaussian_path": str(gaussian_path),
        "control_gated_gaussian_validity_path": str(gaussian_validity_path),
        "valid_pixels": int(learned_validity.sum()),
        "coverage_fraction_of_source_support": float(
            learned_validity.sum() / max(1, source_support.sum())
        ),
        "learned_parameters_added": fusion_parameter_count(fusion)
        + gated_report["learned_parameters_added"],
        "dpt_spherical": False,
        "limitation": "DPT remains independent per tangent face; learned fusion occurs only after all decoders",
    }
    del tangent_depths, token_gates, token_latents, tangent_log_gradients
    if not keep_feature_stores:
        for name, value in list(stores.items()):
            if isinstance(value, str) and Path(value).is_file():
                Path(value).unlink()
                stores[name] = None
    return learned, learned_validity, report


__all__ = [
    "INTERFACE",
    "LOCAL_INPUT_NAMES",
    "TOKEN_LATENT_DIM",
    "LatentConditionedDepthFusion",
    "RayFusionBundle",
    "build_aligned_ray_bundle",
    "fusion_parameter_count",
    "infer_gated_learned_fusion_erp",
    "load_fusion",
    "save_fusion",
]
