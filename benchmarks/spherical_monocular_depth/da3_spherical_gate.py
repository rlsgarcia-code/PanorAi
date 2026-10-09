"""Self-supervised scalar gate for sparse spherical DA3 attention."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from benchmarks.spherical_monocular_depth.da3_sphere import FeatureStoreSpec
from benchmarks.spherical_monocular_depth.da3_spherical_attention import (
    ATTENTION_BLOCK_INDEX,
    TokenStateStoreSpec,
    _open_state_store,
    _validate_backbone,
    sparse_cross_view_attention_components,
)


INTERFACE = "panorai-da3-self-supervised-spherical-gate/v1-experimental"
GATE_INPUT_NAMES = (
    "target_consensus_cosine",
    "neighbor_support_fraction",
    "attention_entropy",
    "relative_correction_rms",
    "normalized_position_displacement",
    "incidence_spread",
)
MAXIMUM_GATE_MAGNITUDE = 0.1


class SphericalAttentionGate:
    """Factory namespace for the fixed 6-16-8-1 gate architecture."""

    @staticmethod
    def create(input_mean: Any | None = None, input_std: Any | None = None) -> Any:
        import torch
        from torch import nn

        class _Gate(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                mean = torch.zeros(len(GATE_INPUT_NAMES), dtype=torch.float32)
                std = torch.ones(len(GATE_INPUT_NAMES), dtype=torch.float32)
                if input_mean is not None:
                    mean.copy_(torch.as_tensor(input_mean, dtype=torch.float32))
                if input_std is not None:
                    std.copy_(torch.as_tensor(input_std, dtype=torch.float32))
                self.register_buffer("input_mean", mean)
                self.register_buffer("input_std", std.clamp_min(1e-6))
                self.encoder = nn.Sequential(
                    nn.Linear(6, 16),
                    nn.GELU(),
                    nn.Linear(16, 8),
                    nn.GELU(),
                )
                self.output = nn.Linear(8, 1)

            def encode(self, values: Any) -> Any:
                normalized = (values - self.input_mean) / self.input_std
                return self.encoder(normalized)

            def forward_with_latent(self, values: Any) -> tuple[Any, Any]:
                latent = self.encode(values)
                gate = MAXIMUM_GATE_MAGNITUDE * torch.tanh(self.output(latent)).squeeze(
                    -1
                )
                return gate, latent

            def forward(self, values: Any) -> Any:
                gate, _ = self.forward_with_latent(values)
                return gate

        return _Gate()


def gate_parameter_count(gate: Any) -> int:
    return int(sum(parameter.numel() for parameter in gate.parameters()))


def _sample_chw(values: np.ndarray, grids: Any, *, device: str) -> Any:
    import torch
    from torch.nn import functional

    tensor = torch.from_numpy(np.ascontiguousarray(values.transpose(0, 3, 1, 2))).to(
        device
    )
    return functional.grid_sample(
        tensor,
        grids,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).permute(0, 2, 3, 1)


def gate_features_and_objective(
    target_base: Any,
    sampled_base: Any,
    correction: Any,
    probabilities: Any,
    geometric_weights: Any,
    grids: Any,
    *,
    self_source_offset: int,
    gaussian_exponent: float,
    identity_regularization: float,
) -> tuple[Any, Any]:
    """Build six gate inputs and quadratic RGB-only consistency coefficients."""

    import torch
    from torch.nn import functional

    source_count, height, width = geometric_weights.shape
    patch_count = height * width
    weights = geometric_weights.reshape(source_count, patch_count)
    denominator = weights.sum(dim=0).clamp_min(1e-12)
    normalized_weights = weights / denominator
    sampled_flat = sampled_base.reshape(source_count, patch_count, -1)
    consensus = torch.sum(normalized_weights[..., None] * sampled_flat, dim=0)
    cosine = functional.cosine_similarity(target_base, consensus, dim=-1)

    neighbor_weights = weights.clone()
    neighbor_weights[self_source_offset] = 0.0
    neighbor_support = neighbor_weights.sum(dim=0) / denominator

    if source_count > 1:
        entropy = -torch.sum(
            probabilities.clamp_min(1e-12) * torch.log(probabilities.clamp_min(1e-12)),
            dim=0,
        ).mean(dim=-1) / np.log(source_count)
    else:
        entropy = torch.zeros(patch_count, device=target_base.device)

    correction_rms = torch.sqrt(torch.mean(correction.square(), dim=-1) + 1e-12)
    base_rms = torch.sqrt(torch.mean(target_base.square(), dim=-1) + 1e-12)
    correction_ratio = correction_rms / base_rms.clamp_min(1e-6)

    yy, xx = torch.meshgrid(
        torch.arange(height, device=grids.device, dtype=grids.dtype),
        torch.arange(width, device=grids.device, dtype=grids.dtype),
        indexing="ij",
    )
    identity = torch.stack(
        (2.0 * (xx + 0.5) / width - 1.0, 2.0 * (yy + 0.5) / height - 1.0),
        dim=-1,
    )
    displacement = torch.linalg.vector_norm(grids - identity[None], dim=-1).reshape(
        source_count, patch_count
    )
    position_displacement = torch.sum(
        normalized_weights * displacement, dim=0
    ) / np.sqrt(8.0)

    incidence = (
        -torch.log(
            geometric_weights.reshape(source_count, patch_count).clamp_min(1e-12)
        )
        / gaussian_exponent
    )
    incidence_mean = torch.sum(normalized_weights * incidence, dim=0)
    incidence_spread = torch.sqrt(
        torch.sum(
            normalized_weights * (incidence - incidence_mean[None]).square(), dim=0
        ).clamp_min(0.0)
    )

    features = torch.stack(
        (
            cosine,
            neighbor_support,
            entropy,
            correction_ratio,
            position_displacement,
            incidence_spread,
        ),
        dim=-1,
    )
    difference = target_base - consensus
    coefficient_a = torch.mean(correction.square(), dim=-1)
    coefficient_b = torch.mean(difference * correction, dim=-1)
    coefficient_c = torch.mean(difference.square(), dim=-1)
    sample_weight = neighbor_support
    objective = torch.stack(
        (
            coefficient_a,
            coefficient_b,
            coefficient_c,
            sample_weight,
            torch.full_like(coefficient_a, float(identity_regularization)),
        ),
        dim=-1,
    )
    return features, objective


def build_gate_samples(
    model: Any,
    state_path: Path,
    base_path: Path,
    state_spec: TokenStateStoreSpec,
    feature_spec: FeatureStoreSpec,
    overlap_plan: Any,
    *,
    device: str,
    identity_regularization: float = 1.0,
    position_transport: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Create depth-free gate samples from one panorama's frozen states."""

    import torch

    if identity_regularization < 0.0 or not np.isfinite(identity_regularization):
        raise ValueError("identity_regularization must be finite and non-negative")
    vit = _validate_backbone(model)
    block = vit.blocks[ATTENTION_BLOCK_INDEX]
    raw = _open_state_store(state_path, state_spec, "r")
    base = _open_state_store(base_path, state_spec, "r")
    height, width = feature_spec.feature_shape_hw
    patch_start = vit.patch_start_idx
    position = vit.interpolate_pos_encoding(
        torch.empty(
            (1, state_spec.token_count, state_spec.channels),
            dtype=torch.float32,
            device=device,
        ),
        height * vit.patch_size,
        width * vit.patch_size,
    )[0, patch_start:]
    position_map = position.reshape(height, width, state_spec.channels)
    position_values = position_map.detach().cpu().numpy()[None]
    feature_parts: list[np.ndarray] = []
    objective_parts: list[np.ndarray] = []
    start = perf_counter()
    with torch.inference_mode():
        for target_index, contributions in enumerate(overlap_plan.targets):
            source_indices = [item.source_index for item in contributions]
            self_offset = source_indices.index(target_index)
            grids = torch.from_numpy(
                np.stack([item.grid_xy for item in contributions], axis=0)
            ).to(device)
            weights = torch.from_numpy(
                np.stack([item.weight for item in contributions], axis=0)
            ).to(device)
            source_raw = np.asarray(
                raw[source_indices, patch_start:], dtype=np.float32
            ).reshape(len(source_indices), height, width, state_spec.channels)
            sampled_raw = _sample_chw(source_raw, grids, device=device)
            source_base = np.asarray(
                base[source_indices, patch_start:], dtype=np.float32
            ).reshape(len(source_indices), height, width, state_spec.channels)
            sampled_base = _sample_chw(source_base, grids, device=device)
            if position_transport:
                repeated_position = np.repeat(
                    position_values, len(source_indices), axis=0
                )
                sampled_position = _sample_chw(repeated_position, grids, device=device)
                transport = position_map[None] - sampled_position
                sampled_raw = sampled_raw + transport
                sampled_base = sampled_base + transport
            target_raw = torch.from_numpy(
                np.array(raw[target_index, patch_start:], dtype=np.float32, copy=True)
            ).to(device)
            target_base = torch.from_numpy(
                np.array(base[target_index, patch_start:], dtype=np.float32, copy=True)
            ).to(device)
            sampled_normalized = block.norm1(sampled_raw).reshape(
                len(source_indices), height * width, state_spec.channels
            )
            correction, probabilities = sparse_cross_view_attention_components(
                block,
                block.norm1(target_raw),
                sampled_normalized,
                weights.reshape(len(source_indices), height * width),
            )
            correction = block.ls1(correction)
            features, objective = gate_features_and_objective(
                target_base,
                sampled_base,
                correction,
                probabilities,
                weights,
                grids,
                self_source_offset=self_offset,
                gaussian_exponent=overlap_plan.gaussian_exponent,
                identity_regularization=identity_regularization,
            )
            valid = (
                torch.isfinite(features).all(dim=-1)
                & torch.isfinite(objective).all(dim=-1)
                & (objective[:, 3] > 1e-4)
                & (objective[:, 0] > 1e-12)
            )
            feature_parts.append(features[valid].float().cpu().numpy())
            objective_parts.append(objective[valid].float().cpu().numpy())
    feature_array = np.concatenate(feature_parts, axis=0)
    objective_array = np.concatenate(objective_parts, axis=0)
    oracle = np.clip(
        -objective_array[:, 1]
        / ((1.0 + objective_array[:, 4]) * objective_array[:, 0] + 1e-12),
        -MAXIMUM_GATE_MAGNITUDE,
        MAXIMUM_GATE_MAGNITUDE,
    )
    return (
        feature_array,
        objective_array,
        {
            "interface": INTERFACE,
            "samples": int(feature_array.shape[0]),
            "input_names": list(GATE_INPUT_NAMES),
            "feature_mean": feature_array.mean(axis=0).tolist(),
            "feature_std": feature_array.std(axis=0).tolist(),
            "oracle_gate_mean": float(oracle.mean()),
            "oracle_gate_median": float(np.median(oracle)),
            "oracle_gate_zero_fraction": float((oracle <= 1e-6).mean()),
            "oracle_gate_one_fraction": float((oracle >= 1.0 - 1e-6).mean()),
            "seconds": perf_counter() - start,
            "state_spec": asdict(state_spec),
            "feature_spec": asdict(feature_spec),
            "position_transport": bool(position_transport),
            "depth_used": False,
        },
    )


def normalized_gate_objective(
    gate_values: Any, objective: Any, *, reduction: str = "mean"
) -> Any:
    """Evaluate the regularized consistency quadratic; gate zero has value one."""

    coefficient_a, coefficient_b, coefficient_c, weight, regularization = (
        objective.unbind(dim=-1)
    )
    value = (
        coefficient_c
        + 2.0 * gate_values * coefficient_b
        + (1.0 + regularization) * gate_values.square() * coefficient_a
    ) / coefficient_c.clamp_min(1e-8)
    weighted = value * weight
    if reduction == "none":
        return weighted
    if reduction != "mean":
        raise ValueError("reduction must be 'none' or 'mean'")
    return weighted.sum() / weight.sum().clamp_min(1e-8)


def save_gate(path: Path, gate: Any, *, metadata: dict[str, Any]) -> tuple[Path, Path]:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {name: value.detach().cpu() for name, value in gate.state_dict().items()}
    save_file(tensors, str(path))
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path, metadata_path


def load_gate(path: Path, *, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    from safetensors.torch import load_file

    tensors = load_file(str(path), device="cpu")
    gate = SphericalAttentionGate.create(tensors["input_mean"], tensors["input_std"])
    missing, unexpected = gate.load_state_dict(tensors, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"gate mismatch: missing={missing}, unexpected={unexpected}")
    gate = gate.to(device).eval()
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return gate, metadata


__all__ = [
    "GATE_INPUT_NAMES",
    "INTERFACE",
    "MAXIMUM_GATE_MAGNITUDE",
    "SphericalAttentionGate",
    "build_gate_samples",
    "gate_features_and_objective",
    "gate_parameter_count",
    "load_gate",
    "normalized_gate_objective",
    "save_gate",
]
