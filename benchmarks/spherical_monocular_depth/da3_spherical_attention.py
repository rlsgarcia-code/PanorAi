"""Sparse panorama-ray attention inside the frozen DA3 ViT.

The official DA3 Metric Large backbone uses independent 2D absolute-position
ViTs for monocular inputs.  This diagnostic preserves every learned tensor and
the original local attention.  At transformer block index 12, queries from a
target patch additionally attend to patches in overlapping views that observe
the same panorama-frame ray.  The correction subtracts the self-only value,
so a single-view neighborhood and ``alpha=0`` are exact neutral operations.

The official DPT remains unchanged and planar per tangent view.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from benchmarks.spherical_monocular_depth.da3_sphere import (
    FeatureStoreSpec,
    _PrecomputedDepthAdapter,
    _open_store,
    decode_shared_feature_store,
    make_spherical_overlap_plan,
)
from benchmarks.spherical_monocular_depth.vit_tangent import (
    NativeTangentPlan,
    _FixedTangentSampler,
)


INTERFACE = "panorai-da3-sparse-spherical-attention/v1-experimental"
ATTENTION_BLOCK_INDEX = 12
EXPECTED_OUTPUT_LAYERS = (4, 11, 17, 23)


@dataclass(frozen=True, slots=True)
class TokenStateStoreSpec:
    view_count: int
    token_count: int
    channels: int
    dtype: str = "float32"

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.view_count, self.token_count, self.channels


def _open_state_store(path: Path, spec: TokenStateStoreSpec, mode: str) -> np.memmap:
    return np.memmap(path, dtype=spec.dtype, mode=mode, shape=spec.shape)


def _rgb_tensor(image: np.ndarray, *, device: str) -> Any:
    import torch

    values = np.asarray(image)
    tensor = torch.from_numpy(
        np.ascontiguousarray(values.astype(np.float32).transpose(2, 0, 1))
    )
    tensor = tensor / 255.0
    mean = tensor.new_tensor((0.485, 0.456, 0.406))[:, None, None]
    std = tensor.new_tensor((0.229, 0.224, 0.225))[:, None, None]
    return ((tensor - mean) / std)[None, None].to(device)


def _validate_backbone(model: Any) -> Any:
    backbone = model.backbone
    vit = backbone.pretrained
    if tuple(backbone.out_layers) != EXPECTED_OUTPUT_LAYERS:
        raise ValueError(
            f"expected DA3 output layers {EXPECTED_OUTPUT_LAYERS}, "
            f"got {tuple(backbone.out_layers)}"
        )
    if vit.alt_start != -1 or vit.rope_start != -1 or vit.rope is not None:
        raise ValueError("this adapter expects the monocular absolute-position DA3 ViT")
    if vit.cat_token:
        raise ValueError("this adapter expects cat_token=False")
    if len(vit.blocks) <= ATTENTION_BLOCK_INDEX:
        raise ValueError("DA3 backbone is too shallow for the attention adapter")
    return vit


def extract_prefix_states(
    model: Any,
    views: Sequence[Any],
    state_path: Path,
    feature_path: Path | None,
    *,
    device: str,
) -> tuple[TokenStateStoreSpec, FeatureStoreSpec, dict[str, Any]]:
    """Run blocks 0..11 and persist states plus the first two DPT levels."""

    import torch

    if not views:
        raise ValueError("views cannot be empty")
    vit = _validate_backbone(model)
    height, width = tuple(views[0].image.shape[:2])
    if height % vit.patch_size or width % vit.patch_size:
        raise ValueError("DA3 view dimensions must be divisible by its patch size")
    feature_shape = (height // vit.patch_size, width // vit.patch_size)
    patch_count = feature_shape[0] * feature_shape[1]
    state_spec = TokenStateStoreSpec(
        view_count=len(views),
        token_count=patch_count + vit.patch_start_idx,
        channels=vit.embed_dim,
    )
    feature_spec = FeatureStoreSpec(
        view_count=len(views),
        level_count=len(EXPECTED_OUTPUT_LAYERS),
        feature_shape_hw=feature_shape,
        channels=vit.embed_dim,
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    states = _open_state_store(state_path, state_spec, "w+")
    features = (
        _open_store(feature_path, feature_spec, "w+")
        if feature_path is not None
        else None
    )
    level_by_block = {
        block_index: level for level, block_index in enumerate(EXPECTED_OUTPUT_LAYERS)
    }
    start = perf_counter()
    for view_index, face in enumerate(views):
        tensor = _rgb_tensor(face.image, device=device)
        with torch.inference_mode():
            state = vit.prepare_tokens_with_masks(tensor)
            for block_index in range(ATTENTION_BLOCK_INDEX):
                state = vit.process_attention(
                    state, vit.blocks[block_index], "local", pos=None
                )
                if features is not None and block_index in level_by_block:
                    level = level_by_block[block_index]
                    normalized = vit.norm(state)[0, 0, vit.patch_start_idx :]
                    features[view_index, level] = (
                        normalized.detach()
                        .float()
                        .cpu()
                        .numpy()
                        .reshape(*feature_shape, vit.embed_dim)
                    )
        states[view_index] = state[0, 0].detach().float().cpu().numpy()
        states.flush()
        if features is not None:
            features.flush()
    return (
        state_spec,
        feature_spec,
        {
            "seconds": perf_counter() - start,
            "state_path": str(state_path),
            "state_shape": list(state_spec.shape),
            "state_bytes": state_path.stat().st_size,
            "feature_path": str(feature_path) if feature_path is not None else None,
            "feature_shape": list(feature_spec.shape),
            "feature_bytes": feature_path.stat().st_size
            if feature_path is not None
            else 0,
            "last_completed_block_index": ATTENTION_BLOCK_INDEX - 1,
        },
    )


def sparse_cross_view_attention_correction(
    block: Any,
    target_normalized_patches: Any,
    sampled_normalized_patches: Any,
    geometric_weights: Any,
) -> Any:
    """Return a frozen-QKV correction relative to self-only attention.

    Args:
        block: the official DA3 transformer block providing Q/K/V and proj.
        target_normalized_patches: ``[P, C]`` target tokens after ``norm1``.
        sampled_normalized_patches: ``[K, P, C]`` ray-aligned source tokens.
        geometric_weights: non-negative ``[K, P]`` spherical support prior.
    """

    correction, _ = sparse_cross_view_attention_components(
        block,
        target_normalized_patches,
        sampled_normalized_patches,
        geometric_weights,
    )
    return correction


def sparse_cross_view_attention_components(
    block: Any,
    target_normalized_patches: Any,
    sampled_normalized_patches: Any,
    geometric_weights: Any,
) -> tuple[Any, Any]:
    """Return the projected correction and source probabilities per head."""

    import torch

    if target_normalized_patches.ndim != 2:
        raise ValueError("target_normalized_patches must have shape [P, C]")
    if sampled_normalized_patches.ndim != 3:
        raise ValueError("sampled_normalized_patches must have shape [K, P, C]")
    if sampled_normalized_patches.shape[1:] != target_normalized_patches.shape:
        raise ValueError("target and sampled token shapes disagree")
    if geometric_weights.shape != sampled_normalized_patches.shape[:2]:
        raise ValueError("geometric_weights must have shape [K, P]")
    if not bool(torch.isfinite(geometric_weights).all()):
        raise ValueError("geometric_weights must be finite")
    if bool((geometric_weights < 0).any()):
        raise ValueError("geometric_weights must be non-negative")
    if bool((geometric_weights.sum(dim=0) <= 0).any()):
        raise ValueError("every target patch needs positive spherical support")

    attention = block.attn
    patch_count, channels = target_normalized_patches.shape
    source_count = sampled_normalized_patches.shape[0]
    head_count = attention.num_heads
    head_dim = channels // head_count
    target_qkv = attention.qkv(target_normalized_patches).reshape(
        patch_count, 3, head_count, head_dim
    )
    source_qkv = attention.qkv(sampled_normalized_patches).reshape(
        source_count, patch_count, 3, head_count, head_dim
    )
    query = attention.q_norm(target_qkv[:, 0])
    self_value = target_qkv[:, 2]
    key = attention.k_norm(source_qkv[:, :, 1])
    value = source_qkv[:, :, 2]
    logits = torch.einsum("phd,kphd->kph", query, key) * attention.scale
    log_prior = torch.log(geometric_weights.clamp_min(1e-12))[:, :, None]
    probability = torch.softmax(logits + log_prior, dim=0)
    mixed_value = torch.sum(probability[..., None] * value, dim=0)
    delta = (mixed_value - self_value).reshape(patch_count, channels)
    return attention.proj_drop(attention.proj(delta)), probability


def compute_local_attention_state_store(
    model: Any,
    state_path: Path,
    base_path: Path,
    state_spec: TokenStateStoreSpec,
    *,
    device: str,
) -> dict[str, Any]:
    """Persist the frozen block-12 local-attention state before its MLP."""

    import torch

    vit = _validate_backbone(model)
    block = vit.blocks[ATTENTION_BLOCK_INDEX]
    states = _open_state_store(state_path, state_spec, "r")
    base = _open_state_store(base_path, state_spec, "w+")
    start = perf_counter()
    with torch.inference_mode():
        for view_index in range(state_spec.view_count):
            target = torch.from_numpy(
                np.array(states[view_index], dtype=np.float32, copy=True)
            ).to(device)[None]
            local = block.attn(block.norm1(target))
            value = target + block.drop_path1(block.ls1(local))
            base[view_index] = value[0].detach().float().cpu().numpy()
            base.flush()
    return {
        "seconds": perf_counter() - start,
        "path": str(base_path),
        "shape": list(state_spec.shape),
        "bytes": base_path.stat().st_size,
        "last_operation": "frozen block-12 local attention before MLP",
    }


def complete_attention_feature_store(
    model: Any,
    state_path: Path,
    feature_path: Path,
    state_spec: TokenStateStoreSpec,
    feature_spec: FeatureStoreSpec,
    overlap_plan: Any,
    *,
    device: str,
    alpha: float,
    position_transport: bool = True,
    gate: Any | None = None,
    base_state_path: Path | None = None,
    gate_map_path: Path | None = None,
    latent_map_path: Path | None = None,
) -> dict[str, Any]:
    """Apply sparse attention at block 12 and run the frozen suffix."""

    import torch
    from torch.nn import functional as functional

    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    vit = _validate_backbone(model)
    states = _open_state_store(state_path, state_spec, "r")
    if gate is not None and base_state_path is None:
        raise ValueError("base_state_path is required when a gate is provided")
    if gate is not None and (gate_map_path is None or latent_map_path is None):
        raise ValueError("gate and latent map paths are required with a gate")
    base_states = (
        _open_state_store(base_state_path, state_spec, "r")
        if base_state_path is not None
        else None
    )
    features = _open_store(feature_path, feature_spec, "r+")
    height, width = feature_spec.feature_shape_hw
    patch_start = vit.patch_start_idx
    block = vit.blocks[ATTENTION_BLOCK_INDEX]
    level_by_block = {
        block_index: level for level, block_index in enumerate(EXPECTED_OUTPUT_LAYERS)
    }
    start = perf_counter()
    total_sources = 0
    gate_values_store = None
    latent_store = None
    gate_sum = 0.0
    gate_abs_sum = 0.0
    gate_min = float("inf")
    gate_max = float("-inf")
    gate_count = 0
    if gate is not None:
        gate_values_store = np.memmap(
            gate_map_path,
            dtype="float32",
            mode="w+",
            shape=(state_spec.view_count, height, width),
        )
        latent_store = np.memmap(
            latent_map_path,
            dtype="float32",
            mode="w+",
            shape=(state_spec.view_count, height, width, 8),
        )
    with torch.inference_mode():
        position = vit.interpolate_pos_encoding(
            torch.empty(
                (1, state_spec.token_count, state_spec.channels),
                dtype=torch.float32,
                device=device,
            ),
            feature_spec.feature_shape_hw[0] * vit.patch_size,
            feature_spec.feature_shape_hw[1] * vit.patch_size,
        )[0, patch_start:]
        position_map = position.reshape(height, width, state_spec.channels)
        position_chw = position_map.permute(2, 0, 1)[None]
        for target_index, contributions in enumerate(overlap_plan.targets):
            target = torch.from_numpy(
                np.array(states[target_index], dtype=np.float32, copy=True)
            ).to(device)[None]
            normalized = block.norm1(target)
            local = block.attn(normalized)
            state = target + block.drop_path1(block.ls1(local))
            if alpha > 0.0:
                source_indices = [item.source_index for item in contributions]
                source_patches = np.asarray(
                    states[source_indices, patch_start:], dtype=np.float32
                ).reshape(len(source_indices), height, width, state_spec.channels)
                source_patches = torch.from_numpy(
                    np.ascontiguousarray(source_patches.transpose(0, 3, 1, 2))
                ).to(device)
                grids = torch.from_numpy(
                    np.stack([item.grid_xy for item in contributions], axis=0)
                ).to(device)
                sampled = functional.grid_sample(
                    source_patches,
                    grids,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                ).permute(0, 2, 3, 1)
                if position_transport:
                    sampled_position = functional.grid_sample(
                        position_chw.expand(len(source_indices), -1, -1, -1),
                        grids,
                        mode="bilinear",
                        padding_mode="border",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)
                    sampled = sampled - sampled_position + position_map[None]
                sampled = block.norm1(sampled).reshape(
                    len(source_indices), height * width, state_spec.channels
                )
                weights = torch.from_numpy(
                    np.stack([item.weight for item in contributions], axis=0)
                ).to(device)
                if gate is None:
                    correction = sparse_cross_view_attention_correction(
                        block,
                        normalized[0, patch_start:],
                        sampled,
                        weights.reshape(len(source_indices), height * width),
                    )
                    full_correction = torch.zeros_like(state)
                    full_correction[0, patch_start:] = correction
                    state = state + alpha * block.drop_path1(block.ls1(full_correction))
                else:
                    from benchmarks.spherical_monocular_depth.da3_spherical_gate import (
                        gate_features_and_objective,
                    )

                    if base_states is None:
                        raise RuntimeError("gated attention lost its base state store")
                    source_base = np.asarray(
                        base_states[source_indices, patch_start:], dtype=np.float32
                    ).reshape(len(source_indices), height, width, state_spec.channels)
                    source_base = torch.from_numpy(
                        np.ascontiguousarray(source_base.transpose(0, 3, 1, 2))
                    ).to(device)
                    sampled_base = functional.grid_sample(
                        source_base,
                        grids,
                        mode="bilinear",
                        padding_mode="border",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)
                    if position_transport:
                        sampled_base = (
                            sampled_base - sampled_position + position_map[None]
                        )
                    correction, probabilities = sparse_cross_view_attention_components(
                        block,
                        normalized[0, patch_start:],
                        sampled,
                        weights.reshape(len(source_indices), height * width),
                    )
                    scaled_correction = block.ls1(correction)
                    target_base = torch.from_numpy(
                        np.array(
                            base_states[target_index, patch_start:],
                            dtype=np.float32,
                            copy=True,
                        )
                    ).to(device)
                    self_offset = source_indices.index(target_index)
                    gate_features, _ = gate_features_and_objective(
                        target_base,
                        sampled_base,
                        scaled_correction,
                        probabilities,
                        weights,
                        grids,
                        self_source_offset=self_offset,
                        gaussian_exponent=overlap_plan.gaussian_exponent,
                        identity_regularization=1.0,
                    )
                    token_gate, token_latent = gate.forward_with_latent(gate_features)
                    state[0, patch_start:] = (
                        state[0, patch_start:]
                        + alpha * token_gate[:, None] * scaled_correction
                    )
                    gate_map = token_gate.reshape(height, width)
                    latent_map = token_latent.reshape(height, width, 8)
                    gate_values_store[target_index] = (
                        gate_map.detach().float().cpu().numpy()
                    )
                    latent_store[target_index] = (
                        latent_map.detach().float().cpu().numpy()
                    )
                    gate_values_store.flush()
                    latent_store.flush()
                    gate_sum += float(gate_map.sum().item())
                    gate_abs_sum += float(gate_map.abs().sum().item())
                    gate_min = min(gate_min, float(gate_map.min().item()))
                    gate_max = max(gate_max, float(gate_map.max().item()))
                    gate_count += gate_map.numel()
            state = state + block.drop_path2(block.ls2(block.mlp(block.norm2(state))))
            for block_index in range(ATTENTION_BLOCK_INDEX + 1, len(vit.blocks)):
                state = vit.blocks[block_index](state, pos=None)
                if block_index in level_by_block:
                    level = level_by_block[block_index]
                    output = vit.norm(state)[0, patch_start:]
                    features[target_index, level] = (
                        output.detach()
                        .float()
                        .cpu()
                        .numpy()
                        .reshape(height, width, state_spec.channels)
                    )
            features.flush()
            total_sources += len(contributions)
    return {
        "seconds": perf_counter() - start,
        "alpha": float(alpha),
        "attention_block_index": ATTENTION_BLOCK_INDEX,
        "attention_block_ordinal": ATTENTION_BLOCK_INDEX + 1,
        "total_target_source_contributions": total_sources,
        "learned_parameters_added": int(
            sum(parameter.numel() for parameter in gate.parameters())
        )
        if gate is not None
        else 0,
        "qkv_source": "frozen official attention weights from block index 12 (13th block)",
        "self_only_neutral": True,
        "absolute_position_transport": bool(position_transport),
        "gated": gate is not None,
        "gate_map_path": str(gate_map_path) if gate_map_path is not None else None,
        "latent_map_path": str(latent_map_path)
        if latent_map_path is not None
        else None,
        "gate_statistics": (
            {
                "mean": gate_sum / gate_count,
                "mean_absolute": gate_abs_sum / gate_count,
                "minimum": gate_min,
                "maximum": gate_max,
                "count": gate_count,
            }
            if gate_count
            else None
        ),
    }


def infer_sparse_attention_erp(
    model: Any,
    rgb: np.ndarray,
    source_support: np.ndarray,
    plan: NativeTangentPlan,
    *,
    scratch_dir: Path,
    device: str = "cpu",
    alpha: float = 1.0,
    maximum_sources_per_target: int = 6,
    position_transport: bool = True,
    keep_feature_stores: bool = False,
    gate_checkpoint: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Infer DA3 with one sparse cross-view attention block."""

    from panorai.data import EquirectangularImage

    if rgb.shape != (*plan.erp_shape_hw, 3):
        raise ValueError("ERP RGB shape disagrees with the frozen plan")
    if source_support.shape != plan.erp_shape_hw or source_support.dtype != np.bool_:
        raise ValueError("source_support must be a boolean native-ERP mask")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    source = EquirectangularImage(rgb)
    views = source.views(
        _FixedTangentSampler(plan.centers_lat_lon_deg),
        size=plan.view_shape_hw,
        fov=(plan.hfov_deg, plan.vfov_deg),
        depth_policy="propagate",
    )
    state_path = scratch_dir / "da3-block11-states-f32.dat"
    feature_path = scratch_dir / "da3-attention-features-f32.dat"
    base_state_path = scratch_dir / "da3-block12-local-states-f32.dat"
    gate_map_path = scratch_dir / "da3-token-gates-f32.dat"
    latent_map_path = scratch_dir / "da3-token-latents-f32.dat"
    state_spec, feature_spec, prefix_report = extract_prefix_states(
        model, list(views), state_path, feature_path, device=device
    )
    overlap_plan = make_spherical_overlap_plan(
        plan,
        feature_shape_hw=feature_spec.feature_shape_hw,
        maximum_sources_per_target=maximum_sources_per_target,
    )
    gate = None
    gate_metadata = None
    base_report = None
    if gate_checkpoint is not None:
        from benchmarks.spherical_monocular_depth.da3_spherical_gate import load_gate

        gate, gate_metadata = load_gate(gate_checkpoint, device=device)
        base_report = compute_local_attention_state_store(
            model, state_path, base_state_path, state_spec, device=device
        )
    attention_report = complete_attention_feature_store(
        model,
        state_path,
        feature_path,
        state_spec,
        feature_spec,
        overlap_plan,
        device=device,
        alpha=alpha,
        position_transport=position_transport,
        gate=gate,
        base_state_path=base_state_path if gate is not None else None,
        gate_map_path=gate_map_path if gate is not None else None,
        latent_map_path=latent_map_path if gate is not None else None,
    )
    depths, decoder_report = decode_shared_feature_store(
        model, feature_path, feature_spec, plan, device=device
    )
    tangent_depth_path = scratch_dir / "da3-tangent-depths-f32.npy"
    if keep_feature_stores:
        np.save(tangent_depth_path, np.stack(depths, axis=0))
    adapter = _PrecomputedDepthAdapter(depths, plan.view_shape_hw)
    predicted = views.map(adapter, input="image", output="depth", units="m")
    reconstructed = predicted.reconstruct(
        blend={"depth": "gaussian"}, modalities=("depth",)
    )
    payload = reconstructed._workflow_data()["depth"]
    metadata = reconstructed._workflow_metadata["depth"]
    validity = np.asarray(metadata["validity"], dtype=bool) & source_support
    radial = np.asarray(payload, dtype=np.float32)
    radial[~validity] = np.nan
    stores = {
        "kept": bool(keep_feature_stores),
        "state_path": str(state_path),
        "feature_path": str(feature_path),
        "base_state_path": str(base_state_path) if gate is not None else None,
        "gate_map_path": str(gate_map_path) if gate is not None else None,
        "latent_map_path": str(latent_map_path) if gate is not None else None,
        "tangent_depth_path": str(tangent_depth_path) if keep_feature_stores else None,
    }
    if not keep_feature_stores:
        state_path.unlink()
        feature_path.unlink()
        if gate is not None:
            base_state_path.unlink()
            gate_map_path.unlink()
            latent_map_path.unlink()
        stores["state_path"] = None
        stores["feature_path"] = None
        stores["base_state_path"] = None
        stores["gate_map_path"] = None
        stores["latent_map_path"] = None
        stores["tangent_depth_path"] = None
        prefix_report["state_path"] = None
        prefix_report["feature_path"] = None
    return (
        radial,
        validity,
        {
            "interface": INTERFACE,
            "prefix": prefix_report,
            "attention": attention_report,
            "local_attention_base": base_report,
            "gate_checkpoint": str(gate_checkpoint)
            if gate_checkpoint is not None
            else None,
            "gate_metadata": gate_metadata,
            "decoder": decoder_report,
            "state_store_spec": asdict(state_spec),
            "feature_store_spec": asdict(feature_spec),
            "overlap_plan": overlap_plan.to_dict(),
            "stores": stores,
            "view_count": len(depths),
            "valid_pixels": int(validity.sum()),
            "coverage_fraction_of_source_support": float(
                validity.sum() / max(1, source_support.sum())
            ),
            "learned_parameters_added": attention_report["learned_parameters_added"],
            "dpt_spherical": False,
            "limitation": (
                "one frozen ViT block receives sparse panorama-ray attention; "
                "the official DPT remains planar per tangent view"
            ),
        },
    )


__all__ = [
    "ATTENTION_BLOCK_INDEX",
    "EXPECTED_OUTPUT_LAYERS",
    "INTERFACE",
    "TokenStateStoreSpec",
    "complete_attention_feature_store",
    "compute_local_attention_state_store",
    "extract_prefix_states",
    "infer_sparse_attention_erp",
    "sparse_cross_view_attention_correction",
    "sparse_cross_view_attention_components",
]
