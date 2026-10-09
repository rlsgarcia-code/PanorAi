"""Canonical spherical feature atlas for frozen DA3 tangent decoders.

This diagnostic is deliberately stronger than target-dependent overlap
consensus.  Each exported ViT level is scattered once into a single ERP
feature field and then gathered back into every tangent lattice.  Therefore a
panorama ray has one canonical feature value before the unchanged per-view DPT
decoders.  The operation adds no learned parameters and does not alter image
or prediction resolution.

The atlas contains already-computed DA3 features.  Consequently it does not
erase the learned planar positional history inside those features; this module
tests canonical same-ray content, not a fully spherical ViT or DPT.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from benchmarks.spherical_monocular_depth.da3_sphere import (
    FeatureStoreSpec,
    _PrecomputedDepthAdapter,
    _open_store,
    decode_shared_feature_store,
    make_spherical_overlap_plan,
)
from benchmarks.spherical_monocular_depth.da3_spherical_attention import (
    EXPECTED_OUTPUT_LAYERS,
    complete_attention_feature_store,
    compute_local_attention_state_store,
    extract_prefix_states,
)
from benchmarks.spherical_monocular_depth.vit_tangent import (
    NativeTangentPlan,
    _FixedTangentSampler,
)


INTERFACE = "panorai-da3-canonical-spherical-feature-atlas/v1-experimental"
DEFAULT_ATLAS_SHAPE_HW = (295, 590)


def _validate_atlas_shape(shape_hw: tuple[int, int]) -> tuple[int, int]:
    height, width = tuple(int(value) for value in shape_hw)
    if height < 2 or width != 2 * height:
        raise ValueError("canonical atlas must have a positive 2:1 ERP shape")
    return height, width


def _feature_specs(plan: NativeTangentPlan, shape_hw: tuple[int, int]) -> list[Any]:
    from panorai.geometry import GnomonicSpec

    return [
        GnomonicSpec(
            center_lat_deg=latitude,
            center_lon_deg=longitude,
            hfov_deg=plan.hfov_deg,
            vfov_deg=plan.vfov_deg,
            output_shape_hw=shape_hw,
        )
        for latitude, longitude in plan.centers_lat_lon_deg
    ]


def _remap_channels(
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    border_mode: int,
    batch_size: int = 30_000,
    channel_batch_size: int = 128,
) -> np.ndarray:
    """Bilinearly sample an HxW or HxWxC array without OpenCV size overflow."""

    import cv2

    source = np.asarray(values, dtype=np.float32)
    sample_x = np.asarray(x, dtype=np.float32).reshape(-1)
    sample_y = np.asarray(y, dtype=np.float32).reshape(-1)
    if sample_x.shape != sample_y.shape:
        raise ValueError("x and y sampling coordinates must have equal shape")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if channel_batch_size < 1:
        raise ValueError("channel_batch_size must be positive")
    tail = source.shape[2:]
    output = np.empty((sample_x.size, *tail), dtype=np.float32)
    for start in range(0, sample_x.size, batch_size):
        stop = min(sample_x.size, start + batch_size)
        map_x = sample_x[start:stop].reshape(-1, 1)
        map_y = sample_y[start:stop].reshape(-1, 1)
        if source.ndim == 2:
            sampled = cv2.remap(
                source,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=border_mode,
            )
            output[start:stop] = sampled.reshape(stop - start)
            continue
        if source.ndim != 3:
            raise ValueError("sampled values must use HW or HWC layout")
        for channel_start in range(0, source.shape[2], channel_batch_size):
            channel_stop = min(
                source.shape[2], channel_start + channel_batch_size
            )
            sampled = cv2.remap(
                source[:, :, channel_start:channel_stop],
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=border_mode,
            )
            output[
                start:stop, channel_start:channel_stop
            ] = sampled.reshape(stop - start, channel_stop - channel_start)
    return output


def _gaussian_face_weight(
    x: np.ndarray,
    y: np.ndarray,
    shape_hw: tuple[int, int],
    hfov_deg: float,
    vfov_deg: float,
) -> np.ndarray:
    """PanorAi tangent reconstruction prior evaluated at fractional pixels."""

    height, width = shape_hw
    plane_x = (2.0 * (np.asarray(x) + 0.5) / width - 1.0) * np.tan(
        np.deg2rad(hfov_deg) / 2.0
    )
    plane_y = (2.0 * (np.asarray(y) + 0.5) / height - 1.0) * np.tan(
        np.deg2rad(vfov_deg) / 2.0
    )
    incidence = 1.0 / np.sqrt(1.0 + plane_x * plane_x + plane_y * plane_y)
    return np.exp(6.0 * (incidence - 1.0)).astype(np.float32)


def canonicalize_feature_store(
    source_path: Path,
    target_path: Path,
    atlas_path: Path,
    support_path: Path,
    store_spec: FeatureStoreSpec,
    plan: NativeTangentPlan,
    *,
    atlas_shape_hw: tuple[int, int] = DEFAULT_ATLAS_SHAPE_HW,
) -> dict[str, Any]:
    """Scatter four per-view feature levels to one ERP field and broadcast it.

    Unsupported samples at the extreme boundary fall back to the original
    per-face value.  Supported samples always originate from the unique atlas.
    """

    import cv2
    from panorai.geometry import (
        erp_pixels_to_rays,
        gnomonic_pixels_to_rays,
        rays_to_erp_pixels,
        rays_to_gnomonic_pixels,
    )

    atlas_shape = _validate_atlas_shape(atlas_shape_hw)
    if store_spec.view_count != len(plan.centers_lat_lon_deg):
        raise ValueError("feature store and tangent plan view counts disagree")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    source = _open_store(source_path, store_spec, "r")
    target = _open_store(target_path, store_spec, "w+")
    feature_shape = store_spec.feature_shape_hw
    specs = _feature_specs(plan, feature_shape)

    atlas_height, atlas_width = atlas_shape
    atlas_spec = FeatureStoreSpec(
        view_count=1,
        level_count=store_spec.level_count,
        feature_shape_hw=atlas_shape,
        channels=store_spec.channels,
    )
    atlas = _open_store(atlas_path, atlas_spec, "w+")
    support = np.memmap(
        support_path,
        dtype="float32",
        mode="w+",
        shape=(store_spec.level_count, atlas_height, atlas_width),
    )
    atlas_y, atlas_x = np.indices(atlas_shape, dtype=np.float64)
    atlas_pixels = np.stack((atlas_x, atlas_y), axis=-1)
    atlas_rays = erp_pixels_to_rays(atlas_pixels, atlas_shape)

    face_y, face_x = np.indices(feature_shape, dtype=np.float64)
    face_pixels = np.stack((face_x, face_y), axis=-1)
    face_to_atlas: list[tuple[np.ndarray, np.ndarray]] = []
    for spec in specs:
        rays = gnomonic_pixels_to_rays(face_pixels, spec)
        projected = rays_to_erp_pixels(rays.rays_xyz, atlas_shape)
        face_to_atlas.append(
            (
                np.asarray(projected.pixels_xy[..., 0], dtype=np.float32),
                np.asarray(projected.pixels_xy[..., 1], dtype=np.float32),
            )
        )

    start = perf_counter()
    level_reports = []
    total_fallback = 0
    total_target = store_spec.view_count * np.prod(feature_shape) * store_spec.level_count
    for level_index in range(store_spec.level_count):
        level_atlas = atlas[0, level_index]
        level_atlas[...] = 0.0
        denominator = support[level_index]
        denominator[...] = 0.0
        contribution_count = 0
        for view_index, spec in enumerate(specs):
            projected = rays_to_gnomonic_pixels(atlas_rays, spec)
            valid = np.asarray(projected.valid, dtype=bool)
            if not valid.any():
                continue
            x = np.asarray(projected.pixels_xy[..., 0], dtype=np.float32)[valid]
            y = np.asarray(projected.pixels_xy[..., 1], dtype=np.float32)[valid]
            weight = _gaussian_face_weight(
                x, y, feature_shape, plan.hfov_deg, plan.vfov_deg
            )
            sampled = _remap_channels(
                source[view_index, level_index],
                x,
                y,
                border_mode=cv2.BORDER_REPLICATE,
            )
            level_atlas[valid] += sampled * weight[:, None]
            denominator[valid] += weight
            contribution_count += int(valid.sum())
        supported = denominator > 0.0
        if not supported.any():
            raise RuntimeError("canonical atlas has no spherical support")
        level_atlas[supported] /= denominator[supported, None]
        level_atlas[~supported] = 0.0
        atlas.flush()
        support.flush()

        level_fallback = 0
        for view_index, (sample_x, sample_y) in enumerate(face_to_atlas):
            values = _remap_channels(
                level_atlas,
                sample_x,
                sample_y,
                border_mode=cv2.BORDER_WRAP,
            ).reshape(*feature_shape, store_spec.channels)
            sampled_support = _remap_channels(
                supported.astype(np.float32),
                sample_x,
                sample_y,
                border_mode=cv2.BORDER_WRAP,
            ).reshape(feature_shape)
            valid = sampled_support >= 1.0 - 1e-5
            level_fallback += int((~valid).sum())
            values[~valid] = source[view_index, level_index][~valid]
            target[view_index, level_index] = values
        target.flush()
        total_fallback += level_fallback
        level_reports.append(
            {
                "level_index": level_index,
                "atlas_supported_cells": int(supported.sum()),
                "atlas_support_fraction": float(supported.mean()),
                "face_contributions": contribution_count,
                "broadcast_fallback_tokens": level_fallback,
            }
        )
    return {
        "source_path": str(source_path),
        "target_path": str(target_path),
        "atlas_path": str(atlas_path),
        "support_path": str(support_path),
        "atlas_shape_hw": list(atlas_shape),
        "atlas_angular_cell_deg": [180.0 / atlas_height, 360.0 / atlas_width],
        "feature_shape_hw": list(feature_shape),
        "level_reports": level_reports,
        "broadcast_fallback_tokens": total_fallback,
        "broadcast_fallback_fraction": float(total_fallback / max(1, total_target)),
        "same_ray_contract": (
            "all supported face tokens are bilinear samples of one canonical ERP "
            "feature field at their panorama-frame ray"
        ),
        "learned_parameters_added": 0,
        "seconds": perf_counter() - start,
    }


def infer_canonical_atlas_erp(
    model: Any,
    rgb: np.ndarray,
    source_support: np.ndarray,
    plan: NativeTangentPlan,
    *,
    scratch_dir: Path,
    gate_checkpoint: Path,
    device: str = "cpu",
    alpha: float = 1.0,
    maximum_sources_per_target: int = 6,
    position_transport: bool = True,
    atlas_shape_hw: tuple[int, int] = DEFAULT_ATLAS_SHAPE_HW,
    keep_feature_stores: bool = False,
    reuse_gated_stores: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run frozen gated DA3, canonicalize all ViT levels, and decode per face."""

    from panorai.data import EquirectangularImage

    from benchmarks.spherical_monocular_depth.da3_spherical_gate import load_gate

    if rgb.shape != (*plan.erp_shape_hw, 3):
        raise ValueError("ERP RGB shape disagrees with the frozen plan")
    if source_support.shape != plan.erp_shape_hw or source_support.dtype != np.bool_:
        raise ValueError("source_support must be a boolean native-ERP mask")
    _validate_atlas_shape(atlas_shape_hw)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    source = EquirectangularImage(rgb)
    views = source.views(
        _FixedTangentSampler(plan.centers_lat_lon_deg),
        size=plan.view_shape_hw,
        fov=(plan.hfov_deg, plan.vfov_deg),
        depth_policy="propagate",
    )
    state_path = scratch_dir / "da3-block11-states-f32.dat"
    feature_path = scratch_dir / "da3-gated-features-f32.dat"
    base_state_path = scratch_dir / "da3-block12-local-states-f32.dat"
    gate_map_path = scratch_dir / "da3-token-gates-f32.dat"
    latent_map_path = scratch_dir / "da3-token-latents-f32.dat"
    canonical_path = scratch_dir / "da3-canonicalized-features-f32.dat"
    atlas_path = scratch_dir / "da3-canonical-atlas-f32.dat"
    atlas_support_path = scratch_dir / "da3-canonical-atlas-support-f32.dat"

    gate, gate_metadata = load_gate(gate_checkpoint, device=device)
    if reuse_gated_stores:
        patch_size = int(model.backbone.pretrained.patch_size)
        feature_shape = (
            plan.view_shape_hw[0] // patch_size,
            plan.view_shape_hw[1] // patch_size,
        )
        feature_spec = FeatureStoreSpec(
            view_count=len(plan.centers_lat_lon_deg),
            level_count=len(EXPECTED_OUTPUT_LAYERS),
            feature_shape_hw=feature_shape,
            channels=int(model.backbone.pretrained.embed_dim),
        )
        required = (
            feature_path,
            gate_map_path,
            latent_map_path,
            scratch_dir / "gated-gaussian-radial-m.npy",
            scratch_dir / "gated-gaussian-validity.npy",
        )
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
        expected_bytes = int(np.prod(feature_spec.shape) * np.dtype("float32").itemsize)
        if feature_path.stat().st_size != expected_bytes:
            raise ValueError("existing gated feature store has an unexpected size")
        prefix_report = {
            "resumed": True,
            "feature_path": str(feature_path),
            "feature_shape": list(feature_spec.shape),
            "feature_bytes": expected_bytes,
        }
        base_report = {"resumed": True, "path": str(base_state_path)}
        attention_report = {
            "resumed": True,
            "gated": True,
            "alpha": float(alpha),
            "absolute_position_transport": bool(position_transport),
            "learned_parameters_added": int(
                sum(parameter.numel() for parameter in gate.parameters())
            ),
        }
    else:
        state_spec, feature_spec, prefix_report = extract_prefix_states(
            model, list(views), state_path, feature_path, device=device
        )
        base_report = compute_local_attention_state_store(
            model, state_path, base_state_path, state_spec, device=device
        )
    overlap_plan = make_spherical_overlap_plan(
        plan,
        feature_shape_hw=feature_spec.feature_shape_hw,
        maximum_sources_per_target=maximum_sources_per_target,
    )
    if not reuse_gated_stores:
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
            base_state_path=base_state_path,
            gate_map_path=gate_map_path,
            latent_map_path=latent_map_path,
        )

    control_path = scratch_dir / "gated-gaussian-radial-m.npy"
    control_validity_path = scratch_dir / "gated-gaussian-validity.npy"
    if reuse_gated_stores:
        control = np.load(control_path, mmap_mode="r")
        control_validity = np.load(control_validity_path, mmap_mode="r")
        if control.shape != plan.erp_shape_hw or control_validity.shape != plan.erp_shape_hw:
            raise ValueError("existing gated Gaussian control has an unexpected shape")
        control_decoder_report = {"resumed": True, "view_count": feature_spec.view_count}
    else:
        control_depths, control_decoder_report = decode_shared_feature_store(
            model, feature_path, feature_spec, plan, device=device
        )
        control_adapter = _PrecomputedDepthAdapter(control_depths, plan.view_shape_hw)
        control_prediction = views.map(
            control_adapter, input="image", output="depth", units="m"
        ).reconstruct(blend={"depth": "gaussian"}, modalities=("depth",))
        control_payload = control_prediction._workflow_data()["depth"]
        control_metadata = control_prediction._workflow_metadata["depth"]
        control_validity = (
            np.asarray(control_metadata["validity"], dtype=bool) & source_support
        )
        control = np.asarray(control_payload, dtype=np.float32)
        control[~control_validity] = np.nan
        np.save(control_path, control)
        np.save(control_validity_path, control_validity)

    atlas_report = canonicalize_feature_store(
        feature_path,
        canonical_path,
        atlas_path,
        atlas_support_path,
        feature_spec,
        plan,
        atlas_shape_hw=atlas_shape_hw,
    )
    depths, decoder_report = decode_shared_feature_store(
        model, canonical_path, feature_spec, plan, device=device
    )
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
        "base_state_path": str(base_state_path),
        "gate_map_path": str(gate_map_path),
        "latent_map_path": str(latent_map_path),
        "canonical_path": str(canonical_path),
        "atlas_path": str(atlas_path),
        "atlas_support_path": str(atlas_support_path),
    }
    if not keep_feature_stores:
        for path in (
            state_path,
            feature_path,
            base_state_path,
            gate_map_path,
            latent_map_path,
            canonical_path,
            atlas_path,
            atlas_support_path,
        ):
            if path.exists():
                path.unlink()
        for name in tuple(stores):
            if name.endswith("_path"):
                stores[name] = None
        prefix_report["state_path"] = None
        prefix_report["feature_path"] = None
        atlas_report["source_path"] = None
        atlas_report["target_path"] = None
        atlas_report["atlas_path"] = None
        atlas_report["support_path"] = None
    return radial, validity, {
        "interface": INTERFACE,
        "prefix": prefix_report,
        "local_attention_base": base_report,
        "attention": attention_report,
        "gate_checkpoint": str(gate_checkpoint),
        "gate_metadata": gate_metadata,
        "canonical_atlas": atlas_report,
        "control_decoder": control_decoder_report,
        "decoder": decoder_report,
        "feature_store_spec": asdict(feature_spec),
        "overlap_plan": overlap_plan.to_dict(),
        "stores": stores,
        "control_gated_gaussian_path": str(control_path),
        "control_gated_gaussian_validity_path": str(control_validity_path),
        "view_count": len(depths),
        "valid_pixels": int(validity.sum()),
        "coverage_fraction_of_source_support": float(
            validity.sum() / max(1, source_support.sum())
        ),
        "learned_parameters_added": attention_report["learned_parameters_added"],
        "dpt_spherical": False,
        "limitation": (
            "the canonical atlas carries view-conditioned planar ViT features; "
            "the unchanged DPT still executes independently in each tangent frame"
        ),
    }
