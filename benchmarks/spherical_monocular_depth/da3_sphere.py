"""Spherical overlap sharing for native-density DA3 tangent features.

This is the first bounded DA3-Sphere experiment.  The official backbone and
DPT weights are unchanged.  Four ViT feature levels are extracted per tangent
view, transported through panorama-frame rays, combined with overlapping
views, and gathered back before the DPT.  It tests feature communication in a
controlled way; the DPT itself is still planar inside each view.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from benchmarks.spherical_monocular_depth.da3_tangent import (
    CANONICAL_FOCAL_PX,
    MODEL_DEPTH_RANGE_M,
)
from benchmarks.spherical_monocular_depth.vit_tangent import (
    NativeTangentPlan,
    _FixedTangentSampler,
    axial_to_radial_tangent,
)


INTERFACE = "panorai-da3-spherical-overlap-features/v1-experimental"


@dataclass(frozen=True, slots=True)
class OverlapContribution:
    source_index: int
    grid_xy: np.ndarray
    weight: np.ndarray
    valid_fraction: float


@dataclass(frozen=True, slots=True)
class SphericalOverlapPlan:
    feature_shape_hw: tuple[int, int]
    view_count: int
    maximum_sources_per_target: int
    gaussian_exponent: float
    targets: tuple[tuple[OverlapContribution, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["targets"] = [
            [
                {
                    "source_index": contribution.source_index,
                    "valid_fraction": contribution.valid_fraction,
                }
                for contribution in contributions
            ]
            for contributions in self.targets
        ]
        return result


@dataclass(frozen=True, slots=True)
class FeatureStoreSpec:
    view_count: int
    level_count: int
    feature_shape_hw: tuple[int, int]
    channels: int
    dtype: str = "float32"

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        height, width = self.feature_shape_hw
        return self.view_count, self.level_count, height, width, self.channels


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


def make_spherical_overlap_plan(
    plan: NativeTangentPlan,
    *,
    feature_shape_hw: tuple[int, int],
    maximum_sources_per_target: int = 6,
    minimum_overlap_fraction: float = 0.01,
    gaussian_exponent: float = 2.0,
) -> SphericalOverlapPlan:
    """Build target-to-source feature sampling maps in panorama coordinates."""

    from panorai.geometry import gnomonic_pixels_to_rays, rays_to_gnomonic_pixels

    if maximum_sources_per_target < 1:
        raise ValueError("maximum_sources_per_target must be positive")
    if not 0.0 <= minimum_overlap_fraction <= 1.0:
        raise ValueError("minimum_overlap_fraction must be in [0, 1]")
    if not np.isfinite(gaussian_exponent) or gaussian_exponent <= 0.0:
        raise ValueError("gaussian_exponent must be finite and positive")
    height, width = feature_shape_hw
    if height < 1 or width < 1:
        raise ValueError("feature_shape_hw must be positive")
    specs = _feature_specs(plan, feature_shape_hw)
    yy, xx = np.indices(feature_shape_hw, dtype=np.float64)
    target_pixels = np.stack((xx, yy), axis=-1)
    target_rays = [
        gnomonic_pixels_to_rays(target_pixels, spec).rays_xyz for spec in specs
    ]
    identity_x = (2.0 * (xx + 0.5) / width - 1.0).astype(np.float32)
    identity_y = (2.0 * (yy + 0.5) / height - 1.0).astype(np.float32)
    identity_grid = np.stack((identity_x, identity_y), axis=-1)
    targets: list[tuple[OverlapContribution, ...]] = []
    for target_index, rays in enumerate(target_rays):
        candidates: list[tuple[float, OverlapContribution]] = []
        for source_index, source_spec in enumerate(specs):
            if source_index == target_index:
                grid = identity_grid.copy()
                valid = np.ones(feature_shape_hw, dtype=bool)
                px = xx
                py = yy
            else:
                projected = rays_to_gnomonic_pixels(rays, source_spec)
                valid = np.asarray(projected.valid, dtype=bool)
                valid_fraction = float(valid.mean())
                if valid_fraction < minimum_overlap_fraction:
                    continue
                px = np.asarray(projected.pixels_xy[..., 0], dtype=np.float64)
                py = np.asarray(projected.pixels_xy[..., 1], dtype=np.float64)
                grid = np.stack(
                    (
                        2.0 * (px + 0.5) / width - 1.0,
                        2.0 * (py + 0.5) / height - 1.0,
                    ),
                    axis=-1,
                ).astype(np.float32)
                grid[~valid] = 0.0
            nx = 2.0 * (px + 0.5) / width - 1.0
            ny = 2.0 * (py + 0.5) / height - 1.0
            weight = np.exp(-gaussian_exponent * (nx * nx + ny * ny))
            weight = np.where(valid, weight, 0.0).astype(np.float32)
            contribution = OverlapContribution(
                source_index=source_index,
                grid_xy=grid,
                weight=weight,
                valid_fraction=float(valid.mean()),
            )
            # Rank by total useful central support, while always retaining self.
            rank = float(weight.mean()) + (1e6 if source_index == target_index else 0.0)
            candidates.append((rank, contribution))
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = [item[1] for item in candidates[:maximum_sources_per_target]]
        if not any(item.source_index == target_index for item in selected):
            raise RuntimeError("overlap plan dropped its target view")
        selected.sort(key=lambda item: item.source_index)
        targets.append(tuple(selected))
    return SphericalOverlapPlan(
        feature_shape_hw=feature_shape_hw,
        view_count=len(specs),
        maximum_sources_per_target=maximum_sources_per_target,
        gaussian_exponent=float(gaussian_exponent),
        targets=tuple(targets),
    )


def _open_store(path: Path, spec: FeatureStoreSpec, mode: str) -> np.memmap:
    return np.memmap(path, dtype=spec.dtype, mode=mode, shape=spec.shape)


def extract_da3_feature_store(
    model: Any,
    views: Sequence[Any],
    path: Path,
    *,
    device: str,
) -> tuple[FeatureStoreSpec, dict[str, Any]]:
    """Extract the four official DINO feature levels into a disk-backed store."""

    import torch

    if not views:
        raise ValueError("views cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    first_shape = tuple(views[0].image.shape[:2])
    if first_shape[0] % 14 or first_shape[1] % 14:
        raise ValueError("DA3 view dimensions must be divisible by 14")
    feature_shape = (first_shape[0] // 14, first_shape[1] // 14)
    store_spec: FeatureStoreSpec | None = None
    store = None
    start = perf_counter()
    for view_index, face in enumerate(views):
        values = np.asarray(face.image)
        if values.shape != (*first_shape, 3):
            raise ValueError("all tangent views must have the same RGB shape")
        tensor = torch.from_numpy(
            np.ascontiguousarray(values.astype(np.float32).transpose(2, 0, 1))
        )
        tensor = tensor / 255.0
        mean = tensor.new_tensor((0.485, 0.456, 0.406))[:, None, None]
        std = tensor.new_tensor((0.229, 0.224, 0.225))[:, None, None]
        tensor = ((tensor - mean) / std)[None, None].to(device)
        with torch.inference_mode():
            features, _ = model.backbone(tensor)
        level_arrays = [
            feature[0][0, 0].detach().float().cpu().numpy() for feature in features
        ]
        if store_spec is None:
            channels = int(level_arrays[0].shape[-1])
            store_spec = FeatureStoreSpec(
                view_count=len(views),
                level_count=len(level_arrays),
                feature_shape_hw=feature_shape,
                channels=channels,
            )
            store = _open_store(path, store_spec, "w+")
        assert store_spec is not None and store is not None
        expected = (feature_shape[0] * feature_shape[1], store_spec.channels)
        if len(level_arrays) != store_spec.level_count or any(
            level.shape != expected for level in level_arrays
        ):
            raise RuntimeError("DA3 backbone feature shape changed between views")
        for level_index, level in enumerate(level_arrays):
            store[view_index, level_index] = level.reshape(
                *feature_shape, store_spec.channels
            )
        store.flush()
    assert store_spec is not None
    return store_spec, {
        "path": str(path),
        "shape": list(store_spec.shape),
        "dtype": store_spec.dtype,
        "bytes": path.stat().st_size,
        "seconds": perf_counter() - start,
    }


def share_feature_store(
    source_path: Path,
    target_path: Path,
    store_spec: FeatureStoreSpec,
    overlap_plan: SphericalOverlapPlan,
    *,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Scatter/gather overlapping features with no learned parameters."""

    import torch
    from torch.nn import functional as F

    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    if overlap_plan.view_count != store_spec.view_count:
        raise ValueError("overlap plan and feature store view counts disagree")
    if overlap_plan.feature_shape_hw != store_spec.feature_shape_hw:
        raise ValueError("overlap plan and feature store shapes disagree")
    source = _open_store(source_path, store_spec, "r")
    target = _open_store(target_path, store_spec, "w+")
    height, width = store_spec.feature_shape_hw
    start = perf_counter()
    total_contributions = 0
    minimum_denominator = float("inf")
    for target_index, contributions in enumerate(overlap_plan.targets):
        source_indices = [item.source_index for item in contributions]
        # Combine all four levels in channels so each spherical transport map
        # is evaluated only once.
        arrays = np.asarray(source[source_indices], dtype=np.float32)
        arrays = arrays.transpose(0, 1, 4, 2, 3).reshape(
            len(source_indices),
            store_spec.level_count * store_spec.channels,
            height,
            width,
        )
        values = torch.from_numpy(np.ascontiguousarray(arrays))
        grids = torch.from_numpy(
            np.stack([item.grid_xy for item in contributions], axis=0)
        )
        weights = torch.from_numpy(
            np.stack([item.weight for item in contributions], axis=0)
        )[:, None]
        sampled = F.grid_sample(
            values,
            grids,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        denominator = weights.sum(dim=0).clamp_min_(1e-12)
        consensus = (sampled * weights).sum(dim=0) / denominator
        consensus = consensus.reshape(
            store_spec.level_count,
            store_spec.channels,
            height,
            width,
        ).permute(0, 2, 3, 1)
        original = torch.from_numpy(
            np.array(source[target_index], dtype=np.float32, copy=True)
        )
        shared = original.lerp(consensus, alpha)
        target[target_index] = shared.numpy()
        target.flush()
        total_contributions += len(contributions)
        minimum_denominator = min(minimum_denominator, float(denominator.min().item()))
    return {
        "source_path": str(source_path),
        "target_path": str(target_path),
        "alpha": float(alpha),
        "total_target_source_contributions": total_contributions,
        "minimum_weight_denominator": minimum_denominator,
        "seconds": perf_counter() - start,
        "bytes": target_path.stat().st_size,
    }


def decode_shared_feature_store(
    model: Any,
    store_path: Path,
    store_spec: FeatureStoreSpec,
    plan: NativeTangentPlan,
    *,
    device: str,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Run the unchanged official DPT on communicated tangent features."""

    import torch

    store = _open_store(store_path, store_spec, "r")
    outputs: list[np.ndarray] = []
    start = perf_counter()
    for view_index in range(store_spec.view_count):
        features = []
        for level_index in range(store_spec.level_count):
            level = np.array(
                store[view_index, level_index], dtype=np.float32, copy=True
            )
            tokens = torch.from_numpy(
                np.ascontiguousarray(level.reshape(-1, level.shape[-1]))
            )
            features.append((tokens[None, None].to(device),))
        with torch.inference_mode():
            output = model.head(
                features,
                plan.view_shape_hw[0],
                plan.view_shape_hw[1],
                patch_start_idx=0,
                chunk_size=None,
            )
            output = model._process_mono_sky_estimation(output)
        canonical_axial = output["depth"][0, 0].detach().float().cpu().numpy()
        axial_m = canonical_axial * (plan.focal_px / CANONICAL_FOCAL_PX)
        radial = axial_to_radial_tangent(axial_m, focal_px=plan.focal_px)
        radial = np.clip(radial, *MODEL_DEPTH_RANGE_M).astype(np.float32, copy=False)
        outputs.append(radial)
    return outputs, {
        "view_count": len(outputs),
        "seconds": perf_counter() - start,
        "decoder": "unchanged official planar DPT per communicated tangent view",
    }


class _PrecomputedDepthAdapter:
    def __init__(
        self, values: Sequence[np.ndarray], expected_shape: tuple[int, int]
    ) -> None:
        self._values = tuple(values)
        self._expected_shape = expected_shape
        self._index = 0

    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.asarray(rgb).shape != (*self._expected_shape, 3):
            raise ValueError("tangent RGB shape disagrees with the frozen plan")
        if self._index >= len(self._values):
            raise RuntimeError("more tangent views requested than were decoded")
        value = self._values[self._index]
        self._index += 1
        return value, np.isfinite(value) & (value > 0.0)


def infer_shared_feature_erp(
    model: Any,
    rgb: np.ndarray,
    source_support: np.ndarray,
    plan: NativeTangentPlan,
    *,
    scratch_dir: Path,
    device: str = "cpu",
    alpha: float = 1.0,
    maximum_sources_per_target: int = 6,
    keep_feature_stores: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run native tangent inference with deterministic spherical feature sharing."""

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
    raw_store = scratch_dir / "da3-vit-features-f32.dat"
    shared_store = scratch_dir / "da3-shared-features-f32.dat"
    store_spec, extraction_report = extract_da3_feature_store(
        model, list(views), raw_store, device=device
    )
    overlap_plan = make_spherical_overlap_plan(
        plan,
        feature_shape_hw=store_spec.feature_shape_hw,
        maximum_sources_per_target=maximum_sources_per_target,
    )
    sharing_report = share_feature_store(
        raw_store, shared_store, store_spec, overlap_plan, alpha=alpha
    )
    depths, decoder_report = decode_shared_feature_store(
        model, shared_store, store_spec, plan, device=device
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
        "raw": extraction_report,
        "shared": sharing_report,
    }
    if not keep_feature_stores:
        raw_store.unlink()
        shared_store.unlink()
        stores["raw"]["path"] = None
        stores["shared"]["target_path"] = None
    return (
        radial,
        validity,
        {
            "interface": INTERFACE,
            "feature_store": stores,
            "feature_store_spec": asdict(store_spec),
            "overlap_plan": overlap_plan.to_dict(),
            "decoder": decoder_report,
            "view_count": len(depths),
            "valid_pixels": int(validity.sum()),
            "coverage_fraction_of_source_support": float(
                validity.sum() / max(1, source_support.sum())
            ),
            "dpt_spherical": False,
            "limitation": (
                "four ViT levels communicate on spherical rays, but the official "
                "DPT remains planar inside each tangent view"
            ),
        },
    )


__all__ = [
    "FeatureStoreSpec",
    "INTERFACE",
    "OverlapContribution",
    "SphericalOverlapPlan",
    "decode_shared_feature_store",
    "extract_da3_feature_store",
    "infer_shared_feature_erp",
    "make_spherical_overlap_plan",
    "share_feature_store",
]
