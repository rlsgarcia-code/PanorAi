"""Native-density tangent-window control for Metric3Dv2 ViT models.

This module deliberately keeps the transformer unchanged.  It samples fixed
gnomonic windows directly from the native angular-density ERP, runs the
perspective-trained model, converts its axial output to radial range, and lets
PanorAi's geometry workflow reconstruct the native ERP.  It is the multiface
control for a later spherical-token port, not that port itself.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sys
import types
from typing import Any, Iterator, Sequence

import numpy as np


INTERFACE = "panorai-metric3dv2-vit-native-tangent/v1-experimental"
CANONICAL_FOCAL_PX = 1000.0
MODEL_DEPTH_RANGE_M = (0.1, 200.0)
MODEL_CONFIGS = {
    "small": {
        "hub_name": "metric3d_vit_small",
        "sha256": "b34b2a2be9148054991cef7e417930e1320602ba7bc503b0ee4e7888543728f6",
        "size_bytes": 150_120_967,
        "decoder": "RAFT-4iter",
    },
    "large": {
        "hub_name": "metric3d_vit_large",
        "sha256": "15328ffc42b528b95f188687418f6f03b3f123eb34ccdbd686c112abbea6d972",
        "size_bytes": 1_647_972_663,
        "decoder": "RAFT-8iter",
    },
}


@dataclass(frozen=True, slots=True)
class NativeTangentPlan:
    """Frozen native-density window layout and camera model."""

    erp_shape_hw: tuple[int, int]
    view_shape_hw: tuple[int, int]
    focal_px: float
    hfov_deg: float
    vfov_deg: float
    overlap_fraction: float
    minimum_latitude_deg: float
    maximum_latitude_deg: float
    centers_lat_lon_deg: tuple[tuple[float, float], ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["centers_lat_lon_deg"] = [
            list(value) for value in self.centers_lat_lon_deg
        ]
        return result


class _FixedTangentSampler:
    def __init__(self, centers: Sequence[tuple[float, float]]) -> None:
        self._centers = tuple(centers)

    def get_tangent_points(self) -> list[tuple[float, float]]:
        return list(self._centers)


def native_erp_focal_px(shape_hw: tuple[int, int]) -> float:
    """Return pixels/radian at the ERP equator."""

    height, width = shape_hw
    if height <= 0 or width != 2 * height:
        raise ValueError("native ERP shape must be positive and exactly 2:1")
    return width / (2.0 * math.pi)


def make_native_tangent_plan(
    erp_shape_hw: tuple[int, int],
    *,
    view_shape_hw: tuple[int, int] = (280, 504),
    overlap_fraction: float = 0.25,
    minimum_latitude_deg: float = -60.0,
    maximum_latitude_deg: float = 90.0,
) -> NativeTangentPlan:
    """Cover supported P74 latitudes without angular minification.

    The view focal length equals the ERP equatorial pixel density.  Therefore
    adjacent tangent samples are approximately one native ERP pixel apart at
    every view centre; no image pyramid or pre-resize is introduced.
    """

    focal = native_erp_focal_px(erp_shape_hw)
    view_height, view_width = view_shape_hw
    if view_height <= 0 or view_width <= 0:
        raise ValueError("view dimensions must be positive")
    if view_height % 28 or view_width % 28:
        raise ValueError("Metric3Dv2 tangent dimensions must be divisible by 28")
    if not 0.0 <= overlap_fraction < 0.75:
        raise ValueError("overlap_fraction must be in [0, 0.75)")
    if not -90.0 <= minimum_latitude_deg < maximum_latitude_deg <= 90.0:
        raise ValueError("latitude interval must lie inside [-90, 90]")

    hfov = math.degrees(2.0 * math.atan(view_width / (2.0 * focal)))
    vfov = math.degrees(2.0 * math.atan(view_height / (2.0 * focal)))
    half_v = vfov / 2.0
    first = max(-90.0 + half_v, minimum_latitude_deg + half_v)
    last = min(90.0 - half_v, maximum_latitude_deg - half_v)
    if last < first:
        raise ValueError("view is taller than the requested latitude interval")
    nominal_step = vfov * (1.0 - overlap_fraction)
    intervals = max(1, int(math.ceil((last - first) / nominal_step)))
    latitudes = np.linspace(first, last, intervals + 1, dtype=np.float64)

    centers: list[tuple[float, float]] = []
    horizontal_step = hfov * (1.0 - overlap_fraction)
    for latitude in latitudes:
        # Longitude distance contracts by cos(latitude).  Keeping this factor
        # avoids hundreds of redundant polar views while retaining overlap.
        circumference = 360.0 * max(math.cos(math.radians(float(latitude))), 0.08)
        count = max(1, int(math.ceil(circumference / horizontal_step)))
        longitudes = -180.0 + (np.arange(count, dtype=np.float64) + 0.5) * (
            360.0 / count
        )
        centers.extend((float(latitude), float(value)) for value in longitudes)

    return NativeTangentPlan(
        erp_shape_hw=tuple(erp_shape_hw),
        view_shape_hw=tuple(view_shape_hw),
        focal_px=float(focal),
        hfov_deg=float(hfov),
        vfov_deg=float(vfov),
        overlap_fraction=float(overlap_fraction),
        minimum_latitude_deg=float(minimum_latitude_deg),
        maximum_latitude_deg=float(maximum_latitude_deg),
        centers_lat_lon_deg=tuple(centers),
    )


def axial_to_radial_tangent(
    axial_depth_m: np.ndarray,
    *,
    focal_px: float,
) -> np.ndarray:
    """Convert pinhole axial depth to radial range using pixel centres."""

    axial = np.asarray(axial_depth_m, dtype=np.float32)
    if axial.ndim != 2:
        raise ValueError("axial depth must be HW")
    if not math.isfinite(focal_px) or focal_px <= 0.0:
        raise ValueError("focal_px must be finite and positive")
    height, width = axial.shape
    x = (np.arange(width, dtype=np.float64) + 0.5 - width / 2.0) / focal_px
    y = (np.arange(height, dtype=np.float64) + 0.5 - height / 2.0) / focal_px
    radial_scale = np.sqrt(1.0 + y[:, None] ** 2 + x[None, :] ** 2)
    return (axial * radial_scale).astype(np.float32)


@contextmanager
def _upstream_import_path(source_root: Path) -> Iterator[None]:
    value = str(source_root)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        try:
            sys.path.remove(value)
        except ValueError:  # pragma: no cover - upstream import mutation
            pass


def load_metric3dv2_vit(
    source_root: Path,
    checkpoint: Path,
    *,
    variant: str,
    device: str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """Load one checksum-identified official Metric3Dv2 ViT checkpoint."""

    if variant not in MODEL_CONFIGS:
        raise ValueError(f"variant must be one of {tuple(MODEL_CONFIGS)}")
    config = MODEL_CONFIGS[variant]
    if checkpoint.stat().st_size != config["size_bytes"]:
        raise RuntimeError("Metric3Dv2 checkpoint size does not match frozen identity")

    import hashlib
    import torch

    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    observed_sha256 = digest.hexdigest()
    if observed_sha256 != config["sha256"]:
        raise RuntimeError(
            "Metric3Dv2 checkpoint SHA-256 does not match frozen identity"
        )

    with _upstream_import_path(source_root):
        model = torch.hub.load(
            str(source_root),
            config["hub_name"],
            source="local",
            pretrain=False,
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise RuntimeError("Metric3Dv2 checkpoint lacks model_state_dict")
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
    missing = tuple(incompatible.missing_keys)
    unexpected = tuple(incompatible.unexpected_keys)
    # The official v2 checkpoints intentionally omit the unused DINO masking
    # token.  It is not exercised by deterministic dense inference.
    expected_missing = ("depth_model.encoder.mask_token",)
    if missing not in ((), expected_missing) or unexpected:
        raise RuntimeError(
            "Metric3Dv2 checkpoint/model mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    decoder = model.depth_model.decoder

    def get_bins(self: Any, bins_num: int) -> Any:
        parameter = next(self.parameters())
        return torch.exp(
            torch.linspace(
                math.log(self.min_val),
                math.log(self.max_val),
                bins_num,
                device=parameter.device,
                dtype=parameter.dtype,
            )
        )

    # Upstream hard-codes CUDA for this lazily-created, non-learned buffer.
    # Bind it to the actual model device without touching checkpoint state.
    decoder.get_bins = types.MethodType(get_bins, decoder)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    report = {
        "variant": variant,
        "hub_name": config["hub_name"],
        "decoder": config["decoder"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": observed_sha256,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "device": device,
    }
    return model, report


class Metric3Dv2TangentInference:
    """Callable adapter used by PanorAi's immutable view workflow."""

    def __init__(self, model: Any, plan: NativeTangentPlan, *, device: str) -> None:
        self.model = model
        self.plan = plan
        self.device = device
        self.calls = 0

    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch

        values = np.asarray(rgb)
        if values.shape != (*self.plan.view_shape_hw, 3):
            raise ValueError("tangent RGB shape disagrees with the frozen plan")
        if values.dtype != np.float32:
            values = values.astype(np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(values.transpose(2, 0, 1)))
        mean = tensor.new_tensor((123.675, 116.28, 103.53))[:, None, None]
        std = tensor.new_tensor((58.395, 57.12, 57.375))[:, None, None]
        tensor = ((tensor - mean) / std)[None].to(self.device)
        with torch.inference_mode():
            prediction, confidence, _ = self.model.inference({"input": tensor})
        axial = prediction[0, 0].detach().float().cpu().numpy()
        canonical_to_real = self.plan.focal_px / CANONICAL_FOCAL_PX
        radial = axial_to_radial_tangent(
            axial * canonical_to_real,
            focal_px=self.plan.focal_px,
        )
        radial = np.clip(radial, *MODEL_DEPTH_RANGE_M).astype(np.float32, copy=False)
        valid = np.isfinite(radial) & (radial > 0.0)
        if confidence is not None:
            confidence_array = confidence.detach().float().cpu().numpy().squeeze()
            valid &= np.isfinite(confidence_array)
        self.calls += 1
        return radial, valid


def infer_native_tangent_erp(
    model: Any,
    rgb: np.ndarray,
    source_support: np.ndarray,
    plan: NativeTangentPlan,
    *,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Infer and Gaussian-fuse native-density tangent predictions."""

    from panorai.data import EquirectangularImage

    if rgb.shape != (*plan.erp_shape_hw, 3):
        raise ValueError("ERP RGB shape disagrees with the frozen plan")
    if source_support.shape != plan.erp_shape_hw or source_support.dtype != np.bool_:
        raise ValueError("source_support must be a boolean native-ERP mask")
    # The south-polar cap is explicitly materialized with the model's neutral
    # RGB padding. Keep it finite for boundary windows, then apply geometric
    # support only after reconstruction; invalid input would inject NaNs into
    # the transformer.
    source = EquirectangularImage(rgb)
    sampler = _FixedTangentSampler(plan.centers_lat_lon_deg)
    views = source.views(
        sampler,
        size=plan.view_shape_hw,
        fov=(plan.hfov_deg, plan.vfov_deg),
        depth_policy="propagate",
    )
    adapter = Metric3Dv2TangentInference(model, plan, device=device)
    predicted = views.map(adapter, input="image", output="depth", units="m")
    reconstructed = predicted.reconstruct(
        blend={"depth": "gaussian"}, modalities=("depth",)
    )
    payload = reconstructed._workflow_data()["depth"]
    metadata = reconstructed._workflow_metadata["depth"]
    validity = np.asarray(metadata["validity"], dtype=bool) & source_support
    radial = np.asarray(payload, dtype=np.float32)
    radial[~validity] = np.nan
    return (
        radial,
        validity,
        {
            "view_count": adapter.calls,
            "workflow": predicted.describe(),
            "reconstruction_blend": metadata["blend"],
            "valid_pixels": int(validity.sum()),
            "coverage_fraction_of_source_support": float(
                validity.sum() / max(1, source_support.sum())
            ),
        },
    )


__all__ = [
    "CANONICAL_FOCAL_PX",
    "INTERFACE",
    "MODEL_CONFIGS",
    "MODEL_DEPTH_RANGE_M",
    "Metric3Dv2TangentInference",
    "NativeTangentPlan",
    "axial_to_radial_tangent",
    "infer_native_tangent_erp",
    "load_metric3dv2_vit",
    "make_native_tangent_plan",
    "native_erp_focal_px",
]
