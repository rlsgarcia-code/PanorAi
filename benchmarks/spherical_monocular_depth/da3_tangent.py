"""Native-density Depth Anything 3 metric tangent control.

The official high-level DA3 API resizes inputs to a configurable processing
resolution.  This benchmark instead calls the unchanged network directly on
the already projected tangent windows.  That keeps the P74 angular sampling
density intact and makes the comparison with the Metric3Dv2 tangent control
meaningful.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys
from typing import Any, Iterator

import numpy as np

from benchmarks.spherical_monocular_depth.vit_tangent import (
    NativeTangentPlan,
    _FixedTangentSampler,
    axial_to_radial_tangent,
)


INTERFACE = "panorai-depth-anything-3-metric-native-tangent/v1-experimental"
MODEL_ID = "depth-anything/DA3METRIC-LARGE"
MODEL_LICENSE = "Apache-2.0"
CHECKPOINT_SHA256 = "bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776"
CHECKPOINT_SIZE_BYTES = 1_336_734_448
CANONICAL_FOCAL_PX = 300.0
MODEL_DEPTH_RANGE_M = (1e-3, 200.0)


@contextmanager
def _upstream_import_path(source_root: Path) -> Iterator[None]:
    source_path = source_root / "src"
    if not source_path.is_dir():
        raise FileNotFoundError(f"DA3 source package not found: {source_path}")
    value = str(source_path)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        try:
            sys.path.remove(value)
        except ValueError:  # pragma: no cover - defensive import cleanup
            pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_da3metric_large(
    source_root: Path,
    checkpoint: Path,
    *,
    device: str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """Load the checksum-pinned official DA3 metric-large network.

    Only the modules required by the monocular metric network are constructed;
    optional exporters, Open3D, Gaussian splatting, and the resizing API are not
    imported or exercised.
    """

    if checkpoint.stat().st_size != CHECKPOINT_SIZE_BYTES:
        raise RuntimeError("DA3 checkpoint size does not match frozen identity")
    observed_sha256 = _sha256(checkpoint)
    if observed_sha256 != CHECKPOINT_SHA256:
        raise RuntimeError("DA3 checkpoint SHA-256 does not match frozen identity")

    from torch import nn

    with _upstream_import_path(source_root):
        from depth_anything_3.model.da3 import DepthAnything3Net
        from depth_anything_3.model.dinov2.dinov2 import DinoV2
        from depth_anything_3.model.dpt import DPT
        from safetensors.torch import load_model

        backbone = DinoV2(
            name="vitl",
            out_layers=[4, 11, 17, 23],
            alt_start=-1,
            qknorm_start=-1,
            rope_start=-1,
            cat_token=False,
        )
        head = DPT(
            dim_in=1024,
            output_dim=1,
            features=256,
            out_channels=[256, 512, 1024, 1024],
        )

        class _CheckpointEnvelope(nn.Module):
            # The official safetensors keys are rooted at ``model.``.
            def __init__(self) -> None:
                super().__init__()
                self.model = DepthAnything3Net(backbone, head)

        envelope = _CheckpointEnvelope()
        missing, unexpected = load_model(
            envelope,
            str(checkpoint),
            strict=True,
            device="cpu",
        )
    if missing or unexpected:
        raise RuntimeError(
            f"DA3 checkpoint/model mismatch: missing={missing}, unexpected={unexpected}"
        )
    model = envelope.model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    report = {
        "model_id": MODEL_ID,
        "model_license": MODEL_LICENSE,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": observed_sha256,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
        "device": device,
        "official_high_level_resize_bypassed": True,
    }
    return model, report


class DA3MetricTangentInference:
    """Callable adapter from one RGB tangent view to metric radial range."""

    def __init__(self, model: Any, plan: NativeTangentPlan, *, device: str) -> None:
        height, width = plan.view_shape_hw
        if height % 14 or width % 14:
            raise ValueError(
                "DA3 tangent dimensions must be divisible by patch size 14"
            )
        self.model = model
        self.plan = plan
        self.device = device
        self.calls = 0

    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch

        values = np.asarray(rgb)
        if values.shape != (*self.plan.view_shape_hw, 3):
            raise ValueError("tangent RGB shape disagrees with the frozen plan")
        tensor = torch.from_numpy(
            np.ascontiguousarray(values.astype(np.float32).transpose(2, 0, 1))
        )
        tensor = tensor / 255.0
        mean = tensor.new_tensor((0.485, 0.456, 0.406))[:, None, None]
        std = tensor.new_tensor((0.229, 0.224, 0.225))[:, None, None]
        tensor = ((tensor - mean) / std)[None, None].to(self.device)
        with torch.inference_mode():
            output = self.model(tensor)
        canonical_axial = output["depth"][0, 0].detach().float().cpu().numpy()
        # Official DA3 metric rule: metres = focal_px * network_depth / 300.
        axial_m = canonical_axial * (self.plan.focal_px / CANONICAL_FOCAL_PX)
        radial = axial_to_radial_tangent(axial_m, focal_px=self.plan.focal_px)
        radial = np.clip(radial, *MODEL_DEPTH_RANGE_M).astype(np.float32, copy=False)
        valid = np.isfinite(radial) & (radial > 0.0)
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
    """Infer DA3 on fixed tangent views and Gaussian-fuse a native ERP."""

    from panorai.data import EquirectangularImage

    if rgb.shape != (*plan.erp_shape_hw, 3):
        raise ValueError("ERP RGB shape disagrees with the frozen plan")
    if source_support.shape != plan.erp_shape_hw or source_support.dtype != np.bool_:
        raise ValueError("source_support must be a boolean native-ERP mask")
    source = EquirectangularImage(rgb)
    sampler = _FixedTangentSampler(plan.centers_lat_lon_deg)
    views = source.views(
        sampler,
        size=plan.view_shape_hw,
        fov=(plan.hfov_deg, plan.vfov_deg),
        depth_policy="propagate",
    )
    adapter = DA3MetricTangentInference(model, plan, device=device)
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
    "CHECKPOINT_SHA256",
    "CHECKPOINT_SIZE_BYTES",
    "DA3MetricTangentInference",
    "INTERFACE",
    "MODEL_DEPTH_RANGE_M",
    "MODEL_ID",
    "MODEL_LICENSE",
    "infer_native_tangent_erp",
    "load_da3metric_large",
]
