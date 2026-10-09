"""External-only adapter for the Depth Any Camera indoor CNN baseline.

No upstream source or checkpoint is copied into PanorAi.  The caller supplies a
Depth Any Camera checkout plus the checksum-verified CNNDepth ResNet-101 indoor
checkpoint.  This module only defines the experiment boundary and the
spherical port glue.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys
import types
from typing import Any, Iterator

import numpy as np


CANONICAL_FOCAL_PX = 519.0
MODEL_DEPTH_RANGE_M = (0.3, 150.0)
CHECKPOINT_SHA256 = (
    "c2a0689860257083b096e0b4133cb75418e25c8c3d22f0e23b7c12465017383c"
)
SOURCE_COMMIT = "371ee299429257bb9a27d1e23b7dc53670e37023"
RGB_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@contextmanager
def _external_import_path(source: Path) -> Iterator[None]:
    value = str(source.resolve())
    sys.path.insert(0, value)
    try:
        yield
    finally:
        try:
            sys.path.remove(value)
        except ValueError:  # pragma: no cover - defensive against upstream code
            pass


def _install_unused_attention_import_shim() -> None:
    """Avoid importing the optional CUDA extension when ``attn_dec=false``.

    Upstream ``cnn_depth.py`` imports its deformable-attention decoder even
    though the published CNNDepth configuration disables that branch.  The
    placeholder is never instantiated or called; a real CNNDepth checkpoint
    load and forward pass remain the integration proof.
    """

    name = "dac.models.defattn_decoder"
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.MSDeformAttnPixelDecoder = None
    sys.modules[name] = module


class SphericalConvWithPostOps:  # materialized lazily so Torch stays optional
    pass


def _port_post_op_convolutions(
    model: Any,
    *,
    max_sampled_elements: int | None,
    angular_step_scale: float | tuple[float, float] = 1.0,
) -> tuple[Any, tuple[dict[str, Any], ...]]:
    """Port upstream Conv2d subclasses without dropping norm/activation."""

    from torch import nn
    from panorai.image_processing.torch import SphericalConv2d

    class _SphericalConvWithPostOps(nn.Module):
        def __init__(self, source: nn.Conv2d) -> None:
            super().__init__()
            self.convolution = SphericalConv2d(
                source,
                max_sampled_elements=max_sampled_elements,
                angular_step_scale=angular_step_scale,
            )
            self.norm = getattr(source, "norm", None)
            self.activation = getattr(source, "activation", None)

        def forward(self, values: Any) -> Any:
            values = self.convolution(values)
            if self.norm is not None:
                values = self.norm(values)
            if self.activation is not None:
                values = self.activation(values)
            return values

    global SphericalConvWithPostOps
    SphericalConvWithPostOps = _SphericalConvWithPostOps
    records: list[dict[str, Any]] = []

    def replace(module: nn.Module, prefix: str = "") -> None:
        for name, child in tuple(module.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            is_post_op_subclass = isinstance(child, nn.Conv2d) and (
                hasattr(child, "norm") or hasattr(child, "activation")
            )
            if is_post_op_subclass:
                replacement = _SphericalConvWithPostOps(child)
                setattr(module, name, replacement)
                records.append(
                    {
                        "path": path,
                        "source_type": type(child).__name__,
                        "target_type": "SphericalConvWithPostOps",
                        "weight_shape": list(child.weight.shape),
                        "angular_step_scale": list(
                            replacement.convolution.angular_step_scale
                        ),
                        "parameter_identity_preserved": (
                            replacement.convolution.weight is child.weight
                            and replacement.convolution.bias is child.bias
                        ),
                        "norm_identity_preserved": replacement.norm
                        is getattr(child, "norm", None),
                    }
                )
            else:
                replace(child, path)

    replace(model)
    return model, tuple(records)


def load_indoor_cnn(
    source: Path,
    config_path: Path,
    checkpoint: Path,
    *,
    spherical: bool,
    max_sampled_elements: int | None = None,
    angular_step_scale: float | tuple[float, float] = 1.0,
) -> tuple[Any, dict[str, Any]]:
    """Load the frozen external CNNDepth checkpoint, optionally ported."""

    import torch

    from protocol import sha256

    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("CNNDepth checkpoint SHA-256 does not match the model card")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_name") != "CNNDepth" or config.get("tgt_f") != 519:
        raise RuntimeError("unexpected CNNDepth indoor configuration")
    config["model"]["pixel_encoder"]["pretrained"] = False
    with _external_import_path(source):
        _install_unused_attention_import_shim()
        from dac.models.cnn_depth import CNNDepth

        model = CNNDepth.build(config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in payload.items()}
    incompatible = model.load_state_dict(state, strict=True)
    port: dict[str, Any] = {
        "ported": False,
        "remaining_planar_spatial_layers": "all (cubemap control)",
    }
    if spherical:
        from panorai.image_processing.torch import port_module_with_report

        model, post_records = _port_post_op_convolutions(
            model,
            max_sampled_elements=max_sampled_elements,
            angular_step_scale=angular_step_scale,
        )
        report = port_module_with_report(
            model,
            max_sampled_elements=max_sampled_elements,
            angular_step_scale=angular_step_scale,
        )
        identities = [
            item["parameter_identity_preserved"] for item in post_records
        ] + [
            layer.parameter_identity_preserved
            for layer in report.layers
            if layer.parameter_identity_preserved is not None
        ]
        if report.remaining_planar_spatial_layers or not all(identities):
            raise RuntimeError("CNNDepth spherical port is incomplete")
        port = {
            "ported": True,
            "post_op_convolutions": list(post_records),
            "core_report": report.to_dict(),
            "all_parameter_identities_preserved": all(identities),
            "remaining_planar_spatial_layers": [],
        }
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "port": port,
    }


def angular_step_scale_for_erp(
    shape: tuple[int, int], *, reference_focal_px: float = CANONICAL_FOCAL_PX
) -> tuple[float, float]:
    """Map a perspective pixel's reference angle onto a native ERP lattice.

    Near the optical axis, one canonical CNNDepth pixel spans approximately
    ``1 / reference_focal_px`` radians.  Multiplying the native ERP pixel
    steps by this pair retains that angular spacing without changing either
    the input or output lattice.
    """

    height, width = shape
    if height < 1 or width < 1:
        raise ValueError("ERP dimensions must be positive")
    if width != 2 * height:
        raise ValueError("angular support requires a full 2:1 ERP")
    if not math.isfinite(reference_focal_px) or reference_focal_px <= 0.0:
        raise ValueError("reference_focal_px must be finite and positive")
    return (
        height / (math.pi * reference_focal_px),
        width / (2.0 * math.pi * reference_focal_px),
    )


def tensor_input(rgb: np.ndarray) -> Any:
    """Convert uint8/float RGB HWC to the checkpoint's normalized NCHW input."""

    import torch

    values = rgb.astype(np.float32) / 255.0
    values = (values - RGB_MEAN) / RGB_STD
    return torch.from_numpy(np.moveaxis(values, -1, 0))[None]


def infer_spherical(model: Any, rgb: np.ndarray) -> np.ndarray:
    """Run the ported CNN on a full ERP without input/prediction resizing."""

    import torch

    height, width = rgb.shape[:2]
    if width != 2 * height:
        raise ValueError("spherical CNNDepth input must be a full 2:1 ERP")
    with torch.inference_mode():
        canonical = model(tensor_input(rgb))[0][0, 0].cpu().numpy()
    effective_focal = width / (2.0 * math.pi)
    radial = canonical * (effective_focal / CANONICAL_FOCAL_PX)
    return np.clip(radial, *MODEL_DEPTH_RANGE_M).astype(np.float32)


def infer_cubemap(model: Any, rgb: np.ndarray, face_size: int) -> np.ndarray:
    """Run the unchanged indoor CNN independently on six native cube faces."""

    import torch
    from panorai.geometry import (
        CUBE_FACE_ORDER,
        cubemap_to_equirectangular,
        equirectangular_to_cubemap,
    )
    from protocol import axial_to_radial

    if face_size < 1:
        raise ValueError("face_size must be positive")
    faces = equirectangular_to_cubemap(
        rgb.astype(np.float32), face_size, interpolation="bilinear"
    )
    radial_faces: dict[str, np.ndarray] = {}
    for face in CUBE_FACE_ORDER:
        face_rgb = np.asarray(faces[face].data)
        with torch.inference_mode():
            canonical = model(tensor_input(face_rgb))[0][0, 0].cpu().numpy()
        axial_m = canonical * ((face_size / 2.0) / CANONICAL_FOCAL_PX)
        radial_faces[face] = axial_to_radial(axial_m)
    result = cubemap_to_equirectangular(
        radial_faces,
        rgb.shape[:2],
        interpolation="bilinear",
        invalid_policy="propagate",
    )
    return np.clip(
        np.asarray(result.data, dtype=np.float32), *MODEL_DEPTH_RANGE_M
    ).astype(np.float32)


__all__ = [
    "CANONICAL_FOCAL_PX",
    "CHECKPOINT_SHA256",
    "MODEL_DEPTH_RANGE_M",
    "SOURCE_COMMIT",
    "angular_step_scale_for_erp",
    "infer_cubemap",
    "infer_spherical",
    "load_indoor_cnn",
    "tensor_input",
]
