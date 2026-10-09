"""External-only adapter for the Matterport3D-trained UniFuse CNN.

The upstream source and checkpoint remain outside PanorAi.  This adapter
rebuilds UniFuse's fixed projection grids at the requested native ERP shape,
loads every learned tensor unchanged, and performs no image or prediction
resize.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Any, Iterator

import numpy as np


CHECKPOINT_SHA256 = (
    "527414545d5000f6d629db3703a800a9aac099e6d511b8f61e9e799d8d3e93b3"
)
SOURCE_COMMIT = "5da3a4e64491c3efff3ce8613ba289147515c832"
MODEL_DEPTH_RANGE_M = (0.001, 10.0)
RGB_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@contextmanager
def _external_import_path(source: Path) -> Iterator[None]:
    value = str((source / "UniFuse").resolve())
    sys.path.insert(0, value)
    try:
        yield
    finally:
        try:
            sys.path.remove(value)
        except ValueError:  # pragma: no cover
            pass


def load_unifuse(
    source: Path, checkpoint: Path, *, height: int, width: int
) -> tuple[Any, dict[str, Any]]:
    """Load learned tensors and rebuild only shape-dependent projection grids."""

    import torch

    from protocol import sha256

    if width != 2 * height or height % 64:
        raise ValueError("UniFuse native ERP shape must be 2:1 and divisible by 64")
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("UniFuse checkpoint SHA-256 does not match")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected_metadata = {
        "dataset": "matterport3d",
        "height": 512,
        "width": 1024,
        "layers": 18,
        "net": "UniFuse",
        "fusion": "cee",
        "se_in_fusion": True,
    }
    observed = {key: payload.get(key) for key in expected_metadata}
    if observed != expected_metadata:
        raise RuntimeError(f"unexpected UniFuse checkpoint metadata: {observed}")
    with _external_import_path(source):
        from networks import UniFuse

        model = UniFuse(
            payload["layers"],
            height,
            width,
            pretrained=False,
            max_depth=MODEL_DEPTH_RANGE_M[1],
            fusion_type=payload["fusion"],
            se_in_fusion=payload["se_in_fusion"],
        )
    target = model.state_dict()
    learned = {
        key: value
        for key, value in payload.items()
        if key in target and not key.startswith("projectors.")
    }
    wrong_shapes = {
        key: (tuple(value.shape), tuple(target[key].shape))
        for key, value in learned.items()
        if value.shape != target[key].shape
    }
    if wrong_shapes:
        raise RuntimeError(f"UniFuse learned tensor shape mismatch: {wrong_shapes}")
    incompatible = model.load_state_dict(learned, strict=False)
    expected_missing = {
        key for key in target if key.startswith("projectors.")
    }
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError(
            f"unexpected missing UniFuse keys: {incompatible.missing_keys}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected UniFuse keys: {incompatible.unexpected_keys}"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    learned_parameter_names = set(dict(model.named_parameters())) - expected_missing
    return model, {
        "checkpoint_training_shape_hw": [512, 1024],
        "native_inference_shape_hw": [height, width],
        "learned_tensor_count": len(learned),
        "learned_parameter_count": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name in learned_parameter_names
        ),
        "rebuilt_nonlearned_projection_grids": sorted(expected_missing),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def tensor_input(rgb: np.ndarray) -> Any:
    """Convert HWC RGB to the checkpoint's ImageNet-normalized NCHW input."""

    import torch

    values = rgb.astype(np.float32) / 255.0
    values = (values - RGB_MEAN) / RGB_STD
    return torch.from_numpy(np.moveaxis(values, -1, 0))[None]


def cubemap_strip(rgb: np.ndarray) -> np.ndarray:
    """Create UniFuse's F/R/B/L/U/D strip without resizing the ERP."""

    from panorai.geometry import CUBE_FACE_ORDER, equirectangular_to_cubemap

    height, width = rgb.shape[:2]
    if width != 2 * height:
        raise ValueError("UniFuse input must be a full 2:1 ERP")
    faces = equirectangular_to_cubemap(
        rgb.astype(np.float32), height // 2, interpolation="bilinear"
    )
    return np.concatenate(
        [np.asarray(faces[name].data) for name in CUBE_FACE_ORDER], axis=1
    )


def infer_unifuse(model: Any, rgb: np.ndarray) -> np.ndarray:
    """Run the panoramic CNN at its constructed shape, with no resize."""

    import torch

    height, width = rgb.shape[:2]
    if (height, width) != (model.equi_h, model.equi_w):
        raise ValueError("input shape differs from the constructed UniFuse grid")
    cube = cubemap_strip(rgb)
    with torch.inference_mode():
        output = model(tensor_input(rgb), tensor_input(cube))["pred_depth"]
    depth = output[0, 0].cpu().numpy()
    if depth.shape != (height, width):
        raise RuntimeError("UniFuse changed the native prediction shape")
    return np.clip(depth, *MODEL_DEPTH_RANGE_M).astype(np.float32)


__all__ = [
    "CHECKPOINT_SHA256",
    "MODEL_DEPTH_RANGE_M",
    "SOURCE_COMMIT",
    "cubemap_strip",
    "infer_unifuse",
    "load_unifuse",
    "tensor_input",
]
