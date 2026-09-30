"""Lazy compatibility adapters for separately installed depth projects.

PanorAi intentionally does not distribute the upstream implementations. This
module contains only PanorAi-authored glue and imports optional dependencies
inside the selected loader.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import tempfile
from typing import Any

from .registry import ModelRegistry


class DepthAdapterUnavailableError(ImportError):
    """An optional upstream depth implementation is unavailable or incompatible."""


_UPSTREAM = {
    "dav2": {
        "name": "Depth Anything V2",
        "license": "Apache-2.0 code; model weights may have additional terms",
        "url": "https://github.com/DepthAnything/Depth-Anything-V2",
    },
    "m3dv2": {
        "name": "Metric3D",
        "license": "BSD-2-Clause code; verify checkpoint terms separately",
        "url": "https://github.com/YvanYin/Metric3D",
    },
    "dust3r": {
        "name": "DUSt3R",
        "license": "CC BY-NC-SA 4.0",
        "url": "https://github.com/naver/dust3r",
    },
    "zoe": {
        "name": "ZoeDepth",
        "license": "Apache-2.0 code; verify model-card terms separately",
        "url": "https://huggingface.co/Intel/zoedepth-nyu-kitti",
    },
}

_import_module = importlib.import_module


def _unavailable(
    key: str, dependency: str, cause: BaseException
) -> DepthAdapterUnavailableError:
    info = _UPSTREAM[key]
    return DepthAdapterUnavailableError(
        f"PanorAi 3.1 is adapter-only for {key!r}; the {info['name']} "
        "implementation is not bundled. Install adapter dependencies with "
        '`pip install "panorai[depth]"`, then install a compatible upstream '
        f"checkout/package from {info['url']} and review its license "
        f"({info['license']}). Required import {dependency!r} failed: {cause}"
    )


def _require_module(key: str, module: str) -> Any:
    try:
        return _import_module(module)
    except Exception as exc:
        raise _unavailable(key, module, exc) from exc


def _required_attribute(key: str, module: Any, name: str) -> Any:
    try:
        return getattr(module, name)
    except Exception as exc:
        module_name = getattr(module, "__name__", repr(module))
        raise _unavailable(key, f"{module_name}.{name}", exc) from exc


@ModelRegistry.register(
    "dav2",
    default_args={
        "checkpoint_path": None,
        "encoder": "vits",
        "dataset": "vkitti",
        "max_depth": 80,
        "device": "mps",
        "return_model": False,
    },
)
def load_dav2_model(
    checkpoint_path: str | Path | None = None,
    encoder: str = "vits",
    dataset: str | None = "vkitti",
    max_depth: float = 80,
    device: str = "mps",
    return_model: bool = False,
) -> Any:
    """Load a separately installed Depth Anything V2 implementation."""

    torch = _require_module("dav2", "torch")
    dpt = _require_module("dav2", "depth_anything_v2.dpt")
    model_type = _required_attribute("dav2", dpt, "DepthAnythingV2")
    model_configs = {
        "vits": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384],
        },
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
    }
    if encoder not in model_configs:
        raise ValueError(f"unsupported Depth Anything V2 encoder: {encoder!r}")
    if checkpoint_path is None:
        from panorai.path_config import get_path

        root = get_path("depth_anything_v2", "checkpoint_dir")
        if root is None:
            raise FileNotFoundError(
                "Provide checkpoint_path or configure "
                "depth_anything_v2.checkpoint_dir in PANORAI_PATHS."
            )
        if dataset:
            filename = f"depth_anything_v2_metric_{dataset}_{encoder}.pth"
        else:
            filename = f"depth_anything_v2_{encoder}.pth"
        checkpoint_path = Path(root) / filename

    model = model_type(**model_configs[encoder], max_depth=max_depth)
    state = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()
    if return_model:
        return model

    def inference(image_array: Any) -> Any:
        return model.infer_image(image_array[..., ::-1])

    return inference


@ModelRegistry.register(
    "m3dv2",
    default_args={
        "backbone": "ViT-Large",
        "cfg_file": None,
        "ckpt_file": None,
        "device": "mps",
    },
)
def load_m3dv2_model(
    backbone: str = "ViT-Large",
    cfg_file: str | Path | None = None,
    ckpt_file: str | Path | None = None,
    device: str = "mps",
) -> Any:
    """Load a separately installed Metric3D implementation."""

    del backbone  # The selected upstream config defines its backbone.
    torch = _require_module("m3dv2", "torch")
    cv2 = _require_module("m3dv2", "cv2")
    try:
        config_type = _required_attribute(
            "m3dv2", _require_module("m3dv2", "mmcv.utils"), "Config"
        )
    except DepthAdapterUnavailableError:
        config_type = _required_attribute(
            "m3dv2", _require_module("m3dv2", "mmengine.config"), "Config"
        )
    model_module = _require_module("m3dv2", "mono.model.monodepth_model")
    get_model = _required_attribute(
        "m3dv2", model_module, "get_configured_monodepth_model"
    )
    if cfg_file is None or ckpt_file is None:
        from panorai.path_config import get_path

        cfg_file = cfg_file or get_path("metric3d", "cfg_file")
        ckpt_file = ckpt_file or get_path("metric3d", "ckpt_file")
    if cfg_file is None or ckpt_file is None:
        raise FileNotFoundError(
            "Provide cfg_file and ckpt_file or configure metric3d paths in "
            "PANORAI_PATHS."
        )

    model = get_model(config_type.fromfile(str(cfg_file)))
    if str(ckpt_file).startswith(("http://", "https://")):
        state = torch.hub.load_state_dict_from_url(str(ckpt_file))[
            "model_state_dict"
        ]
    else:
        state = torch.load(str(ckpt_file), map_location="cpu")["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    def inference(image_array: Any) -> Any:
        height, width = image_array.shape[:2]
        mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float()
        tensor = ((tensor - mean) / std).unsqueeze(0).to(device)
        with torch.no_grad():
            predicted, _, _ = model.inference({"input": tensor})
        depth = predicted[0, 0].detach().cpu().numpy()
        return cv2.resize(depth, (width, height))

    return inference


@ModelRegistry.register(
    "dust3r",
    default_args={
        "checkpoint_path": "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
        "device": "mps",
        "size": 512,
        "return_model": False,
        "max_depth": 80,
    },
)
def load_dust3r_model(
    checkpoint_path: str = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
    device: str = "mps",
    size: int = 512,
    return_model: bool = False,
    max_depth: float = 80,
) -> Any:
    """Load a separately installed DUSt3R implementation."""

    del max_depth  # Preserved for 3.0 call compatibility; DUSt3R returns points.
    torch = _require_module("dust3r", "torch")
    image_module = _require_module("dust3r", "PIL.Image")
    cv2 = _require_module("dust3r", "cv2")
    inference_module = _require_module("dust3r", "dust3r.inference")
    model_module = _require_module("dust3r", "dust3r.model")
    images_module = _require_module("dust3r", "dust3r.utils.image")
    pairs_module = _require_module("dust3r", "dust3r.image_pairs")
    run_inference = _required_attribute("dust3r", inference_module, "inference")
    model_type = _required_attribute(
        "dust3r", model_module, "AsymmetricCroCo3DStereo"
    )
    load_images = _required_attribute("dust3r", images_module, "load_images")
    make_pairs = _required_attribute("dust3r", pairs_module, "make_pairs")
    model = model_type.from_pretrained(checkpoint_path).to(device)
    if return_model:
        return model

    def inference(image_array: Any) -> Any:
        height, width = image_array.shape[:2]
        with tempfile.NamedTemporaryFile(suffix=".png") as temporary:
            image_module.fromarray(image_array).save(temporary.name)
            images = load_images([temporary.name, temporary.name], size=size)
            pairs = make_pairs(
                images, scene_graph="complete", prefilter=None, symmetrize=True
            )
            with torch.no_grad():
                output = run_inference(pairs, model, device, batch_size=1)
        depth = output["pred1"]["pts3d"][0, :, :, -1].detach().cpu().numpy()
        return cv2.resize(depth, (width, height))

    return inference


@ModelRegistry.register("zoe", default_args={"return_model": False})
def load_zoe_model(return_model: bool = False, **kwargs: Any) -> Any:
    """Load ZoeDepth through the separately installed Transformers backend."""

    transformers = _require_module("zoe", "transformers")
    torch = _require_module("zoe", "torch")
    image_module = _require_module("zoe", "PIL.Image")
    processor_type = _required_attribute(
        "zoe", transformers, "AutoImageProcessor"
    )
    model_type = _required_attribute(
        "zoe", transformers, "ZoeDepthForDepthEstimation"
    )
    model_id = kwargs.pop("model_id", "Intel/zoedepth-nyu-kitti")
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected ZoeDepth adapter arguments: {unexpected}")
    processor = processor_type.from_pretrained(model_id)
    model = model_type.from_pretrained(model_id)
    if return_model:
        return model

    def inference(image_array: Any) -> Any:
        image = image_module.fromarray(image_array)
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        post_processed = processor.post_process_depth_estimation(
            outputs, source_sizes=[(image.height, image.width)]
        )
        return post_processed[0]["predicted_depth"].detach().cpu().numpy()

    return inference


__all__ = [
    "DepthAdapterUnavailableError",
    "load_dav2_model",
    "load_dust3r_model",
    "load_m3dv2_model",
    "load_zoe_model",
]
