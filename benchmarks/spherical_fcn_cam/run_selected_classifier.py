#!/usr/bin/env python3
"""Run Places365 ResNet18 or OpenCLIP RN50 densely on one ERP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "docs/_static/tutorials/nature-reserve-forest-erp.jpg"
DEFAULT_INPUT_LICENSE = "CC0-1.0; see docs/_static/tutorials/ATTRIBUTION.md"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.spherical_fcn_cam.run_experiment import (  # noqa: E402
    _cam_statistics,
    _load_erp,
    _peak_rss_bytes,
    _save_heatmap,
    _save_overlay,
    _sha256,
)
from panorai.experimental.deep_learning import (  # noqa: E402
    OpenCLIPRN50Dense,
    Places365ResNet18FCN,
    SphericalAvgPool2d,
    SphericalConv2d,
    SphericalMaxPool2d,
    class_activation_map,
    encode_openclip_prompts,
    load_openclip_rn50,
    load_places365_resnet18,
    port_module_with_report,
    spherical_area_average,
)


SCHEMA = "panorai-selected-spherical-classifier-experiment/v1"
MODELS = ("places365-resnet18", "openclip-rn50")
DEFAULT_OPENCLIP_PROMPTS = (
    "a photo of a bookcase",
    "a photo of a chair",
    "a photo of a door",
    "a photo of a sofa",
    "a photo of a table",
    "a photo of a window",
)


def _normalize(
    tensor: Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> Tensor:
    resolved_mean = torch.tensor(mean, dtype=tensor.dtype)[None, :, None, None]
    resolved_std = torch.tensor(std, dtype=tensor.dtype)[None, :, None, None]
    return (tensor - resolved_mean) / resolved_std


def _port_counts(model: torch.nn.Module) -> dict[str, int]:
    return {
        "spherical_convolutions": sum(
            isinstance(module, SphericalConv2d) for module in model.modules()
        ),
        "spherical_max_pools": sum(
            isinstance(module, SphericalMaxPool2d) for module in model.modules()
        ),
        "spherical_average_pools": sum(
            isinstance(module, SphericalAvgPool2d) for module in model.modules()
        ),
    }


def _save_predictions(
    *,
    rgb: np.ndarray,
    logits: Tensor,
    scores: Tensor,
    labels: Sequence[str],
    selected_indices: Sequence[int],
    model_dir: Path,
    positive_only: bool,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for position, class_index in enumerate(selected_indices, start=1):
        cam = (
            class_activation_map(
                logits,
                class_index,
                output_shape=rgb.shape[:2],
                positive_only=positive_only,
            )[0]
            .cpu()
            .numpy()
        )
        overlay_name = f"cam-{position:02d}-{class_index:04d}.png"
        heatmap_name = f"heatmap-{position:02d}-{class_index:04d}.png"
        _save_overlay(rgb, cam, model_dir / overlay_name)
        _save_heatmap(cam, model_dir / heatmap_name)
        class_score = scores[0, class_index]
        predictions.append(
            {
                "rank": int((scores[0] > class_score).sum()) + 1,
                "class_index": int(class_index),
                "label": str(labels[class_index]),
                "score": float(class_score),
                "cam_min": float(cam.min()),
                "cam_max": float(cam.max()),
                "cam_statistics": _cam_statistics(cam),
                "overlay": overlay_name,
                "heatmap": heatmap_name,
                "heatmap_encoding": "uint16-linear-[0,1]",
            }
        )
    return predictions


def _places_parity(
    model: torch.nn.Module,
) -> tuple[Places365ResNet18FCN, dict[str, Any]]:
    generator = torch.Generator().manual_seed(20261009)
    values = torch.rand((1, 3, 224, 224), generator=generator)
    with torch.inference_mode():
        expected = model(values)
    dense_model = Places365ResNet18FCN(model).eval()
    with torch.inference_mode():
        dense = dense_model.forward_dense(values)
        actual = dense_model(values, spherical_average=False)
    difference = (actual - expected).abs()
    return dense_model, {
        "oracle": "unchanged official Places365 ResNet18 classifier",
        "canonical_crop_hw": [224, 224],
        "feature_shape": list(dense.features.shape),
        "logit_map_shape": list(dense.logits.shape),
        "max_abs_logit_error": float(difference.max()),
        "allclose_rtol": 1e-5,
        "allclose_atol": 1e-6,
        "passed": bool(torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)),
    }


def _openclip_dense_parity(
    model: torch.nn.Module,
    text_features: Tensor,
) -> tuple[OpenCLIPRN50Dense, dict[str, Any]]:
    generator = torch.Generator().manual_seed(20261009)
    values = torch.rand((1, 3, 224, 224), generator=generator)
    dense_model = OpenCLIPRN50Dense(model).eval()
    with torch.inference_mode():
        features = dense_model.forward_features(values)
        actual = dense_model.project_features(features)
        tokens = features.permute(0, 2, 3, 1)
        expected = torch.nn.functional.linear(
            torch.nn.functional.linear(
                tokens,
                model.visual.attnpool.v_proj.weight,
                model.visual.attnpool.v_proj.bias,
            ),
            model.visual.attnpool.c_proj.weight,
            model.visual.attnpool.c_proj.bias,
        ).permute(0, 3, 1, 2)
        original_image = model.encode_image(values, normalize=True)
        original_logits = model.logit_scale.exp() * original_image @ text_features.T
        dense_logits = dense_model.forward_dense(values, text_features).logits
    difference = (actual - expected).abs()
    return dense_model, {
        "oracle": "the same OpenCLIP attention v_proj/c_proj applied token by token",
        "canonical_crop_hw": [224, 224],
        "feature_shape": list(features.shape),
        "dense_embedding_shape": list(actual.shape),
        "max_abs_projection_error": float(difference.max()),
        "exact_projection_reinterpretation": bool(torch.equal(actual, expected)),
        "original_global_logits": original_logits[0].tolist(),
        "dense_mean_logits": dense_logits.mean(dim=(-2, -1))[0].tolist(),
        "global_equivalence_claimed": False,
        "limitation": (
            "The dense head retains v_proj/c_proj but omits the fixed 7x7 query/key "
            "attention and absolute positional embedding. It is local image-text "
            "similarity, not the original global CLIP classifier score."
        ),
    }


def run(
    *,
    model_name: str,
    input_path: Path,
    input_license: str,
    output_dir: Path,
    erp_height: int | None,
    top_k: int,
    prompts: tuple[str, ...],
    threads: int,
    cache_dir: Path | None,
    max_sampled_elements: int,
) -> dict[str, Any]:
    torch.set_num_threads(threads)
    acquisition_started = time.perf_counter()
    if model_name == "places365-resnet18":
        loaded = load_places365_resnet18(
            accept_upstream_terms=True,
            cache_dir=cache_dir,
        )
        dense_model, parity = _places_parity(loaded.model)
        labels = loaded.categories
        text_features = None
        assets = loaded.assets.to_dict()
        mean = loaded.normalization_mean
        std = loaded.normalization_std
        dependency = {"torchvision": __import__("torchvision").__version__}
    else:
        loaded = load_openclip_rn50(
            accept_upstream_terms=True,
            cache_dir=cache_dir,
        )
        text_features = encode_openclip_prompts(
            loaded.model,
            loaded.tokenizer,
            prompts,
        )
        dense_model, parity = _openclip_dense_parity(loaded.model, text_features)
        labels = prompts
        assets = loaded.assets.to_dict()
        mean = loaded.normalization_mean
        std = loaded.normalization_std
        dependency = {"open_clip": loaded.open_clip_version}
    acquisition_seconds = time.perf_counter() - acquisition_started
    if model_name == "places365-resnet18" and not parity["passed"]:
        raise RuntimeError("Places365 planar FCN parity failed")
    if (
        model_name == "openclip-rn50"
        and not parity["exact_projection_reinterpretation"]
    ):
        raise RuntimeError("OpenCLIP dense projection reinterpretation failed")

    rgb, erp = _load_erp(input_path, erp_height)
    normalized = _normalize(erp, mean, std)
    port_report = port_module_with_report(
        dense_model,
        max_sampled_elements=max_sampled_elements,
    )
    if port_report.remaining_planar_spatial_layers:
        raise RuntimeError(
            "planar spatial layers remain: "
            + ", ".join(port_report.remaining_planar_spatial_layers)
        )

    started = time.perf_counter()
    with torch.inference_mode():
        if model_name == "places365-resnet18":
            dense = dense_model.forward_dense(normalized)
            positive_only = True
        else:
            assert text_features is not None
            dense = dense_model.forward_dense(normalized, text_features)
            positive_only = False
        scores = spherical_area_average(dense.logits)
        if model_name == "places365-resnet18":
            selected = scores.softmax(dim=1).topk(top_k, dim=1).indices[0].tolist()
        else:
            selected = list(range(len(labels)))
    inference_seconds = time.perf_counter() - started

    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(model_dir / "input-erp.jpg", quality=95)
    predictions = _save_predictions(
        rgb=rgb,
        logits=dense.logits,
        scores=scores,
        labels=labels,
        selected_indices=selected,
        model_dir=model_dir,
        positive_only=positive_only,
    )
    result = {
        "schema": SCHEMA,
        "model": model_name,
        "assets": assets,
        "planar_parity": parity,
        "erp_input_shape": list(normalized.shape),
        "input_resolution_mode": "source" if erp_height is None else "resized",
        "normalization_mean": list(mean),
        "normalization_std": list(std),
        "spherical_feature_shape": list(dense.features.shape),
        "spherical_logit_map_shape": list(dense.logits.shape),
        "spherical_port": port_report.to_dict(),
        "port_counts": _port_counts(dense_model),
        "all_features_finite": bool(torch.isfinite(dense.features).all()),
        "all_logits_finite": bool(torch.isfinite(dense.logits).all()),
        "parameter_count": sum(
            parameter.numel() for parameter in dense_model.parameters()
        ),
        "prompts": list(prompts) if model_name == "openclip-rn50" else None,
        "predictions": predictions,
        "acquisition_and_load_seconds": acquisition_seconds,
        "spherical_inference_seconds": inference_seconds,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "input": {
            "path": str(input_path),
            "sha256": _sha256(input_path),
            "license": input_license,
        },
        "implementation": {
            "runner_sha256": _sha256(Path(__file__)),
            "classification_adapter_sha256": _sha256(
                ROOT / "panorai/experimental/deep_learning/classification.py"
            ),
            "spherical_core_sha256": _sha256(
                ROOT / "panorai/image_processing/torch.py"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": "cpu",
            "threads": threads,
            **dependency,
        },
        "interpretation": (
            "Dense maps are weak directional evidence from a frozen classifier "
            "or image-text model. They are not calibrated segmentation masks."
        ),
    }
    (model_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--input-license")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--erp-height", type=int, default=224)
    parser.add_argument("--preserve-input-resolution", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--prompt",
        action="append",
        help="OpenCLIP prompt; repeat for an ordered comparison vocabulary",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-sampled-elements", type=int, default=20_000_000)
    parser.add_argument(
        "--accept-upstream-terms",
        action="store_true",
        help="confirm review of the selected model's upstream terms",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.accept_upstream_terms:
        raise ValueError("--accept-upstream-terms is required")
    if not args.preserve_input_resolution and args.erp_height < 64:
        raise ValueError("erp-height must be at least 64")
    if not 1 <= args.top_k <= 20:
        raise ValueError("top-k must be in [1, 20]")
    if args.threads < 1 or args.max_sampled_elements < 1:
        raise ValueError("threads and max-sampled-elements must be positive")
    input_path = args.input.resolve()
    if args.input_license:
        input_license = args.input_license
    elif input_path == DEFAULT_INPUT.resolve():
        input_license = DEFAULT_INPUT_LICENSE
    else:
        raise ValueError("--input-license is required for a custom input")
    prompts = tuple(args.prompt) if args.prompt else DEFAULT_OPENCLIP_PROMPTS
    if any(not prompt.strip() for prompt in prompts):
        raise ValueError("prompts must not be empty")
    result = run(
        model_name=args.model,
        input_path=input_path,
        input_license=input_license,
        output_dir=args.output_dir.resolve(),
        erp_height=None if args.preserve_input_resolution else args.erp_height,
        top_k=args.top_k,
        prompts=prompts,
        threads=args.threads,
        cache_dir=args.cache_dir,
        max_sampled_elements=args.max_sampled_elements,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
