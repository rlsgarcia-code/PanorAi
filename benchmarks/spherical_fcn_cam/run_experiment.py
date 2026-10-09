#!/usr/bin/env python3
"""Run pretrained ImageNet FCN/CAM inference on a real CC0 ERP panorama."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import Tensor
import torchvision

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "docs/_static/tutorials/nature-reserve-forest-erp.jpg"
DEFAULT_INPUT_LICENSE = "CC0-1.0; see docs/_static/tutorials/ATTRIBUTION.md"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panorai.experimental.deep_learning import (  # noqa: E402
    ImageNetFCN,
    SUPPORTED_IMAGENET_MODELS,
    SphericalConv2d,
    SphericalMaxPool2d,
    class_activation_map,
    load_pretrained_imagenet_model,
    port_module_with_report,
    spherical_area_average,
)

SCHEMA = "panorai-spherical-fcn-cam-experiment/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _load_erp(path: Path, height: int | None) -> tuple[np.ndarray, Tensor]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if height is not None:
            image = image.resize((2 * height, height), Image.Resampling.LANCZOS)
        rgb = np.asarray(image).copy()
    if rgb.shape[1] != 2 * rgb.shape[0]:
        raise ValueError(f"input must be a 2:1 ERP; received {rgb.shape[:2]}")
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 255.0
    return rgb, tensor


def _normalize(tensor: Tensor, weights: Any) -> Tensor:
    transform = weights.transforms()
    mean = torch.tensor(transform.mean, dtype=tensor.dtype)[None, :, None, None]
    std = torch.tensor(transform.std, dtype=tensor.dtype)[None, :, None, None]
    return (tensor - mean) / std


def _heat_rgb(cam: np.ndarray) -> np.ndarray:
    clipped = np.clip(cam, 0.0, 1.0)
    red = np.clip(1.5 * clipped, 0.0, 1.0)
    green = np.clip(1.5 - 3.0 * np.abs(clipped - 0.5), 0.0, 1.0)
    blue = np.clip(1.5 * (1.0 - clipped), 0.0, 1.0)
    return np.stack((red, green, blue), axis=-1)


def _save_overlay(rgb: np.ndarray, cam: np.ndarray, path: Path) -> None:
    base = rgb.astype(np.float32) / 255.0
    overlay = np.clip(0.58 * base + 0.42 * _heat_rgb(cam), 0.0, 1.0)
    Image.fromarray(np.round(255.0 * overlay).astype(np.uint8)).save(path)


def _save_heatmap(cam: np.ndarray, path: Path) -> None:
    """Persist a normalized CAM compactly without any display color map."""

    if cam.ndim != 2 or not np.isfinite(cam).all():
        raise ValueError("cam must be a finite HW array")
    encoded = np.round(np.clip(cam, 0.0, 1.0) * 65535.0).astype(np.uint16)
    Image.fromarray(encoded, mode="I;16").save(path)


def _cam_statistics(cam: np.ndarray) -> dict[str, float]:
    """Summarize a normalized ERP CAM with solid-angle weighting."""

    if cam.ndim != 2 or not np.isfinite(cam).all():
        raise ValueError("cam must be a finite HW array")
    height, width = cam.shape
    latitude = np.pi / 2.0 - (np.arange(height) + 0.5) * np.pi / height
    row_weights = np.cos(latitude).clip(min=0.0)
    normalization = float(width * row_weights.sum())
    return {
        "mean": float(cam.mean()),
        "standard_deviation": float(cam.std()),
        "p90": float(np.quantile(cam, 0.9)),
        "spherical_mean": float((cam * row_weights[:, None]).sum() / normalization),
        "spherical_fraction_ge_0_5": float(
            ((cam >= 0.5) * row_weights[:, None]).sum() / normalization
        ),
    }


def _planar_parity(
    model: torch.nn.Module, architecture: str
) -> tuple[ImageNetFCN, dict[str, Any]]:
    generator = torch.Generator().manual_seed(20261008)
    planar = torch.rand((1, 3, 224, 224), generator=generator)
    with torch.inference_mode():
        expected = model(planar)
    fcn = ImageNetFCN(model, architecture).eval()
    with torch.inference_mode():
        dense = fcn.forward_dense(planar)
        actual = fcn(planar, spherical_average=False)
    difference = (actual - expected).abs()
    denominator = expected.abs().clamp_min(torch.finfo(expected.dtype).eps)
    return fcn, {
        "canonical_crop_hw": [224, 224],
        "feature_shape": list(dense.features.shape),
        "logit_map_shape": list(dense.logits.shape),
        "max_abs_logit_error": float(difference.max()),
        "max_relative_logit_error": float((difference / denominator).max()),
        "allclose_rtol": 1e-5,
        "allclose_atol": 1e-6,
        "passed": bool(torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)),
    }


def _selected_class_indices(
    top_indices: list[int], requested_class_indices: tuple[int, ...]
) -> list[int]:
    selected = list(top_indices)
    selected.extend(index for index in requested_class_indices if index not in selected)
    return selected


def run_model(
    model_name: str,
    *,
    input_path: Path,
    output_dir: Path,
    erp_height: int | None,
    top_k: int,
    requested_class_indices: tuple[int, ...] = (),
    threads: int,
    input_license: str,
) -> dict[str, Any]:
    torch.set_num_threads(threads)
    download_started = time.perf_counter()
    loaded = load_pretrained_imagenet_model(model_name, progress=True)
    model = loaded.model
    weights = loaded.weights
    checkpoint = loaded.checkpoint
    download_seconds = time.perf_counter() - download_started

    fcn, planar_parity = _planar_parity(model, model_name)
    if not planar_parity["passed"]:
        raise RuntimeError(f"planar FCN conversion failed for {model_name}")

    rgb, erp = _load_erp(input_path, erp_height)
    normalized = _normalize(erp, weights)
    output_shape = tuple(int(value) for value in normalized.shape[-2:])
    port_report = port_module_with_report(fcn)
    if port_report.remaining_planar_spatial_layers:
        raise RuntimeError(
            "conventional spatial layers remain after spherical port: "
            + ", ".join(port_report.remaining_planar_spatial_layers)
        )
    spherical_convolutions = sum(
        isinstance(module, SphericalConv2d) for module in fcn.modules()
    )
    spherical_pools = sum(
        isinstance(module, SphericalMaxPool2d) for module in fcn.modules()
    )

    started = time.perf_counter()
    with torch.inference_mode():
        dense = fcn.forward_dense(normalized)
        scores = spherical_area_average(dense.logits)
        probabilities = scores.softmax(dim=1)
        _, top_indices = probabilities.topk(top_k, dim=1)
        selected_indices = _selected_class_indices(
            [int(index) for index in top_indices[0]], requested_class_indices
        )
        cams = [
            class_activation_map(
                dense.logits,
                class_index,
                output_shape=output_shape,
            )[0]
            for class_index in selected_indices
        ]
    inference_seconds = time.perf_counter() - started

    categories = weights.meta["categories"]
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(model_dir / "input-erp.jpg", quality=95)
    predictions: list[dict[str, Any]] = []
    top_index_set = {int(index) for index in top_indices[0]}
    requested_index_set = set(requested_class_indices)
    for position, (class_index, cam) in enumerate(zip(selected_indices, cams), start=1):
        class_name = str(categories[class_index])
        probability = probabilities[0, class_index]
        rank = int((probabilities[0] > probability).sum()) + 1
        filename = f"cam-{position:02d}-{class_index:04d}.png"
        heatmap_filename = f"heatmap-{position:02d}-{class_index:04d}.png"
        cam_array = cam.cpu().numpy()
        _save_overlay(rgb, cam_array, model_dir / filename)
        _save_heatmap(cam_array, model_dir / heatmap_filename)
        predictions.append(
            {
                "rank": rank,
                "class_index": class_index,
                "class_name": class_name,
                "probability": float(probability),
                "score": float(scores[0, class_index]),
                "cam_min": float(cam_array.min()),
                "cam_max": float(cam_array.max()),
                "cam_statistics": _cam_statistics(cam_array),
                "overlay": filename,
                "heatmap": heatmap_filename,
                "heatmap_encoding": "uint16-linear-[0,1]",
                "selection_reasons": [
                    reason
                    for reason, selected in (
                        ("top_k", class_index in top_index_set),
                        ("requested", class_index in requested_index_set),
                    )
                    if selected
                ],
            }
        )

    transforms = weights.transforms()
    result = {
        "schema": SCHEMA,
        "implementation": {
            "runner_sha256": _sha256(Path(__file__)),
            "spherical_core_sha256": _sha256(
                ROOT / "panorai/image_processing/torch.py"
            ),
            "fcn_adapter_sha256": _sha256(
                ROOT / "panorai/experimental/deep_learning/fcn.py"
            ),
        },
        "model": model_name,
        "weights": str(weights),
        "weights_url": weights.url,
        "weights_cache_path": checkpoint.cache_path,
        "weights_sha256": checkpoint.sha256,
        "weights_size_bytes": checkpoint.size_bytes,
        "weights_previously_cached": checkpoint.previously_cached,
        "documented_resize_size": list(transforms.resize_size),
        "documented_crop_size": list(transforms.crop_size),
        "documented_min_size": list(weights.meta["min_size"]),
        "normalization_mean": list(transforms.mean),
        "normalization_std": list(transforms.std),
        "parameter_count": sum(parameter.numel() for parameter in fcn.parameters()),
        "planar_fcn_parity": planar_parity,
        "erp_input_shape": list(normalized.shape),
        "input_resolution_mode": "source" if erp_height is None else "resized",
        "spherical_feature_shape": list(dense.features.shape),
        "spherical_logit_map_shape": list(dense.logits.shape),
        "spherical_convolution_count": spherical_convolutions,
        "spherical_pool_count": spherical_pools,
        "spherical_port": port_report.to_dict(),
        "all_features_finite": bool(torch.isfinite(dense.features).all()),
        "all_logits_finite": bool(torch.isfinite(dense.logits).all()),
        "download_and_load_seconds": download_seconds,
        "spherical_inference_seconds": inference_seconds,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "requested_class_indices": list(requested_class_indices),
        "predictions": predictions,
        "input": {
            "path": str(input_path),
            "sha256": _sha256(input_path),
            "license": input_license,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "device": "cpu",
            "threads": threads,
        },
        "interpretation": (
            "The dense class score is weak localization evidence from an ImageNet "
            "classifier, not a semantic-segmentation probability or calibrated mask."
        ),
    }
    with (model_dir / "result.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SUPPORTED_IMAGENET_MODELS, required=True)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--input-license",
        help=(
            "license/provenance statement for a custom input; the tracked "
            "tutorial ERP is recognized as CC0 automatically"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--erp-height", type=int, default=224)
    parser.add_argument(
        "--preserve-input-resolution",
        action="store_true",
        help="run on the source ERP lattice instead of resizing to --erp-height",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--class-index",
        type=int,
        action="append",
        default=[],
        help=(
            "also save the CAM for this zero-based ImageNet-1K class index; "
            "repeat for multiple semantic-pair targets"
        ),
    )
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.preserve_input_resolution and args.erp_height < 64:
        raise ValueError("erp-height must be at least 64")
    if not 1 <= args.top_k <= 20:
        raise ValueError("top-k must be in [1, 20]")
    if any(index < 0 or index >= 1000 for index in args.class_index):
        raise ValueError("class-index must be in [0, 999]")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    input_path = args.input.resolve()
    if args.input_license:
        input_license = args.input_license
    elif input_path == DEFAULT_INPUT.resolve():
        input_license = DEFAULT_INPUT_LICENSE
    else:
        raise ValueError("--input-license is required for a custom input")
    result = run_model(
        args.model,
        input_path=input_path,
        output_dir=args.output_dir.resolve(),
        erp_height=None if args.preserve_input_resolution else args.erp_height,
        top_k=args.top_k,
        requested_class_indices=tuple(dict.fromkeys(args.class_index)),
        threads=args.threads,
        input_license=input_license,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
