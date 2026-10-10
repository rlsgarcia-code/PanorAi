#!/usr/bin/env python3
"""Run a support-aware ImageNet CAM proxy study on frozen P74 panoramas.

This is a development benchmark, not a semantic-segmentation evaluator.  It
keeps the native P74 data external, ports one pretrained ResNet18 to direct
spherical layers, and ranks ImageNet classes only over the observed 150-degree
polar support of each scan.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.nn import functional as F
import torchvision

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.spherical_fcn_cam.run_experiment import (  # noqa: E402
    _heat_rgb,
    _normalize,
    _planar_parity,
)
from benchmarks.spherical_multiview_depth.p74 import (  # noqa: E402
    load_native_angular_rgb,
)
from panorai.experimental.deep_learning import (  # noqa: E402
    SphericalConv2d,
    SphericalMaxPool2d,
    load_pretrained_imagenet_model,
    port_module_with_report,
)

SCHEMA = "panorai-p74-imagenet-proxy-study/v1"

FROZEN_SAMPLE_FILENAMES = {
    "W050": "P-74+MD-04_concluido_408+W_050_rgb.png",
    "W121": "P-74+MD-04_concluido_408+W_121_rgb.png",
    "W150": "P-74+MD-04_concluido_408+W_150_rgb.png",
    "G046": "P-74+MD-05_concluido_326+G046_rgb.png",
    "G100": "P-74+MD-05_concluido_326+G100_rgb.png",
    "G143": "P-74+MD-05_concluido_326+150_G143_rgb.png",
    "M014": "P-74+MD-08_missing_files+M-014_rgb.png",
    "M040": "P-74+MD-08_missing_files+M-040_rgb.png",
    "M069": "P-74+MD-08_missing_files+M-069_rgb.png",
}

# The industrial-name subset was declared before prediction review; unexpected
# recurrent top classes were then added for explanatory visual audit. Names are
# hypotheses/proxy candidates, not ground truth or a preregistered metric set.
PROXY_CANDIDATES = {
    409: "analog clock",
    421: "bannister",
    426: "barometer",
    427: "barrel",
    438: "beaker",
    466: "bullet train",
    489: "chainlink fence",
    506: "coil",
    517: "crane",
    536: "dock",
    540: "drilling platform",
    545: "electric fan",
    556: "fire screen",
    561: "forklift",
    562: "fountain",
    571: "gas pump",
    581: "grille",
    632: "loudspeaker",
    653: "milk can",
    686: "oil filter",
    687: "organ",
    704: "parking meter",
    712: "Petri dish",
    718: "pier",
    727: "planetarium",
    733: "pole",
    743: "prison",
    753: "radiator",
    758: "reel",
    771: "safe",
    782: "screen",
    799: "sliding door",
    818: "spotlight",
    821: "steel arch bridge",
    822: "steel drum",
    839: "suspension bridge",
    844: "switch",
    847: "tank",
    854: "theater curtain",
    877: "turnstile",
    884: "vault",
    897: "washer",
    900: "water tower",
    904: "window screen",
    905: "window shade",
}

# Compact visual audit set; all candidate ranks remain in result.json.
VISUALIZED_INDICES = (
    409,
    421,
    427,
    438,
    466,
    506,
    545,
    556,
    562,
    686,
    687,
    743,
    818,
    821,
    847,
    884,
    897,
)


def spherical_masked_average(logits: Tensor, support: Tensor) -> Tensor:
    """Solid-angle average NCHW logits over a boolean N1HW support mask."""

    if logits.ndim != 4:
        raise ValueError("logits must use NCHW layout")
    if support.ndim != 4 or support.shape[1] != 1:
        raise ValueError("support must use N1HW layout")
    if support.shape[0] != logits.shape[0] or support.shape[-2:] != logits.shape[-2:]:
        raise ValueError("support and logits must share batch and spatial shapes")
    height = logits.shape[-2]
    latitude = (
        math.pi / 2.0
        - (torch.arange(height, device=logits.device, dtype=logits.dtype) + 0.5)
        * math.pi
        / height
    )
    weights = latitude.cos().clamp_min(0)[None, None, :, None]
    weights = weights * support.to(dtype=logits.dtype)
    denominator = weights.sum(dim=(-2, -1))
    if torch.any(denominator <= 0):
        raise ValueError("support must contain at least one valid sample")
    return (logits * weights).sum(dim=(-2, -1)) / denominator


def masked_class_activation_map(
    logits: Tensor,
    class_index: int,
    support: Tensor,
    *,
    output_shape: tuple[int, int],
) -> Tensor:
    """Normalize positive class evidence over valid support and zero invalid ERP pixels."""

    if logits.ndim != 4 or logits.shape[0] != 1:
        raise ValueError("this benchmark expects one NCHW sample")
    if not 0 <= class_index < logits.shape[1]:
        raise ValueError("class index is outside the logits channel range")
    maps = logits[:, class_index : class_index + 1].relu()
    maps = F.interpolate(maps, size=output_shape, mode="bilinear", align_corners=False)
    valid = F.interpolate(
        support.to(dtype=maps.dtype), size=output_shape, mode="nearest"
    ).bool()
    values = maps[valid]
    if values.numel() == 0:
        raise ValueError("support must contain at least one valid sample")
    minimum = values.min()
    scale = (values.max() - minimum).clamp_min(torch.finfo(maps.dtype).eps)
    normalized = (maps - minimum) / scale
    return torch.where(valid, normalized.clamp(0, 1), torch.zeros_like(normalized))[
        :, 0
    ]


def _save_overlay(
    rgb: np.ndarray, cam: np.ndarray, support: np.ndarray, path: Path
) -> None:
    base = rgb.astype(np.float32) / 255.0
    overlay = np.clip(0.58 * base + 0.42 * _heat_rgb(cam), 0.0, 1.0)
    overlay[~support] = 0.35 * base[~support]
    Image.fromarray(np.round(255 * overlay).astype(np.uint8)).save(path, quality=92)


def masked_cam_statistics(cam: np.ndarray, support: np.ndarray) -> dict[str, float]:
    """Summarize a normalized ERP CAM without counting unsupported pixels."""

    if cam.ndim != 2 or support.shape != cam.shape:
        raise ValueError("cam and support must be matching HW arrays")
    if not np.isfinite(cam).all() or not support.any():
        raise ValueError("cam must be finite and support must be nonempty")
    height, width = cam.shape
    latitude = np.pi / 2 - (np.arange(height) + 0.5) * np.pi / height
    weights = np.cos(latitude).clip(min=0)[:, None] * support
    normalization = weights.sum()
    supported_cam = np.where(support, cam, -np.inf)
    peak_row, peak_column = np.unravel_index(np.argmax(supported_cam), cam.shape)
    return {
        "spherical_mean": float((cam * weights).sum() / normalization),
        "spherical_fraction_ge_0_5": float(
            ((cam >= 0.5) * weights).sum() / normalization
        ),
        "peak_longitude_degrees": float(
            ((peak_column + 0.5) / width * 2 * np.pi - np.pi) * 180 / np.pi
        ),
        "peak_latitude_degrees": float(latitude[peak_row] * 180 / np.pi),
    }


def _resolve_sample(image_root: Path) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for sample_id, filename in FROZEN_SAMPLE_FILENAMES.items():
        matches = list(image_root.rglob(filename))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one {filename!r} below {image_root}, found {len(matches)}"
            )
        resolved[sample_id] = matches[0]
    return resolved


def _aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranks: dict[int, list[int]] = defaultdict(list)
    probabilities: dict[int, list[float]] = defaultdict(list)
    for record in records:
        for prediction in record["candidate_predictions"]:
            index = prediction["class_index"]
            ranks[index].append(prediction["rank"])
            probabilities[index].append(prediction["probability"])
    aggregate = []
    for index in PROXY_CANDIDATES:
        values = ranks[index]
        aggregate.append(
            {
                "class_index": index,
                "class_name": PROXY_CANDIDATES[index],
                "median_rank": float(np.median(values)),
                "best_rank": min(values),
                "worst_rank": max(values),
                "top_20_coverage": sum(value <= 20 for value in values),
                "top_100_coverage": sum(value <= 100 for value in values),
                "mean_probability": float(np.mean(probabilities[index])),
                "ranks_by_sample": {
                    record["sample_id"]: values[position]
                    for position, record in enumerate(records)
                },
            }
        )
    return sorted(
        aggregate, key=lambda item: (item["median_rank"], item["class_index"])
    )


def run_study(
    image_root: Path,
    output_dir: Path,
    *,
    height: int = 512,
    threads: int = 6,
) -> dict[str, Any]:
    if height < 64 or threads < 1:
        raise ValueError("height must be at least 64 and threads must be positive")
    torch.set_num_threads(threads)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = _resolve_sample(image_root)

    loaded = load_pretrained_imagenet_model("resnet18", progress=True)
    fcn, parity = _planar_parity(loaded.model, "resnet18")
    if not parity["passed"]:
        raise RuntimeError("planar FCN conversion failed")
    port_report = port_module_with_report(fcn)
    if port_report.remaining_planar_spatial_layers:
        raise RuntimeError("planar spatial layers remain after spherical port")
    fcn.eval()
    categories = loaded.weights.meta["categories"]

    records: list[dict[str, Any]] = []
    for sample_id, path in sample.items():
        rgb, support_np = load_native_angular_rgb(path, (height, 2 * height))
        with Image.open(path) as source_image:
            source_shape_hw = list(source_image.size[::-1])
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None] / 255
        normalized = _normalize(tensor, loaded.weights)
        started = time.perf_counter()
        with torch.inference_mode():
            dense = fcn.forward_dense(normalized)
            support = torch.from_numpy(support_np)[None, None]
            feature_support = F.interpolate(
                support.float(), size=dense.logits.shape[-2:], mode="nearest"
            ).bool()
            scores = spherical_masked_average(dense.logits, feature_support)
            probabilities = scores.softmax(dim=1)[0]
            order = probabilities.argsort(descending=True)
            inverse_rank = torch.empty_like(order)
            inverse_rank[order] = torch.arange(1, order.numel() + 1)
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(exist_ok=True)
        Image.fromarray(rgb).save(sample_dir / "input-erp.jpg", quality=92)
        predictions = []
        for index, declared_name in PROXY_CANDIDATES.items():
            if str(categories[index]) != declared_name:
                raise RuntimeError(
                    f"ImageNet category drift at {index}: {categories[index]!r}"
                )
            item: dict[str, Any] = {
                "class_index": index,
                "class_name": declared_name,
                "rank": int(inverse_rank[index]),
                "score": float(scores[0, index]),
                "probability": float(probabilities[index]),
            }
            if index in VISUALIZED_INDICES:
                with torch.inference_mode():
                    cam = (
                        masked_class_activation_map(
                            dense.logits,
                            index,
                            feature_support,
                            output_shape=(height, 2 * height),
                        )[0]
                        .cpu()
                        .numpy()
                    )
                filename = f"cam-{index:04d}-{declared_name.replace(' ', '-')}.jpg"
                _save_overlay(rgb, cam, support_np, sample_dir / filename)
                item["overlay"] = filename
                item["cam_statistics"] = masked_cam_statistics(cam, support_np)
            predictions.append(item)
        records.append(
            {
                "sample_id": sample_id,
                "source_path": str(path),
                "source_shape_hw": source_shape_hw,
                "erp_shape_hw": [height, 2 * height],
                "valid_support_fraction": float(support_np.mean()),
                "dense_logit_shape": list(dense.logits.shape),
                "inference_seconds": time.perf_counter() - started,
                "top_20": [
                    {
                        "rank": position,
                        "class_index": int(index),
                        "class_name": str(categories[index]),
                        "probability": float(probabilities[index]),
                    }
                    for position, index in enumerate(order[:20].tolist(), start=1)
                ],
                "candidate_predictions": predictions,
            }
        )

    result = {
        "schema": SCHEMA,
        "method": {
            "model": "torchvision ResNet18 DEFAULT ImageNet-1K",
            "weights": str(loaded.weights),
            "checkpoint_sha256": loaded.checkpoint.sha256,
            "fcn_conversion": "global average removed; final Linear reused as 1x1 convolution",
            "spatial_domain": "direct ERP spherical layers; no cube faces or multiface inference",
            "score_aggregation": "solid-angle weighted mean over observed P74 support only",
            "input_shape_nchw": [1, 3, height, 2 * height],
            "planar_parity": parity,
            "spherical_convolution_count": sum(
                isinstance(module, SphericalConv2d) for module in fcn.modules()
            ),
            "spherical_pool_count": sum(
                isinstance(module, SphericalMaxPool2d) for module in fcn.modules()
            ),
            "spherical_port": port_report.to_dict(),
        },
        "sample_selection": {
            "policy": "three frozen panoramas from each P74 W/G/M family, chosen before ImageNet prediction review",
            "sample_ids": list(sample),
        },
        "records": records,
        "aggregate_candidate_ranks": _aggregate(records),
        "limitations": [
            "ImageNet classes are classification labels, not industrial component labels.",
            "CAMs are weak localization evidence and are not calibrated segmentation masks.",
            "The nine-panorama purposive sample is not a P74 census or annotated test set.",
            "Each CAM is normalized independently, so color intensity is not comparable across classes.",
        ],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "device": "cpu",
            "threads": threads,
        },
    }
    with (output_dir / "result.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--threads", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_study(
        args.image_root.resolve(),
        args.output_dir.resolve(),
        height=args.height,
        threads=args.threads,
    )
    print(json.dumps(result["aggregate_candidate_ranks"], indent=2))


if __name__ == "__main__":
    main()
