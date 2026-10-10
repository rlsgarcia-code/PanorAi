#!/usr/bin/env python3
"""Run the frozen P74 proxy study with official InternImage-G weights.

The classifier is evaluated directly on one equirectangular tensor per scan.
All DCNv3 sampling is replaced by differentiable local-tangent spherical
sampling; no perspective faces, cube map, or multiface cache is created.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
import torchvision

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.spherical_fcn_cam.run_p74_imagenet_proxy_study import (  # noqa: E402
    FROZEN_SAMPLE_FILENAMES,
    PROXY_CANDIDATES,
    VISUALIZED_INDICES,
    _aggregate,
    _resolve_sample,
    _save_overlay,
    masked_cam_statistics,
    masked_class_activation_map,
    spherical_masked_average,
)
from benchmarks.spherical_multiview_depth.p74 import (  # noqa: E402
    load_native_angular_rgb,
)
from panorai.experimental.deep_learning import (  # noqa: E402
    INTERNIMAGE_G_REVISION,
    InternImageGDenseClassifier,
    SphericalConv2d,
    SphericalDCNv3,
    load_pretrained_internimage_g,
    port_internimage_to_spherical,
)

SCHEMA = "panorai-p74-internimage-g-proxy-study/v1"
INTERNIMAGE_VISUALIZED_INDICES = tuple(
    dict.fromkeys((*VISUALIZED_INDICES, 561, 653, 822))
)


def _normalize(
    rgb: np.ndarray,
    mean: list[float],
    std: list[float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = (
        torch.from_numpy(rgb.copy())
        .permute(2, 0, 1)
        .to(device=device, dtype=dtype)[None]
        / 255
    )
    center = torch.tensor(mean, device=device, dtype=dtype)[None, :, None, None]
    scale = torch.tensor(std, device=device, dtype=dtype)[None, :, None, None]
    return (tensor - center) / scale


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _actual_head_parity(
    dense_output: Any,
    classifier: InternImageGDenseClassifier,
) -> dict[str, Any]:
    tokens = dense_output.features.flatten(-2).transpose(1, 2).contiguous()
    core = classifier.core
    official = core.head(core.fc_norm(core.clip_projector(tokens)))
    difference = (official - dense_output.global_logits).abs()
    tolerance = 5e-2 if official.dtype == torch.float16 else 2e-5
    return {
        "oracle": "official attention projector, LayerNorm, and ImageNet head",
        "shape": list(official.shape),
        "dtype": str(official.dtype),
        "absolute_and_relative_tolerance": tolerance,
        "maximum_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
        "passed": bool(
            torch.allclose(
                official,
                dense_output.global_logits,
                atol=tolerance,
                rtol=tolerance,
            )
        ),
    }


def _partial_record_path(output_dir: Path, sample_id: str) -> Path:
    return output_dir / sample_id / "record.json"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_dense_proxy_evidence(
    path: Path,
    logits: torch.Tensor,
    support: torch.Tensor,
) -> dict[str, Any]:
    """Persist a compact, auditable proxy-only native lattice."""

    if logits.ndim != 4 or logits.shape[0] != 1:
        raise ValueError("logits must have shape (1,C,H,W)")
    if support.shape != (1, 1, *logits.shape[-2:]):
        raise ValueError("support must have shape (1,1,H,W) matching logits")
    indices = np.asarray(tuple(PROXY_CANDIDATES), dtype=np.int64)
    selected = logits[0, indices.tolist()].detach().float().cpu().numpy()
    valid = support[0, 0].detach().cpu().numpy().astype(bool)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema=np.asarray("panorai-internimage-proxy-evidence/v1"),
        class_indices=indices,
        class_names=np.asarray(tuple(PROXY_CANDIDATES.values())),
        logits=selected,
        support=valid,
    )
    return {
        "path": path.name,
        "shape": list(selected.shape),
        "dtype": str(selected.dtype),
        "class_count": len(indices),
        "support_shape": list(valid.shape),
    }


def _angular_distance_degrees(
    first: dict[str, float], second: dict[str, float]
) -> float:
    lon1 = math.radians(first["peak_longitude_degrees"])
    lat1 = math.radians(first["peak_latitude_degrees"])
    lon2 = math.radians(second["peak_longitude_degrees"])
    lat2 = math.radians(second["peak_latitude_degrees"])
    cosine = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(
        lat2
    ) * math.cos(lon1 - lon2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def compare_to_resnet18(
    records: list[dict[str, Any]], baseline: dict[str, Any]
) -> dict[str, Any]:
    """Compare ranks and directions without treating either model as ground truth."""

    baseline_records = {record["sample_id"]: record for record in baseline["records"]}
    current_ids = {record["sample_id"] for record in records}
    missing = sorted(current_ids - set(baseline_records))
    if missing:
        raise ValueError(f"baseline is missing sample IDs: {missing}")
    rank_deltas: dict[int, list[int]] = {index: [] for index in PROXY_CANDIDATES}
    direction_deltas: dict[int, list[float]] = {
        index: [] for index in VISUALIZED_INDICES
    }
    sample_comparisons = []
    for record in records:
        baseline_record = baseline_records[record["sample_id"]]
        current_predictions = {
            item["class_index"]: item for item in record["candidate_predictions"]
        }
        baseline_predictions = {
            item["class_index"]: item
            for item in baseline_record["candidate_predictions"]
        }
        for index in PROXY_CANDIDATES:
            rank_deltas[index].append(
                current_predictions[index]["rank"] - baseline_predictions[index]["rank"]
            )
            if (
                index in direction_deltas
                and "cam_statistics" in current_predictions[index]
                and "cam_statistics" in baseline_predictions[index]
            ):
                direction_deltas[index].append(
                    _angular_distance_degrees(
                        current_predictions[index]["cam_statistics"],
                        baseline_predictions[index]["cam_statistics"],
                    )
                )
        current_top = {item["class_index"] for item in record["top_20"]}
        baseline_top = {item["class_index"] for item in baseline_record["top_20"]}
        sample_comparisons.append(
            {
                "sample_id": record["sample_id"],
                "top_20_intersection": len(current_top & baseline_top),
                "top_20_jaccard": len(current_top & baseline_top)
                / len(current_top | baseline_top),
                "inference_time_ratio_vs_resnet18": record["inference_seconds"]
                / baseline_record["inference_seconds"],
            }
        )
    candidate_comparisons = []
    for index, name in PROXY_CANDIDATES.items():
        deltas = rank_deltas[index]
        directions = direction_deltas.get(index, [])
        candidate_comparisons.append(
            {
                "class_index": index,
                "class_name": name,
                "median_rank_delta_internimage_minus_resnet18": float(
                    np.median(deltas)
                ),
                "rank_improved_sample_count": sum(delta < 0 for delta in deltas),
                "rank_tied_sample_count": sum(delta == 0 for delta in deltas),
                "mean_peak_direction_distance_degrees": (
                    float(np.mean(directions)) if directions else None
                ),
            }
        )
    return {
        "baseline_schema": baseline.get("schema"),
        "interpretation": (
            "negative rank delta favors InternImage-G; peak distance measures "
            "inter-model directional agreement and is not localization accuracy"
        ),
        "samples": sample_comparisons,
        "candidates": candidate_comparisons,
    }


def run_study(
    image_root: Path,
    output_dir: Path,
    *,
    cache_dir: Path,
    height: int = 512,
    threads: int = 6,
    sample_ids: tuple[str, ...] = tuple(FROZEN_SAMPLE_FILENAMES),
    resume: bool = True,
    max_sampled_elements: int = 16_000_000,
    device_name: str = "auto",
    precision: str = "auto",
    accept_upstream_terms: bool = False,
    baseline_result: Path | None = None,
    save_dense_proxy_maps: bool = False,
) -> dict[str, Any]:
    if height < 64 or threads < 1:
        raise ValueError("height must be at least 64 and threads must be positive")
    unknown = sorted(set(sample_ids) - set(FROZEN_SAMPLE_FILENAMES))
    if unknown:
        raise ValueError(f"unknown frozen sample IDs: {unknown}")
    torch.set_num_threads(threads)
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    if precision == "auto":
        precision = "float32"
    dtype = {"float16": torch.float16, "float32": torch.float32}[precision]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 reference inference requires an accelerator")
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = _resolve_sample(image_root)
    sample = {sample_id: resolved[sample_id] for sample_id in sample_ids}

    loaded = load_pretrained_internimage_g(
        accept_upstream_terms=accept_upstream_terms,
        cache_dir=cache_dir,
        portable_core=True,
    )
    classifier = (
        InternImageGDenseClassifier(loaded.model).eval().to(device=device, dtype=dtype)
    )
    parameter_count = sum(parameter.numel() for parameter in classifier.parameters())
    categories = loaded.categories
    mean = list(loaded.processor.image_mean)
    std = list(loaded.processor.image_std)

    first_rgb, _ = load_native_angular_rgb(
        next(iter(sample.values())), (height, 2 * height)
    )
    planar_crop = first_rgb[:, height // 2 : height // 2 + height]
    with torch.inference_mode():
        planar_dense = classifier.forward_dense(
            _normalize(
                planar_crop,
                mean,
                std,
                device=device,
                dtype=dtype,
            )
        )
        planar_parity = _actual_head_parity(planar_dense, classifier)
    if not planar_parity["passed"]:
        raise RuntimeError("dense InternImage attention-head decomposition failed")
    del planar_dense

    port_report = port_internimage_to_spherical(
        classifier,
        max_sampled_elements=max_sampled_elements,
    )
    if port_report.remaining_dcnv3_layers:
        raise RuntimeError("DCNv3 layers remain after the spherical port")
    if port_report.spatial_layers.remaining_planar_spatial_layers:
        raise RuntimeError("planar spatial layers remain after the spherical port")
    if not port_report.parameter_identity_preserved:
        raise RuntimeError("the spherical port did not preserve parameter identity")
    classifier.eval()

    records: list[dict[str, Any]] = []
    rotation_equivariance: dict[str, Any] | None = None
    existing_result_path = output_dir / "result.json"
    if resume and existing_result_path.is_file():
        existing_result = json.loads(existing_result_path.read_text(encoding="utf-8"))
        rotation_equivariance = existing_result.get("method", {}).get(
            "rotation_equivariance"
        )
    for sample_id, path in sample.items():
        record_path = _partial_record_path(output_dir, sample_id)
        if resume and record_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            dense_record = record.get("dense_proxy_evidence")
            dense_ready = not save_dense_proxy_maps or (
                isinstance(dense_record, dict)
                and (record_path.parent / dense_record.get("path", "")).is_file()
            )
            if record.get("model_revision") == INTERNIMAGE_G_REVISION and dense_ready:
                records.append(record)
                continue

        rgb, support_np = load_native_angular_rgb(path, (height, 2 * height))
        with Image.open(path) as source_image:
            source_shape_hw = list(source_image.size[::-1])
        normalized = _normalize(
            rgb,
            mean,
            std,
            device=device,
            dtype=dtype,
        )
        _synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            dense = classifier.forward_dense(normalized)
            decomposition_error = float(
                (dense.logits.mean(dim=(-2, -1)) - dense.global_logits).abs().max()
            )
            support = torch.from_numpy(support_np).to(device=device)[None, None]
            feature_support = F.interpolate(
                support.float(), size=dense.logits.shape[-2:], mode="nearest"
            ).bool()
            scores = spherical_masked_average(dense.logits, feature_support)
            probabilities = scores.softmax(dim=1)[0]
            order = probabilities.argsort(descending=True)
            inverse_rank = torch.empty_like(order)
            inverse_rank[order] = torch.arange(
                1,
                order.numel() + 1,
                device=order.device,
            )
        _synchronize(device)
        inference_seconds = time.perf_counter() - started
        if rotation_equivariance is None:
            input_shift = normalized.shape[-1] // 8
            output_shift = dense.logits.shape[-1] // 8
            with torch.inference_mode():
                rotated = classifier.forward_dense(
                    torch.roll(normalized, input_shift, dims=-1)
                )
            _synchronize(device)
            expected_logits = torch.roll(dense.logits, output_shift, dims=-1)
            error = rotated.logits - expected_logits
            reference_rms = expected_logits.square().mean().sqrt()
            reference_global_order = dense.global_logits[0].argsort(descending=True)
            rotated_global_order = rotated.global_logits[0].argsort(descending=True)
            reference_top_20 = set(reference_global_order[:20].tolist())
            rotated_top_20 = set(rotated_global_order[:20].tolist())
            reference_global_ranks = torch.empty_like(reference_global_order)
            reference_global_ranks[reference_global_order] = torch.arange(
                reference_global_order.numel(), device=device
            )
            rotated_global_ranks = torch.empty_like(rotated_global_order)
            rotated_global_ranks[rotated_global_order] = torch.arange(
                rotated_global_order.numel(), device=device
            )
            rotation_equivariance = {
                "sample_id": sample_id,
                "longitude_rotation_degrees": 45.0,
                "input_pixel_shift": input_shift,
                "output_cell_shift": output_shift,
                "dense_max_absolute_error": float(error.abs().max()),
                "dense_mean_absolute_error": float(error.abs().mean()),
                "dense_relative_rmse": float(
                    error.square().mean().sqrt()
                    / reference_rms.clamp_min(torch.finfo(error.dtype).eps)
                ),
                "dense_cosine_similarity": float(
                    F.cosine_similarity(
                        rotated.logits.flatten(), expected_logits.flatten(), dim=0
                    )
                ),
                "global_logit_max_absolute_error": float(
                    (rotated.global_logits - dense.global_logits).abs().max()
                ),
                "global_logit_cosine_similarity": float(
                    F.cosine_similarity(
                        rotated.global_logits.flatten(),
                        dense.global_logits.flatten(),
                        dim=0,
                    )
                ),
                "global_top_1_preserved": bool(
                    reference_global_order[0] == rotated_global_order[0]
                ),
                "global_top_20_intersection": len(reference_top_20 & rotated_top_20),
                "global_mean_absolute_rank_change": float(
                    (reference_global_ranks - rotated_global_ranks).abs().float().mean()
                ),
            }
        accelerator_memory = (
            {
                "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
                "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
            }
            if device.type == "mps"
            else None
        )

        sample_dir = output_dir / sample_id
        sample_dir.mkdir(exist_ok=True)
        Image.fromarray(rgb).save(sample_dir / "input-erp.jpg", quality=92)
        dense_proxy_evidence = None
        if save_dense_proxy_maps:
            dense_proxy_evidence = write_dense_proxy_evidence(
                sample_dir / "dense-proxy-evidence.npz",
                dense.logits,
                feature_support,
            )
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
            if index in INTERNIMAGE_VISUALIZED_INDICES:
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
        top_20 = []
        for position, index in enumerate(order[:20].tolist(), start=1):
            top_item: dict[str, Any] = {
                "rank": position,
                "class_index": int(index),
                "class_name": str(categories[index]),
                "probability": float(probabilities[index]),
            }
            if position <= 5:
                top_cam = (
                    masked_class_activation_map(
                        dense.logits,
                        index,
                        feature_support,
                        output_shape=(height, 2 * height),
                    )[0]
                    .cpu()
                    .numpy()
                )
                top_filename = f"top-{position:02d}-class-{index:04d}.jpg"
                _save_overlay(rgb, top_cam, support_np, sample_dir / top_filename)
                top_item["overlay"] = top_filename
                top_item["cam_statistics"] = masked_cam_statistics(top_cam, support_np)
            top_20.append(top_item)

        record = {
            "sample_id": sample_id,
            "model_revision": INTERNIMAGE_G_REVISION,
            "source_path": str(path),
            "source_shape_hw": source_shape_hw,
            "erp_shape_hw": [height, 2 * height],
            "valid_support_fraction": float(support_np.mean()),
            "dense_logit_shape": list(dense.logits.shape),
            "dense_mean_global_max_error": decomposition_error,
            "inference_seconds": inference_seconds,
            "process_peak_rss_bytes": _peak_rss_bytes(),
            "accelerator_memory_after_inference": accelerator_memory,
            "top_20": top_20,
            "candidate_predictions": predictions,
        }
        if dense_proxy_evidence is not None:
            record["dense_proxy_evidence"] = dense_proxy_evidence
        _write_json(record_path, record)
        records.append(record)
        del dense, normalized

    comparison = None
    if baseline_result is not None:
        comparison = compare_to_resnet18(
            records,
            json.loads(baseline_result.read_text(encoding="utf-8")),
        )

    result = {
        "schema": SCHEMA,
        "method": {
            "model": "OpenGVLab InternImage-G 22K-to-1K 512",
            "reported_imagenet_1k_top1_percent": 90.1,
            "parameter_count": parameter_count,
            "repository": loaded.assets.repository,
            "revision": loaded.assets.revision,
            "checkpoint_shards": [
                {
                    "filename": item.filename,
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                }
                for item in loaded.assets.shard_specs
            ],
            "input_shape_nchw": [1, 3, height, 2 * height],
            "training_crop_hw": [loaded.image_size, loaded.image_size],
            "fcn_conversion": (
                "official multiscale fusion and attention classifier retained; "
                "per-token additive class contributions scaled so their spatial "
                "mean exactly reconstructs the global logits"
            ),
            "spatial_domain": (
                "direct ERP spherical layers with local-tangent DCNv3 sampling; "
                "no cube faces or multiface inference"
            ),
            "score_aggregation": (
                "solid-angle weighted mean over observed P74 support only"
            ),
            "planar_attention_head_parity": planar_parity,
            "rotation_equivariance": rotation_equivariance,
            "spherical_dcnv3_count": sum(
                isinstance(module, SphericalDCNv3) for module in classifier.modules()
            ),
            "spherical_convolution_count": sum(
                isinstance(module, SphericalConv2d) for module in classifier.modules()
            ),
            "spherical_port": port_report.to_dict(),
        },
        "sample_selection": {
            "policy": (
                "frozen W/G/M P74 sample declared by the preceding ResNet18 proxy study"
            ),
            "sample_ids": list(sample),
        },
        "records": records,
        "aggregate_candidate_ranks": _aggregate(records),
        "comparison_to_resnet18": comparison,
        "limitations": [
            "ImageNet classes are classification labels, not industrial component labels.",
            "Dense contribution maps are weak localization evidence, not segmentation masks.",
            "The nine-panorama purposive sample is not a P74 census or annotated test set.",
            "Each CAM is normalized independently, so color is not calibrated across classes.",
            "The pure-PyTorch reference core is much slower than the optional CUDA extension.",
        ],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "device": str(device),
            "precision": precision,
            "threads": threads,
        },
    }
    _write_json(output_dir / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--accept-upstream-terms", action="store_true")
    parser.add_argument("--baseline-result", type=Path)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument(
        "--sample-id",
        action="append",
        choices=tuple(FROZEN_SAMPLE_FILENAMES),
        help="limit a run to one or more frozen samples; repeat the option",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-sampled-elements", type=int, default=16_000_000)
    parser.add_argument("--save-dense-proxy-maps", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--precision", choices=("auto", "float16", "float32"), default="auto"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_study(
        args.image_root.resolve(),
        args.output_dir.resolve(),
        cache_dir=args.cache_dir.resolve(),
        height=args.height,
        threads=args.threads,
        sample_ids=tuple(args.sample_id or FROZEN_SAMPLE_FILENAMES),
        resume=not args.no_resume,
        max_sampled_elements=args.max_sampled_elements,
        device_name=args.device,
        precision=args.precision,
        accept_upstream_terms=args.accept_upstream_terms,
        baseline_result=(
            args.baseline_result.resolve() if args.baseline_result is not None else None
        ),
        save_dense_proxy_maps=args.save_dense_proxy_maps,
    )
    print(json.dumps(result["aggregate_candidate_ranks"], indent=2))


if __name__ == "__main__":
    main()
