#!/usr/bin/env python3
"""Train latent-conditioned convex DA3 face fusion without ground-truth depth."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.spherical_monocular_depth.da3_learned_fusion import (  # noqa: E402
    INTERFACE,
    LOCAL_INPUT_NAMES,
    LatentConditionedDepthFusion,
    build_aligned_ray_bundle,
    fusion_parameter_count,
    save_fusion,
)
from benchmarks.spherical_monocular_depth.da3_spherical_attention import (  # noqa: E402
    infer_sparse_attention_erp,
)
from benchmarks.spherical_monocular_depth.da3_tangent import (  # noqa: E402
    load_da3metric_large,
)
from benchmarks.spherical_monocular_depth.protocol import (  # noqa: E402
    json_ready,
    sha256,
)
from benchmarks.spherical_monocular_depth.vit_tangent import (  # noqa: E402
    make_native_tangent_plan,
)
from benchmarks.spherical_multiview_depth.p74 import (  # noqa: E402
    load_native_angular_rgb,
)


SCHEMA = "panorai-p74-da3-latent-conditioned-fusion-training/v1"
P74_PREFIX = "P-74+MD-04_concluido_408+"
DEFAULT_TRAIN_IDS = ("W_050", "W_100", "W_150")
DEFAULT_VALIDATION_IDS = ("W_120",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p74-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-ids", nargs="+", default=list(DEFAULT_TRAIN_IDS))
    parser.add_argument(
        "--validation-ids", nargs="+", default=list(DEFAULT_VALIDATION_IDS)
    )
    parser.add_argument("--erp-height", type=int, default=4128)
    parser.add_argument("--erp-width", type=int, default=8256)
    parser.add_argument("--view-height", type=int, default=812)
    parser.add_argument("--view-width", type=int, default=1400)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--maximum-sources-per-target", type=int, default=6)
    parser.add_argument("--samples-per-panorama", type=int, default=160_000)
    parser.add_argument("--maximum-logit-correction", type=float, default=2.0)
    parser.add_argument("--prior-regularization", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p74-row-chunk", type=int, default=32)
    parser.add_argument("--reuse-samples", action="store_true")
    parser.add_argument(
        "--skip-existing-samples",
        action="store_true",
        help="During generation, reuse any already completed sample file.",
    )
    return parser.parse_args()


def _full_id(short_id: str) -> str:
    return short_id if short_id.startswith("P-74+") else P74_PREFIX + short_id


def _sample_path(args: argparse.Namespace, split: str, panorama_id: str) -> Path:
    return args.output / "samples" / f"{split}-{panorama_id}.npz"


def _save_bundle(path: Path, bundle: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    supported = bundle.validity.any(axis=1)
    np.savez_compressed(
        path,
        depths=bundle.depths[supported],
        local_features=bundle.local_features[supported],
        token_latents=bundle.token_latents[supported],
        token_gate_strength=bundle.token_gate_strength[supported],
        gaussian_weights=bundle.gaussian_weights[supported],
        validity=bundle.validity[supported],
        source_indices=bundle.source_indices[supported],
        flat_indices=bundle.flat_indices[supported],
    )


def _load_samples(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        values = {name: np.asarray(payload[name]) for name in payload.files}
    supported = values["validity"].any(axis=1)
    return {name: value[supported] for name, value in values.items()}


def _generate_split(
    ids: list[str],
    *,
    split: str,
    args: argparse.Namespace,
    model: Any,
    plan: Any,
    rng: np.random.Generator,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, Any]]]:
    samples = []
    reports = []
    feature_shape = (plan.view_shape_hw[0] // 14, plan.view_shape_hw[1] // 14)
    for short_id in ids:
        panorama_id = _full_id(short_id)
        path = _sample_path(args, split, panorama_id)
        if args.skip_existing_samples and path.is_file():
            sample = _load_samples(path)
            samples.append(sample)
            reports.append(
                {
                    "panorama_id": panorama_id,
                    "sample_path": str(path),
                    "sample_sha256": sha256(path),
                    "sample_count": int(sample["depths"].shape[0]),
                    "depth_ground_truth_opened": False,
                    "reused_existing_generation_sample": True,
                }
            )
            continue
        rgb_path = args.p74_root / "images" / f"{panorama_id}_rgb.png"
        rgb, support = load_native_angular_rgb(
            rgb_path, plan.erp_shape_hw, row_chunk=args.p74_row_chunk
        )
        supported_rows = np.flatnonzero(support[:, 0])
        if not np.array_equal(supported_rows, np.arange(supported_rows.size)):
            raise RuntimeError("P74 support is expected to be a contiguous row prefix")
        supported_pixel_count = supported_rows.size * plan.erp_shape_hw[1]
        flat_indices = rng.choice(
            supported_pixel_count, size=args.samples_per_panorama, replace=False
        )
        scratch = args.output / "scratch" / split / panorama_id
        _, _, inference_report = infer_sparse_attention_erp(
            model,
            rgb,
            support,
            plan,
            scratch_dir=scratch,
            device=args.device,
            alpha=1.0,
            maximum_sources_per_target=args.maximum_sources_per_target,
            position_transport=True,
            keep_feature_stores=True,
            gate_checkpoint=args.gate_checkpoint,
        )
        stores = inference_report["stores"]
        tangent_depths = np.load(stores["tangent_depth_path"], mmap_mode="r")
        token_gates = np.memmap(
            stores["gate_map_path"],
            dtype="float32",
            mode="r",
            shape=(len(plan.centers_lat_lon_deg), *feature_shape),
        )
        token_latents = np.memmap(
            stores["latent_map_path"],
            dtype="float32",
            mode="r",
            shape=(len(plan.centers_lat_lon_deg), *feature_shape, 8),
        )
        bundle = build_aligned_ray_bundle(
            plan,
            tangent_depths,
            token_gates,
            token_latents,
            flat_indices,
            maximum_contributions=args.maximum_sources_per_target,
        )
        _save_bundle(path, bundle)
        saved_sample = _load_samples(path)
        samples.append(saved_sample)
        reports.append(
            {
                "panorama_id": panorama_id,
                "rgb_path": str(rgb_path),
                "rgb_sha256": sha256(rgb_path),
                "sample_path": str(path),
                "sample_sha256": sha256(path),
                "sample_count": int(saved_sample["depths"].shape[0]),
                "contribution_histogram": {
                    str(count): int((bundle.validity.sum(axis=1) == count).sum())
                    for count in range(1, bundle.validity.shape[1] + 1)
                },
                "depth_ground_truth_opened": False,
                "npz_opened": False,
                "inference": inference_report,
            }
        )
        del tangent_depths, token_gates, token_latents, bundle, rgb, support
        for value in stores.values():
            if isinstance(value, str) and Path(value).is_file():
                Path(value).unlink()
    return samples, reports


def _reuse_split(
    ids: list[str], *, split: str, args: argparse.Namespace, **_: Any
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, Any]]]:
    samples = []
    reports = []
    for short_id in ids:
        panorama_id = _full_id(short_id)
        path = _sample_path(args, split, panorama_id)
        sample = _load_samples(path)
        samples.append(sample)
        reports.append(
            {
                "panorama_id": panorama_id,
                "sample_path": str(path),
                "sample_sha256": sha256(path),
                "sample_count": int(sample["depths"].shape[0]),
                "depth_ground_truth_opened": False,
                "reused_samples": True,
            }
        )
    return samples, reports


def _concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        name: np.concatenate([part[name] for part in parts], axis=0)
        for name in parts[0]
        if name not in {"flat_indices", "source_indices"}
    }


def _weighted_median_log_depth(data: dict[str, np.ndarray]) -> np.ndarray:
    log_depth = np.log(np.maximum(data["depths"], 1e-6))
    weights = np.where(data["validity"], data["gaussian_weights"], 0.0)
    order = np.argsort(log_depth, axis=1)
    sorted_values = np.take_along_axis(log_depth, order, axis=1)
    sorted_weights = np.take_along_axis(weights, order, axis=1)
    cumulative = np.cumsum(sorted_weights, axis=1)
    threshold = 0.5 * sorted_weights.sum(axis=1, keepdims=True)
    median_index = (cumulative < threshold).sum(axis=1)
    return sorted_values[np.arange(sorted_values.shape[0]), median_index].astype(
        np.float32
    )


def _loss_and_metrics(
    fusion: Any,
    data: dict[str, Any],
    target: Any,
    *,
    prior_regularization: float,
    batch_size: int,
    train: bool,
    optimizer: Any | None = None,
    permutation: Any | None = None,
) -> dict[str, float]:
    import torch
    from torch.nn import functional

    indices = (
        permutation
        if permutation is not None
        else torch.arange(target.shape[0], device=target.device)
    )
    loss_sum = 0.0
    robust_sum = 0.0
    kl_sum = 0.0
    count = 0
    fusion.train(train)
    context = torch.enable_grad() if train else torch.inference_mode()
    with context:
        for start in range(0, indices.numel(), batch_size):
            selected = indices[start : start + batch_size]
            if train:
                optimizer.zero_grad(set_to_none=True)
            fused, learned_weights = fusion(
                data["depths"][selected],
                data["local_features"][selected],
                data["token_latents"][selected],
                data["token_gate_strength"][selected],
                data["gaussian_weights"][selected],
                data["validity"][selected],
            )
            robust = functional.smooth_l1_loss(
                torch.log(fused.clamp_min(1e-6)),
                target[selected],
                beta=0.05,
            )
            prior = torch.where(
                data["validity"][selected],
                data["gaussian_weights"][selected],
                0.0,
            )
            prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            kl = torch.sum(
                learned_weights
                * (
                    torch.log(learned_weights.clamp_min(1e-12))
                    - torch.log(prior.clamp_min(1e-12))
                ),
                dim=-1,
            ).mean()
            loss = robust + prior_regularization * kl
            if train:
                loss.backward()
                optimizer.step()
            size = selected.numel()
            loss_sum += float(loss.detach().item()) * size
            robust_sum += float(robust.detach().item()) * size
            kl_sum += float(kl.detach().item()) * size
            count += size
    return {
        "loss": loss_sum / count,
        "robust_log_consensus_loss": robust_sum / count,
        "kl_from_gaussian_prior": kl_sum / count,
    }


def _to_torch(data: dict[str, np.ndarray], device: str) -> dict[str, Any]:
    import torch

    names = (
        "depths",
        "local_features",
        "token_latents",
        "token_gate_strength",
        "gaussian_weights",
        "validity",
    )
    return {name: torch.from_numpy(data[name]).to(device) for name in names}


def main() -> int:
    args = parse_args()
    if set(map(_full_id, args.train_ids)) & set(map(_full_id, args.validation_ids)):
        raise ValueError("training and validation panoramas must be disjoint")
    args.output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()

    import torch

    torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    plan = make_native_tangent_plan(
        (args.erp_height, args.erp_width),
        view_shape_hw=(args.view_height, args.view_width),
        overlap_fraction=args.overlap,
        minimum_latitude_deg=-60.0,
        maximum_latitude_deg=90.0,
    )
    if args.reuse_samples:
        model = None
        model_report = {"reused_samples": True}
        loader = _reuse_split
    else:
        model, model_report = load_da3metric_large(
            args.source, args.checkpoint, device=args.device
        )
        loader = _generate_split
    train_parts, train_reports = loader(
        list(args.train_ids),
        split="train",
        args=args,
        model=model,
        plan=plan,
        rng=rng,
    )
    validation_parts, validation_reports = loader(
        list(args.validation_ids),
        split="validation",
        args=args,
        model=model,
        plan=plan,
        rng=rng,
    )
    del model
    train_np = _concatenate(train_parts)
    validation_np = _concatenate(validation_parts)
    train_valid = train_np["validity"]
    local_mean = train_np["local_features"][train_valid].mean(axis=0)
    local_std = train_np["local_features"][train_valid].std(axis=0)
    latent_mean = train_np["token_latents"][train_valid].mean(axis=0)
    latent_std = train_np["token_latents"][train_valid].std(axis=0)
    fusion = LatentConditionedDepthFusion.create(
        maximum_logit_correction=args.maximum_logit_correction,
        local_mean=local_mean,
        local_std=np.maximum(local_std, 1e-6),
        latent_mean=latent_mean,
        latent_std=np.maximum(latent_std, 1e-6),
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        fusion.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train = _to_torch(train_np, args.device)
    validation = _to_torch(validation_np, args.device)
    train_target = torch.from_numpy(_weighted_median_log_depth(train_np)).to(
        args.device
    )
    validation_target = torch.from_numpy(_weighted_median_log_depth(validation_np)).to(
        args.device
    )
    baseline_train = _loss_and_metrics(
        fusion,
        train,
        train_target,
        prior_regularization=args.prior_regularization,
        batch_size=args.batch_size,
        train=False,
    )
    baseline_validation = _loss_and_metrics(
        fusion,
        validation,
        validation_target,
        prior_regularization=args.prior_regularization,
        batch_size=args.batch_size,
        train=False,
    )
    history = []
    best_state = None
    best_value = float("inf")
    stale_epochs = 0
    for epoch in range(args.epochs):
        permutation = torch.randperm(train_target.numel(), device=args.device)
        train_metrics = _loss_and_metrics(
            fusion,
            train,
            train_target,
            prior_regularization=args.prior_regularization,
            batch_size=args.batch_size,
            train=True,
            optimizer=optimizer,
            permutation=permutation,
        )
        validation_metrics = _loss_and_metrics(
            fusion,
            validation,
            validation_target,
            prior_regularization=args.prior_regularization,
            batch_size=args.batch_size,
            train=False,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        value = validation_metrics["loss"]
        if value < best_value - 1e-7:
            best_value = value
            best_state = deepcopy(fusion.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("fusion training produced no checkpoint")
    fusion.load_state_dict(best_state)
    selected_train = _loss_and_metrics(
        fusion,
        train,
        train_target,
        prior_regularization=args.prior_regularization,
        batch_size=args.batch_size,
        train=False,
    )
    selected_validation = _loss_and_metrics(
        fusion,
        validation,
        validation_target,
        prior_regularization=args.prior_regularization,
        batch_size=args.batch_size,
        train=False,
    )
    metadata = {
        "schema": SCHEMA,
        "interface": INTERFACE,
        "status": "development-only-not-publication-evidence",
        "architecture": "shared local 8->16, FiLM 8->32, 16->1, masked softmax",
        "parameter_count": fusion_parameter_count(fusion),
        "maximum_logit_correction": args.maximum_logit_correction,
        "local_input_names": list(LOCAL_INPUT_NAMES),
        "model": model_report,
        "token_gate_checkpoint": str(args.gate_checkpoint),
        "token_gate_sha256": sha256(args.gate_checkpoint),
        "plan": plan.to_dict(),
        "protocol": {
            "train_ids": list(map(_full_id, args.train_ids)),
            "validation_ids": list(map(_full_id, args.validation_ids)),
            "held_out_final_id": _full_id("W_121"),
            "depth_ground_truth_used": False,
            "npz_opened": False,
            "pseudo_target": "Gaussian-weighted median of overlapping predicted log-depths",
            "prior_regularization": args.prior_regularization,
            "seed": args.seed,
            "samples_per_panorama": args.samples_per_panorama,
            "maximum_contributions": args.maximum_sources_per_target,
            "variable_cardinality": True,
            "permutation_equivariant": True,
        },
        "training_panorama_reports": train_reports,
        "validation_panorama_reports": validation_reports,
        "baselines": {"train": baseline_train, "validation": baseline_validation},
        "selected": {"train": selected_train, "validation": selected_validation},
        "history": history,
        "runtime_seconds": perf_counter() - start,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "peak_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    checkpoint_path = args.output / "da3-latent-conditioned-fusion.safetensors"
    checkpoint_path, metadata_path = save_fusion(
        checkpoint_path, fusion, metadata=json_ready(metadata)
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "metadata": str(metadata_path),
        "metadata_sha256": sha256(metadata_path),
        "baselines": metadata["baselines"],
        "selected": metadata["selected"],
    }
    result_path = args.output / "training-result.json"
    result_path.write_text(
        json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
