#!/usr/bin/env python3
"""Train the tiny DA3 spherical token gate without reading depth data."""

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

from benchmarks.spherical_monocular_depth.da3_sphere import (  # noqa: E402
    make_spherical_overlap_plan,
)
from benchmarks.spherical_monocular_depth.da3_spherical_attention import (  # noqa: E402
    compute_local_attention_state_store,
    extract_prefix_states,
)
from benchmarks.spherical_monocular_depth.da3_spherical_gate import (  # noqa: E402
    GATE_INPUT_NAMES,
    INTERFACE,
    MAXIMUM_GATE_MAGNITUDE,
    SphericalAttentionGate,
    build_gate_samples,
    gate_parameter_count,
    normalized_gate_objective,
    save_gate,
)
from benchmarks.spherical_monocular_depth.da3_tangent import (  # noqa: E402
    load_da3metric_large,
)
from benchmarks.spherical_monocular_depth.protocol import (  # noqa: E402
    json_ready,
    sha256,
)
from benchmarks.spherical_monocular_depth.vit_tangent import (  # noqa: E402
    _FixedTangentSampler,
    make_native_tangent_plan,
)
from benchmarks.spherical_multiview_depth.p74 import (  # noqa: E402
    load_native_angular_rgb,
)


SCHEMA = "panorai-p74-da3-self-supervised-spherical-gate-training/v1"
P74_PREFIX = "P-74+MD-04_concluido_408+"
DEFAULT_TRAIN_IDS = ("W_050", "W_100", "W_150")
DEFAULT_VALIDATION_IDS = ("W_120",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p74-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    parser.add_argument("--identity-regularization", type=float, default=1.0)
    parser.add_argument("--max-samples-per-panorama", type=int, default=160_000)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p74-row-chunk", type=int, default=32)
    parser.add_argument(
        "--position-transport", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--reuse-samples",
        action="store_true",
        help="Reuse the audited RGB-only sample NPZ files already in --output.",
    )
    return parser.parse_args()


def _full_id(short_id: str) -> str:
    return short_id if short_id.startswith("P-74+") else P74_PREFIX + short_id


def _evaluate_gate(gate: Any, features: Any, objective: Any, batch_size: int) -> dict:
    import torch

    weighted_sum = 0.0
    weight_sum = 0.0
    gate_parts = []
    gate.eval()
    with torch.inference_mode():
        for start in range(0, features.shape[0], batch_size):
            stop = min(features.shape[0], start + batch_size)
            values = gate(features[start:stop])
            losses = normalized_gate_objective(
                values, objective[start:stop], reduction="none"
            )
            weights = objective[start:stop, 3]
            weighted_sum += float(losses.sum().item())
            weight_sum += float(weights.sum().item())
            gate_parts.append(values.cpu())
    gates = torch.cat(gate_parts).numpy()
    return {
        "normalized_objective": weighted_sum / max(weight_sum, 1e-12),
        "gate_mean": float(gates.mean()),
        "gate_median": float(np.median(gates)),
        "gate_p05": float(np.quantile(gates, 0.05)),
        "gate_p95": float(np.quantile(gates, 0.95)),
    }


def _oracle_objective(objective: np.ndarray) -> float:
    a, b, c, weight, regularization = objective.T
    gate = np.clip(
        -b / ((1.0 + regularization) * a + 1e-12),
        -MAXIMUM_GATE_MAGNITUDE,
        MAXIMUM_GATE_MAGNITUDE,
    )
    value = (c + 2.0 * gate * b + (1.0 + regularization) * gate**2 * a) / np.maximum(
        c, 1e-8
    )
    return float(np.sum(weight * value) / np.maximum(weight.sum(), 1e-12))


def _build_split(
    ids: list[str],
    *,
    split: str,
    args: argparse.Namespace,
    model: Any,
    plan: Any,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    from panorai.data import EquirectangularImage

    all_features = []
    all_objectives = []
    reports = []
    for short_id in ids:
        panorama_id = _full_id(short_id)
        rgb_path = args.p74_root / "images" / f"{panorama_id}_rgb.png"
        if not rgb_path.is_file():
            raise FileNotFoundError(rgb_path)
        rgb, _ = load_native_angular_rgb(
            rgb_path, plan.erp_shape_hw, row_chunk=args.p74_row_chunk
        )
        views = EquirectangularImage(rgb).views(
            _FixedTangentSampler(plan.centers_lat_lon_deg),
            size=plan.view_shape_hw,
            fov=(plan.hfov_deg, plan.vfov_deg),
            depth_policy="propagate",
        )
        scratch = args.output / "scratch" / split / panorama_id
        scratch.mkdir(parents=True, exist_ok=True)
        state_path = scratch / "block11-f32.dat"
        base_path = scratch / "block12-local-f32.dat"
        state_spec, feature_spec, prefix_report = extract_prefix_states(
            model, list(views), state_path, None, device=args.device
        )
        base_report = compute_local_attention_state_store(
            model, state_path, base_path, state_spec, device=args.device
        )
        overlap_plan = make_spherical_overlap_plan(
            plan,
            feature_shape_hw=feature_spec.feature_shape_hw,
            maximum_sources_per_target=args.maximum_sources_per_target,
        )
        features, objective, sample_report = build_gate_samples(
            model,
            state_path,
            base_path,
            state_spec,
            feature_spec,
            overlap_plan,
            device=args.device,
            identity_regularization=args.identity_regularization,
            position_transport=args.position_transport,
        )
        raw_samples = features.shape[0]
        if raw_samples > args.max_samples_per_panorama:
            selected = rng.choice(
                raw_samples, size=args.max_samples_per_panorama, replace=False
            )
            selected.sort()
            features = features[selected]
            objective = objective[selected]
        sample_path = args.output / "samples" / f"{split}-{panorama_id}.npz"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sample_path, features=features, objective=objective)
        state_path.unlink()
        base_path.unlink()
        all_features.append(features)
        all_objectives.append(objective)
        reports.append(
            {
                "panorama_id": panorama_id,
                "rgb_path": str(rgb_path),
                "rgb_sha256": sha256(rgb_path),
                "depth_opened": False,
                "prefix": prefix_report,
                "local_attention": base_report,
                "samples": sample_report,
                "raw_sample_count": int(raw_samples),
                "retained_sample_count": int(features.shape[0]),
                "sample_path": str(sample_path),
                "sample_sha256": sha256(sample_path),
            }
        )
        del rgb, views
    return np.concatenate(all_features), np.concatenate(all_objectives), reports


def _load_split(
    ids: list[str], *, split: str, args: argparse.Namespace, **_: Any
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    all_features = []
    all_objectives = []
    reports = []
    for short_id in ids:
        panorama_id = _full_id(short_id)
        sample_path = args.output / "samples" / f"{split}-{panorama_id}.npz"
        with np.load(sample_path) as payload:
            features = np.asarray(payload["features"], dtype=np.float32)
            objective = np.asarray(payload["objective"], dtype=np.float32)
        all_features.append(features)
        all_objectives.append(objective)
        reports.append(
            {
                "panorama_id": panorama_id,
                "sample_path": str(sample_path),
                "sample_sha256": sha256(sample_path),
                "retained_sample_count": int(features.shape[0]),
                "depth_opened": False,
                "reused_audited_rgb_only_samples": True,
            }
        )
    return np.concatenate(all_features), np.concatenate(all_objectives), reports


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
    previous_metadata_path = args.output / "da3-spherical-token-gate.json"
    if args.reuse_samples:
        if not previous_metadata_path.is_file():
            raise FileNotFoundError(previous_metadata_path)
        previous_metadata = json.loads(previous_metadata_path.read_text())
        model = None
        model_report = previous_metadata["model"]
        split_loader = _load_split
    else:
        model, model_report = load_da3metric_large(
            args.source, args.checkpoint, device=args.device
        )
        split_loader = _build_split
    train_features_np, train_objective_np, train_reports = split_loader(
        list(args.train_ids),
        split="train",
        args=args,
        model=model,
        plan=plan,
        rng=rng,
    )
    validation_features_np, validation_objective_np, validation_reports = split_loader(
        list(args.validation_ids),
        split="validation",
        args=args,
        model=model,
        plan=plan,
        rng=rng,
    )
    input_mean = train_features_np.mean(axis=0, dtype=np.float64).astype(np.float32)
    input_std = train_features_np.std(axis=0, dtype=np.float64).astype(np.float32)
    input_std = np.maximum(input_std, 1e-6)
    gate = SphericalAttentionGate.create(input_mean, input_std).to(args.device)
    with torch.no_grad():
        gate.output.weight.zero_()
        gate.output.bias.zero_()
    optimizer = torch.optim.AdamW(
        gate.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_features = torch.from_numpy(train_features_np).to(args.device)
    train_objective = torch.from_numpy(train_objective_np).to(args.device)
    validation_features = torch.from_numpy(validation_features_np).to(args.device)
    validation_objective = torch.from_numpy(validation_objective_np).to(args.device)

    history = []
    best_state = None
    best_value = float("inf")
    stale_epochs = 0
    for epoch in range(args.epochs):
        gate.train()
        permutation = torch.randperm(train_features.shape[0], device=args.device)
        for start_index in range(0, permutation.numel(), args.batch_size):
            indices = permutation[start_index : start_index + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            values = gate(train_features[indices])
            loss = normalized_gate_objective(values, train_objective[indices])
            loss.backward()
            optimizer.step()
        train_metrics = _evaluate_gate(
            gate, train_features, train_objective, args.batch_size
        )
        validation_metrics = _evaluate_gate(
            gate, validation_features, validation_objective, args.batch_size
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        value = validation_metrics["normalized_objective"]
        if value < best_value - 1e-6:
            best_value = value
            best_state = deepcopy(gate.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    gate.load_state_dict(best_state)
    gate.eval()
    final_train = _evaluate_gate(gate, train_features, train_objective, args.batch_size)
    final_validation = _evaluate_gate(
        gate, validation_features, validation_objective, args.batch_size
    )
    metadata = {
        "schema": SCHEMA,
        "interface": INTERFACE,
        "status": "development-only-not-publication-evidence",
        "architecture": "6-16-8-1 GELU-GELU-0.1*tanh",
        "parameter_count": gate_parameter_count(gate),
        "input_names": list(GATE_INPUT_NAMES),
        "latent_width": 8,
        "model": model_report,
        "plan": plan.to_dict(),
        "protocol": {
            "train_ids": list(map(_full_id, args.train_ids)),
            "validation_ids": list(map(_full_id, args.validation_ids)),
            "held_out_final_id": _full_id("W_121"),
            "depth_used": False,
            "npz_opened": False,
            "rgb_only": True,
            "identity_regularization": args.identity_regularization,
            "position_transport": args.position_transport,
            "seed": args.seed,
            "maximum_sources_per_target": args.maximum_sources_per_target,
            "max_samples_per_panorama": args.max_samples_per_panorama,
            "maximum_signed_gate_magnitude": MAXIMUM_GATE_MAGNITUDE,
            "reused_samples": args.reuse_samples,
        },
        "training_panorama_reports": train_reports,
        "validation_panorama_reports": validation_reports,
        "sample_counts": {
            "train": int(train_features.shape[0]),
            "validation": int(validation_features.shape[0]),
        },
        "baselines": {
            "zero_gate_normalized_objective": 1.0,
            "train_oracle_normalized_objective": _oracle_objective(train_objective_np),
            "validation_oracle_normalized_objective": _oracle_objective(
                validation_objective_np
            ),
        },
        "selected": {"train": final_train, "validation": final_validation},
        "history": history,
        "runtime_seconds": perf_counter() - start,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "peak_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    checkpoint_path = args.output / "da3-spherical-token-gate.safetensors"
    checkpoint_path, metadata_path = save_gate(
        checkpoint_path, gate, metadata=json_ready(metadata)
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "metadata": str(metadata_path),
        "metadata_sha256": sha256(metadata_path),
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
