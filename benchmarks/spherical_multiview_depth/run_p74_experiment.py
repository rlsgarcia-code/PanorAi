#!/usr/bin/env python3
"""Run the frozen P74 multiview spherical-depth control experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.spherical_monocular_depth.protocol import (  # noqa: E402
    band_metrics,
    depth_metrics,
    json_ready,
    normal_structure_metrics,
    scale_invariant_structure_metrics,
    seam_score,
    sha256,
    write_binary_ply,
)
from benchmarks.spherical_multiview_depth.p74 import (  # noqa: E402
    load_native_angular_rgb,
    load_registered_pose,
)
from benchmarks.spherical_multiview_depth.refinement import (  # noqa: E402
    INTERFACE,
    DepthPrior,
    DifferentiableSphericalDepthRefiner,
    RefinementOptions,
    SourceView,
    reprojection_score,
)


SCHEMA = "panorai-p74-spherical-multiview-depth-control/v1"
TARGET_ID = "P-74+MD-04_concluido_408+W_121"
OPTIMIZATION_IDS = (
    "P-74+MD-04_concluido_408+W_119",
    "P-74+MD-04_concluido_408+W_124",
)
HELDOUT_ID = "P-74+MD-04_concluido_408+W_122"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refine a native-resolution W121 CNN radial-range prior with "
            "registered W119/W124 views; W122 is opened only after outputs freeze."
        )
    )
    parser.add_argument("--p74-root", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--evaluation-validity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--row-batch", type=int, default=16)
    parser.add_argument("--threads", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--normal-stride", type=int, default=8)
    parser.add_argument("--ply-max-points", type=int, default=1_500_000)
    parser.add_argument("--preview-stride", type=int, default=8)
    return parser.parse_args()


def _paths(root: Path, panorama_id: str) -> tuple[Path, Path]:
    return (
        root / "images" / f"{panorama_id}_rgb.png",
        root / "npzs" / f"{panorama_id}.npz",
    )


def _load_source(
    root: Path,
    target_npz: Path,
    panorama_id: str,
    shape_hw: tuple[int, int],
) -> SourceView:
    rgb_path, npz_path = _paths(root, panorama_id)
    rgb, support = load_native_angular_rgb(rgb_path, shape_hw)
    pose = load_registered_pose(target_npz, npz_path)
    return SourceView(
        rgb=rgb,
        rotation_source_from_target=pose.rotation_source_from_target,
        translation_source_from_target_m=pose.translation_source_from_target_m,
        validity_mask=support,
        view_id=panorama_id,
    )


def _pose_record(source: SourceView) -> dict[str, Any]:
    translation = np.asarray(source.translation_source_from_target_m)
    return {
        "view_id": source.view_id,
        "convention": "X_source = R_source_from_target @ X_target + t",
        "rotation_source_from_target": np.asarray(
            source.rotation_source_from_target
        ).tolist(),
        "translation_source_from_target_m": translation.tolist(),
        "baseline_m": float(np.linalg.norm(translation)),
    }


def _evaluate(
    prediction: np.ndarray,
    truth: np.ndarray,
    validity: np.ndarray,
    *,
    normal_stride: int,
) -> dict[str, Any]:
    return {
        "metric_depth": depth_metrics(prediction, truth, validity),
        "scale_invariant_structure": scale_invariant_structure_metrics(
            prediction, truth, validity
        ),
        "surface_normals": normal_structure_metrics(
            prediction,
            truth,
            validity,
            stride_px=normal_stride,
        ),
        "latitude_bands": band_metrics(prediction, truth, validity),
        "seam": seam_score(prediction, validity),
    }


def _write_native_depth_png(path: Path, radial_range_m: np.ndarray) -> None:
    """Write one uint16 pixel per native depth sample, encoded as depth/15 m."""

    normalized = np.nan_to_num(radial_range_m / 15.0, nan=0.0)
    encoded = np.rint(np.clip(normalized, 0.0, 1.0) * 65535.0).astype(np.uint16)
    if not cv2.imwrite(str(path), encoded):
        raise RuntimeError(f"failed to write {path}")


def _verify_binary_ply(path: Path, *, maximum_radius_m: float) -> dict[str, Any]:
    """Independently parse the emitted PLY and verify its metric radial cap."""

    with path.open("rb") as stream:
        header = bytearray()
        while not header.endswith(b"end_header\n"):
            byte = stream.read(1)
            if not byte:
                raise ValueError(f"truncated PLY header: {path}")
            header.extend(byte)
        header_text = header.decode("ascii")
        vertex_line = next(
            line
            for line in header_text.splitlines()
            if line.startswith("element vertex ")
        )
        count = int(vertex_line.split()[-1])
        dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        )
        vertices = np.fromfile(stream, dtype=dtype, count=count)
    if vertices.size != count:
        raise ValueError(f"truncated PLY body: {path}")
    radius = np.sqrt(
        vertices["x"].astype(np.float64) ** 2
        + vertices["y"].astype(np.float64) ** 2
        + vertices["z"].astype(np.float64) ** 2
    )
    observed = float(radius.max(initial=0.0))
    if observed > maximum_radius_m + 2e-5:
        raise ValueError(
            f"PLY radius {observed} exceeds declared cap {maximum_radius_m}"
        )
    expected_size = len(header) + count * dtype.itemsize
    actual_size = path.stat().st_size
    if expected_size != actual_size:
        raise ValueError(
            f"PLY size mismatch: expected {expected_size}, observed {actual_size}"
        )
    return {
        "vertex_count": count,
        "maximum_radius_m": observed,
        "file_size_bytes": actual_size,
        "binary_little_endian": True,
    }


def _write_panel(
    path: Path,
    target_rgb: np.ndarray,
    truth: np.ndarray,
    maps: dict[str, np.ndarray],
    *,
    stride: int,
) -> None:
    """Write a nearest-sample audit preview; numerical arrays remain native."""

    if stride < 1:
        raise ValueError("preview stride must be positive")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries: list[tuple[str, np.ndarray, str]] = [
        ("W121 RGB", target_rgb, "rgb"),
        ("P74 GT", truth, "depth"),
    ]
    entries.extend((name, value, "depth") for name, value in maps.items())
    columns = 2
    rows = (len(entries) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(18, 4.5 * rows),
        constrained_layout=True,
    )
    flat_axes = np.atleast_1d(axes).reshape(-1)
    depth_axes = []
    depth_image = None
    for axis, (title, value, kind) in zip(flat_axes, entries, strict=False):
        sampled = value[::stride, ::stride]
        if kind == "rgb":
            axis.imshow(sampled)
        else:
            depth_image = axis.imshow(sampled, cmap="turbo", vmin=0.0, vmax=15.0)
            depth_axes.append(axis)
        axis.set_title(title)
        axis.set_axis_off()
    for axis in flat_axes[len(entries) :]:
        axis.set_axis_off()
    if depth_image is not None:
        figure.colorbar(
            depth_image,
            ax=depth_axes,
            fraction=0.018,
            pad=0.01,
            label="radial range (m)",
        )
    figure.suptitle(
        f"P74 native inference {truth.shape[1]}x{truth.shape[0]}; "
        f"preview only uses every {stride}th sample"
    )
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    import torch

    torch.set_num_threads(args.threads)
    prior_memmap = np.load(args.prior, mmap_mode="r")
    if prior_memmap.ndim != 2:
        raise ValueError("prior must be a native HxW radial-range NPY")
    shape_hw = tuple(int(value) for value in prior_memmap.shape)
    prior = np.clip(np.asarray(prior_memmap), 0.3, 15.0).astype(np.float32)
    target_rgb_path, target_npz = _paths(args.p74_root, TARGET_ID)
    target_rgb, target_support = load_native_angular_rgb(target_rgb_path, shape_hw)
    optimization_validity = (
        target_support & np.isfinite(prior) & (prior >= 0.3) & (prior <= 15.0)
    )
    # Deliberately load only W119 and W124 before output freeze.
    sources = tuple(
        _load_source(args.p74_root, target_npz, panorama_id, shape_hw)
        for panorama_id in OPTIMIZATION_IDS
    )
    base_options = RefinementOptions(
        epochs=args.epochs,
        row_batch=args.row_batch,
    )
    variants = {
        "cnn-prior": prior,
    }
    variant_records: dict[str, dict[str, Any]] = {
        "cnn-prior": {
            "role": "frozen-backbone-prior",
            "runtime_seconds": 0.0,
            "options": None,
            "history": [],
        }
    }
    for name, pairwise_weight in (
        ("multiview-no-crf", 0.0),
        ("multiview-crf", base_options.pairwise_weight),
    ):
        options = replace(base_options, pairwise_weight=pairwise_weight)
        print(
            json.dumps(
                {
                    "event": "refinement-start",
                    "variant": name,
                    "shape_hw": shape_hw,
                    "options": options.to_dict(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        start = perf_counter()
        result = DifferentiableSphericalDepthRefiner(options).refine(
            DepthPrior(
                prior,
                optimization_validity,
                provenance=f"{args.prior} sha256={sha256(args.prior)}",
            ),
            target_rgb,
            sources,
            device=args.device,
            progress=lambda row, variant=name: print(
                json.dumps({"event": "epoch", "variant": variant, **row}),
                flush=True,
            ),
        )
        seconds = perf_counter() - start
        prediction = result.radial_range_m.cpu().numpy().astype(np.float32, copy=True)
        log_residual = result.log_residual.cpu().numpy().astype(np.float32, copy=True)
        prediction_path = args.output / f"{TARGET_ID}-{name}-radial-m.npy"
        residual_path = args.output / f"{TARGET_ID}-{name}-log-residual.npy"
        np.save(prediction_path, prediction)
        np.save(residual_path, log_residual)
        _write_native_depth_png(
            args.output / f"{TARGET_ID}-{name}-depth-over-15m.png",
            prediction,
        )
        variants[name] = prediction
        variant_records[name] = {
            "role": "refined-before-heldout-open",
            "runtime_seconds": seconds,
            "options": options.to_dict(),
            "history": list(result.history),
            "prediction": str(prediction_path),
            "prediction_sha256": sha256(prediction_path),
            "log_residual": str(residual_path),
            "log_residual_sha256": sha256(residual_path),
        }
        del result, log_residual
        gc.collect()

    prior_path = args.output / f"{TARGET_ID}-cnn-prior-radial-m.npy"
    np.save(prior_path, prior)
    _write_native_depth_png(
        args.output / f"{TARGET_ID}-cnn-prior-depth-over-15m.png", prior
    )
    variant_records["cnn-prior"].update(
        {
            "prediction": str(prior_path),
            "prediction_sha256": sha256(prior_path),
        }
    )
    validity_path = args.output / f"{TARGET_ID}-optimization-validity.npy"
    np.save(validity_path, optimization_validity)
    freeze_manifest = {
        "schema": SCHEMA,
        "event": "predictions-frozen-before-heldout-open",
        "heldout_id": HELDOUT_ID,
        "target_id": TARGET_ID,
        "optimization_ids": list(OPTIMIZATION_IDS),
        "interface": INTERFACE,
        "shape_hw": list(shape_hw),
        "variants": variant_records,
        "optimization_validity_sha256": sha256(validity_path),
    }
    freeze_path = args.output / "FROZEN-BEFORE-W122.json"
    _json_dump(freeze_path, freeze_manifest)
    freeze_sha256 = sha256(freeze_path)
    print(json.dumps({"event": "outputs-frozen", "sha256": freeze_sha256}), flush=True)

    for name, prediction in variants.items():
        training_scores = []
        for source in sources:
            score = reprojection_score(
                prediction,
                optimization_validity,
                target_rgb,
                source,
                row_batch=args.row_batch,
                device=args.device,
            )
            training_scores.append(score)
            print(
                json.dumps(
                    {"event": "optimization-view-score", "variant": name, **score}
                ),
                flush=True,
            )
        variant_records[name]["optimization_view_reprojection"] = training_scores

    # W122 is first touched here, after every compared prediction is immutable.
    heldout = _load_source(args.p74_root, target_npz, HELDOUT_ID, shape_hw)
    for name, prediction in variants.items():
        score = reprojection_score(
            prediction,
            optimization_validity,
            target_rgb,
            heldout,
            row_batch=args.row_batch,
            device=args.device,
        )
        variant_records[name]["heldout_w122_reprojection"] = score
        print(
            json.dumps({"event": "heldout-score", "variant": name, **score}),
            flush=True,
        )
    del heldout
    gc.collect()

    # Ground truth is evaluation-only and is loaded after optimization and heldout scoring.
    truth = np.load(args.ground_truth, mmap_mode="r")
    evaluation_validity = np.load(args.evaluation_validity, mmap_mode="r")
    if truth.shape != shape_hw or evaluation_validity.shape != shape_hw:
        raise ValueError("ground truth and evaluation validity must match native shape")
    common_validity = (
        np.asarray(evaluation_validity, dtype=bool)
        & np.isfinite(truth)
        & (truth >= 0.3)
        & (truth <= 15.0)
        & optimization_validity
    )
    common_path = args.output / f"{TARGET_ID}-common-evaluation-validity.npy"
    np.save(common_path, common_validity)
    for name, prediction in variants.items():
        variant_records[name]["evaluation"] = _evaluate(
            prediction,
            truth,
            common_validity,
            normal_stride=args.normal_stride,
        )
        ply_path = args.output / f"{TARGET_ID}-{name}-view-15m.ply"
        ply = write_binary_ply(
            ply_path,
            prediction,
            target_rgb,
            common_validity,
            max_points=args.ply_max_points,
        )
        ply["verification"] = _verify_binary_ply(ply_path, maximum_radius_m=15.0)
        ply["sha256"] = sha256(ply_path)
        variant_records[name]["ply"] = ply

    panel_path = args.output / "p74-w121-multiview-depth-panel.png"
    _write_panel(
        panel_path,
        target_rgb,
        truth,
        {
            "CNN prior": variants["cnn-prior"],
            "Multiview, no CRF": variants["multiview-no-crf"],
            "Multiview + CRF": variants["multiview-crf"],
        },
        stride=args.preview_stride,
    )
    report = {
        "schema": SCHEMA,
        "status": "experimental-pose-oracle-control-not-publication-evidence",
        "interface": INTERFACE,
        "dataset": {
            "name": "P74",
            "external": True,
            "target_id": TARGET_ID,
            "optimization_ids": list(OPTIMIZATION_IDS),
            "heldout_id": HELDOUT_ID,
            "depth_semantics": "radial range in metres",
            "metric_cap_m": 15.0,
        },
        "protocol": {
            "native_shape_hw": list(shape_hw),
            "model_or_prediction_resize": False,
            "pyramid": False,
            "native_rgb_coordinate_regrid": (
                "P74 0--150 degree native samples to audited equal-angular ERP; "
                "no angular minification"
            ),
            "optimization_uses_ground_truth": False,
            "heldout_opened_after_prediction_freeze": True,
            "freeze_manifest": str(freeze_path),
            "freeze_manifest_sha256": freeze_sha256,
            "preview_is_metric_input": False,
            "preview_sampling": f"integer stride {args.preview_stride}, display only",
            "common_evaluation_pixels": int(common_validity.sum()),
            "normal_stride_px": args.normal_stride,
        },
        "inputs": {
            "prior": str(args.prior),
            "prior_sha256": sha256(args.prior),
            "ground_truth": str(args.ground_truth),
            "ground_truth_sha256": sha256(args.ground_truth),
            "evaluation_validity": str(args.evaluation_validity),
            "evaluation_validity_sha256": sha256(args.evaluation_validity),
        },
        "poses": [_pose_record(source) for source in sources],
        "variants": variant_records,
        "artifacts": {
            "panel": str(panel_path),
            "panel_sha256": sha256(panel_path),
            "common_evaluation_validity": str(common_path),
            "common_evaluation_validity_sha256": sha256(common_path),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "device": args.device,
            "torch_threads": args.threads,
        },
    }
    report_path = args.output / "results.json"
    _json_dump(report_path, report)
    summary = {
        name: {
            "abs_rel": record["evaluation"]["metric_depth"]["abs_rel"],
            "si_3d_rmse": record["evaluation"]["scale_invariant_structure"][
                "scale_aligned_relative_3d_rmse"
            ],
            "normal_mean_deg": record["evaluation"]["surface_normals"][
                "normal_mean_deg"
            ],
            "heldout_cost": record["heldout_w122_reprojection"][
                "weighted_mean_feature_cost"
            ],
        }
        for name, record in variant_records.items()
    }
    print(json.dumps({"event": "complete", "summary": summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
