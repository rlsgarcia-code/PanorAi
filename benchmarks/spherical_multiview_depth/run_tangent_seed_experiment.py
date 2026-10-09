#!/usr/bin/env python3
"""Native P74 DoG/tangent-RootSIFT depth-seed experiment."""

from __future__ import annotations

import argparse
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
    sha256,
    write_binary_ply,
)
from benchmarks.spherical_multiview_depth.p74 import (  # noqa: E402
    load_native_angular_rgb,
    load_registered_pose,
)
from benchmarks.spherical_multiview_depth.run_p74_experiment import (  # noqa: E402
    OPTIMIZATION_IDS,
    TARGET_ID,
    _evaluate,
    _json_dump,
    _paths,
    _verify_binary_ply,
    _write_native_depth_png,
    _write_panel,
)
from benchmarks.spherical_multiview_depth.tangent_seeds import (  # noqa: E402
    INTERFACE,
    TangentDepthSeedOptions,
    propagate_tangent_depth_seeds,
    sample_erp_scalar,
    solve_tangent_depth_seeds,
)
from panorai.estimators import (  # noqa: E402
    RelativePoseOptions,
    SpatiallyWeightedFivePointSampler,
    SphericalRelativePoseEstimator,
)
from panorai.features import (  # noqa: E402
    FeatureMatcher,
    FeatureMatcherConfig,
    OpenCVTangentDescriptorV2Config,
    SphericalDoGSIFTConfig,
    SphericalDoGSIFTExtractor,
)


SCHEMA = "panorai-p74-tangent-dog-depth-seeds/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect native spherical DoG keypoints, describe 48x48 tangent "
            "RootSIFT patches, refine/freeze pose, and solve depth around a "
            "frozen W121 prior without resizing the ERP."
        )
    )
    parser.add_argument("--p74-root", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--evaluation-validity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-id",
        action="append",
        choices=OPTIMIZATION_IDS,
        help="repeat to select sources; default uses W119 only for the pilot",
    )
    parser.add_argument(
        "--pose-mode",
        choices=("registered", "dog-refined-direction-registered-scale"),
        default="registered",
    )
    parser.add_argument("--max-features", type=int, default=4096)
    parser.add_argument("--dog-octaves", type=int, default=3)
    parser.add_argument("--hypotheses", type=int, default=65)
    parser.add_argument("--range-factor", type=float, default=1.5)
    parser.add_argument("--maximum-reprojection-error-deg", type=float, default=0.35)
    parser.add_argument("--minimum-parallax-deg", type=float, default=0.35)
    parser.add_argument("--maximum-ray-miss-m", type=float, default=0.12)
    parser.add_argument("--threads", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--normal-stride", type=int, default=8)
    parser.add_argument("--ply-max-points", type=int, default=1_500_000)
    parser.add_argument("--preview-stride", type=int, default=8)
    return parser.parse_args()


def _extractor(max_features: int, octaves: int) -> SphericalDoGSIFTExtractor:
    descriptor = OpenCVTangentDescriptorV2Config(
        method="sift",
        keypoint_diameter_in_scales=1.5,
        scale_multipliers=(1.0,),
        orientation_policy="fixed-zero",
        photometric_normalization="local-standardization",
        root_sift=True,
    )
    return SphericalDoGSIFTExtractor(
        SphericalDoGSIFTConfig(
            octaves=octaves,
            max_features=max_features,
            contrast_threshold=0.012,
            patch_size=48,
            descriptor_radius_sigmas=6.0,
            descriptor_config=descriptor,
            convolution_backend="native",
        )
    )


def _matcher() -> FeatureMatcher:
    return FeatureMatcher(
        FeatureMatcherConfig(
            method="flann",
            ratio_test=0.72,
            cross_check=True,
            deduplicate_matches=True,
            angular_dedup_threshold_deg=0.15,
        )
    )


def _estimator() -> SphericalRelativePoseEstimator:
    return SphericalRelativePoseEstimator(
        RelativePoseOptions(
            max_angular_error_deg=1.0,
            max_num_trials=1000,
            stability_trials=6,
            model_competition_trials=128,
            random_seed=7,
            hypothesis_ranking="msac-first",
            nonminimal_refit_max_steps=100,
        ),
        sampler=SpatiallyWeightedFivePointSampler(),
    )


def _angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return float(
        np.degrees(
            np.arccos(
                np.clip(
                    float(left @ right)
                    / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12),
                    -1.0,
                    1.0,
                )
            )
        )
    )


def _rotation_error_deg(estimate: np.ndarray, reference: np.ndarray) -> float:
    relative = np.asarray(estimate) @ np.asarray(reference).T
    return float(
        np.degrees(
            np.arccos(np.clip((float(np.trace(relative)) - 1.0) / 2.0, -1.0, 1.0))
        )
    )


def _pose_record(estimate: Any, registered: Any) -> dict[str, Any]:
    if estimate is None:
        return {"returned": False}
    return {
        "returned": True,
        "rotation_source_from_target": np.asarray(estimate.rotation).tolist(),
        "translation_direction_source_from_target": np.asarray(
            estimate.translation_direction
        ).tolist(),
        "rotation_error_deg": _rotation_error_deg(
            estimate.rotation, registered.rotation_source_from_target
        ),
        "translation_direction_error_deg": _angle_deg(
            estimate.translation_direction,
            registered.translation_source_from_target_m,
        ),
        "inlier_count": int(estimate.num_inliers),
        "match_count": int(estimate.inlier_mask.size),
        "consensus_refit_steps": int(estimate.consensus_refit_steps),
        "median_parallax_deg": float(estimate.median_parallax_deg),
        "quality_accepted": bool(estimate.quality_report.accepted),
        "description": estimate.describe(),
    }


def _sparse_gt_diagnostic(
    result: Any, truth: np.ndarray, validity: np.ndarray
) -> dict[str, Any]:
    gt, _pixels = sample_erp_scalar(truth, result.target_bearings)
    gt_valid, _ = sample_erp_scalar(validity.astype(np.float32), result.target_bearings)
    selection = result.accepted & np.isfinite(gt) & (gt > 0.0) & (gt_valid >= 0.999)
    if not selection.any():
        return {"count": 0}
    prior_error = (
        np.abs(result.prior_range_m[selection] - gt[selection]) / gt[selection]
    )
    solved_error = (
        np.abs(result.solved_range_m[selection] - gt[selection]) / gt[selection]
    )
    return {
        "count": int(selection.sum()),
        "prior_abs_rel": float(np.mean(prior_error)),
        "solved_abs_rel": float(np.mean(solved_error)),
        "fraction_improved": float(np.mean(solved_error < prior_error)),
        "median_prior_abs_rel": float(np.median(prior_error)),
        "median_solved_abs_rel": float(np.median(solved_error)),
    }


def _save_sparse(path: Path, rows: list[dict[str, Any]]) -> None:
    payload: dict[str, np.ndarray] = {}
    for source_index, row in enumerate(rows):
        result = row.pop("_result")
        prefix = f"source_{source_index}_"
        for name in (
            "target_bearings",
            "source_bearings",
            "target_pixels_xy",
            "prior_range_m",
            "solved_range_m",
            "source_range_m",
            "accepted",
            "reprojection_error_deg",
            "parallax_deg",
            "ray_miss_m",
            "confidence",
            "target_scale_deg",
            "descriptor_distance",
        ):
            payload[prefix + name] = np.asarray(getattr(result, name))
    np.savez_compressed(path, **payload)


def main() -> int:
    args = parse_args()
    if args.threads < 1 or args.max_features < 1 or args.dog_octaves < 1:
        raise ValueError("threads, max-features and dog-octaves must be positive")
    cv2.setNumThreads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    source_ids = tuple(args.source_id or (OPTIMIZATION_IDS[0],))
    prior = np.load(args.prior, mmap_mode="r")
    if prior.ndim != 2:
        raise ValueError("prior must have native HxW radial range")
    shape_hw = tuple(int(value) for value in prior.shape)
    seed = np.clip(np.asarray(prior), 0.3, 15.0).astype(np.float32)
    target_path, target_npz = _paths(args.p74_root, TARGET_ID)
    target_rgb, target_support = load_native_angular_rgb(target_path, shape_hw)
    validity = target_support & np.isfinite(seed) & (seed >= 0.3) & (seed <= 15.0)

    extractor = _extractor(args.max_features, args.dog_octaves)
    matcher = _matcher()
    estimator = _estimator()
    timings: dict[str, float] = {}
    started = perf_counter()
    target_features = extractor.extract(
        target_rgb, panorama_id=TARGET_ID, validity_mask=target_support
    )
    timings["target_extract_seconds"] = perf_counter() - started
    print(
        json.dumps(
            {
                "event": "target-extracted",
                "features": len(target_features),
                "seconds": timings["target_extract_seconds"],
            }
        ),
        flush=True,
    )

    options = TangentDepthSeedOptions(
        hypotheses=args.hypotheses,
        range_factor=args.range_factor,
        maximum_reprojection_error_deg=args.maximum_reprojection_error_deg,
        minimum_parallax_deg=args.minimum_parallax_deg,
        maximum_ray_miss_m=args.maximum_ray_miss_m,
    )
    sparse_results = []
    source_records: list[dict[str, Any]] = []
    for source_id in source_ids:
        source_path, source_npz = _paths(args.p74_root, source_id)
        source_rgb, source_support = load_native_angular_rgb(source_path, shape_hw)
        registered = load_registered_pose(target_npz, source_npz)
        started = perf_counter()
        source_features = extractor.extract(
            source_rgb, panorama_id=source_id, validity_mask=source_support
        )
        extract_seconds = perf_counter() - started
        started = perf_counter()
        matches = matcher.match(target_features, source_features)
        match_seconds = perf_counter() - started
        started = perf_counter()
        estimate = estimator.estimate(matches.to_bearing_correspondences())
        pose_seconds = perf_counter() - started
        pose_record = _pose_record(estimate, registered)
        rotation = registered.rotation_source_from_target
        translation = registered.translation_source_from_target_m
        pose_source = "registered-metric"
        if args.pose_mode == "dog-refined-direction-registered-scale":
            if estimate is None:
                raise RuntimeError(f"DoG pose estimation failed for {source_id}")
            rotation = estimate.rotation
            translation = estimate.translation_direction * np.linalg.norm(
                registered.translation_source_from_target_m
            )
            pose_source = "dog-rootsift-refined-direction+registered-baseline"
        target_scales = np.asarray(
            [
                target_features.features[index].scale
                for index in matches.feature_indices_a
            ],
            dtype=np.float64,
        )
        sparse = solve_tangent_depth_seeds(
            seed,
            matches.bearings_a,
            matches.bearings_b,
            rotation,
            translation,
            target_scale_deg=target_scales,
            descriptor_distance=matches.descriptor_distances,
            match_validity=matches.valid,
            options=options,
            descriptor_adapter=(
                "panorai-tangent-opencv-descriptor/v2:48x48:rootsift:"
                "r6:d1.5:fixed-zero:local-standardization"
            ),
            pose_source=pose_source,
        )
        sparse_results.append(sparse)
        source_records.append(
            {
                "source_id": source_id,
                "source_feature_count": len(source_features),
                "match_count": len(matches),
                "mutual_match_count": int(
                    matches.mutual.sum() if matches.mutual is not None else 0
                ),
                "extract_seconds": extract_seconds,
                "match_seconds": match_seconds,
                "pose_seconds": pose_seconds,
                "registered_baseline_m": float(
                    np.linalg.norm(registered.translation_source_from_target_m)
                ),
                "dog_pose_diagnostic": pose_record,
                "depth_seeds": sparse.describe(),
                "_result": sparse,
            }
        )
        print(
            json.dumps(
                {
                    "event": "source-complete",
                    "source_id": source_id,
                    "features": len(source_features),
                    "matches": len(matches),
                    "accepted_depth_seeds": int(sparse.accepted.sum()),
                    "pose": pose_record,
                },
                default=str,
            ),
            flush=True,
        )

    started = perf_counter()
    propagated = propagate_tangent_depth_seeds(
        seed, target_rgb, tuple(sparse_results), options=options
    )
    timings["propagation_seconds"] = perf_counter() - started
    refined = propagated.radial_range_m
    prediction_path = args.output / f"{TARGET_ID}-tangent-seed-radial-m.npy"
    np.save(prediction_path, refined)
    residual_path = args.output / f"{TARGET_ID}-tangent-seed-log-residual.npy"
    weight_path = args.output / f"{TARGET_ID}-tangent-seed-weight.npy"
    changed_path = args.output / f"{TARGET_ID}-tangent-seed-changed.npy"
    np.save(residual_path, propagated.log_residual)
    np.save(weight_path, propagated.accumulated_weight)
    np.save(changed_path, propagated.changed_mask)
    sparse_path = args.output / f"{TARGET_ID}-tangent-depth-seeds.npz"
    _save_sparse(sparse_path, source_records)
    _write_native_depth_png(
        args.output / f"{TARGET_ID}-tangent-seed-depth-over-15m.png", refined
    )
    freeze = {
        "schema": SCHEMA,
        "event": "prediction-frozen-before-ground-truth-open",
        "interface": INTERFACE,
        "shape_hw": list(shape_hw),
        "target_id": TARGET_ID,
        "source_ids": list(source_ids),
        "pose_mode": args.pose_mode,
        "options": options.to_dict(),
        "extractor": extractor.config.to_dict(),
        "matcher": matcher.config.to_dict(),
        "prediction": str(prediction_path),
        "prediction_sha256": sha256(prediction_path),
        "sparse_seeds": str(sparse_path),
        "sparse_seeds_sha256": sha256(sparse_path),
        "changed_pixels": int(propagated.changed_mask.sum()),
        "contributing_seed_count": propagated.contributing_seed_count,
    }
    freeze_path = args.output / "FROZEN-BEFORE-GT.json"
    _json_dump(freeze_path, freeze)

    truth = np.load(args.ground_truth, mmap_mode="r")
    evaluation_validity = np.load(args.evaluation_validity, mmap_mode="r")
    if truth.shape != shape_hw or evaluation_validity.shape != shape_hw:
        raise ValueError("ground truth and evaluation validity must match prior")
    common = (
        validity
        & np.asarray(evaluation_validity, dtype=bool)
        & np.isfinite(truth)
        & (truth >= 0.3)
        & (truth <= 15.0)
    )
    variants = {"cnn-seed": seed, "tangent-seed": refined}
    evaluations: dict[str, Any] = {}
    for name, value in variants.items():
        evaluations[name] = _evaluate(
            value, truth, common, normal_stride=args.normal_stride
        )
        ply_path = args.output / f"{TARGET_ID}-{name}-view-15m.ply"
        ply = write_binary_ply(
            ply_path,
            value,
            target_rgb,
            common,
            max_points=args.ply_max_points,
        )
        ply["verification"] = _verify_binary_ply(ply_path, maximum_radius_m=15.0)
        ply["sha256"] = sha256(ply_path)
        evaluations[name]["ply"] = ply
    for row, sparse in zip(source_records, sparse_results, strict=True):
        row["sparse_ground_truth_diagnostic"] = _sparse_gt_diagnostic(
            sparse, truth, common
        )
    changed_eval = common & propagated.changed_mask
    changed_diagnostic = (
        {
            "pixels": int(changed_eval.sum()),
            "seed_abs_rel": float(
                np.mean(
                    np.abs(seed[changed_eval] - truth[changed_eval])
                    / truth[changed_eval]
                )
            ),
            "tangent_abs_rel": float(
                np.mean(
                    np.abs(refined[changed_eval] - truth[changed_eval])
                    / truth[changed_eval]
                )
            ),
            "fraction_improved": float(
                np.mean(
                    np.abs(refined[changed_eval] - truth[changed_eval])
                    < np.abs(seed[changed_eval] - truth[changed_eval])
                )
            ),
        }
        if changed_eval.any()
        else {"pixels": 0}
    )
    panel_path = args.output / "p74-w121-tangent-depth-seeds-panel.png"
    _write_panel(
        panel_path,
        target_rgb,
        truth,
        {"CNN seed": seed, "DoG+tangent depth seeds": refined},
        stride=args.preview_stride,
    )
    report = {
        "schema": SCHEMA,
        "status": "experimental-registered-pose-control",
        "interface": INTERFACE,
        "protocol": {
            "native_shape_hw": list(shape_hw),
            "image_or_depth_resize": False,
            "detector": "direct spherical DoG on native ERP",
            "descriptor": "48x48 tangent RootSIFT r6/d1.5",
            "descriptor_adapter": "panorai-tangent-opencv-descriptor/v2",
            "matching": "bidirectional Lowe mutual + bilateral spherical NMS",
            "pose": args.pose_mode,
            "pose_refinement": "five-point LO-RANSAC + nonminimal consensus refit",
            "optimized_quantity": "radial range only",
            "ground_truth_used_for_prediction": False,
        },
        "inputs": {
            "prior": str(args.prior),
            "prior_sha256": sha256(args.prior),
            "ground_truth": str(args.ground_truth),
            "ground_truth_sha256": sha256(args.ground_truth),
            "evaluation_validity": str(args.evaluation_validity),
            "evaluation_validity_sha256": sha256(args.evaluation_validity),
        },
        "configuration": {
            "depth": options.to_dict(),
            "extractor": extractor.config.to_dict(),
            "matcher": matcher.config.to_dict(),
            "estimator": estimator.options.to_dict(),
        },
        "sources": source_records,
        "dense_changed_diagnostic": changed_diagnostic,
        "evaluations": evaluations,
        "artifacts": {
            "prediction": str(prediction_path),
            "prediction_sha256": sha256(prediction_path),
            "sparse_seeds": str(sparse_path),
            "sparse_seeds_sha256": sha256(sparse_path),
            "freeze": str(freeze_path),
            "freeze_sha256": sha256(freeze_path),
            "panel": str(panel_path),
            "panel_sha256": sha256(panel_path),
        },
        "timings": timings,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "opencv_threads": args.threads,
        },
    }
    report_path = args.output / "results.json"
    _json_dump(report_path, report)
    print(
        json.dumps(
            {
                "event": "complete",
                "results": str(report_path),
                "seed_abs_rel": evaluations["cnn-seed"]["metric_depth"]["abs_rel"],
                "tangent_abs_rel": evaluations["tangent-seed"]["metric_depth"][
                    "abs_rel"
                ],
                "changed": changed_diagnostic,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
