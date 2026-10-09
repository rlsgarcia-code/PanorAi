#!/usr/bin/env python3
"""Run one pair through the exact optimized public spherical frontend route.

Copy this file and ``run_optimized_public_pair.py`` outside the checkout, then
execute it with a Python environment containing an exact wheel from
``origin/main``. The runner rejects source-checkout PanorAi imports.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
from time import perf_counter
from typing import Any

import numpy as np

try:
    from run_optimized_public_pair import (  # type: ignore[import-not-found]
        P74_FROM_PANORAI,
        P74_ADAPTER,
        _materialize_feature_set,
        _read_rgb,
        adapt_p74_native_polar,
        profile_configuration,
    )
except ImportError:
    from benchmarks.p74_pair_eligibility.run_optimized_public_pair import (
        P74_FROM_PANORAI,
        P74_ADAPTER,
        _materialize_feature_set,
        _read_rgb,
        adapt_p74_native_polar,
        profile_configuration,
    )

SCHEMA = "panorai-unified-optimized-pair/v2"
PANORAI_FROM_DATASET = {
    "matterport360": np.diag((1.0, 1.0, -1.0)),
    "stanford2d3d": np.diag((1.0, -1.0, 1.0)),
    "p74_native_polar": P74_FROM_PANORAI,
}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _row(path: Path, pair_id: str) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if value.get("pair_id") == pair_id:
            return value
    raise KeyError(f"{pair_id!r} is absent from {path}")


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(first @ second) / (
        float(np.linalg.norm(first)) * float(np.linalg.norm(second))
    )
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _pose_record(
    pose: Any | None, evaluation: dict[str, Any], dataset_id: str
) -> dict[str, Any]:
    if pose is None:
        return {"returned": False, "quality_accepted": False}
    adapter = PANORAI_FROM_DATASET[dataset_id]
    estimated_rotation = adapter @ np.asarray(pose.rotation) @ adapter.T
    estimated_translation = adapter @ np.asarray(pose.translation_direction)
    reference = evaluation["reference"]
    expected_rotation = np.asarray(reference["R_to_from"], dtype=np.float64)
    delta = estimated_rotation @ expected_rotation.T
    rotation_error = math.degrees(
        math.acos(float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)))
    )
    translation_error = _angle(
        estimated_translation,
        np.asarray(reference["t_direction_to_from"], dtype=np.float64),
    )
    report = pose.quality_report
    return {
        "returned": True,
        "quality_accepted": bool(report.accepted),
        "quality_rejection_reasons": list(report.rejection_reasons),
        "rotation": np.asarray(pose.rotation).tolist(),
        "translation_direction": np.asarray(pose.translation_direction).tolist(),
        "rotation_error_deg": rotation_error,
        "translation_direction_error_deg": translation_error,
        "primary": rotation_error <= 15.0 and translation_error <= 30.0,
        "strict": rotation_error <= 5.0 and translation_error <= 10.0,
        "precise": rotation_error <= 1.0 and translation_error <= 5.0,
        "catastrophic_accepted": bool(
            report.accepted
            and not (rotation_error <= 15.0 and translation_error <= 30.0)
        ),
        "inlier_count": int(pose.num_inliers),
        "num_trials": int(pose.num_trials),
        "median_parallax_deg": float(pose.median_parallax_deg),
        "cheirality_ratio": float(pose.cheirality_ratio),
        "raw_quality_score": float(report.raw_quality_score),
        "median_residual_deg": float(report.median_residual_deg),
        "p90_residual_deg": float(report.p90_residual_deg),
        "quality_report": report.to_dict(),
        "degenerate": bool(pose.degenerate),
        "degeneracy_reasons": list(pose.degeneracy_reasons),
        "compute_backend": pose.compute_backend,
        "sampling_diagnostics": pose.sampling_diagnostics.to_dict(),
    }


def _prepare(
    pair: dict[str, Any], output_shape_hw: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    from panorai.image_processing import spherical_resize

    source_a = _read_rgb(pair["from_rgb_path"])
    source_b = _read_rgb(pair["to_rgb_path"])
    if pair["dataset_id"] == "p74_native_polar":
        if pair.get("input_adapter") != P74_ADAPTER:
            raise ValueError("P74 input requires the audited native-polar adapter")
        image_a, valid_a = adapt_p74_native_polar(source_a, output_shape_hw)
        image_b, valid_b = adapt_p74_native_polar(source_b, output_shape_hw)
        source = "explicit P74 geometric support mask"
    else:

        def resize(image: np.ndarray) -> np.ndarray:
            if image.shape[:2] == output_shape_hw:
                return image
            return np.clip(
                np.rint(spherical_resize(image, output_shape_hw)), 0.0, 255.0
            ).astype(np.uint8)

        image_a = resize(source_a)
        image_b = resize(source_b)
        valid_a = np.ones(output_shape_hw, dtype=bool)
        valid_b = np.ones(output_shape_hw, dtype=bool)
        source = "explicit full-ERP dataset support; not inferred from pixel values"
    return image_a, image_b, valid_a, valid_b, source


def run(
    pair: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    output_shape_hw: tuple[int, int],
    expected_source_commit: str,
    forbidden_checkout: Path | None,
) -> dict[str, Any]:
    import cv2
    import panorai
    from panorai.estimators import (
        RelativePoseOptions,
        SpatiallyWeightedFivePointSampler,
        SphericalRelativePoseEstimator,
        native_kernels_available,
    )
    from panorai.features import (
        FeatureMatcher,
        FeatureMatcherConfig,
        OpenCVTangentDescriptorV2,
        OpenCVTangentDescriptorV2Config,
        SphericalDoGDetector,
        SphericalDoGDetectorConfig,
        TangentPatchProvider,
        TangentPatchRequest,
    )
    from panorai.image_processing import native_filter_available

    package_path = Path(panorai.__file__).resolve()
    if forbidden_checkout is not None and package_path.is_relative_to(
        forbidden_checkout.resolve()
    ):
        raise RuntimeError(f"PanorAi resolved from forbidden checkout: {package_path}")
    if not native_filter_available() or not native_kernels_available():
        raise RuntimeError("native convolution and pose kernels are required")
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)

    preparation_started = perf_counter()
    image_a, image_b, valid_a, valid_b, validity_source = _prepare(
        pair, output_shape_hw
    )
    preparation_seconds = perf_counter() - preparation_started

    detector_config = SphericalDoGDetectorConfig(
        octaves=3,
        levels_per_octave=3,
        base_sigma_px=1.6,
        contrast_threshold=0.012,
        edge_threshold=10.0,
        refinement_max_iterations=5,
        max_keypoints=4096,
        minimum_valid_support_fraction=0.99,
        angular_dedup_threshold_deg=0.12,
        scale_dedup_log2=0.5,
        selection_policy="equal-area-round-robin",
        selection_grid_shape=(12, 24),
        convolution_backend="native",
    )
    detector = SphericalDoGDetector(detector_config)
    patch_request = TangentPatchRequest(
        output_shape_hw=(48, 48),
        radius_in_scales=6.0,
        minimum_fov_deg=1.0,
        maximum_fov_deg=120.0,
        interpolation="bilinear",
        invalid_policy="propagate",
        minimum_valid_fraction=0.99,
        orientation_policy="upright",
    )
    patch_provider = TangentPatchProvider(max_workers=4)
    descriptor_config = OpenCVTangentDescriptorV2Config(
        method="sift",
        keypoint_diameter_in_scales=1.25,
        scale_multipliers=(1.0,),
        orientation_policy="fixed-zero",
        photometric_normalization="local-standardization",
        minimum_descriptor_valid_fraction=0.99,
        root_sift=True,
    )
    descriptor = OpenCVTangentDescriptorV2(descriptor_config)
    matcher = FeatureMatcher(
        FeatureMatcherConfig(
            method="flann",
            ratio_test=0.72,
            cross_check=False,
            deduplicate_matches=True,
            angular_dedup_threshold_deg=0.15,
        )
    )
    estimator = SphericalRelativePoseEstimator(
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

    total_started = perf_counter()
    started = perf_counter()
    keypoints_a, keypoints_b = detector.detect_batch(
        (image_a, image_b),
        panorama_ids=(pair["from_view_id"], pair["to_view_id"]),
        validity_masks=(valid_a, valid_b),
    )
    detection_seconds = perf_counter() - started
    commits = [keypoints_a.generating_commit, keypoints_b.generating_commit]
    embedded_commits = [value for value in commits if value]
    if embedded_commits and not all(
        expected_source_commit[:8] in value for value in embedded_commits
    ):
        raise RuntimeError(
            f"detector commits {commits} do not contain {expected_source_commit[:8]}"
        )
    detector_provenance = (
        "embedded commit matches expected source"
        if embedded_commits
        else "release wheel omits detector commit; require external wheel hash and Git tree proof"
    )

    started = perf_counter()
    patches_a = patch_provider.materialize(
        image_a, keypoints_a, patch_request, validity_mask=valid_a
    )
    patches_b = patch_provider.materialize(
        image_b, keypoints_b, patch_request, validity_mask=valid_b
    )
    patch_seconds = perf_counter() - started
    started = perf_counter()
    described_a = descriptor.describe(
        patches_a,
        responses=np.asarray(
            [keypoint.response for keypoint in keypoints_a.keypoints], dtype=np.float64
        ),
    )
    described_b = descriptor.describe(
        patches_b,
        responses=np.asarray(
            [keypoint.response for keypoint in keypoints_b.keypoints], dtype=np.float64
        ),
    )
    feature_config = {
        "detector": detector_config.to_dict(),
        "patches": asdict(patch_request),
        "descriptor": descriptor_config.to_dict(),
    }
    features_a = _materialize_feature_set(
        keypoints_a, patches_a, described_a, extractor_config=feature_config
    )
    features_b = _materialize_feature_set(
        keypoints_b, patches_b, described_b, extractor_config=feature_config
    )
    descriptor_seconds = perf_counter() - started
    started = perf_counter()
    matches = matcher.match(features_a, features_b)
    matching_seconds = perf_counter() - started
    valid_match_mask = np.asarray(matches.valid, dtype=bool)
    valid_distances = np.asarray(matches.descriptor_distances)[valid_match_mask]
    valid_ratios = (
        None
        if matches.ratio_scores is None
        else np.asarray(matches.ratio_scores)[valid_match_mask]
    )
    started = perf_counter()
    pose = estimator.estimate(matches.to_bearing_correspondences())
    pose_seconds = perf_counter() - started
    total_seconds = perf_counter() - total_started

    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    rss /= 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return {
        "schema": SCHEMA,
        "pair_id": pair["pair_id"],
        "dataset_id": pair["dataset_id"],
        "partition": pair.get("partition"),
        "resolution_hw": list(output_shape_hw),
        "package": {
            "version": importlib.metadata.version("panorai"),
            "import_path": str(package_path),
            "expected_source_commit": expected_source_commit,
            "detector_generating_commits": commits,
            "detector_provenance": detector_provenance,
        },
        "route": profile_configuration(),
        "native": {
            "convolution_backend": detector_config.convolution_backend,
            "native_filter_available": True,
            "native_pose_kernels_available": True,
            "numpy_fallback_permitted": False,
        },
        "validity": {
            "source": validity_source,
            "derived_from_black_pixels": False,
            "valid_fraction_a": float(valid_a.mean()),
            "valid_fraction_b": float(valid_b.mean()),
        },
        "counts": {
            "keypoints_a": len(keypoints_a),
            "keypoints_b": len(keypoints_b),
            "descriptors_a": len(described_a),
            "descriptors_b": len(described_b),
            "matches": len(matches),
            "valid_matches": int(matches.valid.sum()),
        },
        "matching_diagnostics": {
            "descriptor_distance_median": (
                float(np.median(valid_distances)) if len(valid_distances) else None
            ),
            "descriptor_distance_p90": (
                float(np.quantile(valid_distances, 0.9))
                if len(valid_distances)
                else None
            ),
            "ratio_score_median": (
                float(np.median(valid_ratios))
                if valid_ratios is not None and len(valid_ratios)
                else None
            ),
        },
        "timings_seconds": {
            "input_preparation": preparation_seconds,
            "detection_pair": detection_seconds,
            "detection_per_image": detection_seconds / 2.0,
            "patches_pair": patch_seconds,
            "descriptor_pair": descriptor_seconds,
            "image_ready_per_image": (
                detection_seconds + patch_seconds + descriptor_seconds
            )
            / 2.0,
            "matching": matching_seconds,
            "pose": pose_seconds,
            "pair_total": total_seconds,
        },
        "pose": _pose_record(pose, evaluation, pair["dataset_id"]),
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "opencv": cv2.__version__,
            "opencv_threads": cv2.getNumThreads(),
            "logical_cpu_count": os.cpu_count(),
            "patch_workers": 4,
        },
        "peak_rss_mib": rss,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--forbidden-checkout", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run(
        _row(args.inputs, args.pair_id),
        _row(args.evaluation, args.pair_id),
        output_shape_hw=(args.height, 2 * args.height),
        expected_source_commit=args.expected_source_commit,
        forbidden_checkout=args.forbidden_checkout,
    )
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
