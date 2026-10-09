#!/usr/bin/env python3
"""Benchmark the installed public optimized spherical pair route on P74.

This runner intentionally uses public ``panorai`` imports only.  It is designed
to be copied outside the source checkout and executed by an isolated Python
environment containing a wheel built from an exact ``origin/main`` commit.
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
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np


SCHEMA = "panorai-optimized-public-pair-benchmark/v1"
PROFILE_INTERFACE = "panorai-optimized-public-pair-profile/v1"
P74_ADAPTER = "eq-native-polar-0-150-endpoint-inclusive/v1"
P74_SHADOW_ANGLE_DEG = 30.0
P74_CANONICAL_SHAPE_HW = (2048, 4096)
# Despite the historical adapter naming, this matrix maps PanorAi rays into
# the native P74 scanner/evaluation frame.  Its transpose maps the other way.
P74_FROM_PANORAI = np.asarray(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def profile_configuration() -> dict[str, Any]:
    """Return the exact, serializable route required by VAL-016."""

    return {
        "interface": PROFILE_INTERFACE,
        "route": {
            "detector_class": "SphericalDoGDetector",
            "detector_method": "detect_batch",
            "batch_size": 2,
            "sequential_detection": False,
            "validity_masks": "explicit-per-panorama",
            "patch_provider_class": "TangentPatchProvider",
            "patch_provider_max_workers": 4,
            "multiface_route": False,
            "private_imports": False,
        },
        "detector": {
            "octaves": 3,
            "levels_per_octave": 3,
            "base_sigma_px": 1.6,
            "contrast_threshold": 0.012,
            "edge_threshold": 10.0,
            "refinement_max_iterations": 5,
            "max_keypoints": 4096,
            "minimum_valid_support_fraction": 0.99,
            "angular_dedup_threshold_deg": 0.12,
            "scale_dedup_log2": 0.5,
            "selection_policy": "equal-area-round-robin",
            "selection_grid_shape": [12, 24],
            "convolution_backend": "native",
        },
        "patches": {
            "output_shape_hw": [48, 48],
            "radius_in_scales": 6.0,
            "minimum_fov_deg": 1.0,
            "maximum_fov_deg": 120.0,
            "interpolation": "bilinear",
            "invalid_policy": "propagate",
            "minimum_valid_fraction": 0.99,
            "orientation_policy": "upright",
        },
        "descriptor": {
            "method": "sift",
            "keypoint_diameter_in_scales": 1.25,
            "scale_multipliers": [1.0],
            "orientation_policy": "fixed-zero",
            "photometric_normalization": "local-standardization",
            "minimum_descriptor_valid_fraction": 0.99,
            "root_sift": True,
        },
        "matcher": {
            "method": "flann",
            "ratio_test": 0.72,
            "cross_check": False,
            "deduplicate_matches": True,
            "angular_dedup_threshold_deg": 0.15,
        },
        "pose": {
            "max_angular_error_deg": 1.0,
            "max_num_trials": 1000,
            "stability_trials": 6,
            "model_competition_trials": 128,
            "random_seed": 7,
            "hypothesis_ranking": "msac-first",
            "nonminimal_refit_max_steps": 100,
            "sampler": "spatially-weighted",
        },
    }


def _read_jsonl_row(path: Path, pair_id: str | None) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if pair_id is None or row.get("pair_id") == pair_id:
            return row
    raise ValueError(f"pair {pair_id!r} not found in {path}")


def _read_rgb(path: str) -> np.ndarray:
    import cv2

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _scan_bearings_to_p74_pixels(
    bearings: np.ndarray, *, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    rays = np.asarray(bearings, dtype=np.float64)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    longitude = np.arctan2(-rays[:, 1], rays[:, 0])
    latitude = np.arcsin(np.clip(rays[:, 2], -1.0, 1.0))
    polar = math.pi / 2.0 - latitude
    pixels = np.column_stack(
        (
            np.mod(longitude / (2.0 * math.pi) * (width - 1), width - 1),
            polar / math.radians(150.0) * (height - 1),
        )
    )
    valid = (polar >= -1e-12) & (polar <= math.radians(150.0) + 1e-12)
    return pixels, valid


def adapt_p74_native_polar(
    source: np.ndarray, output_shape_hw: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the audited P74 adapter and return explicit geometric validity."""

    import cv2
    from panorai.data import EquirectangularImage
    from panorai.geometry import erp_pixels_to_rays
    from panorai.image_processing import spherical_resize

    if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        raise ValueError("P74 source must be uint8 HWC RGB")
    observed_height, width = source.shape[:2]
    materialized_height = int(
        round(observed_height / (1.0 - P74_SHADOW_ANGLE_DEG / 180.0))
    )
    observed_rgb = np.empty((observed_height, width, 3), dtype=np.uint8)
    columns = np.arange(width, dtype=np.float64)
    for row_start in range(0, observed_height, 128):
        row_stop = min(observed_height, row_start + 128)
        xx, yy = np.meshgrid(columns, np.arange(row_start, row_stop, dtype=np.float64))
        target_pixels = np.column_stack((xx.ravel(), yy.ravel()))
        panorai_rays = erp_pixels_to_rays(target_pixels, (materialized_height, width))
        native_pixels, valid = _scan_bearings_to_p74_pixels(
            (P74_FROM_PANORAI @ panorai_rays.T).T,
            width=width,
            height=observed_height,
        )
        if not valid.all():
            raise RuntimeError("observed P74 grid escaped native polar support")
        observed_rgb[row_start:row_stop] = cv2.remap(
            source,
            native_pixels[:, 0].reshape(row_stop - row_start, width).astype(np.float32),
            native_pixels[:, 1].reshape(row_stop - row_start, width).astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
    panorama = EquirectangularImage(
        observed_rgb,
        shadow_angle=P74_SHADOW_ANGLE_DEG,
        shadow_padded=False,
    )
    panorama.preprocess()
    validity = np.asarray(panorama.validity("image"), dtype=bool)
    validity &= np.asarray(panorama.support_mask, dtype=bool)
    image = np.clip(
        np.rint(spherical_resize(panorama.image, output_shape_hw)), 0.0, 255.0
    ).astype(np.uint8)
    validity = spherical_resize(
        validity.astype(np.uint8), output_shape_hw, interpolation="nearest"
    ).astype(bool)
    return image, validity


def _materialize_feature_set(
    keypoints: Any,
    patches: Any,
    described: Any,
    *,
    extractor_config: dict[str, Any],
) -> Any:
    """Assemble public feature objects from the explicitly timed public stages."""

    import cv2
    from panorai.features import (
        FaceSetSpec,
        FeatureProvenance,
        SphericalFeature,
        SphericalFeatureSet,
    )

    interface = "panorai-optimized-public-pair-features/v1"
    features: list[Any] = []
    descriptors: list[np.ndarray] = []
    for described_index, keypoint_index in enumerate(
        described.physical_keypoint_ids.tolist()
    ):
        keypoint = keypoints.keypoints[keypoint_index]
        patch = patches.patches[int(described.patch_indices[described_index])]
        output_index = len(features)
        face_id = f"tangent-keypoint-{output_index:06d}"
        provenance = FeatureProvenance(
            interface=interface,
            source_panorama_checksum=keypoints.source_checksum,
            projection_backend="panorai.geometry",
            projection_backend_version=keypoints.projection_backend_version,
            face_id=face_id,
            generating_commit=keypoints.generating_commit,
            selection_reason="spherical-dog-batch-then-parallel-tangent-rootsift",
        )
        centre = (patches.request.output_shape_hw[0] - 1.0) / 2.0
        features.append(
            SphericalFeature(
                feature_id=f"{keypoints.panorama_id}:feature-{output_index:06d}",
                panorama_id=keypoints.panorama_id,
                face_id=face_id,
                pixel_xy=np.asarray((centre, centre), dtype=np.float64),
                source_erp_xy=np.asarray(keypoint.source_erp_xy).copy(),
                bearing_xyz=np.asarray(keypoint.bearing_xyz).copy(),
                response=float(keypoint.response),
                scale=float(keypoint.scale_deg),
                angle_deg=float(described.angles_deg[described_index]),
                octave=int(keypoint.octave),
                descriptor_index=output_index,
                valid=True,
                projection_spec=patch.geometry.projection_spec,
                provenance=provenance,
            )
        )
        descriptors.append(np.asarray(described.descriptors[described_index]))
    descriptor_array = (
        np.stack(descriptors).astype(np.float32, copy=False)
        if descriptors
        else np.empty((0, described.descriptors.shape[1]), dtype=np.float32)
    )
    return SphericalFeatureSet(
        panorama_id=keypoints.panorama_id,
        features=features,
        descriptors=descriptor_array,
        descriptor_type=described.descriptor_type,
        descriptor_metric=described.descriptor_metric,
        extractor_name=described.extractor_name,
        extractor_config=extractor_config,
        backend_name="opencv",
        backend_version=cv2.__version__,
        face_set_spec=FaceSetSpec(
            sampler="per-keypoint-tangent",
            shape_hw=patches.request.output_shape_hw,
            fov_deg=(1.0, 1.0),
        ),
        projection_backend="panorai.geometry",
        projection_backend_version=keypoints.projection_backend_version,
        panorama_checksum=keypoints.source_checksum,
        generating_commit=keypoints.generating_commit,
        interface=interface,
        stability="experimental",
    )


def _peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return value / divisor


def _cpu_name() -> str:
    if sys.platform == "darwin":
        completed = subprocess.run(
            ("sysctl", "-n", "machdep.cpu.brand_string"),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout.strip()
        completed = subprocess.run(
            ("system_profiler", "SPHardwareDataType"),
            check=False,
            capture_output=True,
            text=True,
        )
        for line in completed.stdout.splitlines():
            if line.strip().startswith("Chip:"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _opencv_parallel_framework(cv2: Any) -> str:
    for line in cv2.getBuildInformation().splitlines():
        if "Parallel framework:" in line:
            return line.split(":", 1)[1].strip()
    return "unknown"


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    cosine = float(first @ second) / (
        float(np.linalg.norm(first)) * float(np.linalg.norm(second))
    )
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _pose_errors(pose: Any, evaluation: dict[str, Any] | None) -> dict[str, Any]:
    if pose is None or evaluation is None:
        return {}
    reference = evaluation["reference"]
    expected_rotation = np.asarray(reference["R_to_from"], dtype=np.float64)
    estimated_rotation = (
        P74_FROM_PANORAI
        @ np.asarray(pose.rotation, dtype=np.float64)
        @ P74_FROM_PANORAI.T
    )
    estimated_translation = P74_FROM_PANORAI @ np.asarray(
        pose.translation_direction, dtype=np.float64
    )
    delta = estimated_rotation @ expected_rotation.T
    rotation_error = math.degrees(
        math.acos(float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)))
    )
    translation_error = _angle_deg(
        estimated_translation,
        np.asarray(reference["t_direction_to_from"], dtype=np.float64),
    )
    return {
        "rotation_error_deg": rotation_error,
        "translation_direction_error_deg": translation_error,
        "evaluation_frame": "p74-native-scanner",
        "strict_valid": rotation_error <= 3.0 and translation_error <= 10.0,
        "precise_valid": rotation_error <= 1.0 and translation_error <= 5.0,
    }


def run_pair(
    pair: dict[str, Any],
    *,
    evaluation: dict[str, Any] | None,
    output_shape_hw: tuple[int, int],
    expected_source_commit: str,
) -> dict[str, Any]:
    """Run one optimized pair and return stage-resolved evidence."""

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

    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    configuration = profile_configuration()

    preparation_started = perf_counter()
    source_a = _read_rgb(pair["from_rgb_path"])
    source_b = _read_rgb(pair["to_rgb_path"])
    if pair.get("input_adapter") != P74_ADAPTER:
        raise ValueError(f"unsupported adapter {pair.get('input_adapter')!r}")
    panorama_a, validity_a = adapt_p74_native_polar(source_a, output_shape_hw)
    panorama_b, validity_b = adapt_p74_native_polar(source_b, output_shape_hw)
    preparation_seconds = perf_counter() - preparation_started
    if validity_a.dtype != np.bool_ or validity_b.dtype != np.bool_:
        raise RuntimeError("validity masks must remain explicit boolean arrays")

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
    descriptor_adapter = OpenCVTangentDescriptorV2(descriptor_config)
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
        (panorama_a, panorama_b),
        panorama_ids=(pair["from_view_id"], pair["to_view_id"]),
        validity_masks=(validity_a, validity_b),
    )
    detection_seconds = perf_counter() - started

    started = perf_counter()
    patches_a = patch_provider.materialize(
        panorama_a, keypoints_a, patch_request, validity_mask=validity_a
    )
    patches_b = patch_provider.materialize(
        panorama_b, keypoints_b, patch_request, validity_mask=validity_b
    )
    patch_seconds = perf_counter() - started

    started = perf_counter()
    described_a = descriptor_adapter.describe(
        patches_a,
        responses=np.asarray(
            [keypoint.response for keypoint in keypoints_a.keypoints],
            dtype=np.float64,
        ),
    )
    described_b = descriptor_adapter.describe(
        patches_b,
        responses=np.asarray(
            [keypoint.response for keypoint in keypoints_b.keypoints],
            dtype=np.float64,
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

    started = perf_counter()
    correspondences = matches.to_bearing_correspondences()
    pose = estimator.estimate(correspondences)
    pose_seconds = perf_counter() - started
    total_seconds = perf_counter() - total_started

    pose_record: dict[str, Any] = {"returned": pose is not None}
    if pose is not None:
        pose_record.update(
            rotation=np.asarray(pose.rotation).tolist(),
            translation_direction=np.asarray(pose.translation_direction).tolist(),
            inlier_count=int(pose.num_inliers),
            num_trials=int(pose.num_trials),
            quality_accepted=bool(pose.quality_report.accepted),
            quality_rejection_reasons=list(pose.quality_report.rejection_reasons),
            median_parallax_deg=float(pose.median_parallax_deg),
            cheirality_ratio=float(pose.cheirality_ratio),
        )
        pose_record.update(_pose_errors(pose, evaluation))

    package_version = importlib.metadata.version("panorai")
    package_path = str(Path(panorai.__file__).resolve())
    system = {
        "cpu": _cpu_name(),
        "logical_cpu_count": os.cpu_count(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "opencv": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "opencv_requested_threads": 1,
        "opencv_thread_control_effective": cv2.getNumThreads() == 1,
        "opencv_parallel_framework": _opencv_parallel_framework(cv2),
        "patch_workers": 4,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    overlap = None if evaluation is None else evaluation.get("covariates", {})
    return {
        "schema": SCHEMA,
        "pair_id": pair["pair_id"],
        "overlap": overlap,
        "resolution_hw": list(output_shape_hw),
        "configuration": configuration,
        "resolved_configuration": {
            "detector": detector_config.to_dict(),
            "patches": asdict(patch_request),
            "descriptor": descriptor_config.to_dict(),
            "matcher": matcher.config.to_dict(),
            "pose": estimator.options.to_dict(),
            "sampler": estimator.sampler.name,
        },
        "package": {
            "name": "panorai",
            "version": package_version,
            "import_path": package_path,
            "expected_source_commit": expected_source_commit,
            "detector_generating_commits": [
                keypoints_a.generating_commit,
                keypoints_b.generating_commit,
            ],
        },
        "native": {
            "requested_convolution_backend": detector_config.convolution_backend,
            "native_filter_available": bool(native_filter_available()),
            "native_pose_kernels_available": bool(native_kernels_available()),
            "numpy_fallback_permitted": False,
        },
        "validity": {
            "source": "explicit P74 geometric support mask",
            "derived_from_black_pixels": False,
            "valid_fraction_a": float(validity_a.mean()),
            "valid_fraction_b": float(validity_b.mean()),
        },
        "counts": {
            "keypoints_a": len(keypoints_a),
            "keypoints_b": len(keypoints_b),
            "valid_patches_a": int(patches_a.valid_count),
            "valid_patches_b": int(patches_b.valid_count),
            "descriptors_a": len(described_a),
            "descriptors_b": len(described_b),
            "matches": len(matches),
            "valid_matches": int(matches.valid.sum()),
        },
        "timings_seconds": {
            "input_preparation": preparation_seconds,
            "detection_pair": detection_seconds,
            "detection_per_image": detection_seconds / 2.0,
            "patches_pair": patch_seconds,
            "patches_per_image": patch_seconds / 2.0,
            "descriptor_pair": descriptor_seconds,
            "descriptor_per_image": descriptor_seconds / 2.0,
            "image_ready_per_image": (
                detection_seconds + patch_seconds + descriptor_seconds
            )
            / 2.0,
            "matching": matching_seconds,
            "pose": pose_seconds,
            "pair_total": total_seconds,
            "end_to_end_with_input_preparation": preparation_seconds + total_seconds,
        },
        "pose": pose_record,
        "system": system,
        "peak_rss_mib": _peak_rss_mib(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--pair-id")
    parser.add_argument("--height", type=int, choices=(1024, 2048), required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair = _read_jsonl_row(args.inputs, args.pair_id)
    evaluation = (
        None
        if args.evaluation is None
        else _read_jsonl_row(args.evaluation, pair["pair_id"])
    )
    result = run_pair(
        pair,
        evaluation=evaluation,
        output_shape_hw=(args.height, 2 * args.height),
        expected_source_commit=args.expected_source_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
