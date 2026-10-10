#!/usr/bin/env python3
"""Compare uniform and real-ImageNet-CAM pose sampling on one ERP pair."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAIR_PATH = Path(__file__).with_name("stanford_office3_cam_pair.json")
REFERENCE_PATH = Path(__file__).with_name("stanford_office3_reference.json")
MANUAL_REGIONS_PATH = Path(__file__).with_name("stanford_office3_regions.json")
DEFAULT_DATASET_ROOT = Path(
    "/Users/robinsongarcia/projects/PanorAi-projects/data/training/stanford2d3d"
)
SCHEMA = "panorai-real-cam-pose-prior/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _atomic_json(path: Path, value: dict[str, Any], *, readonly: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if readonly:
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _checked_path(root: Path, relative_path: str, expected_sha256: str) -> Path:
    path = (root / relative_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"checksum mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
    return path


def _preflight(dataset_root: Path) -> dict[str, Any]:
    import torch

    pair = _load_json(PAIR_PATH)
    checked = [
        {
            "kind": "rgb",
            "path": str(
                _checked_path(
                    dataset_root, view["rgb_relative_path"], view["rgb_sha256"]
                )
            ),
            "sha256": view["rgb_sha256"],
        }
        for view in pair["views"]
    ]
    weight_path = (
        Path(torch.hub.get_dir()) / "checkpoints" / pair["model"]["weights_filename"]
    )
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)
    digest = _sha256(weight_path)
    expected_prefix = pair["model"]["weights_filename"].split("-")[-1].split(".")[0]
    if not digest.startswith(expected_prefix):
        raise RuntimeError("cached ImageNet checkpoint hash prefix does not match")
    return {
        "ready": True,
        "dataset_root": str(dataset_root.resolve()),
        "pair_manifest": str(PAIR_PATH),
        "checked_inputs": checked,
        "model_checkpoint": {
            "path": str(weight_path),
            "sha256": digest,
            "size_bytes": weight_path.stat().st_size,
        },
        "manual_regions_used_during_prediction": False,
        "reference_pose_used_during_prediction": False,
        "source_or_model_bytes_will_be_copied": False,
    }


def _read_and_resize(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    from panorai.image_processing import spherical_resize

    image = np.asarray(Image.open(path).convert("RGB"))
    if image.shape[:2] != shape_hw:
        image = np.clip(np.rint(spherical_resize(image, shape_hw)), 0, 255).astype(
            np.uint8
        )
    return image


def _feature_pipeline():
    from panorai.features import SphericalFeaturePipeline

    return SphericalFeaturePipeline.from_preset(
        "sift-flann",
        face_sampler="cube",
        face_shape_hw=768,
        face_fov_deg=95.0,
        max_features=4096,
        ratio_test=0.72,
        angular_dedup_threshold_deg=0.15,
        edge_margin_px=16,
        validity_scale_margin=1.5,
    )


def _pose_estimator():
    from panorai.estimators import (
        RelativePoseOptions,
        SpatiallyWeightedFivePointSampler,
        SphericalRelativePoseEstimator,
    )

    return SphericalRelativePoseEstimator(
        RelativePoseOptions(
            max_angular_error_deg=1.0,
            max_num_trials=1000,
            stability_trials=3,
            model_competition_trials=64,
            random_seed=7,
            hypothesis_ranking="msac-first",
            nonminimal_refit_max_steps=100,
        ),
        sampler=SpatiallyWeightedFivePointSampler(),
    )


def _infer_cams(
    images: tuple[np.ndarray, np.ndarray], pair: dict[str, Any]
) -> tuple[list[dict[int, np.ndarray]], list[dict[str, Any]], dict[str, Any]]:
    import torch

    from panorai.experimental.deep_learning import (
        ImageNetFCN,
        class_activation_map,
        load_pretrained_imagenet_model,
        port_module_with_report,
        spherical_area_average,
    )

    torch.set_num_threads(4)
    architecture = pair["model"]["architecture"]
    loaded = load_pretrained_imagenet_model(architecture, progress=False)
    dense_model = ImageNetFCN(loaded.model, architecture).eval()
    port_report = port_module_with_report(dense_model)
    if port_report.remaining_planar_spatial_layers:
        raise RuntimeError("planar spatial layers remain after spherical port")
    transforms = loaded.weights.transforms()
    mean = torch.tensor(transforms.mean, dtype=torch.float32)[None, :, None, None]
    std = torch.tensor(transforms.std, dtype=torch.float32)[None, :, None, None]
    target_shape = tuple(int(value) for value in pair["cam_input_shape_hw"])
    resized = [
        np.asarray(
            Image.fromarray(image).resize(
                (target_shape[1], target_shape[0]), Image.Resampling.LANCZOS
            )
        ).copy()
        for image in images
    ]
    values = torch.stack(
        [torch.from_numpy(image).permute(2, 0, 1).float() / 255.0 for image in resized]
    )
    normalized = (values - mean) / std
    class_records = list(pair["query"]["classes"]) + list(
        pair["control_query"]["classes"]
    )
    class_ids = [int(item["class_id"]) for item in class_records]
    with torch.inference_mode():
        dense = dense_model.forward_dense(normalized)
        scores = spherical_area_average(dense.logits)
        probabilities = scores.softmax(dim=1)
        maps = {
            class_id: class_activation_map(dense.logits, class_id).cpu().numpy()
            for class_id in class_ids
        }
    categories = loaded.weights.meta["categories"]
    cams: list[dict[int, np.ndarray]] = []
    diagnostics: list[dict[str, Any]] = []
    for view_index in range(len(images)):
        view_maps: dict[int, np.ndarray] = {}
        class_diagnostics = []
        for record in class_records:
            class_id = int(record["class_id"])
            if categories[class_id] != record["class_name"]:
                raise RuntimeError("ImageNet class manifest does not match weights")
            cam = np.asarray(maps[class_id][view_index], dtype=np.float32)
            view_maps[class_id] = cam
            probability = float(probabilities[view_index, class_id])
            class_diagnostics.append(
                {
                    "class_id": class_id,
                    "class_name": record["class_name"],
                    "probability": probability,
                    "rank": int((probabilities[view_index] > probability).sum()) + 1,
                    "cam_shape_hw": list(cam.shape),
                    "cam_nonzero_fraction": float(np.mean(cam > 0.0)),
                    "cam_mean": float(cam.mean()),
                    "cam_standard_deviation": float(cam.std()),
                }
            )
        cams.append(view_maps)
        diagnostics.append({"classes": class_diagnostics})
    model_record = {
        "architecture": architecture,
        "weights": str(loaded.weights),
        "checkpoint": loaded.checkpoint.to_dict(),
        "port": port_report.to_dict(),
        "input_shape_nchw": list(normalized.shape),
        "logit_shape_nchw": list(dense.logits.shape),
    }
    return cams, diagnostics, model_record


def _pose_record(pose: Any) -> dict[str, Any]:
    if pose is None:
        return {"estimated": False, "quality_accepted": False}
    return {
        "estimated": True,
        "quality_accepted": bool(pose.quality_report.accepted),
        "quality_rejection_reasons": list(pose.quality_report.rejection_reasons),
        "rotation": pose.rotation.tolist(),
        "translation_direction": pose.translation_direction.tolist(),
        "num_inliers": int(pose.num_inliers),
        "inlier_ratio": float(pose.num_inliers / len(pose.inlier_mask)),
        "median_residual_deg": float(pose.quality_report.median_residual_deg),
        "median_parallax_deg": float(pose.median_parallax_deg),
        "sampling": pose.sampling_diagnostics.to_dict(),
    }


def _rotation_error_deg(actual: np.ndarray, expected: np.ndarray) -> float:
    def nearest_rotation(value: np.ndarray) -> np.ndarray:
        left, _, right = np.linalg.svd(np.asarray(value, dtype=np.float64))
        sign = np.linalg.det(left @ right)
        return left @ np.diag((1.0, 1.0, sign)) @ right

    relative = nearest_rotation(actual) @ nearest_rotation(expected).T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _direction_error_deg(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = actual / np.linalg.norm(actual)
    expected = expected / np.linalg.norm(expected)
    return math.degrees(math.acos(float(np.clip(actual @ expected, -1.0, 1.0))))


def _reference_pose_in_panorai(
    reference: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    record = reference["relative_pose_dataset_frame"]
    basis = np.asarray(record["panorai_from_dataset_basis"], dtype=np.float64)
    rotation_dataset = np.asarray(record["R_b_from_a"], dtype=np.float64)
    translation_dataset = np.asarray(record["t_direction_b_from_a"], dtype=np.float64)
    return basis.T @ rotation_dataset @ basis, basis.T @ translation_dataset


def _pose_errors(record: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    if not record["estimated"]:
        return {"rotation_error_deg": None, "translation_error_deg": None}
    rotation, translation = _reference_pose_in_panorai(reference)
    return {
        "rotation_error_deg": _rotation_error_deg(
            np.asarray(record["rotation"], dtype=np.float64), rotation
        ),
        "translation_error_deg": _direction_error_deg(
            np.asarray(record["translation_direction"], dtype=np.float64), translation
        ),
    }


def _manual_feature_mask(
    feature_xy: np.ndarray,
    rectangles: list[list[int]],
    shape_hw: tuple[int, int],
) -> np.ndarray:
    height, width = shape_hw
    pixels = np.asarray(feature_xy, dtype=np.float64)
    x = np.mod(pixels[:, 0] + 0.5, width) - 0.5
    y = pixels[:, 1]
    result = np.zeros(len(pixels), dtype=bool)
    for x0, y0, x1, y1 in rectangles:
        result |= (x >= x0 - 0.5) & (x < x1 - 0.5) & (y >= y0 - 0.5) & (y < y1 - 0.5)
    if np.any((y < -0.5) | (y >= height - 0.5)):
        raise ValueError("feature rows are outside the ERP")
    return result


def _cam_alignment(
    cams: list[dict[int, np.ndarray]],
    features: tuple[Any, Any],
    manual: dict[str, Any],
    pair: dict[str, Any],
) -> list[dict[str, Any]]:
    from panorai.object_localization import semantic_region_from_map

    shape_hw = tuple(int(value) for value in pair["feature_shape_hw"])
    records = []
    for view_index, (view, feature_set) in enumerate(
        zip(manual["views"], features, strict=True)
    ):
        for class_record in pair["query"]["classes"]:
            class_id = int(class_record["class_id"])
            rectangles = [
                rectangle
                for region in view["regions"]
                if int(region["class_id"]) == class_id
                for rectangle in region["rectangles_xyxy"]
            ]
            manual_mask = _manual_feature_mask(
                feature_set.source_erp_xy, rectangles, shape_hw
            )
            region = semantic_region_from_map(
                cams[view_index][class_id],
                erp_shape_hw=shape_hw,
                features=feature_set,
                region_id=f"evaluation-{view_index}-{class_id}",
                class_id=class_id,
                class_name=class_record["class_name"],
                threshold=0.0,
                semantic_score=1.0,
                source_id="evaluation-native-cam-lattice",
            )
            weights = np.zeros(len(feature_set), dtype=np.float64)
            weights[region.feature_indices] = region.membership_weights
            target_count = int(manual_mask.sum())
            top_count = max(target_count, 1)
            top_indices = np.argsort(weights)[-top_count:]
            records.append(
                {
                    "view_key": view["key"],
                    "class_id": class_id,
                    "class_name": class_record["class_name"],
                    "manual_feature_fraction": float(manual_mask.mean()),
                    "mean_cam_inside": (
                        float(weights[manual_mask].mean()) if target_count else None
                    ),
                    "mean_cam_outside": (
                        float(weights[~manual_mask].mean())
                        if np.any(~manual_mask)
                        else None
                    ),
                    "top_equal_area_precision": float(manual_mask[top_indices].mean()),
                }
            )
    return records


def _reference_match_mask(matches: Any, reference: dict[str, Any]) -> np.ndarray:
    from panorai.estimators import spherical_tangent_sampson_error

    rotation, translation = _reference_pose_in_panorai(reference)
    x, y, z = translation
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    essential = skew @ rotation
    residuals = spherical_tangent_sampson_error(
        matches.bearings_a, matches.bearings_b, essential, backend="numpy"
    )
    return np.asarray(matches.valid) & (residuals <= math.radians(1.0))


def _prior_enrichment(prior: Any, reference_inliers: np.ndarray) -> dict[str, Any]:
    weights = np.asarray(prior.sampling_weights, dtype=np.float64)
    valid = np.asarray(prior.valid_mask, dtype=bool)
    uniform_fraction = float(reference_inliers.sum() / max(valid.sum(), 1))
    weighted_fraction = float(
        weights[reference_inliers].sum() / max(weights[valid].sum(), 1e-12)
    )
    return {
        "reference_inlier_count": int(reference_inliers.sum()),
        "uniform_reference_inlier_fraction": uniform_fraction,
        "weighted_reference_inlier_fraction": weighted_fraction,
        "enrichment_ratio": (
            weighted_fraction / uniform_fraction if uniform_fraction > 0 else None
        ),
    }


def _write_csv(path: Path, methods: list[dict[str, Any]]) -> None:
    fields = (
        "method",
        "query_role",
        "threshold",
        "profile",
        "estimated",
        "quality_accepted",
        "num_inliers",
        "rotation_error_deg",
        "translation_error_deg",
        "supported_count",
        "weight_ratio",
        "reference_inlier_enrichment",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in methods:
            pose = record["pose"]
            prior = record.get("prior", {})
            errors = record["errors"]
            enrichment = record.get("reference_enrichment", {})
            writer.writerow(
                {
                    "method": record["method"],
                    "query_role": record.get("query_role"),
                    "threshold": record.get("threshold"),
                    "profile": record.get("profile"),
                    "estimated": pose["estimated"],
                    "quality_accepted": pose["quality_accepted"],
                    "num_inliers": pose.get("num_inliers"),
                    "rotation_error_deg": errors["rotation_error_deg"],
                    "translation_error_deg": errors["translation_error_deg"],
                    "supported_count": prior.get("supported_count"),
                    "weight_ratio": prior.get("max_to_min_valid_weight_ratio"),
                    "reference_inlier_enrichment": enrichment.get("enrichment_ratio"),
                }
            )


def run(dataset_root: Path, output_dir: Path) -> dict[str, Any]:
    import cv2

    from panorai.object_localization import (
        SemanticMatchPriorConfig,
        SemanticQuery,
        build_semantic_match_prior,
        semantic_region_from_map,
    )

    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    pair = _load_json(PAIR_PATH)
    shape_hw = tuple(int(value) for value in pair["feature_shape_hw"])
    rgb_paths = [
        _checked_path(dataset_root, view["rgb_relative_path"], view["rgb_sha256"])
        for view in pair["views"]
    ]
    images = tuple(_read_and_resize(path, shape_hw) for path in rgb_paths)
    started = time.perf_counter()
    feature_pipeline = _feature_pipeline()
    features = tuple(
        feature_pipeline.extract(image, panorama_id=view["view_id"])
        for image, view in zip(images, pair["views"], strict=True)
    )
    matches = feature_pipeline.match(features[0], features[1])
    estimator = _pose_estimator()
    baseline = estimator.estimate(matches.to_bearing_correspondences())
    cams, cam_diagnostics, model_record = _infer_cams(images, pair)
    method_records: list[dict[str, Any]] = [
        {"method": "uniform-weight", "pose": _pose_record(baseline)}
    ]
    priors: dict[str, Any] = {}
    query_sweeps = (
        ("target", pair["query"], pair["sweeps"]["cam_thresholds"]),
        (
            "unrelated-control",
            pair["control_query"],
            pair["sweeps"]["control_cam_thresholds"],
        ),
    )
    for query_role, query_record, thresholds in query_sweeps:
        query = SemanticQuery(
            text=query_record["text"],
            class_ids=tuple(int(item["class_id"]) for item in query_record["classes"]),
            vocabulary=query_record["vocabulary"],
        )
        for threshold in thresholds:
            regions = []
            for view_index, feature_set in enumerate(features):
                view_regions = tuple(
                    semantic_region_from_map(
                        cams[view_index][int(class_record["class_id"])],
                        erp_shape_hw=shape_hw,
                        features=feature_set,
                        region_id=(
                            f"cam-{query_role}-{pair['views'][view_index]['key']}-"
                            f"{int(class_record['class_id'])}-t{float(threshold):.2f}"
                        ),
                        class_id=int(class_record["class_id"]),
                        class_name=class_record["class_name"],
                        threshold=float(threshold),
                        semantic_score=float(
                            next(
                                item["probability"]
                                for item in cam_diagnostics[view_index]["classes"]
                                if item["class_id"] == int(class_record["class_id"])
                            )
                        ),
                        source_id="spherical-imagenet-resnet18-native-cam/v1",
                    )
                    for class_record in query_record["classes"]
                )
                regions.append(view_regions)
            for profile in pair["sweeps"]["weight_profiles"]:
                key = f"{query_role}-t{float(threshold):.2f}-{profile['name']}"
                prior = build_semantic_match_prior(
                    query,
                    regions[0],
                    regions[1],
                    matches,
                    SemanticMatchPriorConfig(
                        uniform_mix=float(profile["uniform_mix"]),
                        max_weight_ratio=float(profile["max_weight_ratio"]),
                    ),
                )
                pose = estimator.estimate(prior.to_bearing_correspondences(matches))
                priors[key] = prior
                method_records.append(
                    {
                        "method": "semantic-guided",
                        "query_role": query_role,
                        "query_text": query.text,
                        "query_class_ids": list(query.class_ids),
                        "threshold": float(threshold),
                        "profile": profile["name"],
                        "pose": _pose_record(pose),
                        "prior": prior.describe(),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    cam_arrays = {
        f"view_{view_index}_class_{class_id}": cam
        for view_index, view_cams in enumerate(cams)
        for class_id, cam in view_cams.items()
    }
    cam_path = output_dir / "native_cams.npz"
    np.savez_compressed(cam_path, **cam_arrays)
    prediction = {
        "schema": SCHEMA,
        "pair_manifest_sha256": _sha256(PAIR_PATH),
        "reference_accessed": False,
        "manual_regions_accessed": False,
        "queries": {
            role: {
                "text": record["text"],
                "class_ids": [int(item["class_id"]) for item in record["classes"]],
                "vocabulary": record["vocabulary"],
            }
            for role, record in (
                ("target", pair["query"]),
                ("unrelated-control", pair["control_query"]),
            )
        },
        "counts": {
            "features_a": len(features[0]),
            "features_b": len(features[1]),
            "matches": len(matches),
        },
        "model": model_record,
        "cam_diagnostics": cam_diagnostics,
        "native_cam_artifact": {
            "path": str(cam_path),
            "sha256": _sha256(cam_path),
        },
        "methods": method_records,
        "elapsed_seconds": time.perf_counter() - started,
    }
    prediction_path = output_dir / "prediction.json"
    _atomic_json(prediction_path, prediction, readonly=True)

    # Reference pose and manual rectangles are opened only after prediction is frozen.
    reference = _load_json(REFERENCE_PATH)
    manual = _load_json(MANUAL_REGIONS_PATH)
    reference_inliers = _reference_match_mask(matches, reference)
    evaluated_methods = []
    for record in method_records:
        evaluated = {**record, "errors": _pose_errors(record["pose"], reference)}
        if record["method"] == "semantic-guided":
            key = (
                f"{record['query_role']}-t{record['threshold']:.2f}-{record['profile']}"
            )
            evaluated["reference_enrichment"] = _prior_enrichment(
                priors[key], reference_inliers
            )
        evaluated_methods.append(evaluated)
    evaluation = {
        "schema": "panorai-real-cam-pose-prior-evaluation/v1",
        "prediction_path": str(prediction_path),
        "prediction_sha256": _sha256(prediction_path),
        "reference_opened_after_prediction_freeze": True,
        "manual_regions_opened_after_prediction_freeze": True,
        "methods": evaluated_methods,
        "cam_manual_region_alignment": _cam_alignment(cams, features, manual, pair),
        "dataset_or_model_bytes_distributed": False,
    }
    evaluation_path = output_dir / "evaluation.json"
    _atomic_json(evaluation_path, evaluation)
    _write_csv(output_dir / "pose_comparison.csv", evaluated_methods)
    print(json.dumps(evaluation, indent=2, sort_keys=True))
    return evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(_preflight(args.dataset_root), indent=2, sort_keys=True))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required unless --preflight is used")
    run(args.dataset_root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
