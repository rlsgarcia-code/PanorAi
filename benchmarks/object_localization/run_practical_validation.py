#!/usr/bin/env python3
"""Run a real-panorama development validation of object hypotheses.

The prediction phase reads RGB and frozen manual regions only.  It freezes its
JSON result before this process opens metric depth, reference pose, expected
identity links, or evaluation gates.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
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
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REGIONS_PATH = Path(__file__).with_name("stanford_office3_regions.json")
REFERENCE_PATH = Path(__file__).with_name("stanford_office3_reference.json")
DEFAULT_DATASET_ROOT = Path(
    "/Users/robinsongarcia/projects/PanorAi-projects/"
    "PanorAi-reorg-worktree/datasets/stanford2d3d"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


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
    regions = _load_json(REGIONS_PATH)
    reference = _load_json(REFERENCE_PATH)
    checked: list[dict[str, Any]] = []
    for view in regions["views"]:
        path = _checked_path(
            dataset_root, view["rgb_relative_path"], view["rgb_sha256"]
        )
        checked.append({"kind": "rgb", "path": str(path), "sha256": view["rgb_sha256"]})
    for key in ("depth_a", "depth_b"):
        record = reference[key]
        path = _checked_path(dataset_root, record["relative_path"], record["sha256"])
        checked.append({"kind": key, "path": str(path), "sha256": record["sha256"]})
    return {
        "ready": True,
        "dataset_root": str(dataset_root.resolve()),
        "regions": str(REGIONS_PATH),
        "reference": str(REFERENCE_PATH),
        "checked_inputs": checked,
        "source_bytes_will_be_copied": False,
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


def _region_mask(shape_hw: tuple[int, int], rectangles: list[list[int]]) -> np.ndarray:
    height, width = shape_hw
    result = np.zeros(shape_hw, dtype=np.float32)
    for rectangle in rectangles:
        if len(rectangle) != 4:
            raise ValueError("each region rectangle must be [x0, y0, x1, y1]")
        x0, y0, x1, y1 = (int(value) for value in rectangle)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"rectangle is outside {shape_hw}: {rectangle}")
        result[y0:y1, x0:x1] = 1.0
    return result


def _materialize_regions(
    view: dict[str, Any], features: Any, shape_hw: tuple[int, int]
) -> tuple[Any, ...]:
    from panorai.object_localization import semantic_region_from_map

    results = []
    for record in view["regions"]:
        results.append(
            semantic_region_from_map(
                _region_mask(shape_hw, record["rectangles_xyxy"]),
                erp_shape_hw=shape_hw,
                features=features,
                region_id=record["region_id"],
                class_id=int(record["class_id"]),
                class_name=record["class_name"],
                threshold=0.5,
                semantic_score=float(record["semantic_score"]),
                source_id="stanford-office3-manual-rgb-regions/v1",
            )
        )
    return tuple(results)


def _boundary_variant(
    view: dict[str, Any], shape_hw: tuple[int, int], margin: int
) -> dict[str, Any]:
    """Expand (negative margin) or contract RGB rectangles deterministically."""

    height, width = shape_hw
    result = {**view, "regions": []}
    for region in view["regions"]:
        rectangles = []
        for rectangle in region["rectangles_xyxy"]:
            x0, y0, x1, y1 = (int(value) for value in rectangle)
            adjusted = (
                max(0, x0 + margin),
                max(0, y0 + margin),
                min(width, x1 - margin),
                min(height, y1 - margin),
            )
            if adjusted[0] >= adjusted[2] or adjusted[1] >= adjusted[3]:
                raise ValueError(
                    f"boundary perturbation removed region {region['region_id']}"
                )
            rectangles.append(list(adjusted))
        result["regions"].append({**region, "rectangles_xyxy": rectangles})
    return result


def _robustness_predictions(
    evidence: Any,
    pipeline: Any,
    views: list[dict[str, Any]],
    features: tuple[Any, Any],
    shape_hw: tuple[int, int],
) -> list[dict[str, Any]]:
    variants = (
        ("contract-16px", 16),
        ("contract-32px", 32),
        ("expand-16px", -16),
        ("expand-32px", -32),
    )
    records = []
    for name, margin in variants:
        view_a = _boundary_variant(views[0], shape_hw, margin)
        view_b = _boundary_variant(views[1], shape_hw, margin)
        regions_a = _materialize_regions(view_a, features[0], shape_hw)
        regions_b = _materialize_regions(view_b, features[1], shape_hw)
        result = pipeline.estimate(
            replace(evidence, regions_a=regions_a, regions_b=regions_b)
        )
        records.append(
            {
                "name": name,
                "margin_px": margin,
                "predicted_links": [
                    list(item.region_ids) for item in result.hypotheses
                ],
                "hypothesis_ids": [item.hypothesis_id for item in result.hypotheses],
                "association_evidence": [
                    {
                        "region_ids": [item.region_id_a, item.region_id_b],
                        "match_count": item.match_count,
                        "inlier_count": item.inlier_count,
                        "state": item.state,
                        "reasons": list(item.reasons),
                    }
                    for item in result.associations
                ],
            }
        )
    return records


def _array_or_none(value: np.ndarray | None) -> list[float] | None:
    return None if value is None else np.asarray(value).tolist()


def _prediction_record(
    result: Any, pose: Any, features: tuple[Any, Any], matches: Any
) -> dict[str, Any]:
    associations = [
        {
            "association_id": item.association_id,
            "region_id_a": item.region_id_a,
            "region_id_b": item.region_id_b,
            "class_id": item.class_id,
            "class_name": item.class_name,
            "match_count": item.match_count,
            "inlier_count": item.inlier_count,
            "inlier_match_indices": item.inlier_match_indices.tolist(),
            "ranking_score": item.ranking_score,
            "state": item.state,
            "reasons": list(item.reasons),
        }
        for item in result.associations
    ]
    hypotheses = [
        {
            "hypothesis_id": item.hypothesis_id,
            "association_id": item.association_id,
            "region_ids": list(item.region_ids),
            "class_id": item.class_id,
            "class_name": item.class_name,
            "identity_score": item.identity_score,
            "location": {
                "state": item.location.state,
                "mode": item.location.mode,
                "frame": item.location.frame,
                "units": item.location.units,
                "position_xyz": _array_or_none(item.location.position_xyz),
                "covariance_xyz": _array_or_none(item.location.covariance_xyz),
                "bearing_xyz": _array_or_none(item.location.bearing_xyz),
                "triangulated_count": item.location.triangulated_count,
                "supporting_match_indices": item.location.supporting_match_indices.tolist(),
                "median_parallax_deg": item.location.median_parallax_deg,
                "median_reprojection_error_deg": item.location.median_reprojection_error_deg,
                "location_score": item.location.location_score,
            },
        }
        for item in result.hypotheses
    ]
    return {
        "schema": "panorai-object-localization-practical-prediction/v1",
        "frozen_before_reference_access": True,
        "query": {
            "text": result.query.text,
            "class_ids": list(result.query.class_ids),
            "vocabulary": result.query.vocabulary,
        },
        "counts": {
            "features_a": len(features[0]),
            "features_b": len(features[1]),
            "matches": len(matches),
            "pose_inliers": pose.num_inliers,
            "hypotheses": len(hypotheses),
        },
        "pose": {
            "quality_accepted": pose.quality_report.accepted,
            "quality_rejection_reasons": list(pose.quality_report.rejection_reasons),
            "rotation": pose.rotation.tolist(),
            "translation_direction": pose.translation_direction.tolist(),
            "median_residual_deg": pose.quality_report.median_residual_deg,
            "median_parallax_deg": pose.median_parallax_deg,
        },
        "associations": associations,
        "hypotheses": hypotheses,
    }


def _rotation_error_deg(actual: np.ndarray, expected: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(actual @ expected.T) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _direction_error_deg(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = actual / np.linalg.norm(actual)
    expected = expected / np.linalg.norm(expected)
    return math.degrees(math.acos(float(np.clip(actual @ expected, -1.0, 1.0))))


def _reference_center(
    hypothesis: dict[str, Any],
    associations: dict[str, dict[str, Any]],
    matches: Any,
    features_a: Any,
    depth_a: np.ndarray,
) -> tuple[np.ndarray, int]:
    association = associations[hypothesis["association_id"]]
    match_indices = np.asarray(association["inlier_match_indices"], dtype=np.int64)
    feature_indices = np.asarray(
        matches.feature_indices_a[match_indices], dtype=np.int64
    )
    pixels = np.asarray(features_a.source_erp_xy[feature_indices], dtype=np.float64)
    depth_h, depth_w = depth_a.shape
    feature_h = 1024
    feature_w = 2048
    x = np.floor((pixels[:, 0] + 0.5) * depth_w / feature_w).astype(np.int64)
    y = np.floor((pixels[:, 1] + 0.5) * depth_h / feature_h).astype(np.int64)
    x %= depth_w
    y = np.clip(y, 0, depth_h - 1)
    radial_range = np.asarray(depth_a[y, x], dtype=np.float64)
    valid = np.isfinite(radial_range) & (radial_range > 0.0)
    if not np.any(valid):
        raise RuntimeError(f"no valid reference depth for {hypothesis['region_ids']}")
    bearings = np.asarray(features_a.bearings[feature_indices], dtype=np.float64)
    points = bearings[valid] * radial_range[valid, None]
    return np.median(points, axis=0), int(valid.sum())


def _evaluate(
    prediction: dict[str, Any],
    reference: dict[str, Any],
    matches: Any,
    features_a: Any,
    depth_a: np.ndarray,
) -> dict[str, Any]:
    predicted_links = {tuple(item["region_ids"]) for item in prediction["hypotheses"]}
    expected_links = {tuple(item) for item in reference["expected_links"]}
    true_positives = predicted_links & expected_links
    false_positives = predicted_links - expected_links
    false_negatives = expected_links - predicted_links
    precision = len(true_positives) / max(len(predicted_links), 1)
    recall = len(true_positives) / max(len(expected_links), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    pose_reference = reference["relative_pose_dataset_frame"]
    basis = np.asarray(pose_reference["panorai_from_dataset_basis"], dtype=np.float64)
    estimated_rotation = basis @ np.asarray(prediction["pose"]["rotation"]) @ basis.T
    estimated_translation = basis @ np.asarray(
        prediction["pose"]["translation_direction"]
    )
    rotation_error = _rotation_error_deg(
        estimated_rotation, np.asarray(pose_reference["R_b_from_a"])
    )
    translation_error = _direction_error_deg(
        estimated_translation,
        np.asarray(pose_reference["t_direction_b_from_a"]),
    )

    baseline = float(pose_reference["baseline_m"])
    association_index = {
        item["association_id"]: item for item in prediction["associations"]
    }
    spatial = []
    for hypothesis in prediction["hypotheses"]:
        location = hypothesis["location"]
        if location["state"] != "localized" or location["mode"] != "scale_free_3d":
            continue
        predicted_metric = (
            np.asarray(location["position_xyz"], dtype=np.float64) * baseline
        )
        expected_metric, valid_depth_count = _reference_center(
            hypothesis, association_index, matches, features_a, depth_a
        )
        spatial.append(
            {
                "hypothesis_id": hypothesis["hypothesis_id"],
                "region_ids": hypothesis["region_ids"],
                "support_count": location["triangulated_count"],
                "valid_reference_depth_count": valid_depth_count,
                "predicted_position_m_in_view_a": predicted_metric.tolist(),
                "reference_median_position_m_in_view_a": expected_metric.tolist(),
                "center_error_m": float(
                    np.linalg.norm(predicted_metric - expected_metric)
                ),
            }
        )
    errors = [item["center_error_m"] for item in spatial]
    gates = reference["development_gates"]
    nominal_margin = int(gates["nominal_boundary_margin_px"])
    stress_margin = int(gates["stress_boundary_margin_px"])
    perturbations = []
    for item in prediction["region_boundary_robustness"]:
        links = {tuple(link) for link in item["predicted_links"]}
        true_links = links & expected_links
        unexpected_links = links - expected_links
        perturbations.append(
            {
                **item,
                "exact_expected_identity_set": links == expected_links,
                "unexpected_links": sorted([list(link) for link in unexpected_links]),
                "precision": len(true_links) / max(len(links), 1),
                "fail_closed": not unexpected_links,
            }
        )
    nominal_perturbations = [
        item for item in perturbations if abs(item["margin_px"]) <= nominal_margin
    ]
    stress_perturbations = [
        item for item in perturbations if abs(item["margin_px"]) >= stress_margin
    ]
    nominal_identity_rate = sum(
        item["exact_expected_identity_set"] for item in nominal_perturbations
    ) / len(nominal_perturbations)
    stress_precision = min(item["precision"] for item in stress_perturbations)
    deterministic_ids = len(
        {item["hypothesis_id"] for item in prediction["hypotheses"]}
    ) == len(prediction["hypotheses"])
    gate_results = {
        "pose_quality_accepted": bool(prediction["pose"]["quality_accepted"]),
        "identity_precision": precision >= gates["identity_precision_min"],
        "identity_recall": recall >= gates["identity_recall_min"],
        "nominal_region_boundary_robustness": nominal_identity_rate
        >= gates["minimum_nominal_boundary_identity_rate"],
        "stress_region_boundaries_fail_closed": stress_precision
        >= gates["minimum_stress_boundary_precision"],
        "unique_hypothesis_ids": deterministic_ids,
        "all_expected_hypotheses_localized": len(spatial) == len(expected_links),
        "spatial_center_error": bool(errors)
        and max(errors) <= gates["maximum_spatial_center_error_m"],
        "relative_rotation": rotation_error <= gates["maximum_rotation_error_deg"],
        "relative_translation_direction": translation_error
        <= gates["maximum_translation_direction_error_deg"],
    }
    return {
        "identity": {
            "expected_links": sorted([list(item) for item in expected_links]),
            "predicted_links": sorted([list(item) for item in predicted_links]),
            "true_positives": len(true_positives),
            "false_positives": len(false_positives),
            "false_negatives": len(false_negatives),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "relative_pose": {
            "rotation_error_deg": rotation_error,
            "translation_direction_error_deg": translation_error,
        },
        "spatial": {
            "hypotheses": spatial,
            "median_center_error_m": float(np.median(errors)) if errors else None,
            "maximum_center_error_m": max(errors, default=None),
        },
        "region_boundary_robustness": {
            "nominal_margin_px": nominal_margin,
            "nominal_exact_identity_rate": nominal_identity_rate,
            "stress_margin_px": stress_margin,
            "stress_minimum_precision": stress_precision,
            "variants": perturbations,
        },
        "gates": gate_results,
        "all_gates_passed": all(gate_results.values()),
    }


def _write_csv(path: Path, evaluation: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "hypothesis_id",
                "region_id_a",
                "region_id_b",
                "support_count",
                "center_error_m",
            ),
        )
        writer.writeheader()
        for item in evaluation["spatial"]["hypotheses"]:
            writer.writerow(
                {
                    "hypothesis_id": item["hypothesis_id"],
                    "region_id_a": item["region_ids"][0],
                    "region_id_b": item["region_ids"][1],
                    "support_count": item["support_count"],
                    "center_error_m": item["center_error_m"],
                }
            )


def _draw_regions(
    images: tuple[np.ndarray, np.ndarray],
    views: list[dict[str, Any]],
    path: Path,
) -> None:
    colors = {526: "#ffb000", 453: "#00d4ff"}
    panels = []
    for image, view in zip(images, views, strict=True):
        panel = Image.fromarray(image)
        draw = ImageDraw.Draw(panel)
        for region in view["regions"]:
            color = colors[int(region["class_id"])]
            for rectangle in region["rectangles_xyxy"]:
                draw.rectangle(tuple(rectangle), outline=color, width=5)
            first = region["rectangles_xyxy"][0]
            draw.text((first[0] + 8, first[1] + 8), region["region_id"], fill=color)
        panels.append(panel)
    canvas = Image.new("RGB", (panels[0].width, 2 * panels[0].height), "black")
    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (0, panels[0].height))
    canvas.save(path)


def run(dataset_root: Path, output_dir: Path) -> dict[str, Any]:
    import cv2

    from panorai.object_localization import (
        ObjectLocalizationPipeline,
        PairObjectLocalizationInput,
        SemanticQuery,
    )

    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    regions = _load_json(REGIONS_PATH)
    shape_hw = tuple(int(value) for value in regions["evaluation_shape_hw"])
    views = regions["views"]
    rgb_paths = [
        _checked_path(dataset_root, view["rgb_relative_path"], view["rgb_sha256"])
        for view in views
    ]
    images = tuple(_read_and_resize(path, shape_hw) for path in rgb_paths)

    started = time.perf_counter()
    feature_pipeline = _feature_pipeline()
    features_a = feature_pipeline.extract(images[0], panorama_id=views[0]["view_id"])
    features_b = feature_pipeline.extract(images[1], panorama_id=views[1]["view_id"])
    matches = feature_pipeline.match(features_a, features_b)
    pose = _pose_estimator().estimate(matches.to_bearing_correspondences())
    if pose is None:
        raise RuntimeError("relative pose was not returned")
    regions_a = _materialize_regions(views[0], features_a, shape_hw)
    regions_b = _materialize_regions(views[1], features_b, shape_hw)
    query_record = regions["query"]
    evidence = PairObjectLocalizationInput(
        view_id_a=views[0]["view_id"],
        view_id_b=views[1]["view_id"],
        query=SemanticQuery(
            text=query_record["text"],
            class_ids=tuple(query_record["class_ids"]),
            vocabulary=query_record["vocabulary"],
        ),
        regions_a=regions_a,
        regions_b=regions_b,
        features_a=features_a,
        features_b=features_b,
        matches=matches,
        pose=pose,
        translation_scale=None,
    )
    pipeline = ObjectLocalizationPipeline()
    result = pipeline.estimate(evidence)
    repeated = pipeline.estimate(evidence)
    prediction = _prediction_record(result, pose, (features_a, features_b), matches)
    prediction["hypothesis_ids_repeat_identically"] = [
        item.hypothesis_id for item in result.hypotheses
    ] == [item.hypothesis_id for item in repeated.hypotheses]
    prediction["method_elapsed_seconds"] = time.perf_counter() - started
    prediction["region_boundary_robustness"] = _robustness_predictions(
        evidence,
        pipeline,
        views,
        (features_a, features_b),
        shape_hw,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "prediction.json"
    _atomic_json(prediction_path, prediction, readonly=True)
    prediction_sha256 = _sha256(prediction_path)

    # Reference access begins only after the prediction file is immutable.
    reference = _load_json(REFERENCE_PATH)
    depth_record = reference["depth_a"]
    depth_path = _checked_path(
        dataset_root, depth_record["relative_path"], depth_record["sha256"]
    )
    depth_a = np.load(depth_path, mmap_mode="r")
    evaluation = _evaluate(prediction, reference, matches, features_a, depth_a)
    evaluation["gates"]["repeat_ids_identically"] = prediction[
        "hypothesis_ids_repeat_identically"
    ]
    evaluation["all_gates_passed"] = all(evaluation["gates"].values())
    summary = {
        "schema": "panorai-object-localization-practical-evaluation/v1",
        "status": "PASS" if evaluation["all_gates_passed"] else "FAIL",
        "evidence_class": "post-hoc-real-panorama-development",
        "prediction": str(prediction_path),
        "prediction_sha256": prediction_sha256,
        "reference_opened_after_prediction_freeze": True,
        "dataset_bytes_distributed": False,
        "regions_sha256": _sha256(REGIONS_PATH),
        "reference_sha256": _sha256(REFERENCE_PATH),
        "evaluation": evaluation,
        "limitations": [
            "one Stanford2D3D office pair from one spatial group",
            "manual RGB rectangles are coarse and post-hoc development annotations",
            "ImageNet IDs name region semantics; no detector/CAM quality is evaluated",
            "metric scoring multiplies scale-free output by reference baseline after prediction freeze",
            "regional feature median is evaluated, not physical object centroid",
        ],
    }
    _atomic_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "hypotheses.csv", evaluation)
    _draw_regions(images, views, output_dir / "annotated-regions.png")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.preflight:
        print(json.dumps(_preflight(args.dataset_root), indent=2, sort_keys=True))
        return 0
    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --preflight is used")
    summary = run(args.dataset_root, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
