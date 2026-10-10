#!/usr/bin/env python3
"""Test whether real spherical CAM components support two-view object IDs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.object_localization.run_real_cam_pose_prior import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    MANUAL_REGIONS_PATH,
    PAIR_PATH,
    REFERENCE_PATH,
    _atomic_json,
    _checked_path,
    _feature_pipeline,
    _infer_cams,
    _load_json,
    _manual_feature_mask,
    _pose_estimator,
    _pose_record,
    _preflight,
    _read_and_resize,
    _sha256,
)


SCHEMA = "panorai-real-cam-object-components/v1"


def _region_record(region: Any) -> dict[str, Any]:
    return {
        "region_id": region.region_id,
        "view_id": region.view_id,
        "class_id": region.class_id,
        "class_name": region.class_name,
        "semantic_score": region.semantic_score,
        "feature_count": len(region.feature_indices),
        "feature_indices": region.feature_indices.tolist(),
        "membership_mean": (
            float(region.membership_weights.mean())
            if len(region.membership_weights)
            else None
        ),
        "membership_max": (
            float(region.membership_weights.max())
            if len(region.membership_weights)
            else None
        ),
        "source_id": region.source_id,
    }


def _association_record(association: Any) -> dict[str, Any]:
    return {
        "association_id": association.association_id,
        "region_id_a": association.region_id_a,
        "region_id_b": association.region_id_b,
        "class_id": association.class_id,
        "class_name": association.class_name,
        "match_count": association.match_count,
        "inlier_count": association.inlier_count,
        "semantic_score": association.semantic_score,
        "ranking_score": association.ranking_score,
        "state": association.state,
        "reasons": list(association.reasons),
    }


def _hypothesis_record(hypothesis: Any) -> dict[str, Any]:
    location = hypothesis.location
    return {
        "hypothesis_id": hypothesis.hypothesis_id,
        "class_id": hypothesis.class_id,
        "class_name": hypothesis.class_name,
        "region_ids": list(hypothesis.region_ids),
        "association_id": hypothesis.association_id,
        "identity_score": hypothesis.identity_score,
        "location": {
            "state": location.state,
            "mode": location.mode,
            "triangulated_count": location.triangulated_count,
            "median_parallax_deg": location.median_parallax_deg,
            "median_reprojection_error_deg": location.median_reprojection_error_deg,
            "location_score": location.location_score,
            "position_xyz": (
                None
                if location.position_xyz is None
                else location.position_xyz.tolist()
            ),
        },
    }


def _manual_masks(
    features: tuple[Any, Any], manual: dict[str, Any], shape_hw: tuple[int, int]
) -> list[dict[str, np.ndarray]]:
    results: list[dict[str, np.ndarray]] = []
    for feature_set, view in zip(features, manual["views"], strict=True):
        results.append(
            {
                region["region_id"]: _manual_feature_mask(
                    feature_set.source_erp_xy,
                    region["rectangles_xyxy"],
                    shape_hw,
                )
                for region in view["regions"]
            }
        )
    return results


def _map_component_to_manual(
    feature_indices: np.ndarray,
    class_id: int,
    manual_regions: list[dict[str, Any]],
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    component = np.zeros(len(next(iter(masks.values()))), dtype=bool)
    component[np.asarray(feature_indices, dtype=np.int64)] = True
    candidates = []
    for region in manual_regions:
        if int(region["class_id"]) != int(class_id):
            continue
        target = masks[region["region_id"]]
        intersection = int(np.count_nonzero(component & target))
        union = int(np.count_nonzero(component | target))
        candidates.append(
            {
                "manual_region_id": region["region_id"],
                "intersection": intersection,
                "component_purity": intersection / max(int(component.sum()), 1),
                "manual_coverage": intersection / max(int(target.sum()), 1),
                "feature_iou": intersection / max(union, 1),
            }
        )
    candidates.sort(
        key=lambda item: (
            -item["intersection"],
            -item["feature_iou"],
            item["manual_region_id"],
        )
    )
    best = candidates[0] if candidates and candidates[0]["intersection"] else None
    overlapping = [item for item in candidates if item["intersection"] > 0]
    return {
        "best_manual_region_id": (None if best is None else best["manual_region_id"]),
        "best_component_purity": (None if best is None else best["component_purity"]),
        "best_manual_coverage": (None if best is None else best["manual_coverage"]),
        "best_feature_iou": None if best is None else best["feature_iou"],
        "overlapping_manual_regions": overlapping,
        "merges_multiple_manual_instances": len(overlapping) > 1,
    }


def _entity_name(manual_region_id: str | None) -> str | None:
    if manual_region_id is None:
        return None
    stem, separator, suffix = manual_region_id.rpartition("-")
    return stem if separator and suffix in {"a", "b"} else manual_region_id


def _evaluate_variant(
    variant: dict[str, Any],
    region_objects: tuple[tuple[Any, ...], tuple[Any, ...]],
    result: Any,
    manual: dict[str, Any],
    masks: list[dict[str, np.ndarray]],
    expected_links: set[tuple[str, str]],
) -> dict[str, Any]:
    mappings: list[dict[str, dict[str, Any]]] = []
    mapping_records = []
    for view_index, regions in enumerate(region_objects):
        by_id = {}
        for region in regions:
            mapping = _map_component_to_manual(
                region.feature_indices,
                region.class_id,
                manual["views"][view_index]["regions"],
                masks[view_index],
            )
            by_id[region.region_id] = mapping
            mapping_records.append(
                {
                    "view_key": manual["views"][view_index]["key"],
                    "region_id": region.region_id,
                    "class_id": region.class_id,
                    "feature_count": len(region.feature_indices),
                    **mapping,
                }
            )
        mappings.append(by_id)

    predicted_links = []
    correct_links = []
    false_links = []
    unmapped_hypotheses = []
    for hypothesis in result.hypotheses:
        mapping_a = mappings[0][hypothesis.region_ids[0]]
        mapping_b = mappings[1][hypothesis.region_ids[1]]
        manual_a = mapping_a["best_manual_region_id"]
        manual_b = mapping_b["best_manual_region_id"]
        if manual_a is None or manual_b is None:
            unmapped_hypotheses.append(hypothesis.hypothesis_id)
            continue
        link = (manual_a, manual_b)
        predicted_links.append(link)
        if link in expected_links:
            correct_links.append(link)
        else:
            false_links.append(link)

    expected_entities = {
        (_entity_name(first), _entity_name(second)) for first, second in expected_links
    }
    predicted_entities = {
        (_entity_name(first), _entity_name(second)) for first, second in predicted_links
    }
    return {
        "variant": variant,
        "component_mappings": mapping_records,
        "hypothesis_count": len(result.hypotheses),
        "localized_hypothesis_count": sum(
            item.location.state == "localized" for item in result.hypotheses
        ),
        "predicted_manual_links": [list(item) for item in sorted(set(predicted_links))],
        "correct_manual_links": [list(item) for item in sorted(set(correct_links))],
        "false_manual_links": [list(item) for item in sorted(set(false_links))],
        "unmapped_hypothesis_ids": unmapped_hypotheses,
        "expected_instance_count": len(expected_links),
        "recovered_instance_count": len(set(correct_links)),
        "identity_recall": len(set(correct_links)) / max(len(expected_links), 1),
        "identity_precision": len(set(correct_links))
        / max(len(set(predicted_links)), 1),
        "entity_pairs_recovered": [
            list(item) for item in sorted(predicted_entities & expected_entities)
        ],
        "merged_component_count": sum(
            item["merges_multiple_manual_instances"] for item in mapping_records
        ),
    }


def _write_summary_csv(path: Path, evaluations: list[dict[str, Any]]) -> None:
    fields = (
        "threshold",
        "connectivity",
        "regions_a",
        "regions_b",
        "hypotheses",
        "localized",
        "correct_links",
        "false_links",
        "identity_precision",
        "identity_recall",
        "merged_components",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for evaluation in evaluations:
            variant = evaluation["variant"]
            writer.writerow(
                {
                    "threshold": variant["threshold"],
                    "connectivity": variant["connectivity"],
                    "regions_a": variant["region_count_a"],
                    "regions_b": variant["region_count_b"],
                    "hypotheses": evaluation["hypothesis_count"],
                    "localized": evaluation["localized_hypothesis_count"],
                    "correct_links": len(evaluation["correct_manual_links"]),
                    "false_links": len(evaluation["false_manual_links"]),
                    "identity_precision": evaluation["identity_precision"],
                    "identity_recall": evaluation["identity_recall"],
                    "merged_components": evaluation["merged_component_count"],
                }
            )


def run(dataset_root: Path, output_dir: Path) -> dict[str, Any]:
    import cv2

    from panorai.object_localization import (
        ObjectLocalizationPipeline,
        PairObjectLocalizationInput,
        SemanticQuery,
        SemanticRegionProposalConfig,
        semantic_regions_from_map,
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
    pose = _pose_estimator().estimate(matches.to_bearing_correspondences())
    if pose is None:
        raise RuntimeError("relative pose was not returned")
    cams, cam_diagnostics, model_record = _infer_cams(images, pair)
    query_record = pair["query"]
    query = SemanticQuery(
        text=query_record["text"],
        class_ids=tuple(int(item["class_id"]) for item in query_record["classes"]),
        vocabulary=query_record["vocabulary"],
    )
    variants = []
    runtime_objects = []
    pipeline = ObjectLocalizationPipeline()
    for threshold in (0.25, 0.5, 0.75):
        for connectivity in (4, 8):
            view_regions = []
            proposal_diagnostics = []
            for view_index, feature_set in enumerate(features):
                regions = []
                diagnostics = []
                for class_record in query_record["classes"]:
                    class_id = int(class_record["class_id"])
                    probability = float(
                        next(
                            item["probability"]
                            for item in cam_diagnostics[view_index]["classes"]
                            if item["class_id"] == class_id
                        )
                    )
                    proposals = semantic_regions_from_map(
                        cams[view_index][class_id],
                        erp_shape_hw=shape_hw,
                        features=feature_set,
                        region_id_prefix=(
                            f"cam-{pair['views'][view_index]['key']}-t{threshold:.2f}"
                            f"-c{connectivity}"
                        ),
                        class_id=class_id,
                        class_name=class_record["class_name"],
                        config=SemanticRegionProposalConfig(
                            threshold=threshold,
                            connectivity=connectivity,
                            min_component_cells=1,
                            min_feature_count=3,
                            max_regions=8,
                        ),
                        semantic_score=probability,
                        source_id="spherical-imagenet-resnet18-native-cam-components/v1",
                    )
                    regions.extend(proposals.regions)
                    diagnostics.append(
                        {
                            "class_id": class_id,
                            "class_name": class_record["class_name"],
                            **proposals.describe(),
                        }
                    )
                view_regions.append(tuple(regions))
                proposal_diagnostics.append(diagnostics)
            evidence = PairObjectLocalizationInput(
                view_id_a=pair["views"][0]["view_id"],
                view_id_b=pair["views"][1]["view_id"],
                query=query,
                regions_a=view_regions[0],
                regions_b=view_regions[1],
                features_a=features[0],
                features_b=features[1],
                matches=matches,
                pose=pose,
            )
            result = pipeline.estimate(evidence)
            repeated = pipeline.estimate(evidence)
            variant = {
                "threshold": threshold,
                "connectivity": connectivity,
                "region_count_a": len(view_regions[0]),
                "region_count_b": len(view_regions[1]),
                "proposal_diagnostics": proposal_diagnostics,
                "regions_a": [_region_record(item) for item in view_regions[0]],
                "regions_b": [_region_record(item) for item in view_regions[1]],
                "associations": [
                    _association_record(item) for item in result.associations
                ],
                "hypotheses": [_hypothesis_record(item) for item in result.hypotheses],
                "hypothesis_ids_repeat_identically": [
                    item.hypothesis_id for item in result.hypotheses
                ]
                == [item.hypothesis_id for item in repeated.hypotheses],
            }
            variants.append(variant)
            runtime_objects.append((view_regions, result))

    output_dir.mkdir(parents=True, exist_ok=True)
    cam_path = output_dir / "native_cams.npz"
    np.savez_compressed(
        cam_path,
        **{
            f"view_{view_index}_class_{class_id}": cam
            for view_index, view_cams in enumerate(cams)
            for class_id, cam in view_cams.items()
        },
    )
    prediction = {
        "schema": SCHEMA,
        "pair_manifest_sha256": _sha256(PAIR_PATH),
        "reference_accessed": False,
        "manual_regions_accessed": False,
        "query": {
            "text": query.text,
            "class_ids": list(query.class_ids),
            "vocabulary": query.vocabulary,
        },
        "counts": {
            "features_a": len(features[0]),
            "features_b": len(features[1]),
            "matches": len(matches),
        },
        "pose": _pose_record(pose),
        "model": model_record,
        "cam_diagnostics": cam_diagnostics,
        "native_cam_artifact": {"path": str(cam_path), "sha256": _sha256(cam_path)},
        "variants": variants,
        "elapsed_seconds": time.perf_counter() - started,
    }
    prediction_path = output_dir / "prediction.json"
    _atomic_json(prediction_path, prediction, readonly=True)

    # Manual instance regions and expected links are opened only after freeze.
    manual = _load_json(MANUAL_REGIONS_PATH)
    reference = _load_json(REFERENCE_PATH)
    masks = _manual_masks(features, manual, shape_hw)
    expected_links = {tuple(item) for item in reference["expected_links"]}
    evaluations = [
        _evaluate_variant(
            variant,
            runtime[0],
            runtime[1],
            manual,
            masks,
            expected_links,
        )
        for variant, runtime in zip(variants, runtime_objects, strict=True)
    ]
    evaluation = {
        "schema": "panorai-real-cam-object-components-evaluation/v1",
        "prediction_path": str(prediction_path),
        "prediction_sha256": _sha256(prediction_path),
        "reference_opened_after_prediction_freeze": True,
        "manual_regions_opened_after_prediction_freeze": True,
        "expected_links": [list(item) for item in sorted(expected_links)],
        "evaluations": evaluations,
        "dataset_or_model_bytes_distributed": False,
    }
    _atomic_json(output_dir / "evaluation.json", evaluation)
    _write_summary_csv(output_dir / "summary.csv", evaluations)
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
