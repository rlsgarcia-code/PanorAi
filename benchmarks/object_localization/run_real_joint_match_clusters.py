#!/usr/bin/env python3
"""Test joint spherical match clusters on one frozen repeated-object pair."""

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

from benchmarks.object_localization.run_real_cam_object_components import (  # noqa: E402
    _association_record,
    _entity_name,
    _hypothesis_record,
    _manual_masks,
    _map_component_to_manual,
    _region_record,
)
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
    _pose_estimator,
    _pose_record,
    _preflight,
    _read_and_resize,
    _sha256,
)


SCHEMA = "panorai-real-joint-match-clusters/v1"
CAM_THRESHOLD = 0.25
CAM_CONNECTIVITY = 4
CLUSTER_RADII_DEG = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0)


def _cluster_record(result: Any) -> dict[str, Any]:
    return {
        **result.describe(),
        "clusters": [
            {
                "cluster_id": item.cluster_id,
                "match_count": len(item.match_indices),
                "match_indices": item.match_indices.tolist(),
                "diameter_a_deg": item.diameter_a_deg,
                "diameter_b_deg": item.diameter_b_deg,
                "region_a": _region_record(item.region_a),
                "region_b": _region_record(item.region_b),
            }
            for item in result.clusters
        ],
    }


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

    unique_predicted = set(predicted_links)
    unique_correct = set(correct_links)
    expected_entities = {
        (_entity_name(first), _entity_name(second)) for first, second in expected_links
    }
    predicted_entities = {
        (_entity_name(first), _entity_name(second))
        for first, second in unique_predicted
    }
    return {
        "radius_deg": variant["radius_deg"],
        "component_mappings": mapping_records,
        "hypothesis_count": len(result.hypotheses),
        "localized_hypothesis_count": sum(
            item.location.state == "localized" for item in result.hypotheses
        ),
        "predicted_manual_links": [list(item) for item in sorted(unique_predicted)],
        "correct_manual_links": [list(item) for item in sorted(unique_correct)],
        "false_manual_links": [list(item) for item in sorted(set(false_links))],
        "unmapped_hypothesis_ids": unmapped_hypotheses,
        "expected_instance_count": len(expected_links),
        "mapped_hypothesis_count": len(predicted_links),
        "correct_hypothesis_count": len(correct_links),
        "false_hypothesis_count": len(false_links),
        "strict_hypothesis_precision": len(correct_links)
        / max(len(result.hypotheses), 1),
        "hypothesis_mapping_rate": len(predicted_links)
        / max(len(result.hypotheses), 1),
        "identity_recall": len(unique_correct) / max(len(expected_links), 1),
        "identity_precision": len(unique_correct) / max(len(unique_predicted), 1),
        "entity_pairs_recovered": [
            list(item) for item in sorted(predicted_entities & expected_entities)
        ],
        "merged_component_count": sum(
            item["merges_multiple_manual_instances"] for item in mapping_records
        ),
    }


def _write_summary(path: Path, evaluations: list[dict[str, Any]]) -> None:
    fields = (
        "radius_deg",
        "hypotheses",
        "localized",
        "correct_links",
        "false_links",
        "unmapped_hypotheses",
        "mapped_hypotheses",
        "correct_hypotheses",
        "strict_hypothesis_precision",
        "hypothesis_mapping_rate",
        "identity_precision",
        "identity_recall",
        "merged_components",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for evaluation in evaluations:
            writer.writerow(
                {
                    "radius_deg": evaluation["radius_deg"],
                    "hypotheses": evaluation["hypothesis_count"],
                    "localized": evaluation["localized_hypothesis_count"],
                    "correct_links": len(evaluation["correct_manual_links"]),
                    "false_links": len(evaluation["false_manual_links"]),
                    "unmapped_hypotheses": len(evaluation["unmapped_hypothesis_ids"]),
                    "mapped_hypotheses": evaluation["mapped_hypothesis_count"],
                    "correct_hypotheses": evaluation["correct_hypothesis_count"],
                    "strict_hypothesis_precision": evaluation[
                        "strict_hypothesis_precision"
                    ],
                    "hypothesis_mapping_rate": evaluation["hypothesis_mapping_rate"],
                    "identity_precision": evaluation["identity_precision"],
                    "identity_recall": evaluation["identity_recall"],
                    "merged_components": evaluation["merged_component_count"],
                }
            )


def run(dataset_root: Path, output_dir: Path) -> dict[str, Any]:
    import cv2

    from panorai.object_localization import (
        JointMatchClusterConfig,
        ObjectLocalizationConfig,
        ObjectLocalizationPipeline,
        PairObjectLocalizationInput,
        SemanticQuery,
        SemanticRegionProposalConfig,
        cluster_region_association_matches,
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

    base_regions = []
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
                region_id_prefix=f"cam-{pair['views'][view_index]['key']}-base",
                class_id=class_id,
                class_name=class_record["class_name"],
                config=SemanticRegionProposalConfig(
                    threshold=CAM_THRESHOLD,
                    connectivity=CAM_CONNECTIVITY,
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
        base_regions.append(tuple(regions))
        proposal_diagnostics.append(diagnostics)

    base_evidence = PairObjectLocalizationInput(
        view_id_a=pair["views"][0]["view_id"],
        view_id_b=pair["views"][1]["view_id"],
        query=query,
        regions_a=base_regions[0],
        regions_b=base_regions[1],
        features_a=features[0],
        features_b=features[1],
        matches=matches,
        pose=pose,
    )
    policy = ObjectLocalizationConfig()
    pipeline = ObjectLocalizationPipeline(policy)
    base_result = pipeline.estimate(base_evidence)
    accepted_parents = [
        item for item in base_result.associations if item.state == "accepted"
    ]

    variants = []
    runtime_objects = []
    for radius in CLUSTER_RADII_DEG:
        cluster_results = [
            cluster_region_association_matches(
                association,
                base_evidence,
                JointMatchClusterConfig(
                    max_neighbor_angle_deg=radius,
                    min_matches=3,
                    pose_inliers_only=True,
                ),
            )
            for association in accepted_parents
        ]
        regions_a = tuple(
            region for cluster in cluster_results for region in cluster.regions_a
        )
        regions_b = tuple(
            region for cluster in cluster_results for region in cluster.regions_b
        )
        refined_evidence = PairObjectLocalizationInput(
            view_id_a=base_evidence.view_id_a,
            view_id_b=base_evidence.view_id_b,
            query=query,
            regions_a=regions_a,
            regions_b=regions_b,
            features_a=features[0],
            features_b=features[1],
            matches=matches,
            pose=pose,
        )
        result = pipeline.estimate(refined_evidence)
        repeated = pipeline.estimate(refined_evidence)
        variant = {
            "radius_deg": radius,
            "min_matches": 3,
            "region_count_a": len(regions_a),
            "region_count_b": len(regions_b),
            "clusters": [_cluster_record(item) for item in cluster_results],
            "associations": [_association_record(item) for item in result.associations],
            "hypotheses": [_hypothesis_record(item) for item in result.hypotheses],
            "hypothesis_ids_repeat_identically": [
                item.hypothesis_id for item in result.hypotheses
            ]
            == [item.hypothesis_id for item in repeated.hypotheses],
        }
        variants.append(variant)
        runtime_objects.append(((regions_a, regions_b), result))

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
        "cam_proposal": {
            "threshold": CAM_THRESHOLD,
            "connectivity": CAM_CONNECTIVITY,
            "diagnostics": proposal_diagnostics,
            "regions_a": [_region_record(item) for item in base_regions[0]],
            "regions_b": [_region_record(item) for item in base_regions[1]],
            "associations": [
                _association_record(item) for item in base_result.associations
            ],
        },
        "native_cam_artifact": {"path": str(cam_path), "sha256": _sha256(cam_path)},
        "variants": variants,
        "elapsed_seconds": time.perf_counter() - started,
    }
    prediction_path = output_dir / "prediction.json"
    _atomic_json(prediction_path, prediction, readonly=True)

    # Manual instances and expected links are opened only after prediction freeze.
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
        "schema": "panorai-real-joint-match-clusters-evaluation/v1",
        "prediction_path": str(prediction_path),
        "prediction_sha256": _sha256(prediction_path),
        "reference_opened_after_prediction_freeze": True,
        "manual_regions_opened_after_prediction_freeze": True,
        "expected_links": [list(item) for item in sorted(expected_links)],
        "evaluations": evaluations,
        "dataset_or_model_bytes_distributed": False,
    }
    _atomic_json(output_dir / "evaluation.json", evaluation)
    _write_summary(output_dir / "summary.csv", evaluations)
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
