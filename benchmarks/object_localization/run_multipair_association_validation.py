#!/usr/bin/env python3
"""Validate object association and fail-closed controls on real panorama pairs.

The prediction phase opens RGB and frozen manual regions only. The complete
prediction is made read-only before expected cross-view identities or gates
are opened.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.object_localization.run_practical_validation import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    _atomic_json,
    _checked_path,
    _feature_pipeline,
    _load_json,
    _pose_estimator,
    _read_and_resize,
    _region_mask,
    _sha256,
)

REGIONS_PATH = Path(__file__).with_name("stanford_multipair_regions.json")
REFERENCE_PATH = Path(__file__).with_name("stanford_multipair_reference.json")


def _preflight(dataset_root: Path) -> dict[str, Any]:
    regions = _load_json(REGIONS_PATH)
    reference = _load_json(REFERENCE_PATH)
    pair_ids: set[str] = set()
    checked: list[dict[str, Any]] = []
    for pair in regions["pairs"]:
        pair_id = str(pair["pair_id"])
        if pair_id in pair_ids:
            raise ValueError(f"duplicate pair_id: {pair_id}")
        pair_ids.add(pair_id)
        if len(pair["views"]) != 2:
            raise ValueError(f"{pair_id} must contain exactly two views")
        for view in pair["views"]:
            path = _checked_path(
                dataset_root, view["rgb_relative_path"], view["rgb_sha256"]
            )
            checked.append(
                {
                    "pair_id": pair_id,
                    "view": view["key"],
                    "path": str(path),
                    "sha256": view["rgb_sha256"],
                }
            )
    reference_ids = {str(pair["pair_id"]) for pair in reference["pairs"]}
    if reference_ids != pair_ids:
        raise ValueError("regions and reference pair IDs differ")
    return {
        "ready": True,
        "dataset_root": str(dataset_root.resolve()),
        "pair_count": len(pair_ids),
        "scenario_count": len(pair_ids) * len(regions["controls"]),
        "checked_inputs": checked,
        "dataset_bytes_will_be_copied": False,
    }


def _materialize_regions(
    view: dict[str, Any], features: Any, shape_hw: tuple[int, int]
) -> tuple[Any, ...]:
    from panorai.object_localization import semantic_region_from_map

    return tuple(
        semantic_region_from_map(
            _region_mask(shape_hw, record["rectangles_xyxy"]),
            erp_shape_hw=shape_hw,
            features=features,
            region_id=record["region_id"],
            class_id=int(record["class_id"]),
            class_name=record["class_name"],
            threshold=0.5,
            semantic_score=float(record["semantic_score"]),
            source_id="stanford-multipair-manual-rgb-regions/v1",
        )
        for record in view["regions"]
    )


def _controlled_regions_b(
    regions_b: tuple[Any, ...], control: dict[str, Any]
) -> tuple[Any, ...]:
    operation = control["operation"]
    if operation == "none":
        return regions_b
    if not regions_b:
        raise ValueError("controls require at least one view-B region")
    target_index = 0 if control["target"] == "first" else len(regions_b) - 1
    target = regions_b[target_index]
    if operation == "drop-region-b":
        return tuple(
            region for index, region in enumerate(regions_b) if index != target_index
        )
    if operation == "mismatch-class-b":
        changed = replace(
            target,
            class_id=1000 + int(target.class_id),
            class_name=f"mismatched::{target.class_name}",
        )
        return tuple(
            changed if index == target_index else region
            for index, region in enumerate(regions_b)
        )
    if operation == "duplicate-region-b":
        duplicate = replace(target, region_id=f"{target.region_id}--duplicate")
        return (*regions_b, duplicate)
    raise ValueError(f"unknown control operation: {operation}")


def _result_record(result: Any, repeated: Any) -> dict[str, Any]:
    return {
        "predicted_links": [list(item.region_ids) for item in result.hypotheses],
        "hypothesis_ids": [item.hypothesis_id for item in result.hypotheses],
        "location_modes": [item.location.mode for item in result.hypotheses],
        "ids_repeat_identically": [item.hypothesis_id for item in result.hypotheses]
        == [item.hypothesis_id for item in repeated.hypotheses],
        "associations": [
            {
                "region_ids": [item.region_id_a, item.region_id_b],
                "state": item.state,
                "reasons": list(item.reasons),
                "match_count": item.match_count,
                "inlier_count": item.inlier_count,
                "ranking_score": item.ranking_score,
            }
            for item in result.associations
        ],
    }


def _predict_pair(
    pair: dict[str, Any],
    controls: list[dict[str, Any]],
    dataset_root: Path,
    shape_hw: tuple[int, int],
) -> dict[str, Any]:
    from panorai.object_localization import (
        ObjectLocalizationPipeline,
        PairObjectLocalizationInput,
        SemanticQuery,
    )

    views = pair["views"]
    paths = [
        _checked_path(dataset_root, view["rgb_relative_path"], view["rgb_sha256"])
        for view in views
    ]
    images = tuple(_read_and_resize(path, shape_hw) for path in paths)
    feature_pipeline = _feature_pipeline()
    features_a = feature_pipeline.extract(images[0], panorama_id=views[0]["view_id"])
    features_b = feature_pipeline.extract(images[1], panorama_id=views[1]["view_id"])
    matches = feature_pipeline.match(features_a, features_b)
    pose = _pose_estimator().estimate(matches.to_bearing_correspondences())
    if pose is None:
        raise RuntimeError(f"relative pose was not returned for {pair['pair_id']}")
    regions_a = _materialize_regions(views[0], features_a, shape_hw)
    base_regions_b = _materialize_regions(views[1], features_b, shape_hw)
    query_record = pair["query"]
    query = SemanticQuery(
        text=query_record["text"],
        class_ids=tuple(query_record["class_ids"]),
        vocabulary=query_record["vocabulary"],
    )
    pipeline = ObjectLocalizationPipeline()
    scenario_records = []
    for control in controls:
        evidence = PairObjectLocalizationInput(
            view_id_a=views[0]["view_id"],
            view_id_b=views[1]["view_id"],
            query=query,
            regions_a=regions_a,
            regions_b=_controlled_regions_b(base_regions_b, control),
            features_a=features_a,
            features_b=features_b,
            matches=matches,
            pose=pose,
            translation_scale=None,
        )
        result = pipeline.estimate(evidence)
        repeated = pipeline.estimate(evidence)
        scenario_records.append(
            {"name": control["name"], **_result_record(result, repeated)}
        )
    return {
        "pair_id": pair["pair_id"],
        "area": pair["area"],
        "scene": pair["scene"],
        "feature_count_a": len(features_a),
        "feature_count_b": len(features_b),
        "match_count": len(matches),
        "pose": {
            "quality_accepted": bool(pose.quality_report.accepted),
            "inlier_count": int(pose.num_inliers),
            "degenerate": bool(pose.degenerate),
        },
        "scenarios": scenario_records,
    }


def _metrics(true_positives: int, false_positives: int, false_negatives: int) -> dict:
    predicted = true_positives + false_positives
    expected = true_positives + false_negatives
    precision = true_positives / predicted if predicted else 1.0
    recall = true_positives / expected if expected else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _evaluate(prediction: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    reference_pairs = {pair["pair_id"]: pair for pair in reference["pairs"]}
    totals = {
        "base": [0, 0, 0],
        "controls": [0, 0, 0],
    }
    scenario_results = []
    abstentions: list[bool] = []
    deterministic: list[bool] = []
    evidence_coverage = []
    for pair in prediction["pairs"]:
        pair_reference = reference_pairs[pair["pair_id"]]
        for scenario in pair["scenarios"]:
            expected_record = pair_reference["scenarios"][scenario["name"]]
            expected = {tuple(link) for link in expected_record["expected_links"]}
            predicted = {tuple(link) for link in scenario["predicted_links"]}
            true_links = predicted & expected
            false_links = predicted - expected
            missing_links = expected - predicted
            bucket = "base" if scenario["name"] == "base" else "controls"
            totals[bucket][0] += len(true_links)
            totals[bucket][1] += len(false_links)
            totals[bucket][2] += len(missing_links)
            target = expected_record.get("must_abstain_region_a")
            target_abstained = (
                None if target is None else all(link[0] != target for link in predicted)
            )
            if target_abstained is not None:
                abstentions.append(target_abstained)
            deterministic.append(bool(scenario["ids_repeat_identically"]))
            scenario_results.append(
                {
                    "pair_id": pair["pair_id"],
                    "name": scenario["name"],
                    "expected_links": sorted([list(link) for link in expected]),
                    "predicted_links": sorted([list(link) for link in predicted]),
                    "unexpected_links": sorted([list(link) for link in false_links]),
                    "missing_links": sorted([list(link) for link in missing_links]),
                    "target_abstained": target_abstained,
                    "exact": predicted == expected,
                }
            )
            if scenario["name"] == "base":
                association_index = {
                    tuple(item["region_ids"]): item for item in scenario["associations"]
                }
                for link in sorted(expected):
                    association = association_index.get(link)
                    if not pair["pose"]["quality_accepted"]:
                        blocker = "relative-pose-not-accepted"
                    elif association is None:
                        blocker = "no-class-compatible-proposal"
                    elif (
                        association["match_count"] < 3
                        or association["inlier_count"] < 3
                    ):
                        blocker = "insufficient-region-support"
                    elif "high-region-pose-residual" in association["reasons"]:
                        blocker = "high-region-pose-residual"
                    else:
                        blocker = None
                    evidence_coverage.append(
                        {
                            "pair_id": pair["pair_id"],
                            "region_ids": list(link),
                            "eligible": blocker is None,
                            "blocker": blocker,
                            "promoted": link in predicted,
                            "match_count": (
                                None
                                if association is None
                                else association["match_count"]
                            ),
                            "inlier_count": (
                                None
                                if association is None
                                else association["inlier_count"]
                            ),
                        }
                    )
    base_metrics = _metrics(*totals["base"])
    control_metrics = _metrics(*totals["controls"])
    target_abstention_rate = sum(abstentions) / len(abstentions)
    deterministic_id_rate = sum(deterministic) / len(deterministic)
    accepted_pose_rate = sum(
        pair["pose"]["quality_accepted"] for pair in prediction["pairs"]
    ) / len(prediction["pairs"])
    eligible = [item for item in evidence_coverage if item["eligible"]]
    eligible_promoted = [item for item in eligible if item["promoted"]]
    blockers: dict[str, int] = {}
    for item in evidence_coverage:
        if item["blocker"] is not None:
            blockers[item["blocker"]] = blockers.get(item["blocker"], 0) + 1
    gates = reference["development_gates"]
    gate_results = {
        "base_precision": base_metrics["precision"] >= gates["base_precision_min"],
        "base_recall": base_metrics["recall"] >= gates["base_recall_min"],
        "control_precision": control_metrics["precision"]
        >= gates["control_precision_min"],
        "target_abstention": target_abstention_rate
        >= gates["target_abstention_rate_min"],
        "deterministic_ids": deterministic_id_rate
        >= gates["deterministic_id_rate_min"],
        "accepted_pose_pairs": accepted_pose_rate
        >= gates["accepted_pose_pair_rate_min"],
    }
    return {
        "base": base_metrics,
        "controls": control_metrics,
        "target_abstention_rate": target_abstention_rate,
        "deterministic_id_rate": deterministic_id_rate,
        "accepted_pose_pair_rate": accepted_pose_rate,
        "association_given_eligible_evidence": {
            "expected_identity_count": len(evidence_coverage),
            "eligible_identity_count": len(eligible),
            "promoted_eligible_identity_count": len(eligible_promoted),
            "conditional_recall": len(eligible_promoted) / max(len(eligible), 1),
            "blocker_counts": blockers,
            "identities": evidence_coverage,
        },
        "scenarios": scenario_results,
        "gates": gate_results,
        "all_gates_passed": all(gate_results.values()),
    }


def run(dataset_root: Path, output_dir: Path) -> dict[str, Any]:
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    regions = _load_json(REGIONS_PATH)
    shape_hw = tuple(int(value) for value in regions["evaluation_shape_hw"])
    started = time.perf_counter()
    prediction = {
        "schema": "panorai-object-association-multipair-prediction/v1",
        "regions_sha256": _sha256(REGIONS_PATH),
        "pair_count": len(regions["pairs"]),
        "scenario_count": len(regions["pairs"]) * len(regions["controls"]),
        "pairs": [
            _predict_pair(pair, regions["controls"], dataset_root, shape_hw)
            for pair in regions["pairs"]
        ],
    }
    prediction_elapsed_seconds = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "prediction.json"
    _atomic_json(prediction_path, prediction, readonly=True)

    reference = _load_json(REFERENCE_PATH)
    evaluation = _evaluate(prediction, reference)
    summary = {
        "schema": "panorai-object-association-multipair-evaluation/v1",
        "status": "PASS" if evaluation["all_gates_passed"] else "FAIL",
        "evidence_class": "post-hoc-real-panorama-development",
        "prediction": str(prediction_path),
        "prediction_sha256": _sha256(prediction_path),
        "prediction_elapsed_seconds": prediction_elapsed_seconds,
        "reference_opened_after_prediction_freeze": True,
        "reference_sha256": _sha256(REFERENCE_PATH),
        "dataset_bytes_distributed": False,
        "evaluation": evaluation,
        "limitations": [
            "six selected high-overlap Stanford2D3D pairs are not a population sample",
            "manual RGB rectangles isolate association from detector and CAM quality",
            "ImageNet IDs include visual proxies for exercise equipment and tables",
            "controls mutate frozen region observations rather than acquiring new scenes",
            "no graph or persistent identity beyond one image pair is evaluated",
        ],
    }
    _atomic_json(output_dir / "summary.json", summary)
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
