from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from benchmarks.spherical_fcn_cam.evaluate_semantic_pairs import (
    _peak_to_mask_degrees,
    decode_stanford_semantic,
    evaluate,
    load_pairs,
    localization_metrics,
    preflight,
    spherical_row_weights,
    stanford_class_mask,
    stanford_support,
)
from benchmarks.spherical_fcn_cam.run_experiment import _selected_class_indices
from benchmarks.spherical_fcn_cam.run_public_datasets import (
    semantic_pair_class_indices,
)


ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / "benchmarks/spherical_fcn_cam/semantic_pairs.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rgb_index(index: int) -> tuple[int, int, int]:
    return (index >> 16) & 255, (index >> 8) & 255, index & 255


def _write_semantic(path: Path, indices: np.ndarray) -> None:
    rgb = np.zeros((*indices.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (indices >> 16) & 255
    rgb[..., 1] = (indices >> 8) & 255
    rgb[..., 2] = indices & 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def _write_heatmap(path: Path, values: np.ndarray) -> None:
    encoded = np.round(values * 65535).astype(np.uint16)
    Image.fromarray(encoded, mode="I;16").save(path)


def _write_mapping(path: Path, labels_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "panorai-imagenet-stanford-semantic-pairs/v1",
                "imagenet": {},
                "stanford2d3d": {
                    "semantic_labels_sha256": _sha256(labels_path),
                },
                "pairs": [
                    {
                        "imagenet_index": 453,
                        "imagenet_name": "bookcase",
                        "stanford_class": "bookcase",
                        "relation_kind": "exact",
                        "tier": "primary",
                    }
                ],
            }
        )
    )


def test_frozen_mapping_separates_primary_from_related_pairs() -> None:
    _, primary = load_pairs(MAPPING, include_related=False)
    _, extended = load_pairs(MAPPING, include_related=True)

    assert {pair["stanford_class"] for pair in primary} == {
        "bookcase",
        "chair",
        "door",
        "sofa",
        "table",
    }
    assert {pair["stanford_class"] for pair in extended} == {
        "bookcase",
        "chair",
        "door",
        "sofa",
        "table",
        "window",
    }
    assert semantic_pair_class_indices(include_related=False) == tuple(
        pair["imagenet_index"] for pair in primary
    )
    assert _selected_class_indices([3, 1], (1, 7, 3, 9)) == [3, 1, 7, 9]


def test_official_rgb_index_decoding_and_coarse_mask(tmp_path: Path) -> None:
    indices = np.array([[1, 2], [0, 0x0D0D0D]], dtype=np.uint32)
    path = tmp_path / "semantic.png"
    _write_semantic(path, indices)
    labels = ["<UNK>_0_<UNK>_0_0", "chair_1_office_1_1", "table_1_office_1_1"]

    decoded = decode_stanford_semantic(path)

    assert np.array_equal(decoded, indices)
    assert np.array_equal(
        stanford_support(decoded, labels), [[True, True], [False, False]]
    )
    assert np.array_equal(
        stanford_class_mask(decoded, labels, "chair"),
        [[True, False], [False, False]],
    )
    assert _rgb_index(0x010203) == (1, 2, 3)


def test_localization_metrics_use_exact_spherical_area_and_random_baseline() -> None:
    cam = np.zeros((4, 8), dtype=np.float32)
    target = np.zeros((4, 8), dtype=bool)
    target[1, :2] = True
    cam[target] = 1.0
    support = np.ones_like(target)

    metrics = localization_metrics(cam, target, support)
    weights = spherical_row_weights(4)
    expected_fraction = 2 * weights[1] / (8 * weights.sum())

    assert metrics["target_spherical_fraction_within_support"] == pytest.approx(
        expected_fraction
    )
    assert metrics["cam_mass_inside_target_within_support"] == pytest.approx(1.0)
    assert metrics["cam_mass_lift_over_random"] == pytest.approx(
        1.0 / expected_fraction
    )
    assert metrics["peak_hits_target"] is True
    assert metrics["peak_to_target_degrees"] == pytest.approx(0.0)
    assert metrics["top_area_iou"] == pytest.approx(1.0)
    assert metrics["random_peak_hit_probability"] == pytest.approx(expected_fraction)

    zero_metrics = localization_metrics(np.zeros_like(cam), target, support)
    assert zero_metrics["cam_has_positive_mass"] is False
    assert zero_metrics["cam_mass_inside_target_within_support"] is None
    assert zero_metrics["peak_defined"] is False
    assert zero_metrics["top_area_iou"] is None

    seam_target = np.zeros((1, 8), dtype=bool)
    seam_target[0, 7] = True
    assert _peak_to_mask_degrees(0, 0, seam_target) == pytest.approx(45.0)


def test_preflight_reports_missing_semantics_and_requested_cams(tmp_path: Path) -> None:
    labels_path = tmp_path / "semantic_labels.json"
    labels_path.write_text(json.dumps(["<UNK>_0_<UNK>_0_0", "bookcase_1_office_1_1"]))
    mapping_path = tmp_path / "mapping.json"
    _write_mapping(mapping_path, labels_path)
    result_path = tmp_path / "run/result.json"
    result_path.parent.mkdir()
    result_path.write_text(
        json.dumps(
            {
                "dataset_sample": {
                    "dataset_id": "stanford2d3d",
                    "view_id": "stanford2d3d::area_1::frame",
                },
                "predictions": [],
            }
        )
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"results": [str(result_path)]}))

    report = preflight(
        summary_path,
        tmp_path / "semantic",
        labels_path,
        mapping_path,
    )

    assert report["ready"] is False
    assert report["unique_stanford_views"] == 1
    assert list(report["missing_semantic"]) == ["stanford2d3d::area_1::frame"]
    assert report["results_missing_requested_cams"][0]["missing_class_indices"] == [453]


def test_end_to_end_semantic_pair_evaluation_on_synthetic_erp(tmp_path: Path) -> None:
    labels_path = tmp_path / "semantic_labels.json"
    labels_path.write_text(json.dumps(["<UNK>_0_<UNK>_0_0", "bookcase_1_office_1_1"]))
    mapping_path = tmp_path / "mapping.json"
    _write_mapping(mapping_path, labels_path)

    semantic_root = tmp_path / "semantic"
    semantic_path = semantic_root / "area_1/pano_semantic/frame.png"
    indices = np.ones((2, 4), dtype=np.uint32)
    indices[:, 2:] = 0
    _write_semantic(semantic_path, indices)

    result_path = tmp_path / "run/result.json"
    result_path.parent.mkdir()
    heatmap_path = result_path.parent / "heatmap.png"
    cam = np.zeros((2, 4), dtype=np.float32)
    cam[:, :2] = 1.0
    _write_heatmap(heatmap_path, cam)
    result_path.write_text(
        json.dumps(
            {
                "dataset_sample": {
                    "dataset_id": "stanford2d3d",
                    "view_id": "stanford2d3d::area_1::frame",
                },
                "model": "resnet18",
                "predictions": [
                    {
                        "class_index": 453,
                        "class_name": "bookcase",
                        "probability": 0.25,
                        "score": 2.0,
                        "rank": 4,
                        "heatmap": heatmap_path.name,
                    }
                ],
            }
        )
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"results": [str(result_path)]}))

    output = evaluate(summary_path, semantic_root, labels_path, mapping_path)

    assert output["record_count"] == 1
    assert output["records"][0]["target_present"] is True
    assert output["records"][0]["localization"]["top_area_iou"] == pytest.approx(1.0)
    assert output["aggregates"][0]["positive_peak_hit_rate"] == pytest.approx(1.0)
