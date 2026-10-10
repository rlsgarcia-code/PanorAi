from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmarks.object_localization.run_real_cam_pose_prior import (
    PAIR_PATH,
    REFERENCE_PATH,
    _manual_feature_mask,
    _pose_errors,
    _prior_enrichment,
    _reference_pose_in_panorai,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_prediction_manifest_contains_no_manual_regions_or_reference_pose() -> None:
    pair = _load(PAIR_PATH)
    serialized = json.dumps(pair, sort_keys=True)

    assert pair["schema"] == "panorai-real-cam-pose-pair/v1"
    assert "rectangles_xyxy" not in serialized
    assert "R_b_from_a" not in serialized
    assert "t_direction_b_from_a" not in serialized
    assert [item["class_id"] for item in pair["query"]["classes"]] == [526, 453]
    assert [item["class_id"] for item in pair["control_query"]["classes"]] == [
        130,
        980,
    ]


def test_reference_frame_conversion_round_trips_exactly() -> None:
    reference = _load(REFERENCE_PATH)
    rotation, translation = _reference_pose_in_panorai(reference)
    record = reference["relative_pose_dataset_frame"]
    basis = np.asarray(record["panorai_from_dataset_basis"])

    np.testing.assert_allclose(
        basis @ rotation @ basis.T, np.asarray(record["R_b_from_a"]), atol=1e-15
    )
    np.testing.assert_allclose(
        basis @ translation,
        np.asarray(record["t_direction_b_from_a"]),
        atol=1e-15,
    )
    errors = _pose_errors(
        {
            "estimated": True,
            "rotation": rotation.tolist(),
            "translation_direction": translation.tolist(),
        },
        reference,
    )
    assert errors["rotation_error_deg"] < 1e-5
    assert errors["translation_error_deg"] < 1e-6


def test_manual_feature_mask_handles_split_region_at_erp_seam() -> None:
    pixels = np.asarray(((2047.8, 700.0), (0.2, 700.0), (1000.0, 700.0), (20.0, 100.0)))
    mask = _manual_feature_mask(
        pixels,
        [[1800, 570, 2048, 960], [0, 570, 190, 960]],
        (1024, 2048),
    )

    np.testing.assert_array_equal(mask, np.asarray((True, True, False, False)))


def test_prior_enrichment_reports_weighted_reference_inlier_mass() -> None:
    prior = SimpleNamespace(
        sampling_weights=np.asarray((4.0, 4.0, 1.0, 1.0)),
        valid_mask=np.ones(4, dtype=bool),
    )
    result = _prior_enrichment(prior, np.asarray((True, True, False, False)))

    assert result["uniform_reference_inlier_fraction"] == 0.5
    assert result["weighted_reference_inlier_fraction"] == 0.8
    assert result["enrichment_ratio"] == 1.6
