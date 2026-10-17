from __future__ import annotations

import json

import numpy as np
import pytest

from benchmarks.panoramic_stereo.run_six_method_stage_report import (
    _candidate_minima,
    _choose_probe,
    _load_full_records,
)
from panorai.stereo.classical import RectifiedStereoPair


def test_candidate_minima_are_cost_ranked_and_separated() -> None:
    disparities = np.arange(10, dtype=np.float64)
    costs = np.asarray((9.0, 2.0, 3.0, 1.0, 4.0, 0.5, 2.0, 0.6, 3.0, 4.0))

    candidates = _candidate_minima(disparities, costs, count=3, separation=2.0)

    assert candidates == [(5.0, 0.5), (7.0, 0.6), (3.0, 1.0)]


def test_probe_selection_uses_reference_validity_and_true_source_target() -> None:
    height, width = 40, 96
    xx = np.arange(width, dtype=np.float32)[None, :]
    reference = np.broadcast_to((xx % 11) / 10.0, (height, width)).copy()
    source = np.zeros_like(reference)
    source[:, :-6] = reference[:, 6:]
    support = np.ones((height, width), dtype=bool)
    source_support = support.copy()
    source_support[:, 40:43] = False
    pair = RectifiedStereoPair(reference, source, support, source_support)
    radial = np.full((height, width), 4.0, dtype=np.float32)
    valid = support.copy()
    disparity = np.full((height, width), 6.0, dtype=np.float32)

    (y, x), texture, continuity = _choose_probe(
        pair,
        radial,
        valid,
        disparity,
        max_disparity=16,
    )

    assert 8 <= y < height - 8
    assert 24 <= x < width - 8
    assert source_support[y, x - 6]
    assert texture > 0.0
    assert continuity == 0.0


def test_full_record_loader_keeps_only_completed_provided_pose(tmp_path) -> None:
    arrays = tmp_path / "stereo-bm.npz"
    np.savez(arrays, placeholder=np.asarray((1,), dtype=np.uint8))
    depth = {
        "abs_rel": 0.1,
        "correct_solid_angle_coverage_delta_1_25": 0.4,
        "delta_1_25": 0.8,
        "evaluated_pixels": 10,
        "median_abs_error_m": 0.2,
        "reference_pixels": 20,
        "rmse_m": 0.3,
        "solid_angle_coverage": 0.5,
    }
    pose = {
        "rotation_error_deg": 0.0,
        "translation_direction_error_deg": 0.0,
    }
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "schema": "panorai-panoramic-stereo-real-domains/v2",
                "source_commit": "a" * 40,
                "source_tree_dirty": False,
                "records": [
                    {
                        "dataset_id": "stanford2d3d",
                        "method": "stereo-bm",
                        "pose_mode": "provided",
                        "status": "completed",
                        "pair_id": "essential-stat-2024",
                        "baseline_m": 0.5,
                        "accepted_view_pairs": 6,
                        "rejected_view_pairs": 0,
                        "elapsed_seconds": 1.0,
                        "arrays": str(arrays),
                        "depth": depth,
                        "pose": pose,
                        "private_path": "/not/allowed/in/public/output",
                    },
                    {
                        "dataset_id": "stanford2d3d",
                        "method": "stereo-bm",
                        "pose_mode": "estimate",
                        "status": "completed",
                    },
                    {
                        "dataset_id": "unlisted-external-dataset",
                        "method": "stereo-bm",
                        "pose_mode": "provided",
                        "status": "completed",
                    },
                    {
                        "dataset_id": "stanford2d3d",
                        "method": "unlisted-experimental-method",
                        "pose_mode": "provided",
                        "status": "completed",
                        "private_id": "not-public",
                    },
                ]
            }
        )
    )

    records, summaries = _load_full_records((path,))

    assert set(records) == {("stanford2d3d", "stereo-bm")}
    assert records[("stanford2d3d", "stereo-bm")]["pose_mode"] == "provided"
    assert "private_path" not in records[("stanford2d3d", "stereo-bm")]
    assert summaries[0]["source_commit"] == "a" * 40
    assert len(summaries[0]["sha256"]) == 64


@pytest.mark.parametrize(
    ("schema", "source_commit", "dirty", "message"),
    (
        ("unsupported", "a" * 40, False, "unsupported schema"),
        ("panorai-panoramic-stereo-real-domains/v2", None, False, "source_commit"),
        ("panorai-panoramic-stereo-real-domains/v2", "a" * 40, True, "clean tree"),
    ),
)
def test_full_record_loader_rejects_untraceable_summaries(
    tmp_path, schema, source_commit, dirty, message
) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "schema": schema,
                "source_commit": source_commit,
                "source_tree_dirty": dirty,
                "records": [],
            }
        )
    )

    with pytest.raises(ValueError, match=message):
        _load_full_records((path,))


def test_full_record_loader_rejects_duplicate_public_method(tmp_path) -> None:
    arrays = tmp_path / "record.npz"
    np.savez(arrays, placeholder=np.asarray((1,), dtype=np.uint8))
    record = {
        "dataset_id": "stanford2d3d",
        "method": "stereo-bm",
        "pose_mode": "provided",
        "status": "completed",
        "pair_id": "essential-stat-2024",
        "baseline_m": 0.5,
        "accepted_view_pairs": 6,
        "rejected_view_pairs": 0,
        "elapsed_seconds": 1.0,
        "arrays": str(arrays),
        "depth": {
            "abs_rel": 0.1,
            "correct_solid_angle_coverage_delta_1_25": 0.4,
            "delta_1_25": 0.8,
            "evaluated_pixels": 10,
            "median_abs_error_m": 0.2,
            "reference_pixels": 20,
            "rmse_m": 0.3,
            "solid_angle_coverage": 0.5,
        },
        "pose": {
            "rotation_error_deg": 0.0,
            "translation_direction_error_deg": 0.0,
        },
    }
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "schema": "panorai-panoramic-stereo-real-domains/v2",
                "source_commit": "a" * 40,
                "source_tree_dirty": False,
                "records": [record, record],
            }
        )
    )

    with pytest.raises(ValueError, match="duplicate"):
        _load_full_records((path,))
