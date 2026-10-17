from __future__ import annotations

import json

import numpy as np

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
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "dataset_id": "stanford2d3d",
                        "method": "stereo-bm",
                        "pose_mode": "provided",
                        "status": "completed",
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
                ]
            }
        )
    )

    records = _load_full_records((path,))

    assert set(records) == {("stanford2d3d", "stereo-bm")}
    assert records[("stanford2d3d", "stereo-bm")]["pose_mode"] == "provided"
