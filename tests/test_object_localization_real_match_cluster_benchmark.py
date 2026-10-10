from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np

from benchmarks.object_localization import run_real_joint_match_clusters as benchmark


def test_real_cluster_prediction_phase_freezes_before_manual_or_reference_load() -> (
    None
):
    source = inspect.getsource(benchmark.run)
    freeze = source.index("_atomic_json(prediction_path, prediction, readonly=True)")
    manual = source.index("manual = _load_json(MANUAL_REGIONS_PATH)")
    reference = source.index("reference = _load_json(REFERENCE_PATH)")

    assert freeze < manual
    assert freeze < reference
    assert benchmark.CAM_THRESHOLD == 0.25
    assert benchmark.CLUSTER_RADII_DEG == (1.0, 2.0, 3.0, 5.0, 8.0, 12.0)


def test_evaluation_distinguishes_correct_false_and_unmapped_hypotheses() -> None:
    manual = {
        "views": [
            {
                "key": "a",
                "regions": [
                    {"region_id": "desk-a", "class_id": 526},
                    {"region_id": "other-a", "class_id": 526},
                ],
            },
            {
                "key": "b",
                "regions": [
                    {"region_id": "desk-b", "class_id": 526},
                    {"region_id": "other-b", "class_id": 526},
                ],
            },
        ]
    }
    masks = [
        {
            "desk-a": np.asarray((True, False, False)),
            "other-a": np.asarray((False, True, False)),
        },
        {
            "desk-b": np.asarray((True, False, False)),
            "other-b": np.asarray((False, True, False)),
        },
    ]

    def region(region_id: str, view_id: str, feature_index: int):
        return SimpleNamespace(
            region_id=region_id,
            view_id=view_id,
            class_id=526,
            feature_indices=np.asarray((feature_index,)),
        )

    regions = (
        (
            region("a-correct", "a", 0),
            region("a-false", "a", 0),
            region("a-unmapped", "a", 2),
        ),
        (
            region("b-correct", "b", 0),
            region("b-false", "b", 1),
            region("b-unmapped", "b", 2),
        ),
    )

    def hypothesis(identifier: str, region_a: str, region_b: str):
        return SimpleNamespace(
            hypothesis_id=identifier,
            region_ids=(region_a, region_b),
            location=SimpleNamespace(state="localized"),
        )

    result = SimpleNamespace(
        hypotheses=(
            hypothesis("correct", "a-correct", "b-correct"),
            hypothesis("false", "a-false", "b-false"),
            hypothesis("unmapped", "a-unmapped", "b-unmapped"),
        )
    )
    evaluation = benchmark._evaluate_variant(
        {"radius_deg": 5.0},
        regions,
        result,
        manual,
        masks,
        {("desk-a", "desk-b"), ("other-a", "other-b")},
    )

    assert evaluation["correct_manual_links"] == [["desk-a", "desk-b"]]
    assert evaluation["false_manual_links"] == [["desk-a", "other-b"]]
    assert evaluation["unmapped_hypothesis_ids"] == ["unmapped"]
    assert evaluation["identity_precision"] == 0.5
    assert evaluation["identity_recall"] == 0.5
    assert evaluation["mapped_hypothesis_count"] == 2
    assert evaluation["correct_hypothesis_count"] == 1
    assert evaluation["false_hypothesis_count"] == 1
    assert evaluation["strict_hypothesis_precision"] == 1 / 3
    assert evaluation["hypothesis_mapping_rate"] == 2 / 3
