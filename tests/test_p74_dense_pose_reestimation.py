from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "dense_pose_match_refinement"
    / "run_p74_stage3.py"
)
SPEC = importlib.util.spec_from_file_location("p74_stage3", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class _Frontend:
    _PANORAI_FROM_DATASET = {
        "p74_native_polar": np.asarray(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
    }


def _pose(*, strict: bool, r: float = 0.1, t: float = 0.2, accepted: bool = True):
    return {
        "returned": True,
        "quality_accepted": accepted,
        "rotation_error_deg": r,
        "translation_direction_error_deg": t,
        "broad": strict,
        "strict": strict,
        "precise": strict and r <= 2.0 and t <= 5.0,
    }


def _row(pair: str, initial: dict, candidate: dict) -> dict:
    return {
        "pair_id": pair,
        "status": "evaluated",
        "initial": initial,
        "filtered": candidate,
        "refined": candidate,
        "dense_valid_fraction": 0.5,
        "dense_supported_matches": 20,
        "dense_accepted_matches": 15,
        "refinement_applied_matches": 4,
        "dense_seconds": 0.1,
        "total_seconds": 1.0,
    }


def test_reference_pose_is_converted_to_panorai_frame() -> None:
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.asarray([1.0, 2.0, 3.0])
    evaluation = {
        "dataset_id": "p74_native_polar",
        "reference": {
            "R_to_from": rotation.tolist(),
            "t_to_from_m": translation.tolist(),
        },
    }
    result_rotation, result_translation = MODULE.reference_pose_panorai(
        evaluation, _Frontend
    )
    adapter = _Frontend._PANORAI_FROM_DATASET["p74_native_polar"]
    np.testing.assert_allclose(result_rotation, adapter.T @ rotation @ adapter)
    np.testing.assert_allclose(result_translation, adapter.T @ translation)


def test_build_matches_preserves_serialized_bearings_and_provenance() -> None:
    bearings_a = np.eye(3)
    bearings_b = np.roll(np.eye(3), 1, axis=0)
    method = {"from_view_id": "a", "to_view_id": "b"}
    prediction = {
        "match_bearings_a": bearings_a.tolist(),
        "match_bearings_b": bearings_b.tolist(),
        "configuration": {"matcher": {"ratio_test": 0.72}},
    }
    matches = MODULE.build_matches(method, prediction)

    np.testing.assert_array_equal(matches.bearings_a, bearings_a)
    np.testing.assert_array_equal(matches.bearings_b, bearings_b)
    assert matches.panorama_id_a == "a"
    assert matches.panorama_id_b == "b"
    assert matches.provenance.deduplicated is True
    assert len(matches.provenance.source_checksums) == 2
    assert matches.matcher_config["serialized_distances_available"] is False


def test_stage_gate_requires_recovery_without_regression() -> None:
    invalid = _pose(strict=False, r=20.0, t=40.0, accepted=False)
    correct = _pose(strict=True)
    rows = [
        _row("kept", correct, _pose(strict=True, r=0.2, t=0.3)),
        _row("recovered", invalid, _pose(strict=True, r=1.0, t=2.0)),
    ]
    gate = MODULE.stage_gate(rows)

    assert gate["passed"] is True
    assert gate["recovered_pairs"] == ["recovered"]
    assert gate["strict_loss_pairs"] == []


def test_stage_gate_rejects_loss_and_bounded_accuracy_regression() -> None:
    correct = _pose(strict=True)
    lost = _pose(strict=False, r=6.0, t=2.0, accepted=False)
    rows = [_row("lost", correct, lost)]
    gate = MODULE.stage_gate(rows)

    assert gate["passed"] is False
    assert gate["strict_loss_pairs"] == ["lost"]
    assert gate["checks"]["rotation_regression_bounded"] is False
    assert gate["max_initially_strict_rotation_regression_deg"] == 5.9
    assert gate["max_initially_strict_translation_regression_deg"] == 1.8

    missing = {
        "returned": False,
        "quality_accepted": False,
        "rotation_error_deg": None,
        "translation_direction_error_deg": None,
        "broad": False,
        "strict": False,
        "precise": False,
    }
    missing_gate = MODULE.stage_gate([_row("missing", correct, missing)])
    assert missing_gate["max_initially_strict_rotation_regression_deg"] is None
    assert missing_gate["max_initially_strict_translation_regression_deg"] is None
