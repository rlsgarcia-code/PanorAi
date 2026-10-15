from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_spherical_application_evidence.py"


def _module():
    spec = importlib.util.spec_from_file_location("application_evidence", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_is_outcome_blind_deterministic_and_prefers_independence(
    tmp_path: Path,
) -> None:
    module = _module()
    roots = {}
    split = []
    overlap = []
    for dataset in sorted(module.CORPUS_NAMES):
        root = tmp_path / dataset
        roots[dataset] = root
        for index in range(5):
            group = f"group-{index if index < 2 else 1}"
            left = f"left-{index}"
            right = f"right-{index}"
            for stem in (left, right):
                (root / "rgb" / group).mkdir(parents=True, exist_ok=True)
                (root / "depth" / group).mkdir(parents=True, exist_ok=True)
                (root / "rgb" / group / f"{stem}.png").write_bytes(
                    b"rgb" + bytes([index])
                )
                np.save(root / "depth" / group / f"{stem}.npy", np.ones((2, 4)))
            pair_id = f"{dataset}-{index}"
            split.append(
                {
                    "pair_id": pair_id,
                    "dataset_id": dataset,
                    "partition": "evaluation",
                    "dataset_split": "train",
                    "spatial_group_id": group,
                    "independence_component_id": f"component-{index}",
                    "from_view_id": f"{dataset}::{group}::{left}",
                    "to_view_id": f"{dataset}::{group}::{right}",
                    "covariates": {"baseline_m": 0.2 + index * 0.1},
                    "reference": {
                        "R_to_from": np.eye(3).tolist(),
                        "t_to_from_m": [0.2, 0.0, 0.0],
                    },
                    "forbidden_method_metric": 1.0 - index,
                }
            )
            overlap.append(
                {
                    "pair_id": pair_id,
                    "minimum_directional_cloud_overlap": 0.9 - index * 0.05,
                }
            )

    first = module.select_pairs(split, overlap, roots)
    changed = [dict(item, forbidden_method_metric=-999.0) for item in split]
    second = module.select_pairs(changed, overlap, roots)

    assert first == second
    assert len(first) == 6
    for dataset in roots:
        chosen = [item for item in first if item["dataset_id"] == dataset]
        assert len({item["from"]["view_id"] for item in chosen}) == 3
        assert len({item["spatial_group_id"] for item in chosen}) >= 2
        assert all(item["selection_provenance"]["outcome_blind"] for item in chosen)
        assert all(item["split"] == "development" for item in chosen)
        assert all(
            not item["selection_provenance"]["heldout_eligible"] for item in chosen
        )


def test_immutable_writer_refuses_different_evidence(tmp_path: Path) -> None:
    module = _module()
    target = tmp_path / "evidence.json"
    module.write_immutable_json(target, {"value": 1})
    module.write_immutable_json(target, {"value": 1})

    with pytest.raises(FileExistsError, match="refusing"):
        module.write_immutable_json(target, {"value": 2})


def test_stereo_metrics_distinguish_validity_from_zero_values() -> None:
    module = _module()
    reference = np.full((4, 8), 2.0, dtype=np.float32)
    estimate = reference.copy()
    estimate[0, 0] = 0.0
    valid = np.ones_like(reference, dtype=bool)
    confidence = np.ones_like(reference)

    metrics = module.stereo_metrics(estimate, valid, confidence, reference)

    assert metrics["abs_rel"] == pytest.approx(0.0)
    assert 0.0 < metrics["coverage"] < 1.0


def test_minimum_angular_errors_handles_empty_detections() -> None:
    module = _module()
    expected = np.asarray(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)))

    errors = module._minimum_angular_errors(expected, np.empty((0, 3)))

    np.testing.assert_array_equal(errors, (180.0, 180.0))


def test_nearest_sampler_does_not_turn_nonfinite_coordinates_into_an_index() -> None:
    module = _module()
    values = np.arange(8).reshape(2, 4)

    sampled = module._sample_nearest(
        values, np.asarray(((float("nan"), 1.0), (3.0, 1.0)))
    )

    assert np.isnan(sampled[0])
    assert sampled[1] == 7


def test_core_evaluation_fails_closed_on_protocol_mismatch() -> None:
    module = _module()
    with pytest.raises(ValueError, match="different protocol"):
        module.evaluate_core(
            {
                "contract": module.SELECTION_CONTRACT,
                "protocol_sha256": "wrong",
                "records": [],
            }
        )


def test_selection_policy_is_serializable_and_declares_no_method_outputs() -> None:
    module = _module()
    payload = json.loads(json.dumps(module.SELECTION_POLICY))
    assert payload["method_outputs_used"] is False
    assert payload["partition"] == "evaluation"


def test_slic_analytic_oracles_use_public_wrapper_and_frozen_thresholds() -> None:
    module = _module()

    first = module.evaluate_slic_analytic()
    second = module.evaluate_slic_analytic()

    assert module.canonical_json(first) == module.canonical_json(second)
    assert first["metrics"]["controlled_rotation_stability"] is True
    assert first["metrics"]["planar_low_distortion_limit"] is True
    rotation = first["details"]["rotation"]
    oracle = first["details"]["oracle"]
    assert (
        rotation["minimum_weighted_label_agreement"]
        >= oracle["rotation"]["minimum_weighted_label_agreement"]
    )
    assert rotation["minimum_boundary_f1"] >= oracle["rotation"]["minimum_boundary_f1"]
    assert len(rotation["cases"]) == len(oracle["rotation"]["rotations"]) == 2
    assert all(
        0.0 <= case[metric] <= 1.0
        for case in rotation["cases"]
        for metric in ("weighted_label_agreement", "boundary_f1")
    )
    smallest = first["details"]["planar_limit"][-1]
    assert (
        smallest["label_agreement"]
        >= oracle["planar_limit"]["minimum_smallest_patch_label_agreement"]
    )
    assert (
        smallest["boundary_f1"]
        >= oracle["planar_limit"]["minimum_smallest_patch_boundary_f1"]
    )
    assert all(
        0.0 <= result[metric] <= 1.0
        for result in first["details"]["planar_limit"]
        for metric in ("label_agreement", "boundary_f1")
    )
    assert first["details"]["promotion_limitations"] == []


def test_slic_label_agreement_penalizes_missing_candidate_validity() -> None:
    module = _module()
    reference = np.asarray(((0, 0), (1, 1)), dtype=np.int32)
    candidate = np.asarray(((0, -1), (1, -1)), dtype=np.int32)

    agreement = module._optimal_label_agreement(
        reference, candidate, np.ones(reference.shape, dtype=np.float64)
    )

    assert agreement == pytest.approx(0.5)
