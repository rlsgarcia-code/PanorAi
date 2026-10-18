from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.panoramic_stereo.run_public_multiview_validation import (
    CONFIGURATIONS,
    DEFAULT_MANIFEST,
    _array_payload_sha256,
    _label_error_metrics,
    deterministic_evidence,
    load_manifest,
    load_public_triplet,
    registered_triplet,
)
from benchmarks.panoramic_stereo.run_real_domain_validation import PairData


def _pair(pair_id: str, translation_x: float, source_value: int) -> PairData:
    shape = (4, 8)
    reference = np.zeros((*shape, 3), dtype=np.uint8)
    source = np.full((*shape, 3), source_value, dtype=np.uint8)
    support = np.ones(shape, dtype=bool)
    radial = np.full(shape, 2.0, dtype=np.float32)
    return PairData(
        dataset_id="stanford2d3d",
        pair_id=pair_id,
        reference_rgb=reference,
        source_rgb=source,
        reference_support=support,
        source_support=support.copy(),
        reference_range=radial,
        source_range=radial.copy(),
        rotation_source_from_reference=np.eye(3),
        translation_source_from_reference=np.array(
            (translation_x, 0.0, 0.0), dtype=np.float64
        ),
        provenance={"fixture": "analytic test"},
    )


def test_frozen_public_multiview_manifest_is_outcome_blind_and_balanced() -> None:
    manifest = load_manifest(DEFAULT_MANIFEST)

    assert manifest["protocol_id"] == "VAL-108-public-multiview-full-sphere-stereo"
    assert manifest["selection_policy"]["dense_output_used_for_selection"] is False
    assert len(manifest["triplets"]) == 6
    assert {
        dataset: sum(
            triplet["dataset_id"] == dataset for triplet in manifest["triplets"]
        )
        for dataset in ("stanford2d3d", "matterport360")
    } == {"stanford2d3d": 3, "matterport360": 3}
    assert all(len(item["sources"]) == 2 for item in manifest["triplets"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["selection_policy"].update(
                dense_output_used_for_selection=True
            ),
            "dense-output blind",
        ),
        (
            lambda payload: payload["triplets"][0]["sources"].reverse(),
            "overlap-ranked",
        ),
        (
            lambda payload: payload["triplets"][0]["sources"][0].update(
                baseline_m=0.01
            ),
            "baseline outside frozen policy",
        ),
    ],
)
def test_manifest_rejects_policy_drift(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    payload = deepcopy(json.loads(DEFAULT_MANIFEST.read_text()))
    mutation(payload)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        load_manifest(path)


def test_public_triplet_loader_audits_reference_and_metric_baselines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pairs = {
        "pair-a": _pair("pair-a", 0.5, 10),
        "pair-b": _pair("pair-b", 0.75, 20),
    }

    def fake_loader(dataset_id, root, shape_hw, selected):
        assert dataset_id == "stanford2d3d"
        assert root == tmp_path
        assert shape_hw == (4, 8)
        return pairs[selected["pair_id"]]

    monkeypatch.setattr(
        "benchmarks.panoramic_stereo.run_public_multiview_validation."
        "load_public_pair_selection",
        fake_loader,
    )
    selection = {
        "dataset_id": "stanford2d3d",
        "triplet_id": "fixture",
        "group": "area_fixture",
        "reference": "reference",
        "sources": [
            {
                "pair_id": "pair-a",
                "source": "source-a",
                "baseline_m": 0.5,
                "minimum_directional_cloud_overlap": 0.9,
            },
            {
                "pair_id": "pair-b",
                "source": "source-b",
                "baseline_m": 0.75,
                "minimum_directional_cloud_overlap": 0.8,
            },
        ],
    }

    triplet = load_public_triplet(selection, tmp_path, (4, 8))
    registered = registered_triplet(triplet, source_count=2)

    assert triplet.pairs == (pairs["pair-a"], pairs["pair-b"])
    assert len(registered.sources) == 2
    assert [source.pose.baseline for source in registered.sources] == [0.5, 0.75]
    assert np.array_equal(registered.reference, pairs["pair-a"].reference_rgb)


def test_array_payload_hash_includes_name_dtype_shape_and_values() -> None:
    first = {"a": np.arange(6, dtype=np.float32).reshape(2, 3)}
    reordered = {"a": first["a"].copy()}
    changed = {"a": first["a"].astype(np.float64)}

    assert _array_payload_sha256(first) == _array_payload_sha256(reordered)
    assert _array_payload_sha256(first) != _array_payload_sha256(changed)


def test_label_error_metrics_keep_missing_predictions_out_of_gross_numerator() -> None:
    truth = np.full((1, 4), 2.0, dtype=np.float32)
    prediction = np.array([[2.0, 12.0, np.nan, 2.0]], dtype=np.float32)
    validity = np.array([[True, True, False, True]])
    population = np.ones((1, 4), dtype=bool)
    weights = np.ones((1, 4), dtype=np.float64)

    metrics = _label_error_metrics(
        truth,
        prediction,
        validity,
        population,
        weights,
        min_range_m=0.3,
        max_range_m=12.0,
        hypotheses=48,
    )

    assert metrics["predicted_valid"] == 3
    assert metrics["gross_gt_2_steps_fixed_population_fraction"] == 0.25
    assert metrics["gross_gt_2_steps_conditional_fraction"] == pytest.approx(1 / 3)


def test_deterministic_evidence_excludes_paths_clocks_and_runtimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = load_manifest(DEFAULT_MANIFEST)
    record = {
        "dataset_id": "stanford2d3d",
        "triplet_id": "fixture",
        "configuration": CONFIGURATIONS[0],
        "arrays": "/different/on/every/host.npz",
        "elapsed_seconds": 123.0,
        "populations": {},
    }
    monkey_aggregate = {
        "stanford2d3d": {CONFIGURATIONS[0]: {}},
        "matterport360": {},
    }

    # Keep this unit test independent from numerical aggregation details.
    import benchmarks.panoramic_stereo.run_public_multiview_validation as module

    monkeypatch.setattr(module, "aggregate_records", lambda records: monkey_aggregate)
    evidence = deterministic_evidence(
        manifest_path=DEFAULT_MANIFEST,
        manifest=manifest,
        configuration={"shape": [4, 8]},
        records=[record],
    )

    assert "arrays" not in evidence["records"][0]
    assert "elapsed_seconds" not in evidence["records"][0]
    assert evidence["aggregates"] == monkey_aggregate
