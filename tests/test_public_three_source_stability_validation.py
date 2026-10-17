from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.panoramic_stereo.run_public_three_source_validation import (
    CONFIGURATIONS,
    DEFAULT_MANIFEST,
    PublicSourceSet,
    _populations,
    deterministic_evidence,
    load_manifest,
    load_public_source_set,
    registered_source_set,
)
from benchmarks.panoramic_stereo.run_real_domain_validation import PairData


FROZEN_MANIFEST_SHA256 = (
    "2ea6b5a922bbb1119c89b7e5699705684137ccfe0e3fbfe22c5840ee89eb6b22"
)


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


def test_frozen_three_source_manifest_is_outcome_blind_and_balanced() -> None:
    manifest = load_manifest(DEFAULT_MANIFEST)

    assert manifest["protocol_id"] == "VAL-112-public-three-source-full-sphere-stereo"
    assert manifest["selection_policy"]["dense_output_used_for_selection"] is False
    assert len(manifest["sets"]) == 6
    assert {
        dataset: sum(item["dataset_id"] == dataset for item in manifest["sets"])
        for dataset in ("stanford2d3d", "matterport360")
    } == {"stanford2d3d": 3, "matterport360": 3}
    assert all(len(item["sources"]) == 3 for item in manifest["sets"])
    assert (
        hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest()
        == FROZEN_MANIFEST_SHA256
    )


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
            lambda payload: payload["sets"][0]["sources"].reverse(),
            "overlap-ranked",
        ),
        (
            lambda payload: payload["sets"][0]["sources"][0].update(baseline_m=0.01),
            "baseline outside frozen policy",
        ),
        (
            lambda payload: payload["sets"][4].update(
                group=payload["sets"][3]["group"]
            ),
            "distinct spatial groups",
        ),
    ],
)
def test_three_source_manifest_rejects_policy_drift(
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


def test_source_set_loader_audits_reference_and_all_metric_baselines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pairs = {
        "pair-a": _pair("pair-a", 0.5, 10),
        "pair-b": _pair("pair-b", 0.75, 20),
        "pair-c": _pair("pair-c", 1.0, 30),
    }

    def fake_loader(dataset_id, root, shape_hw, selected):
        assert dataset_id == "stanford2d3d"
        assert root == tmp_path
        assert shape_hw == (4, 8)
        return pairs[selected["pair_id"]]

    monkeypatch.setattr(
        "benchmarks.panoramic_stereo.run_public_three_source_validation."
        "load_public_pair_selection",
        fake_loader,
    )
    selection = {
        "dataset_id": "stanford2d3d",
        "set_id": "fixture",
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
            {
                "pair_id": "pair-c",
                "source": "source-c",
                "baseline_m": 1.0,
                "minimum_directional_cloud_overlap": 0.7,
            },
        ],
    }

    source_set = load_public_source_set(selection, tmp_path, (4, 8))
    registered = registered_source_set(source_set, source_count=3)

    assert source_set.pairs == tuple(pairs.values())
    assert len(registered.sources) == 3
    assert [source.pose.baseline for source in registered.sources] == [
        0.5,
        0.75,
        1.0,
    ]
    assert np.array_equal(registered.reference, pairs["pair-a"].reference_rgb)


def test_three_source_populations_exclude_sky_and_no_return_from_denominator():
    pairs = tuple(
        _pair(f"pair-{index}", 0.5 + 0.1 * index, 10 + index) for index in range(3)
    )
    reference_range = pairs[0].reference_range.copy()
    reference_range[0, 0] = np.nan
    reference_range[0, 1] = 0.0
    source_set = PublicSourceSet(
        dataset_id="stanford2d3d",
        set_id="sky-exclusion-fixture",
        pairs=tuple(
            replace(pair, reference_range=reference_range.copy()) for pair in pairs
        ),
        selection={"fixture": "analytic sky/no-return exclusion"},
    )

    populations, audits, non_sky = _populations(
        source_set,
        np.ones(reference_range.shape, dtype=bool),
        min_range_m=0.3,
        max_range_m=12.0,
    )

    assert not non_sky[0, 0]
    assert not non_sky[0, 1]
    for name, population in populations.items():
        assert not population[0, 0]
        assert not population[0, 1]
        assert audits[name]["excluded_as_sky_pixels"] == 2
        assert audits[name]["non_sky_policy"] == (
            "positive dataset-depth proxy; no RGB/prediction-derived sky mask"
        )


def test_deterministic_three_source_evidence_excludes_paths_and_runtimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = load_manifest(DEFAULT_MANIFEST)
    record = {
        "dataset_id": "stanford2d3d",
        "set_id": "fixture",
        "configuration": next(iter(CONFIGURATIONS)),
        "arrays": "/different/on/every/host.npz",
        "elapsed_seconds": 123.0,
        "populations": {},
    }
    expected = {"stanford2d3d": {}, "matterport360": {}}
    import benchmarks.panoramic_stereo.run_public_three_source_validation as module

    monkeypatch.setattr(module, "aggregate_records", lambda records: expected)
    evidence = deterministic_evidence(
        manifest_path=DEFAULT_MANIFEST,
        manifest=manifest,
        configuration={"shape": [4, 8]},
        records=[record],
    )

    assert "arrays" not in evidence["records"][0]
    assert "elapsed_seconds" not in evidence["records"][0]
    assert evidence["aggregates"] == expected
