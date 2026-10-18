from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from benchmarks.panoramic_stereo.run_full_sphere_stability_validation import (
    make_candidate,
)
from benchmarks.panoramic_stereo.run_public_three_source_validation import (
    DEFAULT_MANIFEST,
    load_manifest,
)
from benchmarks.panoramic_stereo.run_source_count_gate_decomposition import (
    GATE_POLICIES,
    ROUTES,
    aggregate_records,
    candidate_for_policy,
    deterministic_evidence,
)


def _entry(
    *,
    dataset_id: str,
    policy: str,
    route: str,
    valid_weight: float,
    correct_fraction: float,
    gross_fraction: float,
) -> dict:
    return {
        "dataset_id": dataset_id,
        "set_id": "fixture",
        "gate_policy": policy,
        "configuration": route,
        "elapsed_seconds": 5.0,
        "arrays": "/host/path.npz",
        "populations": {
            population: {
                "metrics": {
                    "overall": {
                        "population_weight": 10.0,
                        "predicted_valid_weight": valid_weight,
                        "delta_1_25": correct_fraction,
                    }
                },
                "label_error": {
                    "gross_gt_2_steps_fixed_population_fraction": gross_fraction
                },
            }
            for population in ("observable-union", "observable-intersection")
        },
    }


def test_gate_policies_are_frozen_extreme_single_gate_ablations() -> None:
    assert tuple(GATE_POLICIES) == (
        "baseline",
        "no-matching-cost-gate",
        "no-confidence-gate",
        "no-matching-cost-or-confidence-gates",
        "no-texture-gate",
    )
    assert GATE_POLICIES == {
        "baseline": {},
        "no-matching-cost-gate": {"max_matching_cost": 1.0},
        "no-confidence-gate": {"min_confidence": 0.0},
        "no-matching-cost-or-confidence-gates": {
            "max_matching_cost": 1.0,
            "min_confidence": 0.0,
        },
        "no-texture-gate": {"min_texture_std": 0.0},
    }
    assert ROUTES == ("two-source", "three-source")


def test_candidate_policy_changes_only_declared_stereo_fields() -> None:
    base = make_candidate(
        face_size=32,
        face_fov_deg=100.0,
        min_range_m=0.3,
        max_range_m=12.0,
        hypotheses=8,
        filter_backend="numpy",
    )
    baseline = asdict(base.options.stereo)
    for name, overrides in GATE_POLICIES.items():
        candidate = candidate_for_policy(base, name)
        actual = asdict(candidate.options.stereo)
        expected = {**baseline, **overrides}
        assert actual == expected
        assert candidate.options.planner == base.options.planner
        assert candidate.options.source_aggregation == "median"

    with pytest.raises(ValueError, match="unsupported gate policy"):
        candidate_for_policy(base, "invented")


def test_aggregation_reports_three_minus_two_on_fixed_population() -> None:
    records = []
    for dataset_id in ("stanford2d3d", "matterport360"):
        for policy in GATE_POLICIES:
            records.extend(
                (
                    _entry(
                        dataset_id=dataset_id,
                        policy=policy,
                        route="two-source",
                        valid_weight=6.0,
                        correct_fraction=0.4,
                        gross_fraction=0.2,
                    ),
                    _entry(
                        dataset_id=dataset_id,
                        policy=policy,
                        route="three-source",
                        valid_weight=7.0,
                        correct_fraction=0.5,
                        gross_fraction=0.1,
                    ),
                )
            )

    result = aggregate_records(records)
    comparison = result["stanford2d3d"]["baseline"]["three-minus-two"][
        "observable-union"
    ]
    assert comparison["three_minus_two_coverage"] == pytest.approx(0.1)
    assert comparison["three_minus_two_delta_1_25"] == pytest.approx(0.1)
    assert comparison["three_minus_two_gross_fixed_fraction"] == pytest.approx(-0.1)


def test_deterministic_evidence_excludes_paths_and_runtimes() -> None:
    manifest = load_manifest(DEFAULT_MANIFEST)
    records = [
        _entry(
            dataset_id=dataset_id,
            policy=policy,
            route=route,
            valid_weight=6.0,
            correct_fraction=0.4,
            gross_fraction=0.2,
        )
        for dataset_id in ("stanford2d3d", "matterport360")
        for policy in GATE_POLICIES
        for route in ROUTES
    ]
    evidence = deterministic_evidence(
        manifest_path=DEFAULT_MANIFEST,
        manifest=manifest,
        configuration={"shape": [4, 8]},
        records=records,
    )

    assert evidence["schema"] == "panorai-source-count-gate-decomposition/v1"
    assert evidence["gate_policies"] == {
        name: dict(overrides) for name, overrides in GATE_POLICIES.items()
    }
    assert all("arrays" not in item for item in evidence["records"])
    assert all("elapsed_seconds" not in item for item in evidence["records"])
    assert (
        Path(evidence["selection"]["manifest_sha256"]).name
        == (evidence["selection"]["manifest_sha256"])
    )
