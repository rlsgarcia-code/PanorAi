from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import benchmarks.two_view_pose_probability.run_aligned_analysis as aligned_analysis
import benchmarks.two_view_pose_probability.verify_aligned_analysis as aligned_verifier
from benchmarks.two_view_pose_probability.write_aligned_paper_results import (
    _release_section as paper_release_section,
)
from benchmarks.two_view_pose_probability.verify_aligned_analysis import (
    Audit as AlignedAnalysisAudit,
    EXPECTED_ROUTE_CONFIGURATION,
    PAPER_SCOPE_MARKERS,
    _capture_policy_violations,
    _constant_baseline_mismatches,
    _discrimination_summary,
    _forbidden_feature_paths,
    _independent_probability_metrics,
    _independent_prospective_power,
    _index as aligned_verification_index,
    _metric_mismatches,
    _model_contract_violations,
    _outcome_consistency_violations,
    _paper_figure_bundle,
    _recompute_calibration_rule,
    _recompute_rule_evaluation,
    _recompute_prospective_grid,
    _release_evaluation_violations,
    _raw_route_violations,
    _roc_auc,
    _rule_recomputation_violations,
    _missing_census_markers,
    _missing_paper_scope_markers,
    _uncertainty_violations,
)
from benchmarks.p74_pair_eligibility.run_optimized_public_pair import (
    profile_configuration as optimized_pair_profile_configuration,
)
from benchmarks.two_view_pose_probability.prepare_controlled_timing_manifest import (
    select as select_timing_pairs,
)
from benchmarks.two_view_pose_probability.prepare_controlled_timing_host_gate import (
    validate_route_result as validate_controlled_timing_route_result,
)
from benchmarks.two_view_pose_probability.run_controlled_timing_benchmark import (
    deterministic_order as controlled_timing_order,
    validate_host_gate as validate_controlled_timing_host_gate,
    validate_host_gate_evidence as validate_controlled_timing_gate_evidence,
    validate_selection as validate_controlled_timing_selection,
)
from benchmarks.two_view_pose_probability.summarize_controlled_timing import (
    TIMING_FIELDS as CONTROLLED_TIMING_FIELDS,
    summarize_cells as summarize_controlled_timing_cells,
    summarize_dataset_aggregates as summarize_controlled_timing_datasets,
)

from benchmarks.two_view_pose_probability.build_pair_table import (
    build_rows,
    missingness,
)
from benchmarks.two_view_pose_probability.build_failure_taxonomy import (
    _categories,
    _post_probabilities,
    _select_representative,
)
from benchmarks.two_view_pose_probability.build_aligned_population_table import (
    _outcomes as aligned_outcomes,
    _post as aligned_post,
    _validate_package as validate_aligned_package,
)
from benchmarks.two_view_pose_probability.measure_public_overlap import (
    cloud_overlap,
    equal_area_pixel_grid,
    measure,
)
from benchmarks.two_view_pose_probability.plan_prospective_confirmation import (
    _retrospective_verdict,
    simulate_power,
)
from benchmarks.two_view_pose_probability.prepare_unified_population_replay import (
    _audit_pairs,
    _stable_interleave,
)
from benchmarks.two_view_pose_probability.run_census import (
    audit_split_integrity,
    load_sources,
    run_census,
    standardize,
    validate_pairs,
)
from benchmarks.two_view_pose_probability.run_probability_models import (
    LODO_FIXED_L2,
    design_matrix,
    fit_logistic,
    models_for_profile,
    predict_logistic,
    select_l2_or_fixed,
)
from benchmarks.two_view_pose_probability.run_prospective_confirmation import (
    AUTHORIZATION as PROSPECTIVE_AUTHORIZATION,
    CANDIDATE_SCHEMA as PROSPECTIVE_CANDIDATE_SCHEMA,
    EXPECTED_GATE as PROSPECTIVE_GATE,
    PREDICTION_SCHEMA as PROSPECTIVE_PREDICTION_SCHEMA,
    REFERENCE_SCHEMA as PROSPECTIVE_REFERENCE_SCHEMA,
    REGISTRY_SCHEMA as PROSPECTIVE_REGISTRY_SCHEMA,
    evaluate as evaluate_prospective_confirmation,
    seal as seal_prospective_confirmation,
)
from benchmarks.two_view_pose_probability.run_aligned_analysis import (
    ANALYSIS_STAGE_ORDER,
    _prepare_output_dir,
)
from benchmarks.two_view_pose_probability.run_resumable_population_replay import (
    EXPECTED_ROUTE_SHA256,
    _load_valid_result,
    _slug,
)
from benchmarks.two_view_pose_probability.seal_population_environment import (
    _installed_tree,
    _validate_status as validate_population_environment_status,
)
from benchmarks.two_view_pose_probability.select_release_rule import (
    _selected as selective_rule_selected,
    exact_one_sided_lower,
)
from benchmarks.two_view_pose_probability.summarize_unified_replay import _state
from benchmarks.two_view_pose_probability.render_paper_results import (
    capture_probability_surface,
    primary_post_model,
    summarize_engineering,
    summarize_runtime_overlap,
    wilson_interval,
)
from benchmarks.two_view_pose_probability.render_narrative_figures import (
    prospective_curve,
    replay_rows,
)


def _row(
    pair: str,
    first: str,
    second: str,
    *,
    dataset: str = "dataset-a",
    group: str = "group-a",
) -> dict[str, str]:
    return {
        "pair_id": pair,
        "dataset_id": dataset,
        "spatial_group_id": group,
        "from_view_id": first,
        "to_view_id": second,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prospective_fixture(
    tmp_path: Path,
    *,
    authorization: str = PROSPECTIVE_AUTHORIZATION,
    include_catastrophic_diagnostic: bool = False,
) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_commit = "f" * 40
    candidate = {
        "schema": PROSPECTIVE_CANDIDATE_SCHEMA,
        "authorization": authorization,
        "panorai": {"version": "3.5.0", "source_commit": source_commit},
        "wheel_sha256": "1" * 64,
        "configuration_sha256": "2" * 64,
        "capture_model_sha256": "3" * 64,
        "post_model_sha256": "4" * 64,
        "selective_rule_sha256": "5" * 64,
        "gate": PROSPECTIVE_GATE,
        "deviations": [],
    }
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    retrospective_path = tmp_path / "retrospective-groups.jsonl"
    _write_jsonl(retrospective_path, [{"group_id": "retrospective-only"}])
    registry_rows = []
    prediction_rows = []
    reference_rows = []
    domains = ("matterport-like", "stanford-like", "p74-like")
    for group_index in range(40):
        group_id = f"prospective-group-{group_index:02d}"
        domain_id = domains[group_index % len(domains)]
        for pair_index in range(3):
            pair_id = f"pair-{group_index:02d}-{pair_index}"
            registry_rows.append(
                {
                    "schema": PROSPECTIVE_REGISTRY_SCHEMA,
                    "pair_id": pair_id,
                    "group_id": group_id,
                    "domain_id": domain_id,
                    "image_ids": [f"{pair_id}-a", f"{pair_id}-b"],
                    "capture": {
                        "registered_cloud_overlap_fraction": 0.60,
                        "baseline_m": 0.50,
                    },
                }
            )
            prediction_rows.append(
                {
                    "schema": PROSPECTIVE_PREDICTION_SCHEMA,
                    "pair_id": pair_id,
                    "group_id": group_id,
                    "domain_id": domain_id,
                    "returned": True,
                    "accepted": True,
                    "selected": True,
                    "primary_selected": True,
                    "p_accept_capture": 0.90,
                    "p_precise_capture_given_accept": 0.95,
                    "p_usable_capture": 0.855,
                    "p_precise_post": 0.98,
                    "failure_reason": None,
                }
            )
            reference_rows.append(
                {
                    "schema": PROSPECTIVE_REFERENCE_SCHEMA,
                    "pair_id": pair_id,
                    "rotation_error_deg": 0.20,
                    "translation_direction_error_deg": 0.50,
                }
            )
    if include_catastrophic_diagnostic:
        pair_id = "diagnostic-catastrophic"
        registry_rows.append(
            {
                "schema": PROSPECTIVE_REGISTRY_SCHEMA,
                "pair_id": pair_id,
                "group_id": "prospective-group-00",
                "domain_id": "matterport-like",
                "image_ids": [f"{pair_id}-a", f"{pair_id}-b"],
                "capture": {"registered_cloud_overlap_fraction": 0.60},
            }
        )
        prediction_rows.append(
            {
                "schema": PROSPECTIVE_PREDICTION_SCHEMA,
                "pair_id": pair_id,
                "group_id": "prospective-group-00",
                "domain_id": "matterport-like",
                "returned": True,
                "accepted": True,
                "selected": False,
                "primary_selected": False,
                "p_accept_capture": 0.90,
                "p_precise_capture_given_accept": 0.95,
                "p_usable_capture": 0.855,
                "p_precise_post": 0.20,
                "failure_reason": "excluded from the primary sample",
            }
        )
        reference_rows.append(
            {
                "schema": PROSPECTIVE_REFERENCE_SCHEMA,
                "pair_id": pair_id,
                "rotation_error_deg": 20.0,
                "translation_direction_error_deg": 45.0,
            }
        )
    registry_path = tmp_path / "registry.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    references_path = tmp_path / "references-after-seal.jsonl"
    _write_jsonl(registry_path, registry_rows)
    _write_jsonl(predictions_path, prediction_rows)
    return {
        "candidate": candidate_path,
        "retrospective": retrospective_path,
        "registry": registry_path,
        "predictions": predictions_path,
        "references": references_path,
        "seal": tmp_path / "prediction-seal.json",
        "evaluation": tmp_path / "evaluation",
        "source_commit": source_commit,
        "reference_rows": reference_rows,
    }


def _prospective_seal_args(paths: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        candidate=paths["candidate"],
        registry=paths["registry"],
        predictions=paths["predictions"],
        retrospective_groups=paths["retrospective"],
        future_references=paths["references"],
        expected_package_version="3.5.0",
        expected_source_commit=str(paths["source_commit"]),
        output=paths["seal"],
    )


def test_prospective_confirmation_seals_then_evaluates_40_groups(
    tmp_path: Path,
) -> None:
    paths = _prospective_fixture(tmp_path)

    sealed = seal_prospective_confirmation(_prospective_seal_args(paths))
    _write_jsonl(paths["references"], paths["reference_rows"])
    report = evaluate_prospective_confirmation(
        argparse.Namespace(
            seal=paths["seal"],
            references=paths["references"],
            output_dir=paths["evaluation"],
        )
    )

    assert sealed["predictions"]["primary_selected"] == 120
    assert sealed["predictions"]["primary_groups"] == 40
    assert report["status"] == "PASS"
    assert report["overall"]["selected_precision"] == 1.0
    assert report["overall"]["exact_one_sided_95_lower"] > 0.90
    assert report["overall"]["catastrophic_accepted"] == 0
    assert set(report["domains"]) == {
        "matterport-like",
        "p74-like",
        "stanford-like",
    }
    assert (paths["evaluation"] / "prospective-joined.jsonl").is_file()


def test_prospective_confirmation_rejects_no_go_or_early_references(
    tmp_path: Path,
) -> None:
    no_go = _prospective_fixture(tmp_path / "no-go", authorization="NO_GO")
    with pytest.raises(ValueError, match="not authorized"):
        seal_prospective_confirmation(_prospective_seal_args(no_go))

    early = _prospective_fixture(tmp_path / "early")
    _write_jsonl(early["references"], early["reference_rows"])
    with pytest.raises(FileExistsError, match="before prediction sealing"):
        seal_prospective_confirmation(_prospective_seal_args(early))


def test_prospective_confirmation_counts_every_accepted_pose_for_safety(
    tmp_path: Path,
) -> None:
    paths = _prospective_fixture(
        tmp_path, include_catastrophic_diagnostic=True
    )
    seal_prospective_confirmation(_prospective_seal_args(paths))
    _write_jsonl(paths["references"], paths["reference_rows"])

    report = evaluate_prospective_confirmation(
        argparse.Namespace(
            seal=paths["seal"],
            references=paths["references"],
            output_dir=paths["evaluation"],
        )
    )

    assert report["status"] == "FAIL"
    assert report["overall"]["selected_precision"] == 1.0
    assert report["overall"]["catastrophic_accepted"] == 1
    assert not report["gate_checks"]["maximum_catastrophic_accepted"]


def test_prospective_confirmation_rejects_outcome_leakage_and_mutation(
    tmp_path: Path,
) -> None:
    leaking = _prospective_fixture(tmp_path / "leaking")
    rows = [json.loads(line) for line in leaking["predictions"].read_text().splitlines()]
    rows[0]["rotation_error_deg"] = 0.0
    _write_jsonl(leaking["predictions"], rows)
    with pytest.raises(ValueError, match="reference-only field"):
        seal_prospective_confirmation(_prospective_seal_args(leaking))

    changed = _prospective_fixture(tmp_path / "changed")
    seal_prospective_confirmation(_prospective_seal_args(changed))
    with changed["predictions"].open("a", encoding="utf-8") as stream:
        stream.write("\n")
    _write_jsonl(changed["references"], changed["reference_rows"])
    with pytest.raises(ValueError, match="missing or changed"):
        evaluate_prospective_confirmation(
            argparse.Namespace(
                seal=changed["seal"],
                references=changed["references"],
                output_dir=changed["evaluation"],
            )
        )


def test_load_sources_skips_prediction_manifest_header(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    _write_jsonl(path, [{"manifest": {"pair_count": 1}}, _row("p1", "a", "b")])

    rows, sources = load_sources([path])

    assert rows == [_row("p1", "a", "b")]
    assert sources[0]["pair_rows"] == 1
    assert sources[0]["skipped_manifest_rows"] == 1


def test_reversed_or_duplicate_pairs_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate or reversed"):
        validate_pairs([_row("p1", "a", "b"), _row("p2", "b", "a")])


def test_self_pair_and_duplicate_pair_id_are_rejected() -> None:
    with pytest.raises(ValueError, match="self-pair"):
        standardize([_row("p1", "a", "a")], seed="test")
    with pytest.raises(ValueError, match="duplicate dataset/pair IDs"):
        validate_pairs([_row("p1", "a", "b"), _row("p1", "c", "d")])


def test_shared_image_merges_nominal_groups_into_one_split() -> None:
    rows = [
        _row("p1", "a", "b", group="g1"),
        _row("p2", "b", "c", group="g2"),
        _row("p3", "d", "e", group="g3"),
        _row("p4", "f", "g", group="g4"),
    ]

    standardized, census = standardize(rows, seed="fixed")
    by_pair = {row["pair_id"]: row for row in standardized}

    assert (
        by_pair["p1"]["independence_component_id"]
        == by_pair["p2"]["independence_component_id"]
    )
    assert by_pair["p1"]["split"] == by_pair["p2"]["split"]
    assert census["dataset-a"]["spatial_groups"] == 4
    assert census["dataset-a"]["independence_components"] == 3
    audit_split_integrity(standardized)


def test_split_is_deterministic_and_has_no_image_or_group_leakage() -> None:
    rows = [
        _row(f"p{index}", f"a{index}", f"b{index}", group=f"g{index}")
        for index in range(20)
    ]

    first, census = standardize(rows, seed="fixed")
    second, _ = standardize(list(reversed(rows)), seed="fixed")

    assert [(row["pair_id"], row["split"]) for row in first] == [
        (row["pair_id"], row["split"]) for row in second
    ]
    assert census["dataset-a"]["components_by_split"] == {
        "development": 14,
        "calibration": 3,
        "evaluation": 3,
    }
    audit_split_integrity(first)


def test_run_census_writes_identity_only_reproducible_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "output"
    _write_jsonl(
        source,
        [
            _row("p1", "a", "b", group="g1"),
            _row("p2", "c", "d", group="g2"),
            _row("p3", "e", "f", group="g3"),
        ],
    )

    payload = run_census([source], output, seed="fixed")

    assert payload["totals"] == {
        "datasets": 1,
        "unique_pairs": 3,
        "unique_images": 6,
        "spatial_groups": 3,
        "independence_components": 3,
    }
    pairs = [
        json.loads(line)
        for line in (output / "standardized-pairs.jsonl").read_text().splitlines()
    ]
    assert {row["split"] for row in pairs} == {
        "development",
        "calibration",
        "evaluation",
    }
    assert not any("reference" in row or "quality_report" in row for row in pairs)
    assert (output / "split-manifest.jsonl").is_file()
    assert (output / "census.json").is_file()


def _frozen_pair(pair: str, dataset: str, group: str) -> dict:
    return {
        "pair_id": pair,
        "dataset_id": dataset,
        "spatial_group_id": group,
        "independence_component_id": f"component-{group}",
        "split": "development",
    }


def test_pair_table_separates_features_from_outcomes_and_adapts_sources() -> None:
    pairs = [
        _frozen_pair("public", "matterport360", "building"),
        _frozen_pair("industrial", "p74_native_polar", "family"),
    ]
    predictions = [
        {
            "pair_id": "public",
            "dataset_id": "matterport360",
            "pose_returned": True,
            "quality_accepted": True,
            "method_version": "historical",
            "match_count": 20,
            "inlier_count": 10,
            "quality_report": {
                "raw_quality_score": 0.8,
                "median_residual_deg": 0.2,
                "p90_residual_deg": 0.4,
                "stability": {"requested_trials": 4, "successful_trials": 3},
                "model_competition": {
                    "essential_score_margin": 0.3,
                    "preferred_model": "essential",
                },
            },
        },
        {
            "pair_id": "industrial",
            "dataset_id": "p74_native_polar",
            "pose_returned": True,
            "quality_accepted": False,
            "variant": "coarse-spherical-tangent-dog-rootsift-d1p5",
            "match_count": 40,
            "inlier_count": 20,
            "quality_score": 0.7,
            "descriptor_distance_median": 0.3,
        },
    ]
    evaluated = [
        {
            "pair_id": "public",
            "dataset_id": "matterport360",
            "baseline_m": "2.0",
            "harmonic_median_depth_m": "4.0",
            "baseline_over_harmonic_median_depth": "0.5",
            "rgb_similarity": "0.9",
            "rotation_change_deg": "30.0",
            "rotation_error_deg": "0.5",
            "translation_direction_error_deg": "2.0",
            "strict_success": "True",
        },
        {
            "pair_id": "industrial",
            "dataset_id": "p74_native_polar",
            "baseline_m": 3.0,
            "rotation_error_deg": 10.0,
            "translation_direction_error_deg": 40.0,
            "primary_success": False,
            "strict_success": False,
            "precise_success": False,
            "catastrophic_accepted": False,
        },
    ]
    overlap = [
        {
            "pair_id": "industrial",
            "dataset_id": "p74_native_polar",
            "baseline_m": 3.0,
            "minimum_directional_cloud_overlap": 0.7,
            "symmetric_cloud_overlap": 0.75,
            "symmetric_covisible_fraction": 0.6,
            "overlap_bin": "ge_0.70",
        }
    ]

    features, outcomes, table = build_rows(pairs, predictions, evaluated, overlap)

    assert len(features) == len(outcomes) == len(table) == 2
    assert "outcomes" not in features[0]
    assert "capture" not in outcomes[0]
    assert features[0]["capture"]["predicted_parallax_deg"] == pytest.approx(
        28.0724869359
    )
    assert features[0]["post"]["inlier_ratio"] == 0.5
    assert features[0]["post"]["stability_success_fraction"] == 0.75
    assert outcomes[0]["precise"] is True
    assert outcomes[0]["usable"] is True
    assert features[1]["capture"]["registered_cloud_overlap_min"] == 0.7
    assert features[1]["method"]["frontend_family"] == "optimized-spherical-dog"
    assert table[0]["reference_only"]["rotation_change_deg"] == 30.0
    report = missingness(table)
    assert (
        report["matterport360"]["fields"]["capture.rgb_similarity"][
            "available_fraction"
        ]
        == 1.0
    )
    assert (
        report["p74_native_polar"]["fields"]["capture.representative_scene_distance_m"][
            "available_fraction"
        ]
        == 0.0
    )


def test_lodo_does_not_claim_cross_validation_with_too_few_groups() -> None:
    values = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    target = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    groups = np.asarray(["source-a", "source-a", "source-b", "source-b"])

    selected, candidates, method = select_l2_or_fixed(values, target, groups)

    assert selected == LODO_FIXED_L2
    assert candidates == []
    assert "only 2 source groups" in method


def test_exact_release_bound_matches_preregistered_59_success_reference() -> None:
    assert exact_one_sided_lower(59, 59) == pytest.approx(0.95049239, abs=1e-8)
    assert exact_one_sided_lower(0, 10) == 0.0
    with pytest.raises(ValueError, match="between zero and count"):
        exact_one_sided_lower(11, 10)


def test_prospective_power_simulation_is_deterministic_and_group_aware() -> None:
    first = simulate_power(mean_precision=0.97, icc=0.2, groups=10, repetitions=200)
    second = simulate_power(mean_precision=0.97, icc=0.2, groups=10, repetitions=200)
    assert first == second
    assert 0.0 <= first <= 1.0
    with pytest.raises(ValueError, match="strictly between"):
        simulate_power(mean_precision=1.0, icc=0.2, groups=10, repetitions=10)
    independent = _independent_prospective_power(
        mean_precision=0.97,
        intraclass_correlation=0.2,
        groups=10,
        repetitions=200,
    )
    assert independent == first
    small_grid = _recompute_prospective_grid(
        repetitions=200, group_grid=(5, 10)
    )
    assert len(small_grid) == 24
    assert all(row["selected_pairs"] == 3 * row["independent_groups"] for row in small_grid)


def test_failure_representative_prefers_target_dataset_evaluation_pair() -> None:
    category = _categories()[0]

    def candidate(dataset: str, split: str, pair: str, overlap: float) -> dict:
        return {
            "dataset_id": dataset,
            "split": split,
            "pair_id": pair,
            "capture": {"registered_cloud_overlap_min": overlap},
            "outcomes": {"returned": False},
        }

    selected = _select_representative(
        [
            candidate("p74_native_polar", "development", "lowest", 0.0),
            candidate("matterport360", "evaluation", "public", 0.0),
            candidate("p74_native_polar", "evaluation", "target", 0.05),
        ],
        category,
    )

    assert selected["pair_id"] == "target"


def test_failure_taxonomy_separates_low_overlap_control_from_estimator_failures() -> None:
    categories = {row["id"]: row for row in _categories()}
    low_overlap = {
        "capture": {"registered_cloud_overlap_min": 0.40},
        "outcomes": {
            "returned": True,
            "accepted": False,
            "precise": False,
            "catastrophic_accepted": False,
            "usable": False,
        },
        "post_probability": 0.95,
    }
    eligible = {
        **low_overlap,
        "capture": {"registered_cloud_overlap_min": 0.60},
    }
    no_pose_control = {
        **low_overlap,
        "capture": {"registered_cloud_overlap_min": 0.05},
        "outcomes": {**low_overlap["outcomes"], "returned": False},
    }

    assert categories["no-return-low-overlap"]["role"] == (
        "negative eligibility control"
    )
    assert categories["no-return-low-overlap"]["predicate"](no_pose_control)
    for category_id in ("returned-but-rejected", "overconfident-near-miss"):
        assert categories[category_id]["role"] == "eligible estimator outcome"
        assert not categories[category_id]["predicate"](low_overlap)
        assert categories[category_id]["predicate"](eligible)
    low_catastrophic = {
        **low_overlap,
        "outcomes": {
            **low_overlap["outcomes"],
            "accepted": True,
            "catastrophic_accepted": True,
        },
    }
    eligible_catastrophic = {
        **low_catastrophic,
        "capture": {"registered_cloud_overlap_min": 0.60},
    }
    assert not categories["catastrophic-accepted"]["predicate"](low_catastrophic)
    assert categories["catastrophic-accepted"]["predicate"](
        eligible_catastrophic
    )


def test_failure_taxonomy_uses_requested_post_model(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    rows = [
        {
            "model_id": model_id,
            "dataset_id": "dataset",
            "pair_id": "pair",
            "probability": probability,
        }
        for model_id, probability in (
            ("post-precise-raw-score", 0.1),
            ("post-precise-aligned-orientation", 0.9),
        )
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert _post_probabilities(path, "post-precise-aligned-orientation") == {
        ("dataset", "pair"): 0.9
    }


def test_pair_table_rejects_missing_predictions_and_outcome_contradictions() -> None:
    pairs = [_frozen_pair("p1", "dataset", "group")]
    with pytest.raises(ValueError, match="prediction/pair key mismatch"):
        build_rows(pairs, [], [], [])

    prediction = {
        "pair_id": "p1",
        "dataset_id": "dataset",
        "pose_returned": True,
        "quality_accepted": True,
    }
    evaluation = {
        "pair_id": "p1",
        "dataset_id": "dataset",
        "rotation_error_deg": 0.1,
        "translation_direction_error_deg": 0.2,
        "strict_success": False,
    }
    with pytest.raises(ValueError, match="strict_success contradicts"):
        build_rows(pairs, [prediction], [evaluation], [])


def test_registered_cloud_overlap_uses_bidirectional_metric_support() -> None:
    # Independent Euclidean fixture in metres: the second cloud is a strict
    # subset of the first. At 1 mm tolerance, directional support is 1/2 and
    # 1/1, so minimum overlap is 0.5 and the symmetric value is sqrt(0.5).
    first = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    second = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)

    result = cloud_overlap(first, second, distance_tolerance_m=0.001)

    assert result["from_in_to_fraction"] == 0.5
    assert result["to_in_from_fraction"] == 1.0
    assert result["minimum_directional_cloud_overlap"] == 0.5
    assert result["symmetric_cloud_overlap"] == pytest.approx(np.sqrt(0.5))


def test_public_overlap_identity_fixture_is_complete(tmp_path: Path) -> None:
    # Canonical fixture: ERP pixel centres, +X right/+Y up/+Z forward, radial
    # range in metres, float32. Identical depth and T_world_from_camera imply
    # identical world clouds and independently imply overlap 1.0.
    depth = np.ones((16, 32), dtype=np.float32)
    first_depth = tmp_path / "first.npy"
    second_depth = tmp_path / "second.npy"
    np.save(first_depth, depth)
    np.save(second_depth, depth)
    identity = np.eye(4, dtype=np.float64).tolist()
    views = [
        {
            "view_id": "a",
            "depth": {"path": str(first_depth)},
            "pose": {"matrix": identity},
        },
        {
            "view_id": "b",
            "depth": {"path": str(second_depth)},
            "pose": {"matrix": identity},
        },
    ]
    pairs = [
        {
            "pair_id": "p1",
            "dataset_id": "fixture",
            "spatial_group_id": "room",
            "from_view_id": "a",
            "to_view_id": "b",
            "reference": {"T_to_from_from": identity},
        }
    ]

    rows = measure(
        pairs,
        views,
        sample_points=128,
        distance_tolerance_m=1e-6,
        workers=1,
    )

    assert rows[0]["minimum_directional_cloud_overlap"] == 1.0
    assert rows[0]["symmetric_cloud_overlap"] == 1.0
    assert rows[0]["reference_transform_max_abs_error"] == 0.0
    pixels = equal_area_pixel_grid(depth.shape, 128)
    assert np.all((0 <= pixels[:, 0]) & (pixels[:, 0] < depth.shape[1]))
    assert np.all((0 <= pixels[:, 1]) & (pixels[:, 1] < depth.shape[0]))


def test_probability_model_is_deterministic_and_feature_whitelisted() -> None:
    rows = [
        {"capture": {"overlap": 0.1, "baseline": 1.0}, "outcomes": {"precise": 0}},
        {"capture": {"overlap": 0.2, "baseline": 2.0}, "outcomes": {"precise": 0}},
        {"capture": {"overlap": 0.8, "baseline": 1.0}, "outcomes": {"precise": 1}},
        {"capture": {"overlap": 0.9, "baseline": 2.0}, "outcomes": {"precise": 1}},
    ]
    features = (
        ("capture.overlap", "logit"),
        ("capture.baseline", "log1p"),
    )
    values, complete = design_matrix(rows, features)
    target = np.asarray([0.0, 0.0, 1.0, 1.0])

    first = fit_logistic(values, target, l2=0.1)
    second = fit_logistic(values, target, l2=0.1)
    probability = predict_logistic(values, first)

    assert np.all(complete)
    assert np.array_equal(first, second)
    assert probability[2] > probability[0]
    assert probability[3] > probability[1]
    assert "outcomes.precise" not in {path for path, _ in features}


def test_probability_design_marks_missing_features_without_imputation() -> None:
    values, complete = design_matrix(
        [
            {"capture": {"overlap": 0.5}},
            {"capture": {"overlap": None}},
        ],
        (("capture.overlap", "identity"),),
    )

    assert values.shape == (2, 1)
    assert complete.tolist() == [True, False]


def test_wilson_interval_matches_known_small_sample_reference() -> None:
    low, high = wilson_interval(8, 9)

    assert low == pytest.approx(0.5650002944)
    assert high == pytest.approx(0.9801091124)


def test_narrative_figure_inputs_preserve_frozen_order_and_scenario() -> None:
    replay = {
        "comparisons": [
            {"taxonomy_category": "catastrophic-accepted", "pair_id": "last"},
            {"taxonomy_category": "no-return-low-overlap", "pair_id": "first"},
            {"taxonomy_category": "supported-success", "pair_id": "middle"},
        ]
    }
    assert [row["pair_id"] for row in replay_rows(replay)] == [
        "first",
        "middle",
        "last",
    ]

    plan = {
        "power_grid": [
            {
                "true_precision": 0.97,
                "intraclass_correlation": 0.2,
                "independent_groups": 40,
            },
            {
                "true_precision": 0.97,
                "intraclass_correlation": 0.1,
                "independent_groups": 35,
            },
            {
                "true_precision": 0.97,
                "intraclass_correlation": 0.2,
                "independent_groups": 30,
            },
        ]
    }
    assert [
        row["independent_groups"]
        for row in prospective_curve(plan, true_precision=0.97, icc=0.2)
    ] == [30, 40]


def test_unified_replay_state_distinguishes_pose_and_acceptance_failures() -> None:
    assert _state({"returned": False}) == "no-pose"
    assert _state({"returned": True, "quality_accepted": False}) == (
        "returned-rejected"
    )
    assert (
        _state({"returned": True, "quality_accepted": True, "precise": False})
        == "imprecise-accepted"
    )
    assert (
        _state(
            {
                "returned": True,
                "quality_accepted": True,
                "precise": False,
                "catastrophic_accepted": True,
            }
        )
        == "catastrophic-accepted"
    )
    assert (
        _state({"returned": True, "quality_accepted": True, "precise": True})
        == "precise-accepted"
    )


def test_population_replay_order_interleaves_datasets_deterministically() -> None:
    rows = [
        {
            "dataset_id": dataset,
            "pair_id": f"{dataset}-{index}",
            "from_view_id": f"{dataset}-a-{index}",
            "to_view_id": f"{dataset}-b-{index}",
        }
        for dataset, count in (("large", 4), ("small", 2), ("tiny", 1))
        for index in range(count)
    ]
    first = _stable_interleave(rows, order_seed="fixed")
    second = _stable_interleave(list(reversed(rows)), order_seed="fixed")

    assert [(row["dataset_id"], row["pair_id"]) for row in first] == [
        (row["dataset_id"], row["pair_id"]) for row in second
    ]
    assert [row["dataset_id"] for row in first[:3]] == ["large", "small", "tiny"]
    assert _audit_pairs(rows) == {
        "duplicate_pair_ids": 0,
        "duplicate_or_reversed_pairs": 0,
    }
    with pytest.raises(ValueError, match="duplicate or reversed"):
        _audit_pairs([rows[0], {**rows[0], "pair_id": "another"}])


def test_environment_seal_hashes_installed_tree_deterministically(
    tmp_path: Path,
) -> None:
    package = tmp_path / "panorai"
    package.mkdir()
    (package / "__init__.py").write_text("__version__ = '3.5.0'\n", encoding="utf-8")
    (package / "kernel.so").write_bytes(b"native")
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"not evidence")

    first = _installed_tree(package)
    second = _installed_tree(package)

    assert first == second
    assert first["file_count"] == 2
    assert [record["path"] for record in first["files"]] == [
        "__init__.py",
        "kernel.so",
    ]


def test_environment_seal_requires_exact_replay_entrypoint_and_runner(
    tmp_path: Path,
) -> None:
    python = tmp_path / "venv" / "bin" / "python"
    runner = tmp_path / "runner.py"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    runner.write_text("", encoding="utf-8")
    runner_hash = hashlib.sha256(runner.read_bytes()).hexdigest()
    status = {
        "schema": "panorai-resumable-population-replay-status/v1",
        "state": "running",
        "expected_source_commit": "03c5b36",
        "python": {"requested_executable": str(python)},
        "runner": {"path": str(runner), "sha256": runner_hash},
    }

    validate_population_environment_status(
        status,
        python=python,
        runner=runner,
        expected_source_commit="03c5b36",
        expected_runner_sha256=runner_hash,
    )

    changed = {**status, "runner": {**status["runner"], "sha256": "wrong"}}
    with pytest.raises(ValueError, match="runner hash differs"):
        validate_population_environment_status(
            changed,
            python=python,
            runner=runner,
            expected_source_commit="03c5b36",
            expected_runner_sha256=runner_hash,
        )


def test_aligned_verifier_binds_environment_seal_to_analyzed_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replay_dir = tmp_path / "replay"
    results_dir = replay_dir / "results"
    results_dir.mkdir(parents=True)
    package_root = tmp_path / "venv" / "lib" / "site-packages" / "panorai"
    package_root.mkdir(parents=True)
    package_file = package_root / "__init__.py"
    package_file.write_text("release = '3.5.0'\n", encoding="utf-8")
    record = package_root.parent / "panorai-3.5.0.dist-info" / "RECORD"
    record.parent.mkdir()
    record.write_text("panorai/__init__.py\n", encoding="utf-8")
    wheel = tmp_path / "panorai-3.5.0.whl"
    runner = tmp_path / "runner.py"
    wheel.write_bytes(b"wheel")
    runner.write_bytes(b"runner")
    wheel_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
    runner_hash = hashlib.sha256(runner.read_bytes()).hexdigest()
    monkeypatch.setattr(aligned_verifier, "EXPECTED_WHEEL_SHA256", wheel_hash)
    monkeypatch.setattr(aligned_verifier, "EXPECTED_RUNNER_SHA256", runner_hash)
    tree = aligned_verifier._installed_tree_identity(package_root)
    source_commit = "03c5b36"
    seal = {
        "schema": aligned_verifier.ENVIRONMENT_SEAL_SCHEMA,
        "release": {
            "version": "3.5.0",
            "tag": aligned_verifier.EXPECTED_RELEASE_TAG,
            "tag_object": aligned_verifier.EXPECTED_RELEASE_TAG_OBJECT,
            "source_commit": source_commit,
            "source_tree": aligned_verifier.EXPECTED_SOURCE_TREE,
        },
        "wheel": {"path": str(wheel), "sha256": wheel_hash},
        "runner": {"path": str(runner), "sha256": runner_hash},
        "probe": {
            "version": "3.5.0",
            "package_file": str(package_file),
            "python_executable": str(tmp_path / "venv" / "bin" / "python"),
            "opencv_threads": 1,
            "native_filter_available": True,
            "native_pose_kernels_available": True,
        },
        "hardware": {
            "cpu_model": "test CPU",
            "physical_cpu_count": 4,
            "logical_cpu_count": 8,
            "memory_bytes": 16 * 1024**3,
        },
        "installed_package_tree": {"root": str(package_root), **tree},
        "distribution_record": {
            "path": str(record),
            "sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
        },
        "replay_status": {
            "path": str(replay_dir / "status.json"),
            "total": 2385,
            "completed_at_seal": 463,
            "failures_this_run": 0,
            "python": {
                "requested_executable": str(
                    tmp_path / "venv" / "bin" / "python"
                )
            },
            "runner": {"sha256": runner_hash},
        },
        "scope": "Environment identity only; contains no aggregate pose accuracy.",
    }

    assert not aligned_verifier._environment_seal_violations(
        seal,
        expected_package_version="3.5.0",
        expected_source_commit=source_commit,
        expected_results_dir=results_dir,
    )

    package_file.write_text("changed = True\n", encoding="utf-8")
    violations = aligned_verifier._environment_seal_violations(
        seal,
        expected_package_version="3.5.0",
        expected_source_commit=source_commit,
        expected_results_dir=results_dir,
    )
    assert {violation["reason"] for violation in violations} == {
        "installed package tree changed after sealing"
    }


def test_resumable_replay_accepts_only_complete_native_route_result(
    tmp_path: Path,
) -> None:
    row = {"dataset_id": "dataset", "pair_id": "pair"}
    path = tmp_path / f"{_slug(row)}.json"
    result = {
        "schema": "panorai-unified-optimized-pair/v2",
        **row,
        "package": {
            "version": "3.5.0",
            "expected_source_commit": "abc123",
            "import_path": "/external/site-packages/panorai/__init__.py",
        },
        "resolution_hw": [1024, 2048],
        "native": {
            "convolution_backend": "native",
            "native_filter_available": True,
            "native_pose_kernels_available": True,
            "numpy_fallback_permitted": False,
        },
        "route": optimized_pair_profile_configuration(),
        "validity": {"derived_from_black_pixels": False},
        "system": {"opencv_threads": 16, "patch_workers": 4},
        "matching_diagnostics": {},
        "pose": {"returned": False},
    }
    path.write_text(json.dumps(result), encoding="utf-8")
    assert _load_valid_result(path, row) == result

    result["package"]["version"] = "3.4.1"
    path.write_text(json.dumps(result), encoding="utf-8")
    assert _load_valid_result(path, row) is None
    result["package"]["version"] = "3.5.0"
    path.write_text(json.dumps(result), encoding="utf-8")

    assert (
        _load_valid_result(path, row, expected_source_commit="abc123") == result
    )
    assert _load_valid_result(path, row, expected_source_commit="different") is None

    result["route"]["detector"]["max_keypoints"] = 1024
    path.write_text(json.dumps(result), encoding="utf-8")
    assert _load_valid_result(path, row) is None
    result["route"]["detector"]["max_keypoints"] = 4096

    result["native"]["convolution_backend"] = "numpy"
    path.write_text(json.dumps(result), encoding="utf-8")
    assert _load_valid_result(path, row) is None


def test_independent_verifier_requires_complete_serialized_route() -> None:
    assert EXPECTED_ROUTE_CONFIGURATION == optimized_pair_profile_configuration()
    result = {
        "schema": "panorai-unified-optimized-pair/v2",
        "package": {
            "version": "3.5.0",
            "expected_source_commit": "03c5b36",
            "import_path": "/external/site-packages/panorai/__init__.py",
        },
        "resolution_hw": [1024, 2048],
        "native": {
            "convolution_backend": "native",
            "native_filter_available": True,
            "native_pose_kernels_available": True,
            "numpy_fallback_permitted": False,
        },
        "route": json.loads(json.dumps(EXPECTED_ROUTE_CONFIGURATION)),
        "validity": {"derived_from_black_pixels": False},
        "system": {"opencv_threads": 1, "patch_workers": 4},
        "matching_diagnostics": {},
        "pose": {"returned": False},
    }
    arguments = {
        "expected_sha256": "a" * 64,
        "actual_sha256": "a" * 64,
        "expected_package_version": "3.5.0",
        "expected_source_commit": "03c5b36",
        "expected_opencv_threads": 1,
    }

    assert not _raw_route_violations(result, **arguments)

    result["route"]["detector"]["max_keypoints"] = 1024
    assert _raw_route_violations(result, **arguments) == [
        "serialized route configuration differs"
    ]


def test_aligned_table_rejects_wrong_package_or_source_commit(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    result = {
        "package": {"version": "3.5.0", "expected_source_commit": "abc123"}
    }

    assert validate_aligned_package(
        result,
        path=path,
        expected_package_version="3.5.0",
        expected_source_commit="abc123",
    ) == result["package"]
    with pytest.raises(ValueError, match="unexpected PanorAi version"):
        validate_aligned_package(
            result,
            path=path,
            expected_package_version="3.5.1",
            expected_source_commit="abc123",
        )
    with pytest.raises(ValueError, match="unexpected source commit"):
        validate_aligned_package(
            result,
            path=path,
            expected_package_version="3.5.0",
            expected_source_commit="different",
        )


def test_aligned_table_preserves_translation_orientation_diagnostics() -> None:
    result = {
        "counts": {"matches": 40, "keypoints_a": 100, "keypoints_b": 80},
        "matching_diagnostics": {
            "descriptor_distance_median": 0.2,
            "descriptor_distance_p90": 0.4,
            "ratio_score_median": 0.6,
        },
        "validity": {"valid_fraction_a": 1.0, "valid_fraction_b": 0.8},
        "timings_seconds": {
            "detection_pair": 1.0,
            "patches_pair": 2.0,
            "descriptor_pair": 3.0,
            "matching": 0.1,
            "pose": 0.2,
            "pair_total": 6.3,
        },
        "peak_rss_mib": 500.0,
        "pose": {
            "returned": True,
            "quality_accepted": True,
            "rotation_error_deg": 0.5,
            "translation_direction_error_deg": 2.0,
            "primary": True,
            "strict": True,
            "precise": True,
            "catastrophic_accepted": False,
            "inlier_count": 20,
            "quality_report": {
                "inlier_ratio": 0.5,
                "raw_quality_score": 0.8,
                "median_parallax_deg": 2.0,
                "cheirality_ratio": 0.9,
                "median_residual_deg": 0.1,
                "p90_residual_deg": 0.2,
                "coverage_entropy_a": 0.7,
                "coverage_entropy_b": 0.6,
                "occupied_cells_a": 8,
                "occupied_cells_b": 7,
                "stability": {
                    "requested_trials": 6,
                    "successful_trials": 6,
                    "rotation_p90_deg": 0.4,
                    "translation_p90_deg": 3.0,
                },
                "model_competition": {
                    "essential_score_margin": 0.2,
                    "preferred_model": "essential",
                },
                "translation_orientation": {
                    "cheirality_margin": 0.15,
                    "weighted_cheirality_margin": 0.12,
                    "median_triangulation_angle_deg": 1.5,
                    "ambiguous": False,
                },
            },
        },
    }

    post = aligned_post(result)
    outcomes = aligned_outcomes(result)

    assert post["translation_orientation_cheirality_margin"] == 0.15
    assert post["translation_orientation_weighted_margin"] == 0.12
    assert post["stability_success_fraction"] == 1.0
    assert post["keypoint_count_min"] == 80
    assert outcomes["precise"] is True
    assert outcomes["usable"] is True


def test_aligned_model_profile_adds_frozen_orientation_model_only() -> None:
    historical = {model["model_id"] for model in models_for_profile("historical")}
    aligned = {model["model_id"] for model in models_for_profile("aligned")}

    assert "post-precise-aligned-orientation" not in historical
    assert aligned == historical | {"post-precise-aligned-orientation"}
    with pytest.raises(ValueError, match="unknown model profile"):
        models_for_profile("future-unfrozen-profile")


def test_selective_rule_uses_the_post_model_frozen_in_the_rule() -> None:
    key = ("dataset", "pair")
    feature = {
        "capture": {"registered_cloud_overlap_min": 0.8, "baseline_m": 1.0}
    }
    outcome = {"accepted": True}
    predictions = {
        ("capture-accept-overlap-baseline", *key): {"probability": 0.9},
        ("capture-precise-given-accept-overlap-baseline", *key): {
            "probability": 0.9
        },
        ("post-precise-raw-score", *key): {"probability": 0.1},
        ("post-precise-aligned-orientation", *key): {"probability": 0.9},
    }

    common = {
        "baseline_range": (0.5, 1.5),
        "capture_threshold": 0.8,
        "post_threshold": 0.8,
    }
    assert not selective_rule_selected(
        key,
        outcome,
        feature,
        predictions,
        post_model="post-precise-raw-score",
        **common,
    )
    assert selective_rule_selected(
        key,
        outcome,
        feature,
        predictions,
        post_model="post-precise-aligned-orientation",
        **common,
    )


@pytest.mark.parametrize(
    "verdict", ["NO_GO", "GO_FOR_PROSPECTIVE_CONFIRMATION"]
)
def test_prospective_plan_accepts_only_nonrelease_retrospective_verdicts(
    verdict: str,
) -> None:
    assert _retrospective_verdict({"verdict": verdict}) == verdict


def test_prospective_plan_rejects_release_or_unknown_verdict() -> None:
    with pytest.raises(ValueError, match="unsupported retrospective verdict"):
        _retrospective_verdict({"verdict": "RELEASE"})


def test_aligned_analysis_stage_order_preserves_outcome_sealing() -> None:
    assert ANALYSIS_STAGE_ORDER.index("fit-models") < ANALYSIS_STAGE_ORDER.index(
        "evaluate-models"
    )
    assert ANALYSIS_STAGE_ORDER.index(
        "fit-lodo-models"
    ) < ANALYSIS_STAGE_ORDER.index("evaluate-lodo-models")
    assert ANALYSIS_STAGE_ORDER.index(
        "freeze-selective-rule"
    ) < ANALYSIS_STAGE_ORDER.index("evaluate-selective-rule")
    assert ANALYSIS_STAGE_ORDER.index(
        "render-paper-results"
    ) < ANALYSIS_STAGE_ORDER.index("write-paper-document")


def test_aligned_analysis_requires_a_new_or_empty_output_directory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "analysis"
    assert _prepare_output_dir(output) == output.resolve()
    (output / "partial.txt").write_text("do not mix runs", encoding="utf-8")
    with pytest.raises(ValueError, match="absent or empty"):
        _prepare_output_dir(output)


def test_aligned_analysis_rejects_code_drift_before_creating_output(
    tmp_path: Path,
) -> None:
    code_lock = tmp_path / "analysis-code-lock.json"
    records = aligned_analysis._analysis_code()
    files = {
        name: {"filename": record["filename"], "sha256": record["sha256"]}
        for name, record in records.items()
    }
    files["runner"]["sha256"] = "0" * 64
    code_lock.write_text(
        json.dumps({"schema": aligned_analysis.CODE_LOCK_SCHEMA, "files": files}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="code hash differs for runner"):
        aligned_analysis._validate_analysis_code_lock(code_lock)


def _install_aligned_analysis_fakes(
    monkeypatch: pytest.MonkeyPatch, *, qualifying_rule: bool
) -> None:
    def write(path: Path, content: str = "{}\n") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def build(args: argparse.Namespace) -> None:
        for name in (
            "features.jsonl",
            "outcomes-development-calibration.jsonl",
            "outcomes-evaluation.jsonl",
            "analysis-table.jsonl",
        ):
            write(args.output_dir / name, "")
        write(args.output_dir / "manifest.json")

    def fit(args: argparse.Namespace) -> None:
        assert args.model_profile == "aligned"
        write(args.output_dir / "predictions.jsonl", "")
        write(args.output_dir / "model-card.json")

    def evaluate(args: argparse.Namespace) -> None:
        write(args.output_dir / "evaluation.json")

    def freeze(args: argparse.Namespace) -> None:
        assert args.post_model == "post-precise-aligned-orientation"
        if qualifying_rule:
            write(args.output)
            return
        write(args.output.with_name("release-rule-search.json"))
        raise aligned_analysis.NoQualifyingRuleError("no rule")

    def evaluate_rule(args: argparse.Namespace) -> None:
        write(
            args.output_dir / "release-rule-evaluation.json",
            '{"verdict":"NO_GO"}\n',
        )

    def plan(args: argparse.Namespace) -> None:
        write(args.output_dir / "prospective-power-plan.json")

    def render(args: argparse.Namespace) -> None:
        assert (args.release_rule is None) is (not qualifying_rule)
        assert (args.release_evaluation is None) is (not qualifying_rule)
        write(args.output_dir / "paper-results.json")

    def write_document(args: argparse.Namespace) -> None:
        write(args.output, "# sealed paper results\n")

    monkeypatch.setattr(aligned_analysis, "build_aligned_table", build)
    monkeypatch.setattr(aligned_analysis, "fit_predict", fit)
    monkeypatch.setattr(aligned_analysis, "fit_predict_lodo", fit)
    monkeypatch.setattr(aligned_analysis, "evaluate_models", evaluate)
    monkeypatch.setattr(aligned_analysis, "freeze_rule", freeze)
    monkeypatch.setattr(aligned_analysis, "evaluate_rule", evaluate_rule)
    monkeypatch.setattr(aligned_analysis, "plan_prospective_confirmation", plan)
    monkeypatch.setattr(aligned_analysis, "render_paper_results", render)
    monkeypatch.setattr(aligned_analysis, "write_paper_document", write_document)


@pytest.mark.parametrize("qualifying_rule", [True, False])
def test_aligned_analysis_orchestrates_both_rule_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qualifying_rule: bool,
) -> None:
    _install_aligned_analysis_fakes(monkeypatch, qualifying_rule=qualifying_rule)
    base = tmp_path / "base.jsonl"
    base.write_text("{}\n", encoding="utf-8")
    census = tmp_path / "census.json"
    census.write_text("{}\n", encoding="utf-8")
    results = tmp_path / "results"
    results.mkdir()
    output = tmp_path / "output"
    code_lock = tmp_path / "analysis-code-lock.json"
    code_lock.write_text(
        json.dumps(
            {
                "schema": aligned_analysis.CODE_LOCK_SCHEMA,
                "files": {
                    name: {
                        "filename": record["filename"],
                        "sha256": record["sha256"],
                    }
                    for name, record in aligned_analysis._analysis_code().items()
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = aligned_analysis.run(
        argparse.Namespace(
            base_analysis_table=base,
            census=census,
            analysis_code_lock=code_lock,
            results_dir=results,
            output_dir=output,
            expected_package_version="3.5.0",
            expected_source_commit="source-commit",
        )
    )

    assert manifest["rule_qualified_on_calibration"] is qualifying_rule
    assert manifest["model_profile"] == "aligned"
    assert manifest["expected_package_version"] == "3.5.0"
    assert manifest["artifacts"]["paper_results"]["sha256"]
    assert set(manifest["analysis_code"]) == {
        "runner",
        "table_builder",
        "probability_models",
        "selective_rule",
        "prospective_plan",
        "paper_figures",
        "paper_document",
    }
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "complete"
    if qualifying_rule:
        assert "release_rule" in manifest["artifacts"]
        assert "prospective_plan" in manifest["artifacts"]
        assert "plan-prospective-confirmation" in status["completed_stages"]
    else:
        assert "release_rule_search" in manifest["artifacts"]
        assert "release_rule" not in manifest["artifacts"]
        assert status["scientific_result"] == "no calibration-grid rule qualified"
        assert "evaluate-selective-rule" not in status["completed_stages"]


def test_paper_renderer_prefers_aligned_post_model_with_historical_fallback() -> None:
    historical = {"models": {"post-precise-common": {}}}
    aligned = {
        "models": {
            "post-precise-common": {},
            "post-precise-aligned-orientation": {},
        }
    }

    assert primary_post_model(historical) == "post-precise-common"
    assert primary_post_model(aligned) == "post-precise-aligned-orientation"
    with pytest.raises(ValueError, match="no supported primary post model"):
        primary_post_model({"models": {}})


def test_capture_probability_surface_masks_cells_without_group_support() -> None:
    cards = []
    for model_id in (
        "capture-accept-overlap-baseline",
        "capture-precise-given-accept-overlap-baseline",
    ):
        cards.append(
            {
                "model_id": model_id,
                "features": [
                    {
                        "path": "capture.registered_cloud_overlap_min",
                        "transform": "logit",
                    },
                    {"path": "capture.baseline_m", "transform": "log1p"},
                ],
                "scaler_mean": [0.0, 0.0],
                "scaler_scale": [1.0, 1.0],
                "parameters": [0.0, 0.0, 0.0],
                "platt_parameters": [0.0, 1.0],
            }
        )
    rows = [
        {
            "split": "development",
            "independence_component_id": f"group-{index}",
            "capture": {
                "registered_cloud_overlap_min": 0.8,
                "baseline_m": 1.0,
            },
            "outcomes": {"accepted": True},
        }
        for index in range(8)
    ]

    surface = capture_probability_surface(rows, {"models": cards})
    supported = [row for row in surface if row["supported"]]
    unsupported = [row for row in surface if not row["supported"]]

    assert supported
    assert all(row["support_components"] >= 5 for row in supported)
    assert all(row["p_accept"] == pytest.approx(0.5) for row in supported)
    assert all(row["p_usable"] == pytest.approx(0.25) for row in supported)
    assert unsupported
    assert all(row["p_usable"] is None for row in unsupported)


def test_runtime_overlap_and_engineering_summaries_are_pair_level() -> None:
    rows = []
    for dataset_index, dataset in enumerate(
        ("matterport360", "stanford2d3d", "p74_native_polar")
    ):
        for pair_index, seconds in enumerate((8.0, 12.0)):
            rows.append(
                {
                    "dataset_id": dataset,
                    "independence_component_id": f"{dataset}-g-{pair_index}",
                    "capture": {"registered_cloud_overlap_min": 0.6},
                    "post": {
                        "detection_pair_seconds": seconds * 0.4,
                        "patches_pair_seconds": seconds * 0.2,
                        "descriptor_pair_seconds": seconds * 0.2,
                        "matching_seconds": seconds * 0.1,
                        "pose_seconds": seconds * 0.1,
                        "pair_total_seconds": seconds + dataset_index,
                        "peak_rss_mib": 500.0 + pair_index,
                        "keypoint_count_min": 1000 + pair_index,
                    },
                }
            )

    runtime = summarize_runtime_overlap(rows)
    engineering = summarize_engineering(rows)
    matterport_50_70 = next(
        row
        for row in runtime
        if row["dataset_id"] == "matterport360" and row["overlap_bin"] == "50–70%"
    )

    assert matterport_50_70["pairs"] == 2
    assert matterport_50_70["pair_total_median_seconds"] == pytest.approx(10.0)
    assert matterport_50_70["independence_components"] == 2
    assert engineering["datasets"]["p74_native_polar"]["pairs"] == 2
    assert engineering["resolution"] == "1024x2048"


def test_paper_document_preserves_no_qualifying_rule_as_scientific_result() -> None:
    verdict, body = paper_release_section(
        None,
        None,
        {"candidate_grid": [{"qualifies": False}, {"qualifies": False}]},
    )

    assert verdict == "NO_QUALIFYING_CALIBRATION_RULE"
    assert "2 candidates" in body
    assert "evaluation outcomes were not used" in body


def test_aligned_analysis_verifier_requires_every_check_to_pass() -> None:
    audit = AlignedAnalysisAudit()
    audit.check("first", True, {"value": 1})
    assert audit.passed
    audit.check("second", False, {"value": 2})
    assert not audit.passed
    assert [check["name"] for check in audit.checks] == ["first", "second"]


def test_aligned_analysis_verifier_rejects_duplicate_identity() -> None:
    rows = [
        {"dataset_id": "d", "pair_id": "p"},
        {"dataset_id": "d", "pair_id": "p"},
    ]
    with pytest.raises(ValueError, match="duplicate fixture"):
        aligned_verification_index(rows, ("dataset_id", "pair_id"), "fixture")


def test_capture_predictor_policy_is_operator_visible_and_profile_explicit() -> None:
    good = [
        {
            "model_id": "capture-accept-overlap-baseline",
            "features": [
                {
                    "path": "capture.registered_cloud_overlap_min",
                    "transform": "logit",
                },
                {"path": "capture.baseline_m", "transform": "log1p"},
            ],
        }
    ]
    assert _capture_policy_violations(good, card_label="test") == []

    bad = [
        {
            "model_id": "capture-leaky",
            "features": [
                {"path": "post.inlier_ratio", "transform": "logit"},
                {"path": "reference.rotation_error_deg", "transform": "log1p"},
            ],
        }
    ]
    violations = _capture_policy_violations(bad, card_label="test")
    assert {row["path"] for row in violations} == {
        "post.inlier_ratio",
        "reference.rotation_error_deg",
    }


def test_probability_model_contracts_preserve_conditioning_population() -> None:
    good = [
        {
            "model_id": "capture-precise-given-accept-overlap-baseline",
            "target": "precise",
            "population": "accepted",
        },
        {
            "model_id": "post-precise-aligned-orientation",
            "target": "precise",
            "population": "returned",
        },
    ]
    assert _model_contract_violations(good, card_label="test") == []

    bad = [
        {
            "model_id": "capture-precise-given-accept-overlap-baseline",
            "target": "precise",
            "population": "eligible",
        }
    ]
    violations = _model_contract_violations(bad, card_label="test")
    assert [row["reason"] for row in violations] == [
        "target/population contract mismatch"
    ]


def test_paper_scope_audit_requires_two_view_and_deployment_caveats() -> None:
    document = "\n".join(PAPER_SCOPE_MARKERS)
    assert _missing_paper_scope_markers(document) == []
    missing = _missing_paper_scope_markers(
        document.replace("No result in this document validates dense stereo", "")
    )
    assert missing == ["No result in this document validates dense stereo"]


def test_paper_census_audit_requires_images_pairs_and_groups() -> None:
    totals = {
        "unique_images": 4017,
        "unique_pairs": 2385,
        "independence_components": 69,
    }
    document = "Total: 4,017 unique images, 2,385 unordered pairs, and 69 independence components."
    assert _missing_census_markers(document, totals) == []
    assert _missing_census_markers(document.replace("4,017", "4,016"), totals) == [
        "4,017 unique images"
    ]


def test_group_uncertainty_audit_requires_bootstrap_when_supported() -> None:
    metrics = {
        "brier": 0.1,
        "log_loss": 0.3,
        "ece_10": 0.05,
        "count": 10,
        "independence_components": 5,
        "reliability_bins": [
            {
                "count": 10 if index == 0 else 0,
                "independence_components": 5 if index == 0 else 0,
            }
            for index in range(10)
        ],
        "component_bootstrap": {
            "unit": "independence_component",
            "component_count": 5,
            "repetitions": 10_000,
            "seed": 7,
            "percentile_95": {
                "brier": [0.08, 0.12],
                "log_loss": [0.25, 0.36],
            },
        },
    }
    evaluation = {"models": {"model": {"datasets": {"dataset": metrics}}}}
    assert _uncertainty_violations(evaluation, evaluation_label="test") == []

    metrics["component_bootstrap"] = None
    violations = _uncertainty_violations(evaluation, evaluation_label="test")
    assert [row["reason"] for row in violations] == [
        "missing/invalid component bootstrap"
    ]

    metrics["independence_components"] = 1
    metrics["reliability_bins"][0]["independence_components"] = 1
    assert _uncertainty_violations(evaluation, evaluation_label="test") == []


def test_probability_metric_audit_recomputes_scores_and_reliability() -> None:
    samples = [
        (1.0, 0.8, "group-a"),
        (0.0, 0.2, "group-b"),
        (1.0, 0.6, "group-c"),
        (0.0, 0.1, "group-d"),
    ]
    metrics = _independent_probability_metrics(samples)

    assert metrics["count"] == 4
    assert metrics["positives"] == 2
    assert metrics["independence_components"] == 4
    assert metrics["brier"] == pytest.approx(0.0625)
    assert np.isfinite(metrics["calibration_intercept"])
    assert np.isfinite(metrics["calibration_slope"])
    assert sum(row["count"] for row in metrics["reliability_bins"]) == 4
    assert _metric_mismatches(metrics, metrics, context={}) == []

    tampered = dict(metrics)
    tampered["brier"] = 0.5
    mismatches = _metric_mismatches(tampered, metrics, context={})
    assert [row["field"] for row in mismatches] == ["brier"]

    tampered = dict(metrics)
    tampered["calibration_slope"] = 99.0
    mismatches = _metric_mismatches(tampered, metrics, context={})
    assert [row["field"] for row in mismatches] == ["calibration_slope"]

    bootstrap_samples = [
        (float(index % 2), 0.8 if index % 2 else 0.2, f"group-{index % 5}")
        for index in range(20)
    ]
    bootstrap_metrics = _independent_probability_metrics(
        bootstrap_samples, bootstrap_seed=12345
    )
    assert bootstrap_metrics["component_bootstrap"] == {
        "unit": "independence_component",
        "component_count": 5,
        "repetitions": 10_000,
        "seed": 12345,
        "percentile_95": {
            "brier": pytest.approx([0.04, 0.04]),
            "log_loss": pytest.approx(
                [0.2231435513142097, 0.2231435513142097]
            ),
        },
    }
    tampered = json.loads(json.dumps(bootstrap_metrics))
    tampered["component_bootstrap"]["seed"] = 1
    mismatches = _metric_mismatches(tampered, bootstrap_metrics, context={})
    assert [row["field"] for row in mismatches] == ["component_bootstrap.seed"]

    baseline_metrics = _independent_probability_metrics(
        [(target, 0.5, group) for target, _probability, group in samples]
    )
    reported = {
        **metrics,
        "constant_calibration_prevalence": baseline_metrics,
        "brier_delta_vs_constant": metrics["brier"] - baseline_metrics["brier"],
        "log_loss_delta_vs_constant": (
            metrics["log_loss"] - baseline_metrics["log_loss"]
        ),
    }
    assert (
        _constant_baseline_mismatches(
            reported, metrics, baseline_metrics, context={}
        )
        == []
    )
    reported["brier_delta_vs_constant"] = 1.0
    mismatches = _constant_baseline_mismatches(
        reported, metrics, baseline_metrics, context={}
    )
    assert [row["field"] for row in mismatches] == ["brier_delta_vs_constant"]


def test_selective_rule_recomputation_uses_calibration_only() -> None:
    features = []
    outcomes = []
    predictions = []
    for index in range(30):
        dataset = "matterport360"
        pair_id = f"pair-{index:02d}"
        group = f"group-{index % 5}"
        features.append(
            {
                "dataset_id": dataset,
                "pair_id": pair_id,
                "capture": {
                    "registered_cloud_overlap_min": 0.8,
                    "baseline_m": 1.0,
                },
            }
        )
        outcomes.append(
            {
                "dataset_id": dataset,
                "pair_id": pair_id,
                "split": "calibration",
                "independence_component_id": group,
                "accepted": True,
                "precise": True,
                "catastrophic_accepted": False,
            }
        )
        for model_id in (
            "capture-accept-overlap-baseline",
            "capture-precise-given-accept-overlap-baseline",
            "post-precise-aligned-orientation",
        ):
            predictions.append(
                {
                    "model_id": model_id,
                    "dataset_id": dataset,
                    "pair_id": pair_id,
                    "probability": 0.99,
                }
            )

    recomputed = _recompute_calibration_rule(
        features,
        predictions,
        outcomes,
        post_model="post-precise-aligned-orientation",
    )

    assert recomputed["supported_domains"] == ["matterport360"]
    assert recomputed["chosen"] is not None
    assert recomputed["chosen"]["selected_pairs"] == 30
    assert recomputed["chosen"]["capture_usable_probability_threshold"] == 0.5
    artifact = {
        "candidate_grid": recomputed["candidate_grid"],
        "models": {"post_precision": recomputed["post_model"]},
        "supported_domains_at_freeze": recomputed["supported_domains"],
        "calibration_components_in_envelope_by_dataset": recomputed[
            "domain_components"
        ],
        "capture_envelope": {
            "baseline_m_closed_interval": recomputed["baseline_range"],
            "baseline_support_pairs_development_and_calibration": recomputed[
                "baseline_support_pairs"
            ],
        },
        "chosen_calibration_result": recomputed["chosen"],
    }
    assert (
        _rule_recomputation_violations(artifact, recomputed, qualified=True) == []
    )
    rule = {
        **artifact,
        "models": {
            "capture_acceptance": "capture-accept-overlap-baseline",
            "capture_conditional_precision": (
                "capture-precise-given-accept-overlap-baseline"
            ),
            "post_precision": recomputed["post_model"],
        },
        "thresholds": {
            "capture_usable_probability": 0.5,
            "post_precision_probability": 0.5,
        },
    }
    evaluation_outcomes = [{**row, "split": "evaluation"} for row in outcomes]
    evaluation = _recompute_rule_evaluation(
        rule, features, predictions, evaluation_outcomes
    )
    assert evaluation["verdict"] == "GO_FOR_PROSPECTIVE_CONFIRMATION"
    reported_evaluation = {
        "verdict": evaluation["verdict"],
        "datasets": evaluation["datasets"],
        "pooled_descriptive_only": evaluation["pooled_descriptive_only"],
    }
    assert (
        _release_evaluation_violations(reported_evaluation, evaluation) == []
    )
    outcomes[0]["split"] = "evaluation"
    with pytest.raises(ValueError, match="evaluation outcomes"):
        _recompute_calibration_rule(
            features,
            predictions,
            outcomes,
            post_model="post-precise-aligned-orientation",
        )


def test_paper_figure_bundle_seals_only_expected_aligned_pngs(
    tmp_path: Path,
) -> None:
    analysis_dir = tmp_path / "analysis"
    paper_dir = analysis_dir / "paper-results"
    paper_dir.mkdir(parents=True)
    figure_names = {
        "overlap_response",
        "calibration",
        "post_ablation",
        "runtime_overlap_response",
        "capture_probability_surface",
        "cross_dataset_transfer",
        "selective_rule",
    }
    figures = {}
    for name in sorted(figure_names):
        path = paper_dir / f"{name}.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\nsealed-test-payload")
        figures[name] = str(path)

    records, violations = _paper_figure_bundle(
        {"figures": figures}, analysis_dir, rule_qualified=True
    )

    assert violations == []
    assert set(records) == figure_names
    assert all(len(record["sha256"]) == 64 for record in records.values())

    outside = tmp_path / "outside.png"
    outside.write_bytes(b"\x89PNG\r\n\x1a\nexternal")
    figures["overlap_response"] = str(outside)
    _, violations = _paper_figure_bundle(
        {"figures": figures}, analysis_dir, rule_qualified=True
    )
    assert [row["reason"] for row in violations] == [
        "paper figure is outside aligned paper-results"
    ]

    figures["overlap_response"] = str(paper_dir / "overlap_response.png")
    del figures["selective_rule"]
    _, violations = _paper_figure_bundle(
        {"figures": figures}, analysis_dir, rule_qualified=True
    )
    assert [row["reason"] for row in violations] == ["missing paper figures"]
    records, violations = _paper_figure_bundle(
        {"figures": figures}, analysis_dir, rule_qualified=False
    )
    assert violations == []
    assert set(records) == figure_names - {"selective_rule"}


def test_outcome_consistency_recomputes_pose_thresholds() -> None:
    outcomes = [
        {
            "dataset_id": "matterport360",
            "pair_id": "precise",
            "returned": True,
            "accepted": True,
            "primary": True,
            "strict": True,
            "precise": True,
            "usable": True,
            "catastrophic_accepted": False,
            "rotation_error_deg": 0.5,
            "translation_direction_error_deg": 4.0,
        },
        {
            "dataset_id": "p74_native_polar",
            "pair_id": "no-pose",
            "returned": False,
            "accepted": False,
            "primary": False,
            "strict": False,
            "precise": False,
            "usable": False,
            "catastrophic_accepted": False,
            "rotation_error_deg": None,
            "translation_direction_error_deg": None,
        },
    ]

    assert _outcome_consistency_violations(outcomes) == []

    outcomes[0]["precise"] = False
    outcomes[1]["accepted"] = True
    violations = _outcome_consistency_violations(outcomes)
    assert {(row["pair_id"], row["reason"]) for row in violations} == {
        ("precise", "derived outcome mismatch"),
        ("no-pose", "derived outcome mismatch"),
        ("no-pose", "accepted pose was not returned"),
    }


def test_prediction_feature_leakage_audit_is_recursive() -> None:
    clean = {
        "dataset_id": "matterport360",
        "pair_id": "clean",
        "capture": {"registered_cloud_overlap_min": 0.8},
        "post": {"inlier_ratio": 0.7},
    }
    leaked = {
        **clean,
        "pair_id": "leaked",
        "post": {
            "diagnostics": [
                {"rotation_error_deg": 0.1},
                {"precise": True},
            ]
        },
    }

    assert _forbidden_feature_paths([clean]) == []
    violations = _forbidden_feature_paths([clean, leaked])
    assert {row["path"] for row in violations} == {
        "post.diagnostics.0.rotation_error_deg",
        "post.diagnostics.1.precise",
    }
    assert all(row["pair_id"] == "leaked" for row in violations)


def test_discrimination_summary_uses_target_and_population_contracts() -> None:
    assert _roc_auc([0.0, 1.0, 0.0, 1.0], [0.1, 0.9, 0.4, 0.8]) == 1.0
    assert _roc_auc([0.0, 1.0], [0.5, 0.5]) == 0.5
    assert _roc_auc([1.0, 1.0], [0.2, 0.8]) is None

    outcomes = [
        {
            "dataset_id": "matterport360",
            "pair_id": f"pair-{index}",
            "returned": True,
            "accepted": bool(index % 2),
            "precise": bool(index % 2),
        }
        for index in range(4)
    ]
    predictions = [
        {
            "model_id": "capture-accept-overlap-baseline",
            "dataset_id": "matterport360",
            "pair_id": f"pair-{index}",
            "split": "evaluation",
            "target": "accepted",
            "population": "eligible",
            "probability": probability,
        }
        for index, probability in enumerate((0.1, 0.9, 0.2, 0.8))
    ]

    summary = _discrimination_summary(predictions, outcomes)
    metrics = summary["models"]["capture-accept-overlap-baseline"]
    assert metrics["target"] == "accepted"
    assert metrics["population"] == "eligible"
    assert metrics["datasets"]["matterport360"] == {
        "count": 4,
        "positives": 2,
        "negatives": 2,
        "roc_auc": 1.0,
    }
    assert metrics["macro_roc_auc_supported_datasets"] == 1.0


def test_conceptual_figure_manifest_seals_non_evidence_assets() -> None:
    figure_dir = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "two_view_pose_probability"
        / "figures"
    )
    manifest = json.loads(
        (figure_dir / "conceptual-figures.json").read_text(encoding="utf-8")
    )

    assert manifest["schema"] == "panorai-two-view-pose-conceptual-figures/v1"
    assert set(manifest["figures"]) == {
        "capture_boundary_conditions",
        "graphical_abstract_two_view",
        "post_processing_evidence",
    }
    for record in manifest["figures"].values():
        path = figure_dir / record["filename"]
        payload = path.read_bytes()
        assert payload[:8] == b"\x89PNG\r\n\x1a\n"
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]
        assert int.from_bytes(payload[16:20], "big") == record["width_px"]
        assert int.from_bytes(payload[20:24], "big") == record["height_px"]
        assert "not experimental evidence" in record["role"]


def test_aligned_quantitative_figure_manifest_seals_verified_evidence() -> None:
    figure_dir = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "two_view_pose_probability"
        / "figures"
    )
    manifest = json.loads(
        (figure_dir / "aligned-quantitative-figures.json").read_text(
            encoding="utf-8"
        )
    )

    assert (
        manifest["schema"]
        == "panorai-two-view-pose-aligned-quantitative-figures/v1"
    )
    assert manifest["panorai"] == {
        "source_commit": "03c5b36b28225b24d3909286bf53250d7b532aa3",
        "version": "3.5.0",
    }
    assert manifest["status"].endswith("operational verdict NO-GO")
    assert manifest["controlled_timing"] == {
        "host_gate_sha256": (
            "6d4d9b0f379194817f015ba8474d7667a4a0daac44f41ef5d856ac379265bcf3"
        ),
        "observations_sha256": (
            "648454223db730e9adf7c2f3a343d6878dead7948c1177ec1c60eda6f0a9334b"
        ),
        "raw_observations": 225,
        "repetitions": 3,
        "run_status_sha256": (
            "001c8a9800afea7d671d7da903264c5ccd42ddcceab11e75f847588f8df37adc"
        ),
        "summary_sha256": (
            "80aaeabe9e8da9d78003af38482a514655e0aa057bc33874b0c2f49e8d4b97ee"
        ),
        "unique_pairs": 75,
    }
    assert set(manifest["figures"]) == {
        "calibration_heldout",
        "capture_probability_surface",
        "controlled_runtime_overlap_response",
        "cross_dataset_transfer",
        "overlap_response",
        "post_ablation",
        "runtime_overlap_response",
        "selective_rule_evaluation",
    }
    for record in manifest["figures"].values():
        path = figure_dir / record["filename"]
        payload = path.read_bytes()
        assert payload[:8] == b"\x89PNG\r\n\x1a\n"
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]
        assert int.from_bytes(payload[16:20], "big") == record["width_px"]
        assert int.from_bytes(payload[20:24], "big") == record["height_px"]


def test_tracked_paper_results_report_verified_aligned_summary() -> None:
    study_dir = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "two_view_pose_probability"
    )
    document = (study_dir / "PAPER_RESULTS.md").read_text(encoding="utf-8")

    required_statements = {
        "Status: **NO_GO**",
        "every estimate uses exactly\ntwo spherical panoramas",
        "4,017 unique images, 2,385 unordered pairs, and 69 independence",
        "| Stanford2D3D | 33/150 | 1 | 0.727 | 0.572 | 6 | no |",
        "The frozen primary design requires 40 new groups and 120 selected pairs",
        "independent verification: PASS 59/59",
        "The frozen controlled protocol completed all 225 observations",
        "| P74 | 75 / 25 | 5.731 / 6.584 | 5.607 / 7.869",
    }
    for statement in required_statements:
        assert statement in document

    assert "raw-score post-processing `p_precise_post ≥ 0.80`" not in document
    assert "| Stanford2D3D | 25/150" not in document

    figure_manifest = json.loads(
        (study_dir / "figures" / "aligned-quantitative-figures.json").read_text(
            encoding="utf-8"
        )
    )
    for record in figure_manifest["figures"].values():
        assert f"figures/{record['filename']}" in document


def test_controlled_timing_selection_is_outcome_blind_and_deterministic() -> None:
    rows = [
        {
            "dataset_id": dataset,
            "pair_id": f"{dataset}-{index}",
            "independence_component_id": f"{dataset}-group-{index % 3}",
            "capture": {"registered_cloud_overlap_min": 0.55},
            "post": {"pair_total_seconds": 999.0 - index},
            "outcomes": {"precise": bool(index % 2)},
        }
        for dataset in ("matterport360", "stanford2d3d", "p74_native_polar")
        for index in range(8)
    ]

    selected = select_timing_pairs(rows, pairs_per_cell=5)
    reversed_selected = select_timing_pairs(list(reversed(rows)), pairs_per_cell=5)

    assert selected == reversed_selected
    assert len(selected) == 15
    assert all(row["overlap_bin"] == "50-70" for row in selected)
    assert all("post" not in row and "outcomes" not in row for row in selected)
    assert all(row["selection_uses_algorithm_outcome"] is False for row in selected)
    assert all(
        len(
            {
                row["independence_component_id"]
                for row in selected
                if row["dataset_id"] == dataset
            }
        )
        == 3
        for dataset in ("matterport360", "stanford2d3d", "p74_native_polar")
    )


def test_controlled_timing_repetition_order_is_stable_and_changes_by_repeat() -> None:
    rows = [
        {"dataset_id": "matterport360", "pair_id": f"pair-{index}"}
        for index in range(12)
    ]
    first = controlled_timing_order(rows, 1)
    first_reversed = controlled_timing_order(list(reversed(rows)), 1)
    second = controlled_timing_order(rows, 2)

    assert first == first_reversed
    assert {row["pair_id"] for row in first} == {row["pair_id"] for row in rows}
    assert [row["pair_id"] for row in first] != [
        row["pair_id"] for row in second
    ]
    with pytest.raises(ValueError, match="repetition must be positive"):
        controlled_timing_order(rows, 0)


def test_controlled_timing_host_gate_rejects_contention() -> None:
    commit = "03c5b36b28225b24d3909286bf53250d7b532aa3"
    gate = {
        "schema": "panorai-controlled-timing-host-gate/v1",
        "approved": True,
        "stable_power": True,
        "output_directory_exclusive": True,
        "unrelated_intensive_processes": False,
        "thermal_throttling": False,
        "free_memory_gib": 16.0,
        "required_free_memory_gib": 8.0,
        "route_validation": {
            "validated": True,
            "package_version": "3.5.0",
            "source_commit": commit,
            "wheel_sha256": (
                "e861dafbaa5991aef77dd512b3ef1bf6fdc7967d10fbd236d850bab5c1a5f8a7"
            ),
            "convolution_backend": "native",
            "detector_method": "detect_batch",
            "detector_batch_size": 2,
            "patch_provider_max_workers": 4,
            "numpy_fallback_permitted": False,
            "explicit_validity_masks": True,
            "route_sha256": EXPECTED_ROUTE_SHA256,
            "resolution_hw": [1024, 2048],
            "opencv_threads": 16,
            "patch_workers": 4,
            "result_path": "/external/route-validation.json",
            "result_sha256": "a" * 64,
            "import_path": "/external/venv/site-packages/panorai/__init__.py",
        },
        "system": {
            "cpu_model": "test",
            "physical_cpu_count": 4,
            "logical_cpu_count": 8,
            "ram_gib": 32.0,
            "os": "test",
            "python": "3.12",
            "numpy": "2",
            "opencv": "4",
            "power_source": "AC",
            "power_mode": "automatic",
            "thermal_state": "nominal",
            "thread_environment": {},
        },
    }

    validate_controlled_timing_host_gate(gate, expected_source_commit=commit)
    gate["unrelated_intensive_processes"] = True
    with pytest.raises(ValueError, match="unrelated_intensive_processes=false"):
        validate_controlled_timing_host_gate(gate, expected_source_commit=commit)


def test_controlled_timing_executor_rejects_outcome_field() -> None:
    cells = {
        "lt-10": 0.05,
        "10-25": 0.15,
        "25-50": 0.35,
        "50-70": 0.60,
        "ge-70": 0.80,
    }
    selection = []
    for dataset in ("matterport360", "stanford2d3d", "p74_native_polar"):
        for overlap_bin, overlap in cells.items():
            for index in range(5):
                selection.append(
                    {
                        "schema": "panorai-two-view-controlled-timing-selection/v1",
                        "dataset_id": dataset,
                        "pair_id": f"{dataset}-{overlap_bin}-{index}",
                        "independence_component_id": f"{dataset}-group-{index}",
                        "overlap_bin": overlap_bin,
                        "overlap_lower": 0.0,
                        "overlap_upper": 1.0,
                        "registered_cloud_overlap_min": overlap,
                        "selection_key": f"key-{index}",
                        "selection_uses_algorithm_outcome": False,
                    }
                )
    population = [
        {"dataset_id": row["dataset_id"], "pair_id": row["pair_id"]}
        for row in selection
    ]
    manifest = {
        "schema": "panorai-two-view-controlled-timing-manifest/v1",
        "selection": {"rows": 75},
    }

    validate_controlled_timing_selection(
        selection, manifest, population, population
    )
    selection[0]["outcomes"] = {"precise": True}
    with pytest.raises(ValueError, match="unexpected timing-selection field"):
        validate_controlled_timing_selection(
            selection, manifest, population, population
        )


def test_controlled_timing_host_gate_rejects_checkout_import(tmp_path: Path) -> None:
    commit = "03c5b36b28225b24d3909286bf53250d7b532aa3"
    result = {
        "schema": "panorai-unified-optimized-pair/v2",
        "package": {
            "version": "3.5.0",
            "expected_source_commit": commit,
            "import_path": "/external/venv/site-packages/panorai/__init__.py",
        },
        "resolution_hw": [1024, 2048],
        "native": {
            "convolution_backend": "native",
            "native_filter_available": True,
            "native_pose_kernels_available": True,
            "numpy_fallback_permitted": False,
        },
        "route": optimized_pair_profile_configuration(),
        "validity": {"derived_from_black_pixels": False},
        "system": {"opencv_threads": 16, "patch_workers": 4},
    }
    checkout = tmp_path / "checkout"

    validate_controlled_timing_route_result(
        result,
        expected_source_commit=commit,
        forbidden_checkout=checkout,
    )
    route_path = tmp_path / "route-validation.json"
    route_path.write_text(json.dumps(result))
    route_hash = hashlib.sha256(route_path.read_bytes()).hexdigest()
    gate = {
        "route_validation": {
            "result_path": str(route_path),
            "result_sha256": route_hash,
        }
    }
    validate_controlled_timing_gate_evidence(
        gate,
        expected_source_commit=commit,
        forbidden_checkout=checkout,
    )
    route_path.write_text(json.dumps(result) + "\n")
    with pytest.raises(ValueError, match="result hash changed"):
        validate_controlled_timing_gate_evidence(
            gate,
            expected_source_commit=commit,
            forbidden_checkout=checkout,
        )
    result["route"]["detector"]["max_keypoints"] = 1024
    with pytest.raises(ValueError, match="optimized public route"):
        validate_controlled_timing_route_result(
            result,
            expected_source_commit=commit,
            forbidden_checkout=checkout,
        )
    result["route"]["detector"]["max_keypoints"] = 4096
    result["package"]["import_path"] = str(checkout / "panorai" / "__init__.py")
    with pytest.raises(ValueError, match="forbidden checkout"):
        validate_controlled_timing_route_result(
            result,
            expected_source_commit=commit,
            forbidden_checkout=checkout,
        )


def test_controlled_timing_summary_preserves_pair_sampling_unit() -> None:
    bins = ("lt-10", "10-25", "25-50", "50-70", "ge-70")
    observations = []
    for dataset in ("matterport360", "stanford2d3d", "p74_native_polar"):
        for bin_index, overlap_bin in enumerate(bins):
            for pair_index in range(5):
                for repetition in range(1, 4):
                    seconds = float(bin_index + pair_index + repetition)
                    observations.append(
                        {
                            "dataset_id": dataset,
                            "pair_id": f"{dataset}-{overlap_bin}-{pair_index}",
                            "independence_component_id": (
                                f"{dataset}-group-{pair_index}"
                            ),
                            "overlap_bin": overlap_bin,
                            "registered_cloud_overlap_min": 0.1 * bin_index,
                            "repetition": repetition,
                            "timings_seconds": {
                                field: seconds for field in CONTROLLED_TIMING_FIELDS
                            },
                            "peak_rss_mib": 512.0 + seconds,
                            "counts": {
                                "keypoints_a": 100 + pair_index,
                                "keypoints_b": 80 + pair_index,
                                "matches": 20 + pair_index,
                            },
                        }
                    )

    cells = summarize_controlled_timing_cells(observations)
    datasets = summarize_controlled_timing_datasets(observations)

    assert len(cells) == 15
    assert all(cell["raw_observations"] == 15 for cell in cells)
    assert all(cell["unique_pairs"] == 5 for cell in cells)
    assert all(cell["independence_components"] == 5 for cell in cells)
    assert all(
        cell["metrics"]["pair_total"]["raw_observations"] == 15
        and cell["metrics"]["pair_total"]["unique_pairs"] == 5
        for cell in cells
    )
    assert len(datasets) == 3
    assert all(row["raw_observations"] == 75 for row in datasets)
    assert all(row["unique_pairs"] == 25 for row in datasets)
    assert datasets[0]["workload"]["combined_keypoints"][
        "observation_median"
    ] == 184.0
    assert datasets[0]["workload"]["matches"]["observation_median"] == 22.0
    assert datasets[0]["reference_comparison"]["pair_total"] == {
        "reference_seconds": 8.99,
        "observation_median_ratio": 6.0 / 8.99,
    }
    with pytest.raises(ValueError, match="cell is incomplete"):
        summarize_controlled_timing_cells(observations[:-1])
