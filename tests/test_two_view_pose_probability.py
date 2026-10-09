from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

import benchmarks.two_view_pose_probability.run_aligned_analysis as aligned_analysis

from benchmarks.two_view_pose_probability.build_pair_table import (
    build_rows,
    missingness,
)
from benchmarks.two_view_pose_probability.build_failure_taxonomy import (
    _categories,
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
from benchmarks.two_view_pose_probability.run_aligned_analysis import (
    ANALYSIS_STAGE_ORDER,
    _prepare_output_dir,
)
from benchmarks.two_view_pose_probability.run_resumable_population_replay import (
    _load_valid_result,
    _slug,
)
from benchmarks.two_view_pose_probability.select_release_rule import (
    _selected as selective_rule_selected,
    exact_one_sided_lower,
)
from benchmarks.two_view_pose_probability.summarize_unified_replay import _state
from benchmarks.two_view_pose_probability.render_paper_results import wilson_interval
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


def test_resumable_replay_accepts_only_complete_native_route_result(
    tmp_path: Path,
) -> None:
    row = {"dataset_id": "dataset", "pair_id": "pair"}
    path = tmp_path / f"{_slug(row)}.json"
    result = {
        "schema": "panorai-unified-optimized-pair/v2",
        **row,
        "package": {"version": "3.5.0", "expected_source_commit": "abc123"},
        "native": {
            "convolution_backend": "native",
            "native_filter_available": True,
            "native_pose_kernels_available": True,
            "numpy_fallback_permitted": False,
        },
        "route": {
            "route": {
                "detector_method": "detect_batch",
                "patch_provider_max_workers": 4,
            }
        },
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

    result["native"]["convolution_backend"] = "numpy"
    path.write_text(json.dumps(result), encoding="utf-8")
    assert _load_valid_result(path, row) is None


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


def test_aligned_analysis_requires_a_new_or_empty_output_directory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "analysis"
    assert _prepare_output_dir(output) == output.resolve()
    (output / "partial.txt").write_text("do not mix runs", encoding="utf-8")
    with pytest.raises(ValueError, match="absent or empty"):
        _prepare_output_dir(output)


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

    monkeypatch.setattr(aligned_analysis, "build_aligned_table", build)
    monkeypatch.setattr(aligned_analysis, "fit_predict", fit)
    monkeypatch.setattr(aligned_analysis, "fit_predict_lodo", fit)
    monkeypatch.setattr(aligned_analysis, "evaluate_models", evaluate)
    monkeypatch.setattr(aligned_analysis, "freeze_rule", freeze)
    monkeypatch.setattr(aligned_analysis, "evaluate_rule", evaluate_rule)
    monkeypatch.setattr(aligned_analysis, "plan_prospective_confirmation", plan)
    monkeypatch.setattr(aligned_analysis, "render_paper_results", render)


@pytest.mark.parametrize("qualifying_rule", [True, False])
def test_aligned_analysis_orchestrates_both_rule_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qualifying_rule: bool,
) -> None:
    _install_aligned_analysis_fakes(monkeypatch, qualifying_rule=qualifying_rule)
    base = tmp_path / "base.jsonl"
    base.write_text("{}\n", encoding="utf-8")
    results = tmp_path / "results"
    results.mkdir()
    output = tmp_path / "output"

    manifest = aligned_analysis.run(
        argparse.Namespace(
            base_analysis_table=base,
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
