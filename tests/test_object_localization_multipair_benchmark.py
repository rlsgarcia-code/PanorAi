from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "benchmarks"
    / "object_localization"
    / "run_multipair_association_validation.py"
)
REGIONS = RUNNER.with_name("stanford_multipair_regions.json")
REFERENCE = RUNNER.with_name("stanford_multipair_reference.json")


def _runner():
    spec = importlib.util.spec_from_file_location(
        "multipair_association_runner", RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_multipair_manifest_has_six_real_pairs_and_fixed_controls() -> None:
    regions = json.loads(REGIONS.read_text(encoding="utf-8"))
    assert regions["schema"] == "panorai-object-association-multipair-regions/v1"
    assert len(regions["pairs"]) == 6
    assert {pair["area"] for pair in regions["pairs"]} == {
        "area_2",
        "area_3",
        "area_4",
    }
    assert {pair["scene"] for pair in regions["pairs"]} >= {"office_3", "lobby_1"}
    assert [item["name"] for item in regions["controls"]] == [
        "base",
        "absent-b-last",
        "semantic-mismatch-first",
        "duplicate-b-first",
    ]

    pair_ids = [pair["pair_id"] for pair in regions["pairs"]]
    assert len(pair_ids) == len(set(pair_ids))
    for pair in regions["pairs"]:
        assert len(pair["views"]) == 2
        for view in pair["views"]:
            assert len(view["rgb_sha256"]) == 64
            for region in view["regions"]:
                assert region["region_id"].startswith("s")
                assert 0 <= region["class_id"] < 1000
                for x0, y0, x1, y1 in region["rectangles_xyxy"]:
                    assert 0 <= x0 < x1 <= 2048
                    assert 0 <= y0 < y1 <= 1024


def test_reference_encodes_fail_closed_control_expectations() -> None:
    regions = json.loads(REGIONS.read_text(encoding="utf-8"))
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    by_pair = {pair["pair_id"]: pair for pair in reference["pairs"]}
    assert set(by_pair) == {pair["pair_id"] for pair in regions["pairs"]}

    for pair in reference["pairs"]:
        base = pair["scenarios"]["base"]["expected_links"]
        assert len(base) >= 2
        assert pair["scenarios"]["absent-b-last"]["expected_links"] == base[:-1]
        assert (
            pair["scenarios"]["semantic-mismatch-first"]["expected_links"] == base[1:]
        )
        assert pair["scenarios"]["duplicate-b-first"]["expected_links"] == base[1:]
        for name in (
            "absent-b-last",
            "semantic-mismatch-first",
            "duplicate-b-first",
        ):
            assert pair["scenarios"][name]["must_abstain_region_a"] in {
                base[0][0],
                base[-1][0],
            }


@dataclass(frozen=True)
class _Region:
    region_id: str
    class_id: int
    class_name: str


def test_controls_drop_mismatch_and_duplicate_without_mutating_inputs() -> None:
    module = _runner()
    regions = (_Region("b0", 10, "first"), _Region("b1", 20, "last"))

    dropped = module._controlled_regions_b(
        regions, {"operation": "drop-region-b", "target": "last"}
    )
    mismatched = module._controlled_regions_b(
        regions, {"operation": "mismatch-class-b", "target": "first"}
    )
    duplicated = module._controlled_regions_b(
        regions, {"operation": "duplicate-region-b", "target": "first"}
    )

    assert dropped == (regions[0],)
    assert mismatched[0].class_id == 1010
    assert regions[0].class_id == 10
    assert [item.region_id for item in duplicated] == ["b0", "b1", "b0--duplicate"]


def test_evaluator_counts_every_unexpected_promotion() -> None:
    module = _runner()
    reference = {
        "development_gates": {
            "base_precision_min": 1.0,
            "base_recall_min": 1.0,
            "control_precision_min": 1.0,
            "target_abstention_rate_min": 1.0,
            "deterministic_id_rate_min": 1.0,
            "accepted_pose_pair_rate_min": 1.0,
        },
        "pairs": [
            {
                "pair_id": "pair",
                "scenarios": {
                    "base": {"expected_links": [["a0", "b0"]]},
                    "control": {
                        "expected_links": [],
                        "must_abstain_region_a": "a0",
                    },
                },
            }
        ],
    }
    prediction = {
        "pairs": [
            {
                "pair_id": "pair",
                "pose": {"quality_accepted": True},
                "scenarios": [
                    {
                        "name": "base",
                        "predicted_links": [["a0", "b0"]],
                        "ids_repeat_identically": True,
                        "associations": [
                            {
                                "region_ids": ["a0", "b0"],
                                "match_count": 5,
                                "inlier_count": 4,
                                "reasons": [],
                            }
                        ],
                    },
                    {
                        "name": "control",
                        "predicted_links": [["a0", "b-wrong"]],
                        "ids_repeat_identically": True,
                    },
                ],
            }
        ]
    }

    evaluation = module._evaluate(prediction, reference)

    assert evaluation["base"]["precision"] == 1.0
    assert evaluation["controls"]["false_positives"] == 1
    assert evaluation["controls"]["precision"] == 0.0
    assert evaluation["target_abstention_rate"] == 0.0
    assert (
        evaluation["association_given_eligible_evidence"]["conditional_recall"] == 1.0
    )
    assert not evaluation["all_gates_passed"]


def test_prediction_artifact_excludes_nondeterministic_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    module = _runner()
    regions = {
        "evaluation_shape_hw": [8, 16],
        "controls": [{"name": "base"}],
        "pairs": [{"pair_id": "pair"}],
    }
    reference = {"pairs": [], "development_gates": {}}
    monkeypatch.setattr(
        module,
        "_load_json",
        lambda path: regions if path == module.REGIONS_PATH else reference,
    )
    monkeypatch.setattr(
        module,
        "_predict_pair",
        lambda pair, controls, dataset_root, shape_hw: {"pair_id": pair["pair_id"]},
    )
    monkeypatch.setattr(
        module,
        "_evaluate",
        lambda prediction, expected: {"all_gates_passed": True},
    )

    first = module.run(tmp_path / "dataset", tmp_path / "first")
    second = module.run(tmp_path / "dataset", tmp_path / "second")
    first_prediction = (tmp_path / "first" / "prediction.json").read_bytes()
    second_prediction = (tmp_path / "second" / "prediction.json").read_bytes()

    assert first_prediction == second_prediction
    assert b"elapsed" not in first_prediction
    assert first["prediction_sha256"] == second["prediction_sha256"]
    assert first["prediction_elapsed_seconds"] >= 0.0
    assert second["prediction_elapsed_seconds"] >= 0.0
