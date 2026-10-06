from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from benchmarks.real_pair_relative_pose.run_benchmark import (
    _PANORAI_FROM_OPENCV,
    _check_relative_oracle,
    _relative_oracle,
    _selected_indices,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_pair_oracle_maps_panorai_camera_1_points_to_camera_2() -> None:
    rotation_1 = Rotation.from_euler("xyz", [8.0, -5.0, 3.0], degrees=True).as_matrix()
    rotation_2 = Rotation.from_euler("xyz", [-4.0, 7.0, 16.0], degrees=True).as_matrix()
    center_1 = np.array([0.2, -0.4, 0.7])
    center_2 = np.array([1.1, 0.3, -0.2])
    relative_rotation, relative_direction = _relative_oracle(
        center_1, rotation_1, center_2, rotation_2
    )

    point_world = np.array([4.0, 2.0, 5.0])
    point_1 = _PANORAI_FROM_OPENCV @ (rotation_1.T @ (point_world - center_1))
    point_2 = _PANORAI_FROM_OPENCV @ (rotation_2.T @ (point_world - center_2))
    metric_translation = _PANORAI_FROM_OPENCV @ (rotation_2.T @ (center_1 - center_2))

    np.testing.assert_allclose(
        point_2, relative_rotation @ point_1 + metric_translation, atol=1e-12
    )
    np.testing.assert_allclose(
        relative_direction,
        metric_translation / np.linalg.norm(metric_translation),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        relative_rotation.T @ relative_rotation, np.eye(3), atol=1e-12
    )
    np.testing.assert_allclose(np.linalg.det(relative_rotation), 1.0, atol=1e-12)
    _check_relative_oracle()


def test_real_pair_sampling_windows_are_disjoint_and_grid_aligned() -> None:
    selected, maximum, base = _selected_indices((6.0, 20.0), 5.0, 4)

    assert base == 6.0
    assert maximum == 74
    assert selected == {
        0: (0, 0),
        1: (0, 1),
        2: (0, 2),
        3: (0, 3),
        70: (1, 0),
        71: (1, 1),
        72: (1, 2),
        73: (1, 3),
    }


def test_combined_real_pair_summary_matches_frozen_derived_rows() -> None:
    benchmark = ROOT / "benchmarks/real_pair_relative_pose"
    development = benchmark / "results/raw.csv"
    heldout = benchmark / "results/heldout/raw.csv"
    combined = json.loads(
        (benchmark / "results/COMBINED_SUMMARY.json").read_text(encoding="utf-8")
    )
    rows = []
    for path in (development, heldout):
        with path.open(encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream))

    assert _sha256(development) == combined["development"]["raw_csv_sha256"]
    assert _sha256(heldout) == combined["heldout"]["raw_csv_sha256"]
    assert (
        _sha256(benchmark / "HELDOUT_PROTOCOL.md")
        == combined["heldout"]["protocol_sha256"]
    )

    for variant, key in (
        ("count-first", "count_first"),
        ("msac-first-refit", "msac_first_refit"),
    ):
        returned = [
            row
            for row in rows
            if row["variant"] == variant and row["returned"] == "True"
        ]
        rotation = np.asarray([float(row["rotation_error_deg"]) for row in returned])
        translation = np.asarray(
            [float(row["translation_error_deg"]) for row in returned]
        )
        recorded = combined["combined"][key]
        assert len(returned) == recorded["returned_count"]
        np.testing.assert_allclose(
            np.median(rotation), recorded["rotation_error_median_deg"]
        )
        np.testing.assert_allclose(
            np.quantile(rotation, 0.95, method="higher"),
            recorded["rotation_error_p95_deg"],
        )
        np.testing.assert_allclose(
            np.median(translation), recorded["translation_error_median_deg"]
        )
        np.testing.assert_allclose(
            np.quantile(translation, 0.95, method="higher"),
            recorded["translation_error_p95_deg"],
        )
        assert (
            int(np.sum((rotation < 5.0) & (translation < 10.0)))
            == recorded["strict_count"]
        )
        assert (
            int(np.sum((rotation < 1.0) & (translation < 5.0)))
            == recorded["high_precision_count"]
        )


def test_translation_observability_study_preserves_its_negative_result() -> None:
    """The rejected GEO-017 experiment remains auditable from derived data."""

    benchmark = ROOT / "benchmarks/translation_observability"
    raw = benchmark / "results/heldout/raw.csv"
    summary_path = benchmark / "results/heldout/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with raw.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert _sha256(benchmark / "HELDOUT_PROTOCOL.md") == (
        "b2f61a8cc2f2a87a305fbb046c3a49b6de84bd21c3906ca9ce4d9194187f97d1"
    )
    assert _sha256(raw) == (
        "4a9c167586d64874556e42da61381599f2a99a93d8c0286c8f5c99965e39888c"
    )
    assert _sha256(summary_path) == (
        "307e3004736d4bae32909267ee14793d2b366e8d28b1966078238c69d74c91ca"
    )
    assert len(rows) == 24
    assert summary["measurements"]["row_count"] == len(rows)

    variants = {
        name: sorted(
            (row for row in rows if row["variant"] == name),
            key=lambda row: int(row["pair_index"]),
        )
        for name in ("msac-first-refit", "fixed-rotation-translation-gated")
    }
    baseline = variants["msac-first-refit"]
    candidate = variants["fixed-rotation-translation-gated"]
    assert len(baseline) == len(candidate) == 12
    assert [row["pair_index"] for row in baseline] == [
        row["pair_index"] for row in candidate
    ]

    common = [
        (base, trial)
        for base, trial in zip(baseline, candidate, strict=True)
        if base["returned"] == trial["returned"] == "True"
    ]
    assert len(common) == 10
    baseline_rotation = np.asarray(
        [float(base["rotation_error_deg"]) for base, _ in common]
    )
    candidate_rotation = np.asarray(
        [float(trial["rotation_error_deg"]) for _, trial in common]
    )
    baseline_translation = np.asarray(
        [float(base["translation_error_deg"]) for base, _ in common]
    )
    candidate_translation = np.asarray(
        [float(trial["translation_error_deg"]) for _, trial in common]
    )

    np.testing.assert_allclose(candidate_rotation, baseline_rotation, atol=1e-12)
    assert np.median(candidate_translation) > np.median(baseline_translation)
    assert np.quantile(candidate_translation, 0.95, method="higher") > np.quantile(
        baseline_translation, 0.95, method="higher"
    )
    assert int(np.sum(candidate_translation < baseline_translation)) == 5
    assert int(np.sum(candidate_translation > baseline_translation)) == 5

    def success_count(
        items: list[dict[str, str]], r_limit: float, t_limit: float
    ) -> int:
        return sum(
            row["returned"] == "True"
            and float(row["rotation_error_deg"]) < r_limit
            and float(row["translation_error_deg"]) < t_limit
            for row in items
        )

    assert success_count(candidate, 5.0, 10.0) == success_count(baseline, 5.0, 10.0)
    assert success_count(candidate, 1.0, 5.0) == 3
    assert success_count(baseline, 1.0, 5.0) == 4
    assert sum(row["accepted"] == "True" for row in candidate) == 7
    assert sum(row["accepted"] == "True" for row in baseline) == 8
