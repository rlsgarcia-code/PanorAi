from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree


SCRIPT = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "p74_pair_eligibility"
    / "build_pose_eligible_queue.py"
)
SPEC = importlib.util.spec_from_file_location("p74_pair_eligibility", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_native_projection_uses_p74_sign_period_and_support() -> None:
    points = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )

    x, y, radial, supported = MODULE.native_pixels_from_local(
        points, height=151, width=361
    )

    np.testing.assert_array_equal(x, [0, 90, 270, 0])
    np.testing.assert_array_equal(y, [90, 90, 90, 150])
    np.testing.assert_allclose(radial, 1.0)
    np.testing.assert_array_equal(supported, [True, True, True, False])


def test_classification_separates_negative_ambiguous_stress_and_primary() -> None:
    common = {
        "minimum_cloud_overlap": 0.50,
        "primary_cloud_overlap": 0.70,
        "minimum_covisible_overlap": 0.03,
        "minimum_directional_overlap": 0.03,
        "minimum_coverage": 0.08,
        "primary_coverage": 0.15,
        "minimum_parallax_samples": 64,
        "minimum_parallax_deg": 0.5,
    }
    negative, _ = MODULE.classify_pair(
        scene_cloud_overlap=0.009,
        minimum_directional_cloud_overlap=0.009,
        covisible_overlap=0.20,
        minimum_directional=0.009,
        coverage=0.2,
        parallax_count=100,
        parallax_median_rad=math.radians(2),
        **common,
    )
    ambiguous, _ = MODULE.classify_pair(
        scene_cloud_overlap=0.60,
        minimum_directional_cloud_overlap=0.49,
        covisible_overlap=0.20,
        minimum_directional=0.035,
        coverage=0.2,
        parallax_count=100,
        parallax_median_rad=math.radians(2),
        **common,
    )
    stress, _ = MODULE.classify_pair(
        scene_cloud_overlap=0.60,
        minimum_directional_cloud_overlap=0.55,
        covisible_overlap=0.10,
        minimum_directional=0.08,
        coverage=0.12,
        parallax_count=100,
        parallax_median_rad=math.radians(2),
        **common,
    )
    primary, _ = MODULE.classify_pair(
        scene_cloud_overlap=0.75,
        minimum_directional_cloud_overlap=0.72,
        covisible_overlap=0.20,
        minimum_directional=0.18,
        coverage=0.20,
        parallax_count=100,
        parallax_median_rad=math.radians(2),
        **common,
    )

    assert (negative, ambiguous, stress, primary) == (
        "no_overlap_negative",
        "ambiguous",
        "pose_stress",
        "pose_primary",
    )


def test_scene_cloud_intersection_is_bidirectional_in_common_frame() -> None:
    left_points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    right_points = np.asarray([[0.1, 0.0, 0.0], [5.0, 0.0, 0.0]])
    left = SimpleNamespace(sampled_scene=left_points, scene_tree=cKDTree(left_points))
    right = SimpleNamespace(
        sampled_scene=right_points, scene_tree=cKDTree(right_points)
    )

    result = MODULE.scene_cloud_intersection_metrics(
        left, right, distance_tolerance_m=0.25
    )

    assert result["left_in_right_fraction"] == 0.5
    assert result["right_in_left_fraction"] == 0.5
    assert result["symmetric_fraction"] == 0.5
    assert result["matched_distance_m"]["median"] == 0.1


def _pair(pair_id: str, left: str, right: str, overlap: float, status: str) -> dict:
    return {
        "pair_id": pair_id,
        "family_id": "family",
        "from_view_id": left,
        "to_view_id": right,
        "from_rgb_path": f"{left}.png",
        "to_rgb_path": f"{right}.png",
        "scene_cloud_intersection": {"symmetric_fraction": overlap},
        "difficulty_reference": {"value": overlap},
        "symmetric_covisible_fraction": overlap,
        "selection": {"status": status},
    }


def test_spanning_forest_does_not_require_invalid_clique_closure() -> None:
    pairs = [
        _pair("ab", "a", "b", 0.8, "pose_primary"),
        _pair("bc", "b", "c", 0.6, "pose_primary"),
        _pair("ac", "a", "c", 0.0, "no_overlap_negative"),
    ]

    selected = MODULE.maximum_spanning_forest(["a", "b", "c"], pairs)

    assert selected == ["ab", "bc"]


def test_queue_uses_only_eligible_pairs_and_is_deterministic() -> None:
    pairs = [
        _pair("p0", "a", "b", 0.01, "ambiguous"),
        _pair("p1", "a", "c", 0.05, "pose_stress"),
        _pair("p2", "a", "d", 0.20, "pose_primary"),
        _pair("p3", "a", "e", 0.80, "pose_primary"),
    ]

    first = MODULE.stratified_queue(pairs, 3)
    second = MODULE.stratified_queue(list(reversed(pairs)), 3)

    assert first == second
    assert [row["pair_id"] for row in first] == ["p1", "p2", "p3"]
    assert all(row["pair_id"] != "p0" for row in first)
