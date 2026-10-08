from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "p74_pair_eligibility"
    / "run_overlap_response.py"
)
SPEC = importlib.util.spec_from_file_location("p74_overlap_response", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _pair(family: str, pair_id: str, overlap: float) -> dict:
    return {
        "family_id": family,
        "pair_id": pair_id,
        "difficulty_reference": {"value": overlap},
    }


def test_overlap_bins_have_frozen_boundaries() -> None:
    values = [0.0, 0.0999, 0.10, 0.2499, 0.25, 0.4999, 0.50, 0.6999, 0.70, 1.0]
    assert [MODULE.overlap_bin(value) for value in values] == [
        "lt_0.10",
        "lt_0.10",
        "0.10_to_0.25",
        "0.10_to_0.25",
        "0.25_to_0.50",
        "0.25_to_0.50",
        "0.50_to_0.70",
        "0.50_to_0.70",
        "ge_0.70",
        "ge_0.70",
    ]


def test_balanced_selection_is_prediction_free_and_deterministic() -> None:
    values = [0.05, 0.15, 0.30, 0.60, 0.80]
    rows = [
        _pair(family, f"{family}-{bin_index}-{item}", overlap)
        for family in ("a", "b")
        for bin_index, overlap in enumerate(values)
        for item in range(3)
    ]
    first = MODULE.select_balanced(rows, per_family_bin=2, seed="fixed")
    second = MODULE.select_balanced(
        list(reversed(rows)), per_family_bin=2, seed="fixed"
    )

    assert first == second
    assert len(first) == 20
    assert MODULE.overlap_bin(first[0]["difficulty_reference"]["value"]) == "ge_0.70"
    counts = {}
    for row in first:
        key = (
            row["family_id"],
            MODULE.overlap_bin(row["difficulty_reference"]["value"]),
        )
        counts[key] = counts.get(key, 0) + 1
    assert set(counts.values()) == {2}


def test_relative_transform_maps_from_local_to_to_local() -> None:
    rotation_from = np.eye(3)
    translation_from = np.asarray([2.0, 0.0, 0.0])
    rotation_to = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation_to = np.asarray([1.0, 1.0, 0.0])
    rotation, translation = MODULE.relative_transform(
        rotation_from, translation_from, rotation_to, translation_to
    )
    point_from = np.asarray([3.0, 2.0, 1.0])
    scene = rotation_from @ point_from + translation_from
    expected_to = rotation_to.T @ (scene - translation_to)

    np.testing.assert_allclose(rotation @ point_from + translation, expected_to)


def test_wilson_interval_contains_observed_rate() -> None:
    low, high = MODULE.wilson_interval(6, 9)
    assert low < 6 / 9 < high


def test_best_profile_freezes_spherical_convolution_refine_and_sift() -> None:
    profile = MODULE.BEST_PROFILE

    assert profile["detector"] == {
        "family": "coarse-dog",
        "octaves": 3,
        "contrast_threshold": 0.012,
        "max_keypoints": 4096,
        "proposal_height": 512,
        "proposal_width": 1024,
        "proposal_resampling": "spherical-gaussian",
        "proposal_prefilter_sigma_px": 1.0,
        "proposal_prefilter_intermediate_height": 1024,
        "fine_verification": "tangent-dog",
        "fine_candidate_multiplier": 2.0,
        "convolution_backend": "native",
    }
    assert profile["descriptor"]["descriptor_radius_sigmas"] == 6.0
    assert profile["descriptor"]["patch_size"] == 48
    assert profile["descriptor"]["keypoint_diameter_in_scales"] == 1.5
    assert profile["descriptor"]["root_sift"] is True
    assert profile["matcher"]["ratio_test"] == 0.72
    assert profile["estimator"]["profile"] == "full"


def test_direct_profile_is_full_spherical_dog_and_not_coarse() -> None:
    profile = MODULE.DIRECT_SPHERICAL_PROFILE

    assert profile["detector"] == {
        "family": "dog",
        "implementation": "SphericalDoGDetector",
        "octaves": 3,
        "levels_per_octave": 3,
        "base_sigma_px": 1.6,
        "contrast_threshold": 0.012,
        "edge_threshold": 10.0,
        "max_keypoints": 4096,
        "selection_policy": "equal-area-round-robin",
        "convolution_backend": "native",
        "localization": "second-order-taylor-tangent-east-north-scale-level",
    }
    assert profile["descriptor"]["descriptor_radius_sigmas"] == 6.0
    assert profile["descriptor"]["patch_size"] == 48
    assert profile["descriptor"]["keypoint_diameter_in_scales"] == 1.5
    assert profile["descriptor"]["root_sift"] is True
    assert profile["matcher"]["ratio_test"] == 0.72

    pipeline = MODULE.exact_direct_spherical_pipeline(None)
    assert type(pipeline).__name__ == "SphericalDoGSIFTPipeline"
    assert pipeline.detector_config.convolution_backend == "native"
