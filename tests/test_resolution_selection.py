from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from panorai.features import (
    ResolutionSelectionObservation,
    ResolutionSelectionPolicy,
    SphericalFeaturePipeline,
    select_feature_resolution,
)
from panorai.features._resolution_selection import _retention


class _Features:
    def __init__(self, bearings: np.ndarray) -> None:
        self.bearings = np.asarray(bearings, dtype=np.float64)

    def __len__(self) -> int:
        return len(self.bearings)


class _Matches:
    def __init__(self, count: int) -> None:
        self.valid = np.ones(count, dtype=bool)

    def __len__(self) -> int:
        return len(self.valid)


def _bearings(count: int = 120) -> np.ndarray:
    index = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (index + 0.5) / count
    radius = np.sqrt(1.0 - z * z)
    longitude = index * math.pi * (3.0 - math.sqrt(5.0))
    return np.stack((radius * np.cos(longitude), z, radius * np.sin(longitude)), axis=1)


def _rotation_y(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.asarray(
        (
            (math.cos(angle), 0.0, math.sin(angle)),
            (0.0, 1.0, 0.0),
            (-math.sin(angle), 0.0, math.cos(angle)),
        )
    )


def _observation(
    width: int,
    *,
    rotation_deg: float,
    inliers: int,
    coverage: float,
    accepted: bool = True,
    bearings: np.ndarray | None = None,
) -> ResolutionSelectionObservation:
    points = _bearings() if bearings is None else bearings
    matches = _Matches(120)
    mask = np.zeros(120, dtype=bool)
    mask[:inliers] = True
    quality = SimpleNamespace(
        accepted=accepted,
        rejection_reasons=() if accepted else ("weak-angular-coverage",),
        coverage_entropy_a=coverage,
        coverage_entropy_b=coverage * 0.98,
        stability=SimpleNamespace(
            successful_trials=6,
            rotation_p90_deg=0.02,
        ),
    )
    pose = SimpleNamespace(
        rotation=_rotation_y(rotation_deg),
        inlier_mask=mask,
        quality_report=quality,
    )
    return ResolutionSelectionObservation(
        erp_shape_hw=(width // 2, width),
        face_shape_hw=(width // 4, width // 4),
        features_a=_Features(points),
        features_b=_Features(points),
        matches=matches,
        pose=pose,
        runtime_seconds=width / 1000.0,
    )


def test_selects_first_level_on_converged_plateau() -> None:
    low = _observation(1024, rotation_deg=1.0, inliers=80, coverage=0.80)
    middle = _observation(2048, rotation_deg=0.05, inliers=100, coverage=0.94)
    high = _observation(4096, rotation_deg=0.0, inliers=110, coverage=0.96)

    report = select_feature_resolution((high, low, middle))

    assert report.decision == "converged-minimum"
    assert report.converged_plateau
    assert report.selected_erp_shape_hw == (1024, 2048)
    assert not report.transitions[0].converged
    assert "rotation-not-converged" in report.transitions[0].rejection_reasons
    assert report.transitions[1].converged
    assert report.transitions[1].rotation_delta_deg == pytest.approx(0.05)
    assert report.transitions[1].effective_inlier_retention == pytest.approx(100 / 110)
    json.dumps(report.to_dict())


def test_geometric_repeatability_detects_resolution_dependent_features() -> None:
    low = _observation(2048, rotation_deg=0.0, inliers=100, coverage=0.9)
    shifted = _bearings() @ _rotation_y(2.0).T
    high = _observation(
        4096,
        rotation_deg=0.0,
        inliers=100,
        coverage=0.9,
        bearings=shifted,
    )

    report = select_feature_resolution((low, high))

    assert not report.transitions[0].converged
    assert "low-feature-repeatability" in report.transitions[0].rejection_reasons
    assert report.selected_erp_shape_hw == (2048, 4096)
    assert report.decision == "highest-resolution-fallback"


def test_early_convergence_is_not_selected_when_higher_level_breaks_plateau() -> None:
    low = _observation(1024, rotation_deg=0.02, inliers=100, coverage=0.9)
    middle = _observation(2048, rotation_deg=0.0, inliers=105, coverage=0.91)
    high = _observation(4096, rotation_deg=1.0, inliers=110, coverage=0.92)

    report = select_feature_resolution((low, middle, high))

    assert report.transitions[0].converged
    assert not report.transitions[1].converged
    assert report.selected_erp_shape_hw == (2048, 4096)
    assert report.decision == "highest-resolution-fallback"


def test_highest_level_fallback_is_explicit_and_rejected_pose_is_not_selected() -> None:
    accepted = select_feature_resolution(
        (_observation(2048, rotation_deg=0.0, inliers=80, coverage=0.8),)
    )
    assert accepted.decision == "highest-resolution-fallback"
    assert not accepted.converged_plateau

    rejected = select_feature_resolution(
        (
            _observation(
                2048,
                rotation_deg=0.0,
                inliers=10,
                coverage=0.1,
                accepted=False,
            ),
        )
    )
    assert rejected.decision == "no-selection"
    assert rejected.selected_level is None
    assert "weak-angular-coverage" in rejected.decision_reasons


def test_weighted_effective_inliers_and_contract_validation() -> None:
    observation = _observation(2048, rotation_deg=0.0, inliers=4, coverage=0.8)
    weights = np.ones(120, dtype=np.float64)
    weights[:4] = (1.0, 1.0, 0.0, 0.0)
    weighted = ResolutionSelectionObservation(
        erp_shape_hw=observation.erp_shape_hw,
        face_shape_hw=observation.face_shape_hw,
        features_a=observation.features_a,
        features_b=observation.features_b,
        matches=observation.matches,
        pose=observation.pose,
        inlier_weights=weights,
    )
    report = select_feature_resolution((weighted,))
    assert report.levels[0].num_inliers == 4
    assert report.levels[0].effective_inlier_count == 2.0

    with pytest.raises(ValueError, match="unique ERP widths"):
        select_feature_resolution((observation, observation))
    with pytest.raises(ValueError, match="must not exceed 1"):
        ResolutionSelectionPolicy(min_feature_repeatability=1.1)
    with pytest.raises(ValueError, match="inlier_weights"):
        ResolutionSelectionObservation(
            erp_shape_hw=(1024, 2048),
            face_shape_hw=(512, 512),
            features_a=observation.features_a,
            features_b=observation.features_b,
            matches=observation.matches,
            pose=observation.pose,
            inlier_weights=np.ones(2),
        )


def test_retention_is_finite_and_bounded() -> None:
    assert _retention(0.0, 0.0) == 1.0
    assert _retention(5.0, 0.0) == 1.0
    assert _retention(10.0, 5.0) == 1.0


def test_real_feature_and_match_objects_feed_the_report() -> None:
    rng = np.random.default_rng(20261006)
    panorama = rng.integers(0, 256, size=(256, 512, 3), dtype=np.uint8)
    pipeline = SphericalFeaturePipeline.from_preset(
        "sift-flann",
        face_sampler="cube",
        face_fov_deg=95.0,
        face_shape_hw=(128, 128),
        edge_margin_px=4,
        max_features=256,
    )
    features_a = pipeline.extract(panorama, panorama_id="resolution-a")
    features_b = pipeline.extract(panorama, panorama_id="resolution-b")
    matches = pipeline.match(features_a, features_b)

    report = select_feature_resolution(
        (
            ResolutionSelectionObservation(
                erp_shape_hw=panorama.shape[:2],
                face_shape_hw=(128, 128),
                features_a=features_a,
                features_b=features_b,
                matches=matches,
                pose=None,
            ),
        )
    )

    assert len(features_a) > 0
    assert len(matches) > 0
    assert report.levels[0].feature_count_a == len(features_a)
    assert report.levels[0].match_count == len(matches)
    assert report.decision == "no-selection"
    assert report.decision_reasons == (
        "no-converged-plateau",
        "highest-resolution-unusable",
        "pose-not-returned",
    )
