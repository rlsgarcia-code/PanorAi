from __future__ import annotations

import numpy as np
import pytest

from benchmarks.panoramic_stereo.run_real_domain_validation import (
    depth_metrics,
    load_public_pair_selection,
    make_matcher,
    relative_pose,
)
from panorai.applications import (
    ADCensusStereoDepth,
    CensusMGMStereoDepth,
    ELASStereoDepth,
    PatchMatchStereoDepth,
    StereoBMDepth,
    StereoSGBMDepth,
)


def test_relative_pose_uses_source_from_reference_convention():
    rotation_a = np.eye(3)
    center_a = np.asarray((0.0, 0.0, 0.0))
    angle = np.deg2rad(15.0)
    rotation_b = np.asarray(
        (
            (np.cos(angle), 0.0, np.sin(angle)),
            (0.0, 1.0, 0.0),
            (-np.sin(angle), 0.0, np.cos(angle)),
        )
    )
    center_b = np.asarray((0.5, 0.0, 0.0))

    rotation, translation = relative_pose(rotation_a, center_a, rotation_b, center_b)
    point_a = np.asarray((1.0, 0.25, 4.0))
    point_world = rotation_a.T @ point_a + center_a
    expected_b = rotation_b @ (point_world - center_b)

    np.testing.assert_allclose(rotation @ point_a + translation, expected_b)


def test_depth_metrics_distinguish_zero_from_invalid_and_weight_solid_angle():
    reference = np.full((4, 8), 2.0, dtype=np.float32)
    estimate = reference.copy()
    validity = np.ones((4, 8), dtype=bool)
    support = np.ones((4, 8), dtype=bool)
    validity[0] = False
    estimate[0] = np.nan
    reference[1, 0] = 0.0

    metrics = depth_metrics(estimate, validity, reference, support)

    assert metrics["solid_angle_coverage"] < 1.0
    assert metrics["abs_rel"] == pytest.approx(0.0)
    assert metrics["delta_1_25"] == pytest.approx(1.0)
    assert metrics["correct_solid_angle_coverage_delta_1_25"] == pytest.approx(
        metrics["solid_angle_coverage"]
    )
    assert metrics["reference_pixels"] == 31


def test_explicit_public_pair_loader_fails_before_asset_access(tmp_path):
    with pytest.raises(ValueError, match="unsupported public dataset"):
        load_public_pair_selection("unknown", tmp_path, (8, 16), {})

    with pytest.raises(ValueError, match="missing fields"):
        load_public_pair_selection(
            "stanford2d3d",
            tmp_path,
            (8, 16),
            {"pair_id": "pair-only"},
        )


@pytest.mark.parametrize(
    ("method", "expected"),
    (
        ("stereo-bm", StereoBMDepth),
        ("ad-census", ADCensusStereoDepth),
        ("stereo-sgbm", StereoSGBMDepth),
        ("census-tmgm", CensusMGMStereoDepth),
        ("elas-style", ELASStereoDepth),
        ("patchmatch", PatchMatchStereoDepth),
    ),
)
def test_real_domain_runner_builds_each_declared_matcher(method, expected):
    class Args:
        max_disparity = 32
        block_size = 5
        uniqueness_ratio = 5
        left_right_tolerance = 1.5
        speckle_window_size = 0

    assert isinstance(make_matcher(Args(), method), expected)
