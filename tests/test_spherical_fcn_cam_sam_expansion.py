from __future__ import annotations

import numpy as np
import pytest

from benchmarks.spherical_fcn_cam.run_p74_cam_sam_expansion_pilot import (
    advance_chart,
    angular_distance_degrees,
    consensus_frontiers,
    greedy_spherical_peaks,
    maximum_pairwise_iou,
    normalized_proxy_ensemble,
    spherical_dilate,
)
from panorai.geometry import GnomonicSpec


def test_proxy_ensemble_normalizes_each_channel_before_weighting() -> None:
    evidence = {
        "class_indices": np.asarray([10, 20]),
        "logits": np.asarray([[[0.0, 2.0]], [[0.0, 10.0]]]),
        "support": np.asarray([[True, True]]),
    }

    actual = normalized_proxy_ensemble(evidence, {10: 1.0, 20: 3.0})

    np.testing.assert_allclose(actual, [[0.0, 1.0]])


def test_residual_peaks_wrap_longitude_and_obey_geodesic_separation() -> None:
    values = np.zeros((4, 8))
    values[1, 0] = 1.0
    values[1, 7] = 0.9  # adjacent across the longitude seam
    values[2, 4] = 0.8

    peaks = greedy_spherical_peaks(
        values,
        np.ones_like(values, dtype=bool),
        np.zeros_like(values, dtype=bool),
        maximum_count=3,
        minimum_separation_degrees=60.0,
        relative_threshold=0.1,
    )

    assert [(item["row"], item["column"]) for item in peaks] == [(1, 0), (2, 4)]


def test_spherical_dilation_wraps_longitude_but_not_latitude() -> None:
    mask = np.zeros((5, 7), dtype=bool)
    mask[0, 0] = True

    dilated = spherical_dilate(mask, 1)

    assert dilated[0, -1]
    assert dilated[0, 1]
    assert dilated[1, 0]
    assert not dilated[-1, 0]


def test_consensus_frontier_uses_two_of_three_majority_not_union() -> None:
    masks = np.zeros((3, 20, 20), dtype=bool)
    masks[:2, :5, 8:12] = True
    masks[0, -5:, 8:12] = True  # union-only bottom contact

    consensus, envelope, frontiers = consensus_frontiers(
        masks, band_pixels=5, minimum_pixels=10
    )

    assert [item["side"] for item in frontiers] == ["top"]
    assert consensus[:5].sum() == 20
    assert envelope[-5:].sum() == 20


def test_maximum_pairwise_iou_allows_two_masks_to_outvote_one_alternative() -> None:
    masks = np.zeros((3, 10, 10), dtype=bool)
    masks[0, 2:7, 2:7] = True
    masks[1, 2:7, 2:7] = True
    masks[2, 8:, 8:] = True

    assert maximum_pairwise_iou(masks) == pytest.approx(1.0)


def test_advance_chart_moves_beyond_frontier_and_keeps_overlap_prompt() -> None:
    spec = GnomonicSpec(
        center_lat_deg=10.0,
        center_lon_deg=20.0,
        hfov_deg=30.0,
        vfov_deg=30.0,
        output_shape_hw=(100, 100),
    )

    advanced, _, prompt_xy = advance_chart(spec, (49.5, 0.0), step_fraction=0.65)

    assert angular_distance_degrees(
        (spec.center_lon_deg, spec.center_lat_deg),
        (advanced.center_lon_deg, advanced.center_lat_deg),
    ) == pytest.approx(19.5)
    assert -0.5 <= prompt_xy[0] <= 99.5
    assert -0.5 <= prompt_xy[1] <= 99.5
