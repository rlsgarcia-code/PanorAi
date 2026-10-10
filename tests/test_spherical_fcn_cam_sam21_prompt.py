from __future__ import annotations

import numpy as np
import pytest

from benchmarks.spherical_fcn_cam.run_p74_cam_sam21_prompt_study import (
    cam_component_mask,
    connected_component_at_anchor,
    majority_mask,
    prompt_from_component,
)


def test_majority_mask_keeps_two_of_three_agreement() -> None:
    masks = np.zeros((3, 8, 8), dtype=bool)
    masks[:2, 2:6, 3:5] = True
    masks[2, :2, :2] = True

    actual = majority_mask(masks)

    assert actual.sum() == 8
    assert actual[3, 4]
    assert not actual[0, 0]


def test_connected_component_chooses_nearest_when_anchor_is_background() -> None:
    mask = np.zeros((12, 14), dtype=bool)
    mask[1:4, 1:4] = True
    mask[7:11, 9:13] = True

    actual = connected_component_at_anchor(mask, (8.0, 8.0))

    assert actual.sum() == 16
    assert actual[8, 10]
    assert not actual[2, 2]


def test_prompt_from_component_has_inside_positives_and_outside_negatives() -> None:
    component = np.zeros((100, 120), dtype=bool)
    component[20:85, 45:70] = True

    prompt = prompt_from_component(component, (55.0, 50.0))

    assert prompt["positive_count"] == 3
    assert prompt["negative_count"] >= 2
    for (x, y), label in zip(prompt["points_xy"], prompt["labels"], strict=True):
        assert bool(component[round(y), round(x)]) is (label == 1)
    x1, y1, x2, y2 = prompt["box_xyxy"]
    assert (x1, y1, x2, y2) == (33.0, 8.0, 81.0, 96.0)


def test_cam_component_uses_thresholded_region_at_anchor() -> None:
    evidence = np.zeros((80, 100), dtype=np.float32)
    evidence[20:60, 40:65] = 0.8
    evidence[5:15, 5:15] = 0.9

    component, threshold = cam_component_mask(
        evidence,
        (50.0, 40.0),
        relative_threshold=0.7,
        absolute_threshold=0.35,
    )

    assert threshold == pytest.approx(0.56)
    assert component.sum() == 1000
    assert component[40, 50]
    assert not component[10, 10]


def test_cam_component_has_local_fallback_for_isolated_low_peak() -> None:
    evidence = np.zeros((128, 128), dtype=np.float32)
    evidence[64, 64] = 0.1

    component, threshold = cam_component_mask(
        evidence, (64.0, 64.0), fallback_radius=10
    )

    assert threshold == 0.35
    assert 300 <= component.sum() <= 320
    assert component[64, 64]
