from __future__ import annotations

import numpy as np
import pytest

from benchmarks.spherical_fcn_cam.run_p74_cam_sam_prompt_pilot import (
    erp_pixel_from_lon_lat,
    make_prompt_spec,
    ordered_mask_records,
    pairwise_mask_iou,
    select_candidate,
)


def _record() -> dict[str, object]:
    return {
        "candidate_predictions": [
            {
                "class_index": 653,
                "class_name": "milk can",
                "rank": 1,
                "cam_statistics": {
                    "peak_latitude_degrees": 64.5,
                    "peak_longitude_degrees": 143.25,
                },
            }
        ]
    }


def test_candidate_and_prompt_spec_are_fixed_before_segmentation() -> None:
    candidate = select_candidate(_record(), "milk can")
    spec = make_prompt_spec(candidate, fov_deg=70.0, face_size=1024)

    assert candidate["rank"] == 1
    assert spec.center_lat_deg == 64.5
    assert spec.center_lon_deg == 143.25
    assert spec.output_shape_hw == (1024, 1024)
    assert spec.hfov_deg == spec.vfov_deg == 70.0


def test_candidate_without_rendered_peak_fails_explicitly() -> None:
    record = _record()
    record["candidate_predictions"][0].pop("cam_statistics")  # type: ignore[index,union-attr]
    with pytest.raises(ValueError, match="no rendered CAM statistics"):
        select_candidate(record, "milk can")


def test_erp_direction_mapping_respects_pixel_centers_and_wrap() -> None:
    assert erp_pixel_from_lon_lat(-180.0, 90.0, (180, 360)) == pytest.approx(
        (359.5, -0.5)
    )
    assert erp_pixel_from_lon_lat(0.0, 0.0, (180, 360)) == pytest.approx((179.5, 89.5))
    assert erp_pixel_from_lon_lat(180.0, -90.0, (180, 360)) == pytest.approx(
        (359.5, 179.5)
    )


def test_all_multimask_outputs_remain_in_decoder_order() -> None:
    masks = np.zeros((3, 5, 7), dtype=bool)
    masks[0, 2, 3] = True
    masks[1, 1:4, 2:5] = True
    masks[2] = True
    scores = np.asarray([0.15, 0.91, 0.42])
    erp_masks = [np.full((4, 8), index % 2 == 0) for index in range(3)]

    records = ordered_mask_records(masks, scores, erp_masks)

    assert [item["decoder_index"] for item in records] == [0, 1, 2]
    assert [item["predicted_iou"] for item in records] == pytest.approx(
        [0.15, 0.91, 0.42]
    )
    assert [item["display_number"] for item in records] == [1, 2, 3]
    assert all(item["contains_positive_prompt"] for item in records)


def test_mask_count_mismatch_fails_instead_of_dropping_an_alternative() -> None:
    masks = np.zeros((3, 2, 2), dtype=bool)
    with pytest.raises(ValueError, match="counts differ"):
        ordered_mask_records(masks, np.zeros(2), [np.zeros((2, 2), dtype=bool)] * 3)


def test_pairwise_iou_reports_similarity_without_ranking_masks() -> None:
    masks = np.zeros((3, 2, 3), dtype=bool)
    masks[0, 0, :2] = True
    masks[1, 0, 1:] = True
    masks[2] = masks[0]

    matrix = pairwise_mask_iou(masks)

    np.testing.assert_allclose(
        matrix,
        [[1.0, 1.0 / 3.0, 1.0], [1.0 / 3.0, 1.0, 1.0 / 3.0], [1.0, 1.0 / 3.0, 1.0]],
    )
