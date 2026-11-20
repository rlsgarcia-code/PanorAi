from __future__ import annotations

from io import BytesIO
import json

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from panorai_workbench.api import app
from panorai_workbench.line_refinement import refine_erp_line, robust_line_suggestion


client = TestClient(app)


def test_robust_fit_recommends_known_edge_without_mutating_manual_line() -> None:
    height, width = 100, 180
    edge = np.zeros((height, width), dtype=bool)
    x = np.arange(20, 160)
    y = np.rint(32.0 + 0.18 * x).astype(int)
    edge[y, x] = True
    manual = np.array([[24.0, 41.0], [154.0, 64.4]])
    original = manual.copy()

    result = robust_line_suggestion(edge, manual, band_radius_px=10.0)

    assert result is not None
    np.testing.assert_array_equal(manual, original)
    recommended = np.asarray(result["recommended_chart_xy"])
    expected_y = 32.0 + 0.18 * recommended[:, 0]
    assert np.max(np.abs(recommended[:, 1] - expected_y)) < 0.6
    assert result["inlier_count"] >= 100


def test_refinement_uses_gnomonic_crop_canny_and_public_geometry() -> None:
    height, width = 128, 256
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[height // 2 :] = 220
    manual = [[95.5, 67.5], [159.5, 67.5]]
    result = refine_erp_line(
        image,
        {
            "manual_erp_xy": manual,
            "chart": {
                "center_lat_deg": 0.0,
                "center_lon_deg": 0.0,
                "hfov_deg": 90.0,
                "vfov_deg": 70.0,
                "output_shape_hw": [128, 160],
            },
            "band_radius_px": 14.0,
            "threshold_low": 0.02,
            "threshold_high": 0.05,
        },
    )

    assert result["status"] == "suggested"
    assert result["manual_erp_xy"] == manual
    recommended = np.asarray(result["recommended_erp_xy"])
    assert np.max(np.abs(recommended[:, 1] - 63.5)) < 0.6
    assert result["execution"]["edge_detector"] == "opencv.Canny"
    assert result["execution"]["edge_domain"] == "gnomonic-crop"
    assert result["execution"]["projection_public_api"] == (
        "panorai.geometry.equirectangular_to_gnomonic"
    )
    assert result["execution"]["fit_owner"] == ("panorai-workbench-annotation-assist")
    assert result["execution"]["automatic_accept"] is False
    assert result["execution"]["reported_coordinate_grid"] == (
        "original-erp-pixel-centres"
    )
    assert result["crop_png_base64"]


def test_refinement_keeps_crop_and_empty_evidence_without_supported_edge() -> None:
    image = np.full((128, 256, 3), 80, dtype=np.uint8)
    manual = [[95.5, 67.5], [159.5, 67.5]]
    result = refine_erp_line(
        image,
        {
            "manual_erp_xy": manual,
            "chart": {
                "center_lat_deg": 0.0,
                "center_lon_deg": 0.0,
                "hfov_deg": 90.0,
                "vfov_deg": 70.0,
                "output_shape_hw": [128, 160],
            },
            "band_radius_px": 14.0,
            "threshold_low": 0.02,
            "threshold_high": 0.05,
        },
    )

    assert result["status"] == "no-supported-edge"
    assert result["manual_erp_xy"] == manual
    assert result["crop_png_base64"]
    assert result["candidate_count"] == 0
    assert result["edge_points_crop_xy"] == []


def test_refinement_uses_bounded_adaptive_pass_for_low_contrast_edge() -> None:
    image = np.zeros((128, 256, 3), dtype=np.uint8)
    image[64:] = 6
    result = refine_erp_line(
        image,
        {
            "manual_erp_xy": [[95.5, 67.5], [159.5, 67.5]],
            "chart": {
                "center_lat_deg": 0.0,
                "center_lon_deg": 0.0,
                "hfov_deg": 90.0,
                "vfov_deg": 70.0,
                "output_shape_hw": [128, 160],
            },
            "band_radius_px": 14.0,
            "threshold_low": 0.035,
            "threshold_high": 0.09,
        },
    )

    assert result["status"] == "suggested"
    assert result["parameters"]["adaptive_pass"] is True
    assert result["parameters"]["selected_threshold_low"] == 0.0175
    assert result["parameters"]["selected_threshold_high"] == 0.045


def test_large_erp_keeps_original_coordinate_grid_with_local_crop() -> None:
    image = np.zeros((512, 1024, 3), dtype=np.uint8)
    image[256:] = 220
    manual = [[383.5, 259.5], [639.5, 259.5]]
    result = refine_erp_line(
        image,
        {
            "manual_erp_xy": manual,
            "chart": {
                "center_lat_deg": 0.0,
                "center_lon_deg": 0.0,
                "hfov_deg": 90.0,
                "vfov_deg": 70.0,
                "output_shape_hw": [256, 320],
            },
            "band_radius_px": 18.0,
            "threshold_low": 0.02,
            "threshold_high": 0.05,
        },
    )

    assert result["status"] == "suggested"
    assert result["manual_erp_xy"] == manual
    assert result["execution"]["edge_domain"] == "gnomonic-crop"
    assert result["parameters"]["crop_shape_hw"][0] < 256
    crop_height, crop_width = result["parameters"]["crop_shape_hw"]
    assert crop_width <= 320
    assert crop_height * crop_width < 256 * 320
    assert result["execution"]["reported_coordinate_grid"] == (
        "original-erp-pixel-centres"
    )


def test_refinement_endpoint_accepts_image_and_versioned_request() -> None:
    height, width = 128, 256
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[height // 2 :] = 220
    encoded = BytesIO()
    Image.fromarray(image).save(encoded, format="PNG")
    request = {
        "manual_erp_xy": [[95.5, 67.5], [159.5, 67.5]],
        "chart": {
            "center_lat_deg": 0.0,
            "center_lon_deg": 0.0,
            "hfov_deg": 90.0,
            "vfov_deg": 70.0,
            "output_shape_hw": [128, 160],
        },
        "band_radius_px": 14.0,
        "threshold_low": 0.02,
        "threshold_high": 0.05,
    }

    response = client.post(
        "/api/v1/interactions/refine-line",
        data={"request": json.dumps(request)},
        files={"image": ("edge.png", encoded.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema"] == "panorai-workbench-line-refinement/v1"
    assert payload["status"] == "suggested"
    assert payload["execution"]["automatic_accept"] is False
