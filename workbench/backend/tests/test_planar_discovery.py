from __future__ import annotations

from io import BytesIO
import json

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from panorai_workbench.api import app
from panorai_workbench.planar_discovery import discover_planar_orientations
from panorai_workbench.samples import STEREO_TEXTURE


client = TestClient(app)


def _indoor() -> np.ndarray:
    with Image.open(STEREO_TEXTURE) as source:
        return np.asarray(source.convert("RGB"))


def _request(**updates):
    result = {
        "revision": "backend-indoor-v1",
        "options": {
            "atlas_face_size": 256,
            "minimum_bundle_lines": 2,
            "minimum_bundle_support_deg": 4.0,
            "maximum_lines": 300,
        },
    }
    result.update(updates)
    return result


def test_public_discovery_translation_is_orientation_only_and_reviewable() -> None:
    payload = discover_planar_orientations(_indoor(), _request())

    assert payload["schema"] == "panorai-workbench-planar-discovery/v1"
    assert payload["interface"] == "panorai-spherical-planar-discovery/v1-experimental"
    assert payload["lines"]
    assert payload["edge_preview_erp_xy"]
    assert len(payload["bundles"]) >= 2
    assert all("suggestions" in item for item in payload["bundles"])
    assert payload["proposals"]
    assert all(item["offset_m"] is None for item in payload["proposals"])
    assert payload["diagnostics"]["metric_geometry_published"] is False
    assert payload["pipeline"]["steps"][2]["stage"] == "DetectAndFitGnomonicLines"


def test_direction_expansion_decision_is_scoped_to_one_bundle() -> None:
    initial = discover_planar_orientations(_indoor(), _request())
    bundle = next(item for item in initial["bundles"] if item["suggestions"])
    suggestion = bundle["suggestions"][0]
    reviewed = discover_planar_orientations(
        _indoor(),
        _request(
            revision="backend-indoor-suggestion-v2",
            bundle_line_decisions={
                bundle["bundle_id"]: {suggestion["line_id"]: "rejected"}
            },
        ),
    )

    same_bundle = next(
        item for item in reviewed["bundles"] if item["bundle_id"] == bundle["bundle_id"]
    )
    decision = next(
        item for item in same_bundle["suggestions"] if item["line_id"] == suggestion["line_id"]
    )
    assert decision["state"] == "rejected"
    assert reviewed["diagnostics"]["metric_geometry_published"] is False


def test_reviewed_bundle_pair_is_replayed_through_public_request() -> None:
    initial = discover_planar_orientations(_indoor(), _request())
    candidate = initial["proposals"][0]
    reviewed = discover_planar_orientations(
        _indoor(),
        _request(
            revision="backend-indoor-v2",
            plane_bundle_groups={"reviewed-floor": candidate["bundle_ids"]},
        ),
    )

    proposal = next(
        item for item in reviewed["proposals"] if item["proposal_id"] == "reviewed-floor"
    )
    assert reviewed["state"] == "reviewed"
    assert proposal["source"] == "human"
    assert proposal["state"] == "accepted"
    assert proposal["offset_m"] is None


def test_discovery_endpoint_accepts_an_erp_and_never_accepts_geometry() -> None:
    image = _indoor()
    stream = BytesIO()
    Image.fromarray(image).save(stream, format="PNG")

    response = client.post(
        "/api/v1/interactions/discover-planar-orientations",
        data={"request": json.dumps(_request())},
        files={"image": ("indoor.png", stream.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["lines"]
    assert payload["bundles"]
    assert all(item["state"] == "draft" for item in payload["proposals"])
    assert all(item["offset_m"] is None for item in payload["proposals"])


def test_discovery_reports_the_exact_operator_selected_working_resolution() -> None:
    payload = discover_planar_orientations(
        _indoor(),
        _request(
            options={
                "atlas_face_size": 768,
                "canny_low": 25,
                "canny_high": 105,
                "local_contrast_clip_limit": 2.0,
                "minimum_segment_px": 14,
                "minimum_edge_support": 0.3,
                "minimum_bundle_lines": 2,
                "minimum_bundle_support_deg": 4.0,
                "maximum_lines": 300,
            }
        ),
    )

    assert payload["effective_options"]["atlas_face_size"] == 768
    assert payload["effective_options"]["canny_low"] == 25
    assert payload["effective_options"]["canny_high"] == 105
    assert payload["effective_options"]["local_contrast_clip_limit"] == 2.0
    assert payload["effective_options"]["minimum_segment_px"] == 14
    assert payload["effective_options"]["minimum_edge_support"] == 0.3
