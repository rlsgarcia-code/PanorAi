from __future__ import annotations

import numpy as np
import pytest

from panorai_workbench.service import (
    _epipolar_normal,
    _jsonable,
    analyze_pair,
    lift_erp_points,
    single_view_academy_status,
)


def test_epipolar_normal_is_unit_and_orthogonal_to_valid_locus() -> None:
    essential = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    bearing_a = np.array([1.0, 0.0, 0.0])
    normal = np.asarray(_epipolar_normal(essential, bearing_a))
    assert np.isclose(np.linalg.norm(normal), 1.0)
    assert np.isclose(normal @ np.array([1.0, 0.0, 0.0]), 0.0)


def test_jsonable_removes_nonfinite_numpy_scalars() -> None:
    payload = _jsonable({"array": np.array([1, 2]), "bad": np.float64(np.inf)})
    assert payload == {"array": [1, 2], "bad": None}


def test_academy_status_exposes_supported_api_and_library_gaps() -> None:
    payload = single_view_academy_status()
    assert payload["schema"] == "panorai-workbench-academy-capabilities/v1"
    assert payload["available"][0]["public_api"] == (
        "panorai.geometry.erp_pixels_to_rays"
    )
    assert {item["id"] for item in payload["requires_panorai_update"]} == {
        "great-circle-line-fit",
        "vanishing-direction-estimator",
        "single-view-calibration",
    }


def test_lift_erp_points_uses_canonical_pixel_centres() -> None:
    payload = lift_erp_points([[3.5, 1.5]], (4, 8))
    np.testing.assert_allclose(payload["rays_xyz"], [[0.0, 0.0, 1.0]], atol=1e-12)
    assert payload["execution"]["public_api"] == ("panorai.geometry.erp_pixels_to_rays")


def test_pair_analysis_routes_pose_through_public_application(monkeypatch) -> None:
    pytest.importorskip("panorai.applications")
    from panorai.applications import SphericalRelativePose
    from panorai.features import SphericalFeaturePipeline

    class FeatureSet:
        features: tuple = ()

        @staticmethod
        def describe() -> dict:
            return {"count": 0}

    class Matches:
        @staticmethod
        def describe() -> dict:
            return {"count": 0}

        @staticmethod
        def __len__() -> int:
            return 0

    seen: dict[str, object] = {}
    feature_set = FeatureSet()
    matches = Matches()

    monkeypatch.setattr(
        SphericalFeaturePipeline,
        "extract",
        lambda self, image, *, panorama_id=None: feature_set,
    )
    monkeypatch.setattr(
        SphericalFeaturePipeline,
        "match",
        lambda self, reference, source: matches,
    )

    def estimate_matches(self, observed):
        seen["application"] = self
        seen["matches"] = observed
        return None

    monkeypatch.setattr(SphericalRelativePose, "estimate_matches", estimate_matches)

    image = np.zeros((16, 32, 3), dtype=np.uint8)
    payload = analyze_pair(image, image)

    assert isinstance(seen["application"], SphericalRelativePose)
    assert seen["matches"] is matches
    assert payload["execution"]["api_level"] == "applications"
    assert payload["execution"]["public_api"] == (
        "panorai.applications.SphericalRelativePose"
    )
