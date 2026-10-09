from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarks.spherical_monocular_depth.vit_tangent import (
    axial_to_radial_tangent,
    infer_native_tangent_erp,
    make_native_tangent_plan,
    native_erp_focal_px,
)


def test_native_erp_focal_matches_pixels_per_radian() -> None:
    assert native_erp_focal_px((100, 200)) == pytest.approx(200 / (2 * math.pi))


def test_tangent_plan_is_native_density_and_deterministic() -> None:
    first = make_native_tangent_plan(
        (4128, 8256), view_shape_hw=(280, 504), overlap_fraction=0.25
    )
    second = make_native_tangent_plan(
        (4128, 8256), view_shape_hw=(280, 504), overlap_fraction=0.25
    )
    assert first == second
    assert first.focal_px == pytest.approx(8256 / (2 * math.pi))
    assert first.hfov_deg == pytest.approx(
        math.degrees(2 * math.atan(504 / (2 * first.focal_px)))
    )
    assert first.vfov_deg == pytest.approx(
        math.degrees(2 * math.atan(280 / (2 * first.focal_px)))
    )
    assert len(first.centers_lat_lon_deg) > 100
    assert min(lat for lat, _ in first.centers_lat_lon_deg) > -60.0
    assert max(lat for lat, _ in first.centers_lat_lon_deg) < 90.0


def test_tangent_plan_rejects_model_incompatible_shape() -> None:
    with pytest.raises(ValueError, match="divisible by 28"):
        make_native_tangent_plan((100, 200), view_shape_hw=(281, 504))


def test_axial_to_radial_uses_pixel_centres() -> None:
    axial = np.full((2, 2), 2.0, dtype=np.float32)
    radial = axial_to_radial_tangent(axial, focal_px=1.0)
    expected = 2.0 * math.sqrt(1.0 + 0.5**2 + 0.5**2)
    np.testing.assert_allclose(radial, expected, rtol=0.0, atol=1e-6)


def test_axial_to_radial_preserves_optical_axis_limit() -> None:
    axial = np.asarray([[3.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        axial_to_radial_tangent(axial, focal_px=1000.0), axial
    )


def test_shadow_support_never_injects_nan_into_vit_input() -> None:
    torch = pytest.importorskip("torch")

    class FiniteModel:
        def inference(self, data):
            assert torch.isfinite(data["input"]).all()
            batch, _, height, width = data["input"].shape
            depth = torch.ones((batch, 1, height, width), dtype=torch.float32)
            confidence = torch.ones_like(depth)
            return depth, confidence, {}

    plan = make_native_tangent_plan(
        (56, 112), view_shape_hw=(28, 28), overlap_fraction=0.25
    )
    rgb = np.full((56, 112, 3), 127, dtype=np.uint8)
    support = np.ones((56, 112), dtype=bool)
    support[47:] = False
    depth, validity, report = infer_native_tangent_erp(
        FiniteModel(), rgb, support, plan, device="cpu"
    )
    assert np.isfinite(depth[validity]).all()
    assert not validity[47:].any()
    assert report["view_count"] == len(plan.centers_lat_lon_deg)
