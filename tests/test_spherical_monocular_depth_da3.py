from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarks.spherical_monocular_depth.da3_tangent import (
    CANONICAL_FOCAL_PX,
    DA3MetricTangentInference,
    infer_native_tangent_erp,
)
from benchmarks.spherical_monocular_depth.vit_tangent import make_native_tangent_plan


def test_da3_metric_rule_and_axial_to_radial_are_both_applied() -> None:
    torch = pytest.importorskip("torch")

    class ConstantModel:
        def __call__(self, tensor):
            _, views, _, height, width = tensor.shape
            raw = CANONICAL_FOCAL_PX / 1000.0
            return {"depth": torch.full((1, views, height, width), raw)}

    plan = make_native_tangent_plan(
        (56, 112), view_shape_hw=(28, 28), overlap_fraction=0.25
    )
    adapter = DA3MetricTangentInference(ConstantModel(), plan, device="cpu")
    radial, valid = adapter(np.full((28, 28, 3), 127, dtype=np.uint8))
    axial_m = plan.focal_px / 1000.0
    x = (0.5 - 14.0) / plan.focal_px
    y = (0.5 - 14.0) / plan.focal_px
    assert radial[0, 0] == pytest.approx(
        axial_m * math.sqrt(1.0 + x * x + y * y), rel=1e-6
    )
    assert valid.all()


def test_da3_adapter_uses_b_n_c_h_w_and_imagenet_normalization() -> None:
    torch = pytest.importorskip("torch")

    class InspectModel:
        def __call__(self, tensor):
            assert tensor.shape == (1, 1, 3, 28, 28)
            expected = torch.tensor(
                [(1.0 - 0.485) / 0.229, (1.0 - 0.456) / 0.224, (1.0 - 0.406) / 0.225]
            )
            torch.testing.assert_close(tensor[0, 0, :, 0, 0], expected)
            return {"depth": torch.ones((1, 1, 28, 28))}

    plan = make_native_tangent_plan((56, 112), view_shape_hw=(28, 28))
    DA3MetricTangentInference(InspectModel(), plan, device="cpu")(
        np.full((28, 28, 3), 255, dtype=np.uint8)
    )


def test_da3_adapter_rejects_non_patch_aligned_shape() -> None:
    plan = make_native_tangent_plan((56, 112), view_shape_hw=(28, 28))
    object.__setattr__(plan, "view_shape_hw", (29, 28))
    with pytest.raises(ValueError, match="patch size 14"):
        DA3MetricTangentInference(object(), plan, device="cpu")


def test_shadow_support_never_injects_nan_into_da3_input() -> None:
    torch = pytest.importorskip("torch")

    class FiniteModel:
        def __call__(self, tensor):
            assert torch.isfinite(tensor).all()
            batch, views, _, height, width = tensor.shape
            return {"depth": torch.ones((batch, views, height, width))}

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
