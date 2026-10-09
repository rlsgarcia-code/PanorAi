from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmarks.spherical_monocular_depth.da3_sphere import (
    FeatureStoreSpec,
    make_spherical_overlap_plan,
    share_feature_store,
)
from benchmarks.spherical_monocular_depth.vit_tangent import make_native_tangent_plan


def _write_store(path: Path, spec: FeatureStoreSpec, values: np.ndarray) -> None:
    store = np.memmap(path, dtype=spec.dtype, mode="w+", shape=spec.shape)
    store[...] = values
    store.flush()


def test_single_view_full_sharing_is_identity(tmp_path: Path) -> None:
    base = make_native_tangent_plan(
        (56, 112), view_shape_hw=(28, 28), overlap_fraction=0.25
    )
    plan = type(base)(
        erp_shape_hw=base.erp_shape_hw,
        view_shape_hw=base.view_shape_hw,
        focal_px=base.focal_px,
        hfov_deg=base.hfov_deg,
        vfov_deg=base.vfov_deg,
        overlap_fraction=base.overlap_fraction,
        minimum_latitude_deg=base.minimum_latitude_deg,
        maximum_latitude_deg=base.maximum_latitude_deg,
        centers_lat_lon_deg=((0.0, 0.0),),
    )
    overlap = make_spherical_overlap_plan(plan, feature_shape_hw=(2, 2))
    spec = FeatureStoreSpec(1, 2, (2, 2), 3)
    values = np.arange(np.prod(spec.shape), dtype=np.float32).reshape(spec.shape)
    source = tmp_path / "source.dat"
    target = tmp_path / "target.dat"
    _write_store(source, spec, values)
    share_feature_store(source, target, spec, overlap, alpha=1.0)
    observed = np.memmap(target, dtype=spec.dtype, mode="r", shape=spec.shape)
    np.testing.assert_allclose(observed, values, rtol=0.0, atol=1e-5)


def test_zero_alpha_preserves_features_with_multiple_views(tmp_path: Path) -> None:
    plan = make_native_tangent_plan(
        (56, 112), view_shape_hw=(28, 28), overlap_fraction=0.25
    )
    overlap = make_spherical_overlap_plan(
        plan, feature_shape_hw=(2, 2), maximum_sources_per_target=4
    )
    spec = FeatureStoreSpec(len(plan.centers_lat_lon_deg), 1, (2, 2), 2)
    rng = np.random.default_rng(7)
    values = rng.normal(size=spec.shape).astype(np.float32)
    source = tmp_path / "source.dat"
    target = tmp_path / "target.dat"
    _write_store(source, spec, values)
    share_feature_store(source, target, spec, overlap, alpha=0.0)
    observed = np.memmap(target, dtype=spec.dtype, mode="r", shape=spec.shape)
    np.testing.assert_array_equal(observed, values)


def test_overlap_consensus_preserves_constant_features(tmp_path: Path) -> None:
    plan = make_native_tangent_plan(
        (56, 112), view_shape_hw=(28, 28), overlap_fraction=0.25
    )
    overlap = make_spherical_overlap_plan(
        plan, feature_shape_hw=(2, 2), maximum_sources_per_target=4
    )
    spec = FeatureStoreSpec(len(plan.centers_lat_lon_deg), 2, (2, 2), 3)
    values = np.full(spec.shape, 3.25, dtype=np.float32)
    source = tmp_path / "source.dat"
    target = tmp_path / "target.dat"
    _write_store(source, spec, values)
    report = share_feature_store(source, target, spec, overlap, alpha=1.0)
    observed = np.memmap(target, dtype=spec.dtype, mode="r", shape=spec.shape)
    np.testing.assert_allclose(observed, values, rtol=0.0, atol=2e-5)
    assert report["minimum_weight_denominator"] > 0.0


def test_longitude_seam_views_communicate() -> None:
    base = make_native_tangent_plan(
        (56, 112), view_shape_hw=(28, 28), overlap_fraction=0.25
    )
    plan = type(base)(
        erp_shape_hw=base.erp_shape_hw,
        view_shape_hw=base.view_shape_hw,
        focal_px=base.focal_px,
        hfov_deg=40.0,
        vfov_deg=40.0,
        overlap_fraction=base.overlap_fraction,
        minimum_latitude_deg=base.minimum_latitude_deg,
        maximum_latitude_deg=base.maximum_latitude_deg,
        centers_lat_lon_deg=((0.0, -179.0), (0.0, 179.0)),
    )
    overlap = make_spherical_overlap_plan(
        plan,
        feature_shape_hw=(4, 4),
        maximum_sources_per_target=2,
        minimum_overlap_fraction=0.01,
    )
    assert [[item.source_index for item in target] for target in overlap.targets] == [
        [0, 1],
        [0, 1],
    ]


def test_overlap_plan_rejects_invalid_controls() -> None:
    plan = make_native_tangent_plan((56, 112), view_shape_hw=(28, 28))
    with pytest.raises(ValueError, match="maximum_sources"):
        make_spherical_overlap_plan(
            plan, feature_shape_hw=(2, 2), maximum_sources_per_target=0
        )
    with pytest.raises(ValueError, match="gaussian_exponent"):
        make_spherical_overlap_plan(
            plan, feature_shape_hw=(2, 2), gaussian_exponent=0.0
        )
