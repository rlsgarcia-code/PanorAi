from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from benchmarks.spherical_monocular_depth.da3_canonical_atlas import (
    _remap_channels,
    canonicalize_feature_store,
)
from benchmarks.spherical_monocular_depth.da3_sphere import FeatureStoreSpec
from benchmarks.spherical_monocular_depth.vit_tangent import make_native_tangent_plan


def _write_store(path: Path, spec: FeatureStoreSpec, values: np.ndarray) -> None:
    store = np.memmap(path, dtype=spec.dtype, mode="w+", shape=spec.shape)
    store[...] = values
    store.flush()


def _duplicate_view_plan() -> object:
    base = make_native_tangent_plan(
        (56, 112), view_shape_hw=(56, 84), overlap_fraction=0.25
    )
    return type(base)(
        erp_shape_hw=base.erp_shape_hw,
        view_shape_hw=base.view_shape_hw,
        focal_px=base.focal_px,
        hfov_deg=base.hfov_deg,
        vfov_deg=base.vfov_deg,
        overlap_fraction=base.overlap_fraction,
        minimum_latitude_deg=base.minimum_latitude_deg,
        maximum_latitude_deg=base.maximum_latitude_deg,
        centers_lat_lon_deg=((0.0, 15.0), (0.0, 15.0)),
    )


def test_erp_remap_wraps_longitude_for_multichannel_values() -> None:
    values = np.stack(
        (np.arange(4, dtype=np.float32), 10.0 + np.arange(4, dtype=np.float32)),
        axis=-1,
    )[None]
    observed = _remap_channels(
        values,
        np.array([-0.25, 3.75]),
        np.array([0.0, 0.0]),
        border_mode=cv2.BORDER_WRAP,
    )
    np.testing.assert_allclose(observed[0], observed[1], rtol=0.0, atol=1e-6)


def test_remap_preserves_all_1024_feature_channels() -> None:
    values = np.arange(2 * 3 * 1024, dtype=np.float32).reshape(2, 3, 1024)
    observed = _remap_channels(
        values,
        np.array([1.0]),
        np.array([0.0]),
        border_mode=cv2.BORDER_REPLICATE,
    )
    assert observed.shape == (1, 1024)
    np.testing.assert_array_equal(observed[0], values[0, 1])


def test_canonical_atlas_broadcasts_one_field_to_duplicate_views(
    tmp_path: Path,
) -> None:
    plan = _duplicate_view_plan()
    spec = FeatureStoreSpec(2, 2, (4, 6), 3)
    values = np.empty(spec.shape, dtype=np.float32)
    values[0] = 1.0
    values[1] = 3.0
    source = tmp_path / "source.dat"
    target = tmp_path / "target.dat"
    atlas = tmp_path / "atlas.dat"
    support = tmp_path / "support.dat"
    _write_store(source, spec, values)

    report = canonicalize_feature_store(
        source,
        target,
        atlas,
        support,
        spec,
        plan,
        atlas_shape_hw=(16, 32),
    )
    observed = np.memmap(target, dtype=spec.dtype, mode="r", shape=spec.shape)
    np.testing.assert_allclose(
        observed[0, :, 1:-1, 1:-1],
        observed[1, :, 1:-1, 1:-1],
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        observed[:, :, 1:-1, 1:-1], 2.0, rtol=0.0, atol=1e-5
    )
    assert report["learned_parameters_added"] == 0
    assert report["atlas_shape_hw"] == [16, 32]
    assert "one canonical ERP" in report["same_ray_contract"]


def test_canonical_atlas_preserves_constant_features_including_fallback(
    tmp_path: Path,
) -> None:
    plan = _duplicate_view_plan()
    spec = FeatureStoreSpec(2, 1, (4, 6), 2)
    values = np.full(spec.shape, 7.25, dtype=np.float32)
    source = tmp_path / "source.dat"
    target = tmp_path / "target.dat"
    _write_store(source, spec, values)
    canonicalize_feature_store(
        source,
        target,
        tmp_path / "atlas.dat",
        tmp_path / "support.dat",
        spec,
        plan,
        atlas_shape_hw=(12, 24),
    )
    observed = np.memmap(target, dtype=spec.dtype, mode="r", shape=spec.shape)
    np.testing.assert_allclose(observed, values, rtol=0.0, atol=1e-5)


def test_canonical_atlas_rejects_non_erp_shape(tmp_path: Path) -> None:
    plan = _duplicate_view_plan()
    spec = FeatureStoreSpec(2, 1, (4, 6), 1)
    source = tmp_path / "source.dat"
    _write_store(source, spec, np.ones(spec.shape, dtype=np.float32))
    with pytest.raises(ValueError, match="2:1 ERP"):
        canonicalize_feature_store(
            source,
            tmp_path / "target.dat",
            tmp_path / "atlas.dat",
            tmp_path / "support.dat",
            spec,
            plan,
            atlas_shape_hw=(12, 25),
        )
