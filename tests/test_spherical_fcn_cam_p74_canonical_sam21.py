from __future__ import annotations

import numpy as np

from benchmarks.spherical_fcn_cam.run_p74_exhaustive_segmentation import (
    _native_from_canonical,
)


def test_p74_native_backprojection_uses_polar_limit_not_erp_rows() -> None:
    canonical_height = 180
    canonical_width = 360
    canonical_rows = np.broadcast_to(
        np.arange(canonical_height, dtype=np.uint16)[:, None],
        (canonical_height, canonical_width),
    )

    native = _native_from_canonical(canonical_rows, (101, 201))

    assert native[0, 0] == 0
    # The P74 raster stops at 150 degrees polar angle.  Treating it as an ERP
    # would incorrectly map the final row to 179 instead of 150.
    assert native[-1, 0] == 150


def test_p74_native_longitude_is_reversed_and_wraps_at_both_edges() -> None:
    canonical = np.broadcast_to(np.arange(360, dtype=np.uint16)[None, :], (180, 360))

    native = _native_from_canonical(canonical, (101, 201))

    assert native[50, 0] == 180
    assert native[50, -1] == 180
    assert native[50, 100] == 0
    assert native[50, 50] == 90
    assert native[50, 150] == 270


def test_p74_label_backprojection_preserves_discrete_ids() -> None:
    labels = np.zeros((180, 360), dtype=np.uint16)
    labels[:, :120] = 3
    labels[:, 120:240] = 7
    labels[:, 240:] = 11

    native = _native_from_canonical(labels, (101, 201))

    assert set(np.unique(native)) == {3, 7, 11}
