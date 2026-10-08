from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from panorai.data import XYZImageDataset
from panorai.data import xyz_image_dataset as xyz_module


def test_xyz_image_dataset_requires_explicit_source_and_frame() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        XYZImageDataset(coordinate_frame="scanner-local")
    with pytest.raises(ValueError, match="coordinate_frame"):
        XYZImageDataset(files=["sample_2x2.ply"], coordinate_frame="")


def test_xyz_image_dataset_reports_geometry_and_validity(monkeypatch) -> None:
    xyz = np.asarray(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            [[0.0, 0.0, 0.0], [np.nan, 0.0, 1.0]],
        ],
        dtype=np.float64,
    )
    rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)

    monkeypatch.setattr(
        xyz_module,
        "read_ply_xyz_image",
        lambda path: (xyz.copy(), rgb.copy()),
    )
    dataset = XYZImageDataset(
        files=[Path("sample_2x2.ply")],
        coordinate_frame="scanner-local-right-up-forward",
        units="metres",
        shadow_angle=30.0,
    )

    sample = dataset[0]

    np.testing.assert_array_equal(sample["rgb_image"], rgb)
    np.testing.assert_array_equal(sample["xyz_image"], xyz)
    np.testing.assert_allclose(
        sample["radial_depth"][:1], np.asarray([[1.0, 2.0]])
    )
    np.testing.assert_array_equal(
        sample["validity_mask"], np.asarray([[True, True], [False, False]])
    )
    assert sample["coordinate_frame"] == "scanner-local-right-up-forward"
    assert sample["units"] == "metres"
    assert sample["shadow_angle"] == 30.0
