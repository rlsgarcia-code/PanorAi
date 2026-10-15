from __future__ import annotations

from collections import deque
import math
import warnings

import numpy as np
import pytest

from panorai.image_processing import spherical_slic
from panorai.image_processing._slic import (
    _assign_labels,
    _erp_rays,
    _solid_angle_weights,
)


def _is_periodically_connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if not len(points):
        return False
    height, width = mask.shape
    start = tuple(int(value) for value in points[0])
    seen = {start}
    queue = deque([start])
    while queue:
        y, x = queue.popleft()
        neighbours = [(y, (x - 1) % width), (y, (x + 1) % width)]
        if y > 0:
            neighbours.append((y - 1, x))
        if y + 1 < height:
            neighbours.append((y + 1, x))
        for neighbour in neighbours:
            if mask[neighbour] and neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return len(seen) == int(np.count_nonzero(mask))


def test_spherical_slic_is_deterministic_connected_and_area_balanced() -> None:
    shape = (48, 96)
    image = np.full((*shape, 3), 0.5, dtype=np.float32)

    first = spherical_slic(image, n_segments=12, max_num_iter=15, min_size_factor=0.25)
    second = spherical_slic(image, n_segments=12, max_num_iter=15, min_size_factor=0.25)

    np.testing.assert_array_equal(first, second)
    assert first.shape == shape
    assert first.dtype == np.int32
    assert len(np.unique(first)) == 12
    assert all(_is_periodically_connected(first == label) for label in np.unique(first))

    weights = _solid_angle_weights(shape)
    assert math.isclose(float(weights.sum()), 4.0 * math.pi, rel_tol=0.0, abs_tol=1e-12)
    areas = np.asarray([weights[first == label].sum() for label in np.unique(first)])
    assert areas.max() / areas.min() < 1.15


def test_spherical_slic_keeps_a_colored_object_connected_across_erp_seam() -> None:
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    image[..., 2] = 96
    image[:, :6] = (255, 0, 0)
    image[:, -6:] = (255, 0, 0)

    labels = spherical_slic(
        image,
        n_segments=8,
        compactness=2.0,
        max_num_iter=15,
        min_size_factor=0.25,
        start_label=0,
    )

    np.testing.assert_array_equal(labels[:, 0], labels[:, -1])
    equator_label = labels[image.shape[0] // 2, 0]
    assert _is_periodically_connected(labels == equator_label)
    assert labels[image.shape[0] // 2, image.shape[1] // 2] != equator_label


def test_spherical_slic_mask_excludes_invalid_pixels_and_preserves_start_label() -> (
    None
):
    y, x = np.indices((30, 60))
    image = np.stack(
        (
            x / 59.0,
            y / 29.0,
            np.full_like(x, 0.25, dtype=np.float64),
        ),
        axis=-1,
    ).astype(np.float32)
    mask = np.ones(image.shape[:2], dtype=bool)
    mask[:5] = False
    mask[-4:] = False
    mask[12:15, 22:29] = False

    labels = spherical_slic(
        image,
        n_segments=9,
        compactness=6.0,
        max_num_iter=12,
        start_label=7,
        mask=mask,
    )

    np.testing.assert_array_equal(labels[~mask], -1)
    assert labels[mask].min() == 7
    assert all(
        _is_periodically_connected(labels == label) for label in np.unique(labels[mask])
    )


def test_spherical_slic_without_connectivity_retains_requested_cluster_ids() -> None:
    image = np.random.default_rng(4).random((20, 40, 3), dtype=np.float32)
    labels = spherical_slic(
        image,
        n_segments=7,
        compactness=5.0,
        max_num_iter=3,
        enforce_connectivity=False,
        start_label=3,
    )

    assert labels.min() >= 3
    assert labels.max() <= 9
    assert labels.dtype == np.int32


def test_small_disconnected_antipodal_mask_is_not_treated_as_a_local_cap() -> None:
    image = np.full((16, 32, 3), 0.5, dtype=np.float32)
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[(2, 13, 2, 13), (3, 19, 19, 3)] = True

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        labels = spherical_slic(
            image,
            n_segments=4,
            max_num_iter=2,
            enforce_connectivity=False,
            start_label=0,
            mask=mask,
        )

    assert not [item for item in caught if issubclass(item.category, RuntimeWarning)]
    assert len(np.unique(labels[mask])) == 4
    np.testing.assert_array_equal(labels[~mask], -1)


def test_spherical_assignment_is_invariant_under_a_common_3d_rotation() -> None:
    rng = np.random.default_rng(8)
    rays = rng.normal(size=(80, 3))
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    centers = rays[[2, 17, 31, 48, 67]]
    colors = rng.normal(size=(80, 3))
    center_colors = colors[[2, 17, 31, 48, 67]]
    angle = np.deg2rad(73.0)
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )

    original = _assign_labels(
        colors,
        rays,
        center_colors,
        centers,
        compactness=7.0,
        angular_scale=0.8,
    )
    rotated = _assign_labels(
        colors,
        rays @ rotation.T,
        center_colors,
        centers @ rotation.T,
        compactness=7.0,
        angular_scale=0.8,
    )

    np.testing.assert_array_equal(rotated, original)


def test_spatial_distance_is_intrinsic_not_an_erp_or_gnomonic_pixel_distance() -> None:
    rays = _erp_rays((18, 36))
    equator_row = 8
    polar_row = 0
    left_column = 10
    right_column = 14

    equator_distance = np.arccos(
        np.clip(
            rays[equator_row, left_column] @ rays[equator_row, right_column],
            -1.0,
            1.0,
        )
    )
    polar_distance = np.arccos(
        np.clip(
            rays[polar_row, left_column] @ rays[polar_row, right_column],
            -1.0,
            1.0,
        )
    )

    # The raster displacement is four pixels in both cases. A planar ERP or
    # per-view pixel metric would make them equal; intrinsic spherical distance
    # correctly contracts the longitude arc near the pole.
    assert polar_distance < 0.12 * equator_distance


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda: spherical_slic(np.zeros((8, 16))), ValueError, "HWC"),
        (lambda: spherical_slic(np.zeros((8, 16, 4))), ValueError, "three RGB"),
        (
            lambda: spherical_slic(np.full((8, 16, 3), np.nan)),
            ValueError,
            "finite",
        ),
        (
            lambda: spherical_slic(np.full((8, 16, 3), 2.0)),
            ValueError,
            r"\[0, 1\]",
        ),
        (
            lambda: spherical_slic(np.zeros((8, 16, 3)), n_segments=0),
            ValueError,
            "positive integer",
        ),
        (
            lambda: spherical_slic(np.zeros((8, 16, 3)), compactness=-1),
            ValueError,
            "non-negative",
        ),
        (
            lambda: spherical_slic(
                np.zeros((8, 16, 3)), mask=np.ones((8, 16), dtype=np.uint8)
            ),
            TypeError,
            "boolean",
        ),
        (
            lambda: spherical_slic(
                np.zeros((8, 16, 3)), mask=np.zeros((8, 16), dtype=bool)
            ),
            ValueError,
            "at least one",
        ),
        (
            lambda: spherical_slic(np.zeros((8, 16, 3)), enforce_connectivity="yes"),
            TypeError,
            "boolean",
        ),
        (
            lambda: spherical_slic(
                np.zeros((8, 16, 3)), start_label=np.iinfo(np.int32).max
            ),
            ValueError,
            "too large",
        ),
    ],
)
def test_spherical_slic_rejects_invalid_public_inputs(call, error, message) -> None:
    with pytest.raises(error, match=message):
        call()
