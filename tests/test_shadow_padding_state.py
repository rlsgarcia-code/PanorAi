import numpy as np
import pytest

from panorai.data import EquirectangularImage


def _observed_panorama() -> np.ndarray:
    image = np.zeros((150, 360, 3), dtype=np.uint8)
    image[:75, :, 0] = 37
    image[75:, :, 1] = 91
    image[60:65] = 0
    return image


def _materialized_panorama() -> np.ndarray:
    observed = _observed_panorama()
    return np.pad(observed, ((0, 30), (0, 0), (0, 0)))


def test_unpadded_shadow_is_materialized_once_with_explicit_support():
    panorama = EquirectangularImage(
        _observed_panorama(),
        shadow_angle=30.0,
        shadow_padded=False,
    )

    with pytest.raises(ValueError, match="call preprocess"):
        panorama.views(size=8)

    panorama.preprocess()

    assert panorama.shape == (180, 360, 3)
    assert panorama.shadow_angle == 30.0
    assert panorama.shadow_padded is True
    assert np.array_equal(panorama.image, _materialized_panorama())
    assert np.all(panorama.support_mask[:150])
    assert not np.any(panorama.support_mask[150:])
    assert not np.any(panorama._workflow_support["image"][150:])
    assert not np.any(panorama.validity("image")[150:])

    first_data = panorama.image.copy()
    first_support = panorama.support_mask.copy()
    panorama.preprocess()

    assert panorama.shape == (180, 360, 3)
    assert np.array_equal(panorama.image, first_data)
    assert np.array_equal(panorama.support_mask, first_support)


def test_already_padded_shadow_is_not_padded_again_and_matches_projection():
    from_observed = EquirectangularImage(
        _observed_panorama(), shadow_angle=30.0, shadow_padded=False
    )
    from_observed.preprocess()
    already_padded = EquirectangularImage(
        _materialized_panorama(), shadow_angle=30.0, shadow_padded=True
    )

    already_padded.preprocess()

    assert np.array_equal(already_padded.image, from_observed.image)
    assert np.array_equal(already_padded.support_mask, from_observed.support_mask)
    views_a = from_observed.views("cube", size=12, fov=90.0)
    views_b = already_padded.views("cube", size=12, fov=90.0)
    assert len(views_a) == len(views_b) == 6
    for face_a, face_b in zip(views_a, views_b):
        np.testing.assert_array_equal(face_a.image, face_b.image, strict=True)
        assert np.array_equal(face_a.support_mask, face_b.support_mask)


def test_black_pixels_inside_observed_region_remain_supported():
    panorama = EquirectangularImage(
        _materialized_panorama(), shadow_angle=30.0, shadow_padded=True
    )

    assert np.all(panorama.image[62] == 0)
    assert np.all(panorama.support_mask[62])
    assert not np.any(panorama.support_mask[150:])


def test_shadow_state_propagates_through_clone_and_modalities():
    image = _materialized_panorama().astype(np.float32)
    panorama = EquirectangularImage(
        image, shadow_angle=30.0, shadow_padded=True
    ).with_depth(np.ones(image.shape[:2], dtype=np.float32))
    panorama = panorama.with_labels(np.zeros(image.shape[:2], dtype=np.int32))

    clone = panorama.clone()

    for value in (panorama, clone):
        assert value.shadow_angle == 30.0
        assert value.shadow_padded is True
        assert not np.any(value.support_mask[150:])
        assert not np.any(value._workflow_support["image"][150:])
        assert not np.any(value._workflow_support["depth"][150:])
        assert not np.any(value._workflow_support["labels"][150:])


@pytest.mark.parametrize(
    ("shadow_angle", "shadow_padded", "error"),
    [
        (-1.0, False, ValueError),
        (180.0, False, ValueError),
        (float("nan"), False, ValueError),
        (0.0, True, ValueError),
        (30.0, "yes", TypeError),
    ],
)
def test_invalid_shadow_state_is_rejected(shadow_angle, shadow_padded, error):
    with pytest.raises(error):
        EquirectangularImage(
            np.zeros((4, 8, 3), dtype=np.uint8),
            shadow_angle=shadow_angle,
            shadow_padded=shadow_padded,
        )


def test_materialized_shadow_angle_cannot_be_changed_or_padded_twice():
    panorama = EquirectangularImage(
        _materialized_panorama(), shadow_angle=30.0, shadow_padded=True
    )

    with pytest.raises(ValueError, match="already shadow-padded"):
        panorama.preprocess(shadow_padded=False)
    with pytest.raises(ValueError, match="cannot change shadow_angle"):
        panorama.preprocess(shadow_angle=20.0)
