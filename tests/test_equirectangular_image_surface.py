from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

import panorai as pa
from panorai.data import DataFactory, RasterFieldSpec, SphericalTranspose
from panorai.geometry import GnomonicSpec
from panorai.ops.spherical import GaussianBlur


def _rgb(height: int = 8, width: int = 16) -> np.ndarray:
    rows, columns = np.indices((height, width))
    return np.stack(
        (
            columns * 255 // max(1, width - 1),
            rows * 255 // max(1, height - 1),
            (rows + columns) * 255 // max(1, height + width - 2),
        ),
        axis=-1,
    ).astype(np.uint8)


def test_image_factories_own_pixels_and_expose_pillow_style_properties(
    tmp_path,
) -> None:
    source = _rgb()
    path = tmp_path / "source.png"
    Image.fromarray(source).save(path)

    panorama = pa.EquirectangularImage.open(path)
    source[:] = 0

    assert panorama.size == (16, 8)
    assert panorama.shape_hw == (8, 16)
    assert panorama.width == 16
    assert panorama.height == 8
    assert panorama.mode == "RGB"
    assert panorama.format == "PNG"
    assert panorama.backend == "numpy"
    assert panorama.dtype == np.uint8
    assert panorama.image.max() == 255

    created = pa.EquirectangularImage.new("RGB", (12, 6), (10, 20, 30))
    assert created.size == (12, 6)
    np.testing.assert_array_equal(created.image[0, 0], (10, 20, 30))

    chw = np.moveaxis(_rgb(6, 12), -1, 0)
    from_chw = pa.EquirectangularImage.fromarray(chw, layout="CHW")
    assert from_chw.image.shape == (6, 12, 3)
    chw[:] = 0
    assert from_chw.image.max() == 255


@pytest.mark.parametrize("pattern", ("spherical-grid", "directions", "checkerboard"))
def test_sample_patterns_are_deterministic_owned_spherical_fixtures(pattern) -> None:
    first = pa.EquirectangularImage.sample(pattern, size=(24, 12))
    second = pa.EquirectangularImage.sample(pattern, size=(24, 12))

    assert first.size == (24, 12)
    assert first.mode == "RGB"
    assert first.dtype == np.uint8
    assert first.backend == "numpy"
    assert first.support("image").all()
    assert first.validity("image").all()
    np.testing.assert_array_equal(first.image, second.image)

    first.image[:] = 0
    assert second.image.max() > 0


def test_direction_sample_encodes_canonical_pixel_centre_rays() -> None:
    from panorai.geometry import erp_pixels_to_rays

    width, height = 8, 4
    sample = pa.EquirectangularImage.sample("directions", size=(width, height))
    rows, columns = np.indices((height, width), dtype=np.float64)
    pixels_xy = np.stack((columns, rows), axis=-1)
    rays = erp_pixels_to_rays(pixels_xy, (height, width))
    expected = np.rint((rays + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    np.testing.assert_array_equal(sample.image, expected)


def test_sample_rejects_unknown_or_non_string_patterns() -> None:
    with pytest.raises(ValueError, match="pattern must be one of"):
        pa.EquirectangularImage.sample("photograph")
    with pytest.raises(TypeError, match="pattern must be a string"):
        pa.EquirectangularImage.sample(None)  # type: ignore[arg-type]


def test_constant_creates_scalar_and_channel_images_in_geometry_order() -> None:
    scalar = pa.EquirectangularImage.constant(2.5, height=3, width=6)
    rgb = pa.EquirectangularImage.constant(
        (10, 20, 30), height=2, width=4, dtype=np.uint8
    )

    assert scalar.image.shape == (3, 6)
    assert scalar.size == (6, 3)
    assert scalar.dtype == np.float32
    assert scalar.mode == "F"
    np.testing.assert_array_equal(scalar.image, np.full((3, 6), 2.5, np.float32))

    assert rgb.image.shape == (2, 4, 3)
    assert rgb.size == (4, 2)
    assert rgb.dtype == np.uint8
    assert rgb.mode == "RGB"
    np.testing.assert_array_equal(rgb.image[0, 0], (10, 20, 30))
    assert rgb.support("image").all()
    assert rgb.validity("image").all()


def test_constant_preserves_explicit_support_and_validity() -> None:
    support = np.ones((3, 6), dtype=bool)
    support[:, 0] = False
    valid = np.ones((3, 6), dtype=bool)
    valid[1, 2] = False

    panorama = pa.EquirectangularImage.constant(
        0.0,
        height=3,
        width=6,
        support_mask=support,
        valid=valid,
    )

    np.testing.assert_array_equal(panorama.support("image"), support)
    np.testing.assert_array_equal(panorama.validity("image"), valid)


@pytest.mark.parametrize(
    ("value", "height", "width", "dtype", "error", "message"),
    (
        (0.0, 0, 4, np.float32, ValueError, "at least 2"),
        (0.0, 2, True, np.float32, TypeError, "integer pair"),
        ((), 2, 4, np.float32, ValueError, "non-empty channel vector"),
        ([[1.0]], 2, 4, np.float32, ValueError, "scalar or a non-empty"),
        (float("nan"), 2, 4, np.float32, ValueError, "finite"),
        (1 + 2j, 2, 4, np.float32, TypeError, "real numeric"),
        (1.0, 2, 4, object, TypeError, "real numeric"),
    ),
)
def test_constant_rejects_invalid_dimensions_values_and_dtypes(
    value, height, width, dtype, error, message
) -> None:
    with pytest.raises(error, match=message):
        pa.EquirectangularImage.constant(
            value,
            height=height,
            width=width,
            dtype=dtype,
        )


def test_save_and_data_factory_use_the_owned_equirectangular_surface(tmp_path) -> None:
    panorama = pa.EquirectangularImage.fromarray(_rgb())
    path = tmp_path / "roundtrip.png"
    panorama.save(path)

    reopened = pa.EquirectangularImage.open(path)
    through_factory = DataFactory.from_file(str(path), "equirectangular")
    np.testing.assert_array_equal(reopened.image, panorama.image)
    np.testing.assert_array_equal(through_factory.image, panorama.image)


def test_copy_and_image_style_transforms_are_immutable() -> None:
    panorama = pa.EquirectangularImage.fromarray(_rgb())
    copied = panorama.copy()
    thumbnail = panorama.thumbnail((8, 8))

    copied.image[:] = 0
    assert panorama.image.max() == 255
    assert thumbnail.size == (8, 4)
    assert panorama.size == (16, 8)
    already_small = panorama.thumbnail((64, 64))
    assert already_small is not panorama
    np.testing.assert_array_equal(already_small.image, panorama.image)
    extreme = pa.EquirectangularImage.new("L", (100, 2))
    with pytest.raises(ValueError, match="cannot preserve this aspect ratio"):
        extreme.thumbnail((2, 2))


def test_resize_transforms_all_fields_with_modality_specific_sampling() -> None:
    rgb = _rgb()
    depth = np.linspace(1.0, 3.0, 8 * 16, dtype=np.float32).reshape(8, 16)
    depth_valid = np.ones((8, 16), dtype=bool)
    depth_valid[2:4, 4:6] = False
    depth[~depth_valid] = np.nan
    labels = (np.indices((8, 16))[1] % 4).astype(np.int16)
    support = np.ones((8, 16), dtype=bool)
    support[:, 0] = False
    panorama = (
        pa.EquirectangularImage(rgb, support_mask=support)
        .with_depth(depth, valid=depth_valid, units="m")
        .with_labels(labels)
    )

    resized = panorama.resize((8, 4))

    assert resized.image.shape == (4, 8, 3)
    assert resized.depth.shape == resized.labels.shape == (4, 8)
    assert resized.image.dtype == np.uint8
    assert resized.depth.dtype == np.float32
    assert resized.labels.dtype == np.int16
    assert set(np.unique(resized.labels)).issubset(set(np.unique(labels)))
    assert resized.support("image").shape == (4, 8)
    assert resized.validity("depth").shape == (4, 8)
    assert resized.field_spec("labels").interpolation == "nearest"
    assert panorama.image.shape == (8, 16, 3)
    assert np.isnan(panorama.depth[~depth_valid]).all()


def test_yaw_rotation_is_an_exact_longitude_shift_with_nearest_sampling() -> None:
    height, width = 10, 20
    values = np.broadcast_to(np.arange(width), (height, width)).astype(np.int16)
    panorama = pa.EquirectangularImage.from_field(
        "labels",
        values,
        spec=RasterFieldSpec.categorical(),
    )

    rotated = panorama.rotate(
        yaw_deg=360.0 / width,
        interpolation="nearest",
    )

    np.testing.assert_array_equal(rotated.labels, np.roll(values, 1, axis=1))
    with pytest.raises(ValueError, match=r"shape \(3, 3\)"):
        panorama.rotate(45)


def test_named_transposes_have_spherical_not_array_axis_semantics() -> None:
    values = np.arange(6 * 10).reshape(6, 10).astype(np.int16)
    panorama = pa.EquirectangularImage(values)

    longitude = panorama.transpose(SphericalTranspose.FLIP_LONGITUDE)
    latitude = panorama.transpose("flip-latitude")
    antipodal = panorama.transpose(pa.SphericalTranspose.ANTIPODAL)

    np.testing.assert_array_equal(longitude.image, values[:, ::-1])
    np.testing.assert_array_equal(latitude.image, values[::-1])
    np.testing.assert_array_equal(
        antipodal.image, np.roll(values[::-1], values.shape[1] // 2, axis=1)
    )
    assert longitude.shape == panorama.shape
    with pytest.raises(ValueError, match="transpose method"):
        panorama.transpose("rotate-90")


def test_crop_returns_typed_face_and_paste_reprojects_without_mutation() -> None:
    rgb = _rgb(12, 24)
    panorama = pa.EquirectangularImage(rgb)
    spec = GnomonicSpec(
        center_lat_deg=15.0,
        center_lon_deg=35.0,
        hfov_deg=80.0,
        vfov_deg=60.0,
        output_shape_hw=(7, 9),
    )

    face = panorama.crop(spec)
    pasted = panorama.paste(face)

    assert face.spec == spec
    assert face.image.shape == (7, 9, 3)
    assert face._erp_shape_hw == panorama.shape_hw
    assert pasted.image.shape == rgb.shape
    assert pasted.support("image").all()
    assert pasted.validity("image").shape == panorama.shape_hw
    np.testing.assert_array_equal(panorama.image, rgb)
    with pytest.raises(TypeError, match="GnomonicFace"):
        panorama.paste(face.image)


def test_alpha_composite_is_explicit_and_rejects_categorical_fields() -> None:
    base = pa.EquirectangularImage.new("RGB", (24, 12), (0, 0, 0))
    face = pa.EquirectangularImage.new("RGB", (24, 12), (200, 100, 50)).crop(
        GnomonicSpec(output_shape_hw=(7, 9))
    )
    mixed = base.composite(face, blend="alpha", alpha=0.25)
    region = mixed.image[..., 0] > 0

    assert region.any()
    assert set(np.unique(mixed.image[..., 0])).issubset({0, 50})
    with pytest.raises(ValueError, match="interval"):
        base.composite(face, blend="alpha", alpha=1.5)

    labels = np.zeros((12, 24), dtype=np.int16)
    typed = base.with_labels(labels)
    typed_face = typed.crop(GnomonicSpec(output_shape_hw=(7, 9)))
    with pytest.raises(TypeError, match="categorical"):
        typed.composite(typed_face, blend="alpha", alpha=0.5)


def test_filter_delegates_to_registered_spherical_operation() -> None:
    values = np.zeros((9, 18), dtype=np.float32)
    values[:, 0] = 1.0
    panorama = pa.EquirectangularImage(values)
    operation = GaussianBlur(ksize=3, sigma=0.8, backend="numpy")

    filtered = panorama.filter(operation)
    expected = operation(values)

    np.testing.assert_array_equal(filtered.image, expected)
    assert filtered.support("image").shape == values.shape
    np.testing.assert_array_equal(panorama.image, values)


def test_float_export_requires_an_explicit_safe_range(tmp_path) -> None:
    values = np.linspace(0.0, 1.0, 8 * 16 * 3, dtype=np.float32).reshape(8, 16, 3)
    panorama = pa.EquirectangularImage(values)
    exported = panorama.to_pil()
    assert exported.mode == "RGB"
    assert exported.size == (16, 8)

    path = tmp_path / "float.png"
    panorama.save(path)
    assert Image.open(path).mode == "RGB"

    bad = pa.EquirectangularImage(values * 2.0)
    with pytest.raises(ValueError, match="outside value_range"):
        bad.to_pil()


def test_torch_resize_and_rotation_preserve_layout_device_and_autograd() -> None:
    torch = pytest.importorskip("torch")
    values = torch.arange(8 * 16, dtype=torch.float64).reshape(1, 8, 16)
    values.requires_grad_()
    panorama = pa.EquirectangularImage(values)

    resized = panorama.resize((8, 4))
    rotated = panorama.rotate(yaw_deg=22.5, interpolation="nearest")
    loss = resized.image.sum() + rotated.image.sum()
    loss.backward()

    assert tuple(resized.image.shape) == (1, 4, 8)
    assert tuple(rotated.image.shape) == tuple(values.shape)
    assert resized.image.dtype == values.dtype
    assert resized.image.device == values.device
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


def test_euler_yaw_matches_the_explicit_canonical_matrix() -> None:
    values = np.arange(8 * 16, dtype=np.float64).reshape(8, 16)
    yaw = math.radians(45.0)
    matrix = np.asarray(
        (
            (math.cos(yaw), 0.0, math.sin(yaw)),
            (0.0, 1.0, 0.0),
            (-math.sin(yaw), 0.0, math.cos(yaw)),
        )
    )
    panorama = pa.EquirectangularImage(values)
    named = panorama.rotate(yaw_deg=45.0, interpolation="nearest")
    explicit = panorama.rotate(matrix, interpolation="nearest")
    np.testing.assert_array_equal(named.image, explicit.image)
