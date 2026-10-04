from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import numpy as np
import pytest

pytest.importorskip(
    "panorai._native._geometry", reason="optional geometry extension is not built"
)

from panorai.data.equirectangular_image import EquirectangularImage  # noqa: E402
from panorai.geometry import (  # noqa: E402
    CUBE_FACE_ORDER,
    GnomonicProjector,
    GnomonicSpec,
    cubemap_to_equirectangular,
)
from panorai.geometry import _engine, _native  # noqa: E402


def _faces(shape=(19, 23), *, dtype=np.float32, channels=3):
    result = {}
    for index, face in enumerate(CUBE_FACE_ORDER):
        values = np.random.default_rng(300 + index).random(
            (*shape, channels) if channels is not None else shape
        )
        result[face] = values.astype(dtype)
    return result


def _python_result(monkeypatch, faces, output_shape):
    extension = _native._native_geometry
    monkeypatch.setattr(_native, "_native_geometry", None)
    try:
        return cubemap_to_equirectangular(faces, output_shape).data.copy()
    finally:
        monkeypatch.setattr(_native, "_native_geometry", extension)


def _python_gnomonic_batch(monkeypatch, image, plan, interpolation):
    extension = _native._native_geometry
    monkeypatch.setattr(_native, "_native_geometry", None)
    try:
        return tuple(
            item.copy()
            for item in _engine._gnomonic_batch_from_equirectangular(
                image, plan, interpolation=interpolation
            )
        )
    finally:
        monkeypatch.setattr(_native, "_native_geometry", extension)


def _python_workflow(monkeypatch, image, valid, count):
    extension = _native._native_geometry
    monkeypatch.setattr(_native, "_native_geometry", None)
    try:
        views = EquirectangularImage(image, valid=valid).views(
            "fibonacci",
            count=count,
            size=(13, 19),
            fov=(107, 73),
        )
        result = views.reconstruct(blend="gaussian")
        return (
            tuple(face.image.copy() for face in views),
            result.image.copy(),
            result.validity("image"),
            result.support_mask.copy(),
        )
    finally:
        monkeypatch.setattr(_native, "_native_geometry", extension)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("channels", [None, 1, 4])
def test_native_cubemap_matches_selective_numpy_bit_exactly(
    monkeypatch, dtype, channels
) -> None:
    faces = _faces(dtype=dtype, channels=channels)
    output_shape = (47, 96)

    expected = _python_result(monkeypatch, faces, output_shape)
    actual = cubemap_to_equirectangular(faces, output_shape).data

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == dtype


def test_native_cubemap_preserves_seams_poles_and_canonical_ties(monkeypatch) -> None:
    faces = {}
    yy, xx = np.indices((31, 37), dtype=np.float64)
    for index, face in enumerate(CUBE_FACE_ORDER):
        faces[face] = np.stack(
            (
                np.full_like(xx, index),
                xx + index * 100.0,
                yy + index * 1000.0,
            ),
            axis=-1,
        )
    output_shape = (63, 128)

    expected = _python_result(monkeypatch, faces, output_shape)
    actual = cubemap_to_equirectangular(faces, output_shape).data

    np.testing.assert_array_equal(actual, expected)
    # The first/last ERP columns straddle the -pi/+pi seam and must both use
    # the canonical back face without an accidental horizontal face wrap.
    equator = output_shape[0] // 2
    assert np.all(actual[equator, (0, -1), 0] == CUBE_FACE_ORDER.index("back"))
    assert actual[0, output_shape[1] // 2, 0] == CUBE_FACE_ORDER.index("up")
    assert actual[-1, output_shape[1] // 2, 0] == CUBE_FACE_ORDER.index("down")


@pytest.mark.parametrize(
    ("faces", "kwargs"),
    [
        (_faces(dtype=np.float32), {"interpolation": "nearest"}),
        (_faces(dtype=np.uint8), {"interpolation": "nearest"}),
        (
            _faces(dtype=np.float32),
            {
                "invalid_policy": "renormalize",
                "validity_masks": {
                    face: np.ones((19, 23), dtype=bool) for face in CUBE_FACE_ORDER
                },
                "min_valid_weight": 0.5,
            },
        ),
    ],
)
def test_unsupported_modes_stay_on_python_fallback(monkeypatch, faces, kwargs) -> None:
    def forbidden(*args, **arguments):
        raise AssertionError("native geometry route was selected")

    monkeypatch.setattr(_native, "native_cubemap_to_equirectangular", forbidden)

    result = cubemap_to_equirectangular(faces, (41, 80), **kwargs)

    assert result.data.shape[:2] == (41, 80)


def test_mixed_face_dtypes_stay_on_python_fallback(monkeypatch) -> None:
    faces = _faces(dtype=np.float32)
    faces["right"] = faces["right"].astype(np.float64)

    def forbidden(*args, **arguments):
        raise AssertionError("native geometry route was selected")

    monkeypatch.setattr(_native, "native_cubemap_to_equirectangular", forbidden)

    result = cubemap_to_equirectangular(faces, (41, 80))

    assert result.data.dtype == np.float32


def test_native_wrapper_accepts_noncontiguous_faces(monkeypatch) -> None:
    contiguous = _faces(shape=(17, 29), dtype=np.float64, channels=2)
    faces = {face: values[:, ::-1] for face, values in contiguous.items()}
    assert not faces["front"].flags.c_contiguous

    expected = _python_result(monkeypatch, faces, (39, 78))
    actual = cubemap_to_equirectangular(faces, (39, 78)).data

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("channels", [None, 3])
@pytest.mark.parametrize("interpolation", ["nearest", "bilinear"])
def test_native_arbitrary_gnomonic_generation_matches_numpy(
    monkeypatch, dtype, channels, interpolation
) -> None:
    shape = (37, 74) if channels is None else (37, 74, channels)
    image = np.random.default_rng(211).random(shape).astype(dtype)
    image = image[:, ::-1]
    assert not image.flags.c_contiguous
    specs = [
        GnomonicSpec(89.0, 179.5, 103.0, 61.0, 37.0, (11, 17)),
        GnomonicSpec(-89.0, -179.5, 97.0, 77.0, -29.0, (13, 19)),
        GnomonicSpec(7.0, 13.0, 111.0, 69.0, 91.0, (9, 23)),
        GnomonicSpec(-31.0, 122.0, 83.0, 105.0, 181.0, (15, 11)),
        GnomonicSpec(51.0, -77.0, 119.0, 87.0, 12.5, (17, 21)),
        GnomonicSpec(-4.0, 0.0, 90.0, 90.0, -6.0, (12, 18)),
        GnomonicSpec(22.0, 61.0, 73.0, 99.0, 44.0, (8, 25)),
    ]
    plan = _engine._gnomonic_batch_forward_plan(specs, image.shape[:2], image)

    expected = _python_gnomonic_batch(monkeypatch, image, plan, interpolation)
    actual = _engine._gnomonic_batch_from_equirectangular(
        image, plan, interpolation=interpolation
    )

    assert len(actual) == 7
    for result, oracle in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, oracle)
        assert result.dtype == dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("channels", [None, 3])
@pytest.mark.parametrize("count", [1, 6, 14, 42])
def test_native_multiface_gaussian_workflow_matches_numpy(
    monkeypatch, dtype, channels, count
) -> None:
    shape = (35, 70) if channels is None else (35, 70, channels)
    image = np.random.default_rng(919).random(shape).astype(dtype)
    valid = np.ones((35, 70), dtype=bool)
    valid[3:8, 11:19] = False
    valid[20:27, 51:58] = False
    image[3:8, 11:19, ...] = np.nan
    image[20:27, 51:58, ...] = np.asarray(1e6, dtype=dtype)
    expected_faces, expected, expected_valid, expected_support = _python_workflow(
        monkeypatch, image, valid, count
    )

    views = EquirectangularImage(image, valid=valid).views(
        "fibonacci",
        count=count,
        size=(13, 19),
        fov=(107, 73),
    )
    result = views.reconstruct(blend="gaussian")

    for face, oracle in zip(views, expected_faces, strict=True):
        np.testing.assert_array_equal(face.image, oracle)
    np.testing.assert_allclose(
        result.image, expected, atol=2e-7, rtol=0.0, equal_nan=True
    )
    np.testing.assert_array_equal(result.validity("image"), expected_valid)
    np.testing.assert_array_equal(result.support_mask, expected_support)
    assert result.image.dtype == dtype


def test_public_multiface_workflow_selects_both_native_kernels(monkeypatch) -> None:
    generation_calls = []
    reconstruction_calls = []
    original_generation = _native.native_equirectangular_to_gnomonic_batch
    original_reconstruction = _native.native_gnomonic_gaussian_to_equirectangular

    def record_generation(image, plans, *, interpolation):
        generation_calls.append((len(plans), interpolation))
        return original_generation(image, plans, interpolation=interpolation)

    def record_reconstruction(faces, masks, plans, output_shape):
        reconstruction_calls.append((len(faces), output_shape))
        return original_reconstruction(faces, masks, plans, output_shape)

    monkeypatch.setattr(
        _native, "native_equirectangular_to_gnomonic_batch", record_generation
    )
    monkeypatch.setattr(
        _native,
        "native_gnomonic_gaussian_to_equirectangular",
        record_reconstruction,
    )

    views = EquirectangularImage(_faces((31, 62))["front"]).views(
        "fibonacci", count=17, size=(15, 21), fov=(101, 79)
    )
    result = views.reconstruct(blend="gaussian")

    assert generation_calls == [(17, "bilinear")]
    assert reconstruction_calls == [(17, (31, 62))]
    assert result.image.shape == (31, 62, 3)


def test_unsupported_multiface_modes_keep_python_fallback(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("native arbitrary-face route was selected")

    original_generation = _native.native_equirectangular_to_gnomonic_batch
    specs = [GnomonicSpec(0.0, 0.0, 90.0, 90.0, 0.0, (9, 11))]
    labels = np.arange(18 * 36, dtype=np.int16).reshape(18, 36)
    plan = _engine._gnomonic_batch_forward_plan(specs, labels.shape, labels)
    monkeypatch.setattr(_native, "native_equirectangular_to_gnomonic_batch", forbidden)
    projected = _engine._gnomonic_batch_from_equirectangular(
        labels, plan, interpolation="nearest"
    )
    assert projected[0].dtype == np.int16
    monkeypatch.setattr(
        _native,
        "native_equirectangular_to_gnomonic_batch",
        original_generation,
    )

    monkeypatch.setattr(
        _native, "native_gnomonic_gaussian_to_equirectangular", forbidden
    )
    label_views = (
        EquirectangularImage(np.ones((18, 36), dtype=np.float32))
        .with_labels(labels)
        .views("fibonacci", count=9, size=(7, 9))
    )
    result = label_views.reconstruct(modalities="labels", blend="closest")
    assert result.labels.dtype == np.int16

    monkeypatch.setattr(_native, "native_equirectangular_to_gnomonic_batch", forbidden)
    template = GnomonicProjector(
        GnomonicSpec(output_shape_hw=(3, 3)), interpolation="bilinear"
    )
    custom_views = EquirectangularImage(np.ones((18, 36), dtype=np.float32)).views(
        "fibonacci", count=7, size=(7, 9), projector=template
    )
    assert len(custom_views) == 7

    monkeypatch.setattr(
        _native,
        "native_equirectangular_to_gnomonic_batch",
        original_generation,
    )
    valid = np.ones((18, 36), dtype=bool)
    valid[4:8, 9:15] = False
    depth_views = (
        EquirectangularImage(np.ones((18, 36), dtype=np.float32))
        .with_depth(np.full((18, 36), 2.0, dtype=np.float32), valid=valid)
        .views(
            "fibonacci",
            count=7,
            size=(7, 9),
            depth_policy="renormalize",
            min_valid_weight=0.5,
        )
    )
    depth_result = depth_views.reconstruct(modalities="depth", blend="gaussian")
    assert depth_result.depth.shape == (18, 36)


def test_native_kernel_releases_gil_during_sampling() -> None:
    faces = _faces(shape=(1024, 1024), dtype=np.float32, channels=None)
    output_shape = (2048, 4096)
    plan = _engine._cubemap_selective_back_plan(
        output_shape, faces["front"].shape, faces["front"]
    )
    ordered_faces = tuple(faces[face] for face in CUBE_FACE_ORDER)
    plans = tuple(
        (face_plan.flat_indices, face_plan.map_x, face_plan.map_y)
        for face_plan in plan.faces
    )
    started = threading.Event()
    finished = threading.Event()

    def sample() -> None:
        started.set()
        for _ in range(4):
            _native.native_cubemap_to_equirectangular(
                ordered_faces, plans, output_shape
            )
        finished.set()

    worker = threading.Thread(target=sample)
    worker.start()
    assert started.wait(timeout=2.0)
    # If the extension holds the GIL, this sleep cannot reacquire it until the
    # worker has returned and set ``finished``.
    time.sleep(0.005)
    observed_while_running = not finished.is_set()
    worker.join(timeout=30.0)

    assert not worker.is_alive()
    assert observed_while_running


def _assert_gil_released(operation, *, timeout=15.0) -> None:
    started = threading.Event()
    finished = threading.Event()

    def run() -> None:
        started.set()
        operation()
        finished.set()

    worker = threading.Thread(target=run)
    worker.start()
    assert started.wait(timeout=2.0)
    time.sleep(0.005)
    observed_while_running = not finished.is_set()
    worker.join(timeout=timeout)

    assert not worker.is_alive()
    assert observed_while_running


def test_native_arbitrary_face_generation_releases_gil() -> None:
    image = np.random.default_rng(808).random((1024, 2048, 3), dtype=np.float32)
    specs = [
        GnomonicSpec(
            center_lat_deg=(index % 5) * 12.0 - 24.0,
            center_lon_deg=index * 137.5,
            hfov_deg=97.0,
            vfov_deg=83.0,
            roll_deg=index * 7.0,
            output_shape_hw=(512, 512),
        )
        for index in range(16)
    ]
    plan = _engine._gnomonic_batch_forward_plan(specs, image.shape[:2], image)

    def sample_repeatedly() -> None:
        native_plans = tuple(face.pixels_xy for face in plan.faces)
        for _ in range(4):
            _native.native_equirectangular_to_gnomonic_batch(
                image, native_plans, interpolation="bilinear"
            )

    _assert_gil_released(sample_repeatedly)


def test_native_arbitrary_face_gaussian_reconstruction_releases_gil() -> None:
    source = np.random.default_rng(809).random((512, 1024, 3), dtype=np.float32)
    views = EquirectangularImage(source).views(
        "fibonacci", count=42, size=(128, 128), fov=(97.0, 83.0)
    )
    values = [face.image for face in views]
    masks = [face._workflow_metadata["image"]["validity"] for face in views]
    specs = [face.spec for face in views]
    plan = views._get_batch_back_plan(source.shape[:2], values, specs)
    native_plans = tuple(
        (
            face.flat_indices,
            face.map_x,
            face.map_y,
            face.center_score,
        )
        for face in plan.faces
    )

    def reconstruct_repeatedly() -> None:
        for _ in range(4):
            _native.native_gnomonic_gaussian_to_equirectangular(
                values, masks, native_plans, source.shape[:2]
            )

    _assert_gil_released(reconstruct_repeatedly)


def test_native_multiface_workflow_is_safe_under_concurrent_callers() -> None:
    source = np.random.default_rng(810).random((128, 256, 3), dtype=np.float32)

    def execute(_: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        views = EquirectangularImage(source).views(
            "fibonacci", count=14, size=(64, 64), fov=(97.0, 83.0)
        )
        result = views.reconstruct(blend="gaussian")
        return result.image, result.validity("image"), result.support_mask

    expected = execute(0)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(execute, range(16)))

    for actual in results:
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(actual[2], expected[2])
