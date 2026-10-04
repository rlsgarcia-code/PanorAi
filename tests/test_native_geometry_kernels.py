from __future__ import annotations

import threading
import time

import numpy as np
import pytest

pytest.importorskip(
    "panorai._native._geometry", reason="optional geometry extension is not built"
)

from panorai.geometry import (  # noqa: E402
    CUBE_FACE_ORDER,
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
        _native.native_cubemap_to_equirectangular(ordered_faces, plans, output_shape)
        finished.set()

    worker = threading.Thread(target=sample)
    worker.start()
    assert started.wait(timeout=2.0)
    # If the extension holds the GIL, this sleep cannot reacquire it until the
    # worker has returned and set ``finished``.
    time.sleep(0.005)
    observed_while_running = not finished.is_set()
    worker.join(timeout=10.0)

    assert not worker.is_alive()
    assert observed_while_running
