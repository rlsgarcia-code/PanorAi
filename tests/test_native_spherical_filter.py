from __future__ import annotations

import threading
import time

import numpy as np
import pytest

native = pytest.importorskip(
    "panorai._native._geometry", reason="optional geometry extension is not built"
)

from panorai.image_processing import (  # noqa: E402
    native_filter_available,
    spherical_filter2d,
    spherical_gaussian_blur,
)
from panorai.image_processing import _native  # noqa: E402
from panorai.features._spherical_detector import (  # noqa: E402
    _dog_extrema_candidates_mask_numpy,
)
from panorai.image_processing._native import (  # noqa: E402
    native_extrema_available,
    native_spherical_extrema3d,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("channels", [None, 1, 4])
def test_native_spherical_convolution_matches_numpy_exactly(dtype, channels) -> None:
    shape = (37, 74) if channels is None else (37, 74, channels)
    image = np.random.default_rng(71).normal(size=shape).astype(dtype)[:, ::-1]
    kernel = np.array(
        [[1.0, -2.0, 3.0], [0.5, 0.0, -0.25], [-1.0, 4.0, 2.0]],
        dtype=np.float64,
    )

    expected = spherical_filter2d(image, kernel, backend="numpy")
    actual = spherical_filter2d(image, kernel, backend="native")

    if dtype == np.float32:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, atol=4e-13, rtol=0.0)
    assert actual.dtype == dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(9, 18), (37, 74), (64, 128)])
def test_native_spherical_extrema_matches_numpy_bit_exactly(dtype, shape) -> None:
    rng = np.random.default_rng(804 + shape[0])
    scales = tuple(rng.normal(size=shape).astype(dtype) for _ in range(3))
    angular_step = np.pi / shape[0]

    expected = _dog_extrema_candidates_mask_numpy(*scales, angular_step, 0.2)
    actual = native_spherical_extrema3d(*scales, angular_step, 0.2)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.bool_


def test_native_spherical_extrema_preserves_seam_and_poles() -> None:
    shape = (17, 34)
    previous = np.zeros(shape, dtype=np.float64)
    current = np.zeros(shape, dtype=np.float64)
    following = np.zeros(shape, dtype=np.float64)
    current[0, 0] = 2.0
    current[-1, -1] = -2.0
    current[shape[0] // 2, 0] = 1.0
    current[shape[0] // 2, -1] = 0.75
    angular_step = np.pi / shape[0]

    expected = _dog_extrema_candidates_mask_numpy(
        previous, current, following, angular_step, 0.1
    )
    actual = native_spherical_extrema3d(previous, current, following, angular_step, 0.1)

    np.testing.assert_array_equal(actual, expected)


def test_auto_backend_dispatches_compatible_convolution(monkeypatch) -> None:
    calls = []
    implementation = _native.native_spherical_filter2d

    def record(image, kernel, angular_step):
        calls.append((image.shape, kernel.shape, angular_step))
        return implementation(image, kernel, angular_step)

    monkeypatch.setattr(_native, "native_spherical_filter2d", record)
    result = spherical_gaussian_blur(np.ones((17, 34, 3), dtype=np.float32), 5, 1.0)

    assert calls and calls[0][:2] == ((17, 34, 3), (5, 5))
    np.testing.assert_allclose(result, 1.0, atol=1e-7, rtol=0.0)


def test_integer_input_uses_numpy_fallback(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("native route selected for unsupported uint8 input")

    monkeypatch.setattr(_native, "native_spherical_filter2d", forbidden)
    result = spherical_gaussian_blur(np.ones((9, 18), dtype=np.uint8))
    assert result.dtype == np.float64


def test_native_entry_point_rejects_invalid_buffers() -> None:
    assert native_filter_available()
    with pytest.raises(TypeError, match="float32 or float64"):
        native.spherical_filter2d(
            np.zeros((4, 8), dtype=np.uint8), np.ones((3, 3)), 0.1
        )
    with pytest.raises(ValueError, match="positive and odd"):
        native.spherical_filter2d(
            np.zeros((4, 8), dtype=np.float32), np.ones((2, 2)), 0.1
        )
    assert native_extrema_available()
    with pytest.raises(ValueError, match="identical shape and dtype"):
        native.spherical_extrema3d(
            np.zeros((4, 8), dtype=np.float32),
            np.zeros((4, 8), dtype=np.float64),
            np.zeros((4, 8), dtype=np.float32),
            0.1,
            0.01,
        )


def test_native_spherical_convolution_releases_gil() -> None:
    image = np.random.default_rng(3).random((512, 1024, 3), dtype=np.float32)
    kernel = np.ones((5, 5), dtype=np.float64) / 25.0
    started = threading.Event()
    finished = threading.Event()

    def run() -> None:
        started.set()
        spherical_filter2d(image, kernel, backend="native")
        finished.set()

    worker = threading.Thread(target=run)
    worker.start()
    assert started.wait(timeout=2.0)
    time.sleep(0.005)
    observed_while_running = not finished.is_set()
    worker.join(timeout=15.0)

    assert not worker.is_alive()
    assert observed_while_running
