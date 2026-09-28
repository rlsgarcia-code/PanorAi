import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path
import pytest

try:
    import numpy as np
except Exception as e:  # pragma: no cover - skip if numpy missing
    pytest.skip(f"Skipping blender tests because numpy import failed: {e}", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


@pytest.fixture()
def blender_modules(monkeypatch):
    """Load blender modules with stubbed dependencies."""
    panorai_pkg = ModuleType("panorai")
    panorai_pkg.__path__ = [str(ROOT / "panorai")]
    sys.modules.setdefault("panorai", panorai_pkg)

    bl_pkg = ModuleType("panorai.blenders")
    bl_pkg.__path__ = [str(ROOT / "panorai" / "blenders")]
    sys.modules.setdefault("panorai.blenders", bl_pkg)

    # Stub scipy.ndimage required by gaussian module
    scipy_stub = ModuleType("scipy")
    nd_stub = ModuleType("scipy.ndimage")
    nd_stub.gaussian_filter = lambda img, sigma=1: img
    monkeypatch.setitem(sys.modules, "scipy", scipy_stub)
    monkeypatch.setitem(sys.modules, "scipy.ndimage", nd_stub)

    base_mod = _load_module("panorai.blenders.base_blenders", "panorai/blenders/base_blenders.py")
    registry_mod = _load_module("panorai.blenders.registry", "panorai/blenders/registry.py")
    avg_mod = _load_module("panorai.blenders.average", "panorai/blenders/average.py")
    gauss_mod = _load_module("panorai.blenders.gaussian", "panorai/blenders/gaussian.py")

    return SimpleNamespace(
        AverageBlender=avg_mod.AverageBlender,
        GaussianBlender=gauss_mod.GaussianBlender,
    )


def test_average_blender_basic(blender_modules):
    B = blender_modules.AverageBlender()
    images = [
        np.array([[[1], [0]], [[1], [0]]], dtype=np.float32),
        np.array([[[2], [2]], [[0], [0]]], dtype=np.float32),
    ]
    masks = [
        np.array([[1, 0], [1, 0]], dtype=np.float32),
        np.array([[1, 1], [0, 0]], dtype=np.float32),
    ]
    out = B.blend(images, masks)
    expected = np.array([[[1.5], [2.0]], [[1.0], [0.0]]], dtype=np.float32)
    assert np.allclose(out, expected)


def test_average_blender_length_mismatch(blender_modules):
    B = blender_modules.AverageBlender()
    imgs = [np.zeros((1, 1, 1), dtype=np.float32)]
    with pytest.raises(ValueError):
        B.blend(imgs, [])


def test_average_blender_uses_masks_not_pixel_values(blender_modules):
    blender = blender_modules.AverageBlender()
    images = [
        np.zeros((1, 2, 3), dtype=np.float32),
        np.array([[[2, 2, 2], [9, 9, 9]]], dtype=np.float32),
    ]
    masks = [
        np.array([[True, True]]),
        np.array([[True, False]]),
    ]

    output, support = blender.blend(images, masks, return_mask=True)

    assert np.allclose(output, [[[1, 1, 1], [0, 0, 0]]])
    assert np.array_equal(support, [[True, True]])


def test_gaussian_blender_weighted(monkeypatch, blender_modules):
    weights1 = np.array([[1, 0], [0, 0]], dtype=np.float32)
    weights2 = np.array([[0, 1], [0, 0]], dtype=np.float32)
    calls = {"n": 0}

    class DummyConfig:
        def __init__(self):
            self.x_points = 2
            self.y_points = 2
        def update(self, **kw):
            pass

    class DummyProjector:
        def __init__(self):
            self.config = DummyConfig()
        def back_project(self, arr, eq_shape, return_mask=False):
            w = weights1 if calls["n"] == 0 else weights2
            calls["n"] += 1
            projected = np.dstack([w, w, w])
            support = w > 0
            return (projected, support) if return_mask else projected

    def fake_dist(*a, **k):
        return np.ones((2, 2), dtype=np.float32)

    gaussian_mod = sys.modules["panorai.blenders.gaussian"]
    monkeypatch.setattr(gaussian_mod, "get_distribution", fake_dist)

    blender = blender_modules.GaussianBlender(
        fov_deg=90,
        projector=DummyProjector(),
        tangent_points=[(0, 0), (0, 0)],
    )

    imgs = [
        np.full((2, 2, 1), 10, dtype=np.float32),
        np.full((2, 2, 1), 20, dtype=np.float32),
    ]
    masks = [np.ones((2, 2), dtype=np.float32) for _ in imgs]

    out = blender.blend(imgs, masks)
    expected = np.array([[[10.0], [20.0]], [[0.0], [0.0]]], dtype=np.float32)
    assert np.allclose(out, expected)


def test_gaussian_blender_length_mismatch(blender_modules):
    class DummyConfig:
        def __init__(self):
            self.x_points = 1
            self.y_points = 1
        def update(self, **kw):
            pass

    class DummyProjector:
        def __init__(self):
            self.config = DummyConfig()
        def back_project(self, arr, eq_shape, return_mask=False):
            support = np.ones(eq_shape, dtype=bool)
            return (arr, support) if return_mask else arr

    blender = blender_modules.GaussianBlender(
        fov_deg=90,
        projector=DummyProjector(),
        tangent_points=[(0, 0)],
    )

    imgs = [np.zeros((1, 1, 1), dtype=np.float32), np.zeros((1, 1, 1), dtype=np.float32)]
    masks = [np.ones((1, 1), dtype=np.float32)]
    with pytest.raises(ValueError):
        blender.blend(imgs, masks)
