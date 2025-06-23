import importlib.util
import sys
from types import ModuleType
from pathlib import Path
import pytest
np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    calls = {"fromarray": 0, "show": 0}

    class DummyImage:
        def show(self):
            calls["show"] += 1

    def fake_fromarray(arr):
        calls["fromarray"] += 1
        return DummyImage()

    pil_module = ModuleType("PIL")
    image_sub = ModuleType("PIL.Image")
    image_sub.fromarray = fake_fromarray
    pil_module.Image = image_sub
    monkeypatch.setitem(sys.modules, "PIL", pil_module)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_sub)

    cv2_stub = ModuleType("cv2")
    cv2_stub.error = Exception
    cv2_stub.INTER_NEAREST = 0
    cv2_stub.INTER_LINEAR = 1
    cv2_stub.INTER_CUBIC = 2
    cv2_stub.INTER_LANCZOS4 = 3
    cv2_stub.BORDER_CONSTANT = 0
    cv2_stub.BORDER_REPLICATE = 1
    cv2_stub.BORDER_REFLECT = 2
    cv2_stub.BORDER_WRAP = 3
    cv2_stub.BORDER_REFLECT_101 = 4
    cv2_stub.BORDER_TRANSPARENT = 5
    cv2_stub.remap = lambda *a, **k: a[0]
    monkeypatch.setitem(sys.modules, "cv2", cv2_stub)

    preproc_pkg = ModuleType("panorai.preprocessing")
    preproc_mod = ModuleType("panorai.preprocessing.preprocessor")

    class DummyPreprocessor:
        @staticmethod
        def preprocess_eq(img, **kw):
            return img

    preproc_mod.Preprocessor = DummyPreprocessor
    preproc_pkg.preprocessor = preproc_mod
    monkeypatch.setitem(sys.modules, "panorai.preprocessing", preproc_pkg)
    monkeypatch.setitem(sys.modules, "panorai.preprocessing.preprocessor", preproc_mod)

    yield calls


@pytest.fixture
def EquirectangularImage(patch_dependencies):
    module = _load_module("panorai.data.equirectangular_image", "panorai/data/equirectangular_image.py")
    return module.EquirectangularImage, patch_dependencies


@pytest.fixture
def GnomonicFace(patch_dependencies):
    module = _load_module("panorai.data.gnomonic_image", "panorai/data/gnomonic_image.py")
    return module.GnomonicFace, patch_dependencies


def test_equirectangular_show_invokes_pil(EquirectangularImage):
    EQ, calls = EquirectangularImage
    img = EQ(np.zeros((2, 2, 3), dtype=np.uint8))
    img.show()
    assert calls["fromarray"] == 1
    assert calls["show"] == 1


def test_gnomonic_show_invokes_pil(GnomonicFace):
    GF, calls = GnomonicFace
    face = GF(np.zeros((2, 2, 3), dtype=np.uint8), lat=0, lon=0, fov=90)
    face.show()
    assert calls["fromarray"] == 1
    assert calls["show"] == 1
