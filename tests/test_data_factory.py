import importlib.util
import sys
from types import ModuleType
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Stub implementations used for patching
# ---------------------------------------------------------------------------
class Array(list):
    def copy(self):
        return Array(self)


def asarray(obj):
    return Array([0])


class DummyImage:
    def convert(self, mode):
        return self


def open_image(path):
    return DummyImage()


class StubEQ:
    def __init__(self, data):
        self.data = data


class StubGF:
    def __init__(self, data, lat=0.0, lon=0.0, fov=90.0):
        self.data = data
        self.lat = lat
        self.lon = lon
        self.fov = fov


class StubGFS:
    def __init__(self, faces, channel_name="default"):
        self.faces = faces
        self.channel_name = channel_name
        self.blender = None

    def attach_blender(self, name, **kwargs):
        self.blender = name


def _load_module(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    """Patch heavy external modules with lightweight stubs."""

    # Minimal package structure
    panorai_pkg = ModuleType("panorai")
    panorai_pkg.__path__ = [str(ROOT / "panorai")]
    monkeypatch.setitem(sys.modules, "panorai", panorai_pkg)

    data_pkg = ModuleType("panorai.data")
    data_pkg.__path__ = [str(ROOT / "panorai" / "data")]
    monkeypatch.setitem(sys.modules, "panorai.data", data_pkg)

    # NumPy stub
    numpy_stub = ModuleType("numpy")
    numpy_stub.asarray = asarray
    numpy_stub.ndarray = Array
    monkeypatch.setitem(sys.modules, "numpy", numpy_stub)

    # PIL stub
    pil_module = ModuleType("PIL")
    image_sub = ModuleType("PIL.Image")
    image_sub.open = open_image
    image_sub.Image = DummyImage
    pil_module.Image = image_sub
    monkeypatch.setitem(sys.modules, "PIL", pil_module)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_sub)

    # Modules required by DataFactory
    eq_module = ModuleType("panorai.data.equirectangular_image")
    eq_module.EquirectangularImage = StubEQ
    monkeypatch.setitem(sys.modules, "panorai.data.equirectangular_image", eq_module)

    gf_module = ModuleType("panorai.data.gnomonic_image")
    gf_module.GnomonicFace = StubGF
    monkeypatch.setitem(sys.modules, "panorai.data.gnomonic_image", gf_module)

    set_module = ModuleType("panorai.data.gnomonic_imageset")
    set_module.GnomonicFaceSet = StubGFS
    monkeypatch.setitem(sys.modules, "panorai.data.gnomonic_imageset", set_module)

    yield

    # Ensure DataFactory gets re-imported with fresh stubs each time
    cleanup_modules = [
        "panorai.data.factory",
        "panorai.data.equirectangular_image",
        "panorai.data.gnomonic_image",
        "panorai.data.gnomonic_imageset",
        "panorai.data",
        "panorai",
        "PIL.Image",
        "PIL",
        "numpy",
    ]
    for m in cleanup_modules:
        sys.modules.pop(m, None)


@pytest.fixture
def DataFactory(patch_dependencies):
    module = _load_module("panorai.data.factory", "panorai/data/factory.py")
    return module.DataFactory


def test_from_array_returns_equirectangular(DataFactory):
    arr = [[[0, 0, 0] for _ in range(2)] for _ in range(2)]
    obj = DataFactory.from_array(arr, "equirectangular")
    assert isinstance(obj, StubEQ)
    assert obj.data == arr


def test_from_file(DataFactory, tmp_path):
    file_path = tmp_path / "img.png"
    file_path.write_bytes(b"fake")

    obj = DataFactory.from_file(str(file_path), "equirectangular")
    assert isinstance(obj, StubEQ)
    assert isinstance(obj.data, Array)


def test_from_dict_returns_gnomonic_face(DataFactory):
    data = {"r": Array([1]), "g": Array([2])}
    obj = DataFactory.from_dict(data, "gnomonic_face")
    assert isinstance(obj, StubGF)
    assert obj.data is data


def test_from_pil_creates_gnomonic_face(DataFactory):
    img = DummyImage()
    obj = DataFactory.from_pil(img, "gnomonic_face")
    assert isinstance(obj, StubGF)
    assert isinstance(obj.data, Array)


def test_from_list_multiple_faces_attaches_blender(DataFactory):
    faces = [StubGF("a"), StubGF("b")]
    face_set = DataFactory.from_list(faces, channel_name="rgb")
    assert isinstance(face_set, StubGFS)
    assert face_set.faces == faces
    assert face_set.channel_name == "rgb"
    assert face_set.blender == "average"


def test_invalid_data_type_raises(DataFactory):
    with pytest.raises(ValueError):
        DataFactory.from_array(Array([1]), "unknown")


def test_from_file_missing(DataFactory, tmp_path):
    missing = tmp_path / "none.png"
    with pytest.raises(FileNotFoundError):
        DataFactory.from_file(str(missing), "equirectangular")


def test_from_dict_invalid_type_raises(DataFactory):
    with pytest.raises(ValueError):
        DataFactory.from_dict({"r": Array([1])}, "unknown")


def test_from_pil_invalid_type_raises(DataFactory):
    img = DummyImage()
    with pytest.raises(ValueError):
        DataFactory.from_pil(img, "unknown")


def test_from_file_invalid_type_raises(DataFactory, tmp_path):
    file_path = tmp_path / "img.png"
    file_path.write_bytes(b"fake")
    with pytest.raises(ValueError):
        DataFactory.from_file(str(file_path), "unknown")
