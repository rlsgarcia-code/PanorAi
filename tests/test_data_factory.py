import importlib.util
import sys
from types import ModuleType
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Minimal package structure to avoid importing heavy dependencies
panorai_pkg = ModuleType("panorai")
panorai_pkg.__path__ = [str(ROOT / "panorai")]
sys.modules.setdefault("panorai", panorai_pkg)

# Subpackage placeholder
data_pkg = ModuleType("panorai.data")
data_pkg.__path__ = [str(ROOT / "panorai" / "data")]
sys.modules.setdefault("panorai.data", data_pkg)

# Minimal NumPy stub
numpy_stub = ModuleType("numpy")

class Array(list):
    def copy(self):
        return Array(self)

def asarray(obj):
    return Array([0])

numpy_stub.asarray = asarray
numpy_stub.ndarray = Array
sys.modules.setdefault("numpy", numpy_stub)

# Minimal PIL.Image stub
pil_module = ModuleType("PIL")

class DummyImage:
    def convert(self, mode):
        return self

def open_image(path):
    return DummyImage()

image_sub = ModuleType("PIL.Image")
image_sub.open = open_image
image_sub.Image = DummyImage

pil_module.Image = image_sub
sys.modules.setdefault("PIL", pil_module)
sys.modules.setdefault("PIL.Image", image_sub)

# Stub classes used by DataFactory
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
    pass

# Register stub modules so DataFactory.import uses them
eq_module = ModuleType("panorai.data.equirectangular_image")
eq_module.EquirectangularImage = StubEQ
sys.modules.setdefault("panorai.data.equirectangular_image", eq_module)

gf_module = ModuleType("panorai.data.gnomonic_image")
gf_module.GnomonicFace = StubGF
sys.modules.setdefault("panorai.data.gnomonic_image", gf_module)

set_module = ModuleType("panorai.data.gnomonic_imageset")
set_module.GnomonicFaceSet = StubGFS
sys.modules.setdefault("panorai.data.gnomonic_imageset", set_module)


def _load_module(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module

# Load DataFactory after stubs are set
factory_module = _load_module("panorai.data.factory", "panorai/data/factory.py")
DataFactory = factory_module.DataFactory


def test_from_array_returns_equirectangular():
    arr = [[[0, 0, 0] for _ in range(2)] for _ in range(2)]
    obj = DataFactory.from_array(arr, "equirectangular")
    assert isinstance(obj, StubEQ)
    assert obj.data == arr


def test_from_file(tmp_path):
    file_path = tmp_path / "img.png"
    file_path.write_bytes(b"fake")

    obj = DataFactory.from_file(str(file_path), "equirectangular")
    assert isinstance(obj, StubEQ)
    assert isinstance(obj.data, Array)
