import builtins
import sys
import importlib.util
from types import ModuleType
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

def _load_module(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # Ensure fresh imports using the lightweight stubs below by clearing any
    # previously imported modules from earlier tests.  Without this, modules
    # like ``panorai.data.gnomonic_image`` may already be loaded with the real
    # ``numpy`` dependency which breaks the stubbed environment used here.
    for mod in [
        "panorai.data.equirectangular_image",
        "panorai.data.gnomonic_image",
        "panorai.data.gnomonic_imageset",
        "panorai.data.spherical_data",
        "panorai.data.multi_data",
        "panorai.data.multi_handler",
    ]:
        sys.modules.pop(mod, None)

    panorai_pkg = ModuleType("panorai")
    panorai_pkg.__path__ = [str(ROOT / "panorai")]
    monkeypatch.setitem(sys.modules, "panorai", panorai_pkg)

    numpy_stub = ModuleType("numpy")
    class Array(list):
        def __init__(self, data):
            super().__init__(data if isinstance(data, list) else list(data))
        @property
        def shape(self):
            return (len(self), len(self[0])) if self and isinstance(self[0], list) else (len(self),)
        @property
        def ndim(self):
            return len(self.shape)
        @property
        def dtype(self):
            return float
        def copy(self):
            return Array([row[:] if isinstance(row, list) else row for row in self])
        def astype(self, *a, **k):
            return self.copy()
        def __add__(self, other):
            return Array([[v + other for v in row] for row in self])
        __radd__ = __add__
        def mean(self):
            return sum(sum(r) for r in self) / sum(len(r) for r in self)
    def array(obj, dtype=None):
        return Array(obj)
    def zeros(shape, dtype=float):
        return Array([[0]*shape[1] for _ in range(shape[0])])
    def ones(shape, dtype=float):
        return Array([[1]*shape[1] for _ in range(shape[0])])
    def full(shape, fill, dtype=float):
        return Array([[fill]*shape[1] for _ in range(shape[0])])
    def array_equal(a, b):
        def _tolist(x):
            if hasattr(x, "tolist"):
                return x.tolist()
            return [ _tolist(i) for i in x ] if isinstance(x, list) else x
        return _tolist(a) == _tolist(b)
    def all(arr):
        return builtins.all(arr) if not isinstance(arr, bool) else arr
    numpy_stub.ndarray = Array
    numpy_stub.array = array
    numpy_stub.zeros = zeros
    numpy_stub.ones = ones
    numpy_stub.full = full
    numpy_stub.array_equal = array_equal
    numpy_stub.all = all
    numpy_stub.float32 = float
    monkeypatch.setitem(sys.modules, "numpy", numpy_stub)
    pil_module = ModuleType("PIL")
    image_sub = ModuleType("PIL.Image")
    image_sub.open = lambda *a, **k: None
    image_sub.Image = type("Image", (), {})
    pil_module.Image = image_sub
    monkeypatch.setitem(sys.modules, "PIL", pil_module)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_sub)

    utils_mod = ModuleType("panorai.utils")
    class ImageResizer: ...
    class ResizerConfig:
        def create_resizer(self):
            return ImageResizer()
    utils_mod.ImageResizer = ImageResizer
    utils_mod.ResizerConfig = ResizerConfig
    utils_mod.setup_logging = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "panorai.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "panorai.utils.resizer", utils_mod)
    shape_mod = ModuleType("panorai.utils.shape_manager")
    class ShapeManager:
        @staticmethod
        def to_numpy(x, dtype=float):
            return x
    shape_mod.ShapeManager = ShapeManager
    monkeypatch.setitem(sys.modules, "panorai.utils.shape_manager", shape_mod)
    exc_mod = ModuleType("panorai.utils.exceptions")
    class PanoraiError(Exception):
        pass
    class InvalidDataError(PanoraiError):
        pass
    class ChannelMismatchError(PanoraiError):
        pass
    class MetadataValidationError(PanoraiError):
        pass
    class DataConversionError(PanoraiError):
        pass
    class MissingChannelError(PanoraiError):
        pass
    class FaceSetError(PanoraiError):
        pass
    class ImageProcessingError(PanoraiError):
        pass
    exc_mod.PanoraiError = PanoraiError
    exc_mod.InvalidDataError = InvalidDataError
    exc_mod.ChannelMismatchError = ChannelMismatchError
    exc_mod.MetadataValidationError = MetadataValidationError
    exc_mod.DataConversionError = DataConversionError
    exc_mod.MissingChannelError = MissingChannelError
    exc_mod.FaceSetError = FaceSetError
    exc_mod.ImageProcessingError = ImageProcessingError
    monkeypatch.setitem(sys.modules, "panorai.utils.exceptions", exc_mod)

    factory_pkg = ModuleType("panorai.factory")
    factory_pkg.__path__ = [str(ROOT / "panorai" / "factory")]
    monkeypatch.setitem(sys.modules, "panorai.factory", factory_pkg)

    factory_mod = ModuleType("panorai.factory.panorai_factory")
    class DummyProjection:
        def __init__(self, lat=0.0, lon=0.0, fov=90.0, **kw):
            self.config = {"phi1_deg": lat, "lam0_deg": lon, "fov_deg": fov}
        def project(self, arr):
            return arr + 1
        def back_project(self, arr, shape):
            return full(shape, arr.mean())
    class DummySampler:
        def get_tangent_points(self):
            return [(0,0), (1,1)]
    class DummyBlender:
        def blend(self, arrays, _):
            result = arrays[0].copy()
            for arr in arrays[1:]:
                for i,row in enumerate(arr):
                    for j,val in enumerate(row):
                        result[i][j] += val
            for i,row in enumerate(result):
                for j in range(len(row)):
                    result[i][j] /= len(arrays)
            return result
    class PanoraiFactory:
        @staticmethod
        def get_projection(name, lat, lon, fov, **kw):
            return DummyProjection(lat=lat, lon=lon, fov=fov)
        @staticmethod
        def get_sampler(name, **kw):
            return DummySampler()
        @staticmethod
        def get_blender(name, **kw):
            return DummyBlender()
    factory_mod.PanoraiFactory = PanoraiFactory
    monkeypatch.setitem(sys.modules, "panorai.factory.panorai_factory", factory_mod)

    preproc_pkg = ModuleType("panorai.preprocessing")
    monkeypatch.setitem(sys.modules, "panorai.preprocessing", preproc_pkg)
    preproc_mod = ModuleType("panorai.preprocessing.preprocessor")
    class DummyPreprocessor:
        @staticmethod
        def preprocess_eq(x, **kw):
            return x + 1
    preproc_mod.Preprocessor = DummyPreprocessor
    monkeypatch.setitem(sys.modules, "panorai.preprocessing.preprocessor", preproc_mod)

    pcd_pkg = ModuleType("panorai.pcd")
    monkeypatch.setitem(sys.modules, "panorai.pcd", pcd_pkg)
    pcd_mod = ModuleType("panorai.pcd.handler")
    class DummyPCD: ...
    class DummyPCDHandler:
        @staticmethod
        def gnomonic_face_to_pcd(*a, **k):
            return DummyPCD()
        @staticmethod
        def equirectangular_image_to_pcd(*a, **k):
            return DummyPCD()
        @staticmethod
        def gnomonic_faceset_to_pcd(*a, **k):
            return DummyPCD()
    pcd_mod.PCDHandler = DummyPCDHandler
    monkeypatch.setitem(sys.modules, "panorai.pcd.handler", pcd_mod)
    yield
    for mod in [
        "panorai.data.equirectangular_image",
        "panorai.data.gnomonic_image",
        "panorai.data.gnomonic_imageset",
        "panorai.data.spherical_data",
        "panorai.data.multi_data",
        "panorai.data.multi_handler",
    ]:
        sys.modules.pop(mod, None)

@pytest.fixture

def data_modules(patch_dependencies):
    eq = _load_module("panorai.data.equirectangular_image", "panorai/data/equirectangular_image.py")
    gf = _load_module("panorai.data.gnomonic_image", "panorai/data/gnomonic_image.py")
    gfs = _load_module("panorai.data.gnomonic_imageset", "panorai/data/gnomonic_imageset.py")
    return eq, gf, gfs

def test_gnomonic_face_basic(data_modules):
    np = sys.modules["numpy"]
    eq, gf, _ = data_modules
    arr = np.ones((2, 2))
    face = gf.GnomonicFace(arr, lat=10, lon=20, fov=90)
    assert face.lat == 10 and face.lon == 20
    assert face.projection.config["phi1_deg"] == 10
    assert "lat=10" in repr(face)
    clone = face.clone()
    assert clone.lat == face.lat and np.array_equal(clone.data, face.data)
    assert clone is not face

def test_gnomonic_face_to_equirectangular_and_pcd(data_modules):
    np = sys.modules["numpy"]
    eq_mod, gf_mod, _ = data_modules
    arr = np.ones((2, 2))
    face = gf_mod.GnomonicFace(arr, lat=0, lon=0, fov=90)
    eq_image = face.to_equirectangular((4, 4))
    assert isinstance(eq_image, eq_mod.EquirectangularImage)
    assert eq_image.shape == (4, 4)
    with pytest.raises(ValueError):
        face.to_pcd()
    depth = np.ones((2, 2))
    pcd = face.to_pcd(depth=depth)
    assert type(pcd).__name__ == "DummyPCD"

def test_equirectangular_image_workflow(data_modules):
    np = sys.modules["numpy"]
    eq_mod, gf_mod, gfs_mod = data_modules
    data = np.zeros((2, 4))
    eq_img = eq_mod.EquirectangularImage(data, lat=0, lon=0)
    assert eq_img.sampler.get_tangent_points() == [(0, 0), (1, 1)]
    eq_img.attach_projection("gnomonic", lat=5, lon=6, fov=70)
    assert eq_img.projection.config["phi1_deg"] == 5
    eq_img.preprocess(delta_lat=1, delta_lon=2)
    assert eq_img.lat == 1 and eq_img.lon == 2
    assert eq_img.data == data + 1
    face = eq_img.to_gnomonic(10, 20, 90)
    assert isinstance(face, gf_mod.GnomonicFace)
    assert face.data == eq_img.data + 1
    face_set = eq_img.to_gnomonic_face_set(fov=90)
    assert isinstance(face_set, gfs_mod.GnomonicFaceSet)
    assert len(face_set) == 2
    pts = eq_img.augment_with_rotations([(0, 0)], [(1, 2)])
    assert (1, 2) in pts and (0, 0) in pts
    clone = eq_img.clone()
    assert np.array_equal(clone.data, eq_img.data)
    assert clone is not eq_img
    assert eq_img.shape == (2, 4)
    pcd = eq_img.to_pcd()
    assert type(pcd).__name__ == "DummyPCD"

def test_gnomonic_face_set_operations(data_modules):
    np = sys.modules["numpy"]
    eq_mod, gf_mod, gfs_mod = data_modules
    face1 = gf_mod.GnomonicFace(np.zeros((2, 2)), 0, 0, 90)
    face2 = gf_mod.GnomonicFace(np.ones((2, 2)), 1, 1, 90)
    fs = gfs_mod.GnomonicFaceSet([face1, face2])
    assert len(fs) == 2
    assert fs[0] is face1
    fs.add_face(face1)
    assert len(fs) == 3
    fs.apply_to_all(lambda f: setattr(f, "lat", f.lat + 1))
    assert face1.lat == 1 and face2.lat == 2
    fs.attach_blender("average")
    eq_image = fs.to_equirectangular((4, 4))
    assert isinstance(eq_image, eq_mod.EquirectangularImage)
    assert abs(eq_image.data[0][0] - 1/3) < 1e-6
    with pytest.raises(ValueError):
        gfs_mod.GnomonicFaceSet([]).to_equirectangular((2, 2))
    depth = np.ones((2, 2))
    assert type(fs.to_pcd(depth=depth)).__name__ == "DummyPCD"
    clone = fs.clone()
    assert len(clone) == len(fs)
    assert clone.blender is fs.blender

def test_faceset_to_pcd_eq_shape_channels(data_modules, monkeypatch):
    np = sys.modules["numpy"]
    _, gf_mod, gfs_mod = data_modules
    face = gf_mod.GnomonicFace(np.zeros((2, 2)), 0, 0, 90)
    fs = gfs_mod.GnomonicFaceSet([face])

    captured = {}
    pcd_mod = sys.modules["panorai.pcd.handler"]

    def fake(*a, **k):
        captured["shape"] = k.get("eq_shape")
        return object()

    monkeypatch.setattr(pcd_mod.PCDHandler, "gnomonic_faceset_to_pcd", staticmethod(fake))

    depth = np.ones((2, 2))
    fs.to_pcd(depth=depth, eq_shape=(4, 8, 3))
    assert captured["shape"] == (4, 8)
