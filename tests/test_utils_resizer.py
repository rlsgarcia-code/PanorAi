import importlib.util
import sys
from types import ModuleType
from pathlib import Path
import pytest


class Array:
    def __init__(self, shape):
        self._shape = shape
        self.ndim = len(shape)
        self.dtype = "float32"

    @property
    def shape(self):
        return self._shape


def zeros(shape, dtype=None):
    return Array(shape)


def ones(shape, dtype=None):
    return Array(shape)

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


@pytest.fixture()
def resizer_module(monkeypatch):
    """Load panorai.utils.resizer with stubbed dependencies."""
    call_log = {"skimage": [], "cv2": []}

    skimage_stub = ModuleType("skimage")
    transform_stub = ModuleType("skimage.transform")

    def stub_resize(img, new_shape, mode=None, anti_aliasing=None):
        call_log["skimage"].append((img.shape, new_shape))
        return zeros(new_shape)

    transform_stub.resize = stub_resize
    skimage_stub.transform = transform_stub
    monkeypatch.setitem(sys.modules, "skimage", skimage_stub)
    monkeypatch.setitem(sys.modules, "skimage.transform", transform_stub)

    cv2_stub = ModuleType("cv2")

    def stub_cv2_resize(img, size, interpolation=None):
        call_log["cv2"].append((img.shape, size, interpolation))
        shape = (size[1], size[0]) + ((img.shape[2],) if img.ndim == 3 else ())
        return zeros(shape)

    cv2_stub.resize = stub_cv2_resize
    cv2_stub.INTER_LINEAR = 1
    monkeypatch.setitem(sys.modules, "cv2", cv2_stub)

    numpy_stub = ModuleType("numpy")
    numpy_stub.ndarray = Array
    monkeypatch.setitem(sys.modules, "numpy", numpy_stub)

    module = _load_module("panorai.utils.resizer", "panorai/utils/resizer.py")

    monkeypatch.setattr(module, "resize", stub_resize)
    monkeypatch.setattr(module.cv2, "resize", stub_cv2_resize)

    yield module, call_log

    for m in ["panorai.utils.resizer", "skimage", "skimage.transform", "cv2"]:
        sys.modules.pop(m, None)



def test_resize_skimage_paths(resizer_module):
    module, log = resizer_module
    ImageResizer = module.ImageResizer

    arr2d = ones((4, 6))
    resizer = ImageResizer(resize_factor=0.5, method="skimage")
    out2 = resizer.resize_image(arr2d)
    assert log["skimage"][-1] == ((4, 6), (2, 3))
    assert out2.shape == (2, 3)

    arr3d = ones((8, 4, 3))
    out3 = resizer.resize_image(arr3d)
    assert log["skimage"][-1] == ((8, 4, 3), (4, 2, 3))
    assert out3.shape == (4, 2, 3)


def test_resize_cv2_path(resizer_module):
    module, log = resizer_module
    ImageResizer = module.ImageResizer

    arr = ones((4, 5, 3))
    resizer = ImageResizer(resize_factor=0.5, method="cv2")
    out = resizer.resize_image(arr)
    # cv2.resize expects size=(width, height)
    assert log["cv2"][-1] == ((4, 5, 3), (2, 2), resizer.interpolation)
    assert out.shape == (2, 2, 3)


def test_resize_no_upsample(resizer_module):
    module, log = resizer_module
    ImageResizer = module.ImageResizer

    arr = ones((4, 4))
    resizer = ImageResizer(resize_factor=2.0, method="skimage")
    out = resizer.resize_image(arr, upsample=False)
    assert log["skimage"][-1] == ((4, 4), (2, 2))
    assert out.shape == (2, 2)


def test_invalid_method_raises(resizer_module):
    module, _ = resizer_module
    ImageResizer = module.ImageResizer
    resizer = ImageResizer(resize_factor=2.0, method="bad")
    with pytest.raises(ValueError):
        resizer.resize_image(zeros((2, 2)))


def test_resizer_config_creation_and_repr(resizer_module):
    module, _ = resizer_module
    ResizerConfig = module.ResizerConfig
    cfg = ResizerConfig(
        resize_factor=1.5,
        method="cv2",
        mode="constant",
        anti_aliasing=False,
        interpolation=7,
    )
    resizer = cfg.create_resizer()
    assert isinstance(resizer, module.ImageResizer)
    assert resizer.resize_factor == 1.5
    assert resizer.method == "cv2"
    assert resizer.mode == "constant"
    assert resizer.anti_aliasing is False
    assert resizer.interpolation == 7

    rep = repr(cfg)
    for token in [
        "resize_factor=1.5",
        "method='cv2'",
        "mode='constant'",
        "anti_aliasing=False",
        "interpolation=7",
    ]:
        assert token in rep

