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


@pytest.fixture()
def prep_module(monkeypatch):
    """Load transformations module with stubbed cv2 and skimage."""
    call_log = {"remap": [], "resize": []}

    cv2_stub = ModuleType("cv2")

    def stub_remap(img, map_x, map_y, interpolation=None, borderMode=None):
        call_log["remap"].append((map_x, map_y))
        ix = np.round(map_y).astype(int) % img.shape[0]
        iy = np.round(map_x).astype(int) % img.shape[1]
        return img[ix, iy]

    cv2_stub.remap = stub_remap
    cv2_stub.INTER_LINEAR = 1
    cv2_stub.BORDER_WRAP = 0
    monkeypatch.setitem(sys.modules, "cv2", cv2_stub)

    skimage_stub = ModuleType("skimage")
    transform_stub = ModuleType("skimage.transform")

    def stub_resize(img, new_shape, mode=None, anti_aliasing=None, preserve_range=True):
        call_log["resize"].append(new_shape)
        if img.ndim == 3 and len(new_shape) == 2:
            new_shape = (*new_shape, img.shape[2])
        return np.zeros(new_shape, dtype=img.dtype)

    transform_stub.resize = stub_resize
    skimage_stub.transform = transform_stub
    monkeypatch.setitem(sys.modules, "skimage", skimage_stub)
    monkeypatch.setitem(sys.modules, "skimage.transform", transform_stub)

    module = _load_module(
        "panorai.preprocessing.transformations",
        "panorai/preprocessing/transformations.py",
    )
    yield module, call_log

    for m in [
        "panorai.preprocessing.transformations",
        "cv2",
        "skimage",
        "skimage.transform",
    ]:
        sys.modules.pop(m, None)


def test_extend_height_increases(prep_module):
    module, _ = prep_module
    P = module.PreprocessEquirectangularImage
    img = np.ones((2, 2), dtype=np.uint8)
    extended = P.extend_height(img, 60)
    assert extended.shape[0] > img.shape[0]
    assert np.array_equal(extended[:2], img)
    assert np.all(extended[2:] == 0)


def test_extend_and_undo_height_are_inverse(prep_module):
    module, _ = prep_module
    P = module.PreprocessEquirectangularImage
    img = np.arange(150 * 4, dtype=np.float32).reshape(150, 4)
    extended = P.extend_height(img, 30)
    restored = P.undo_extend_height(extended, 30)
    assert extended.shape == (180, 4)
    assert np.array_equal(restored, img)


def test_preprocess_skips_materialized_shadow_padding(prep_module):
    module, _ = prep_module
    P = module.PreprocessEquirectangularImage
    img = np.ones((180, 360, 3), dtype=np.uint8)
    out = P.preprocess(img, shadow_angle=30, shadow_padded=True)
    assert out.shape == img.shape
    assert np.array_equal(out, img)


def test_rotate_changes_coords(monkeypatch, prep_module):
    module, log = prep_module
    P = module.PreprocessEquirectangularImage
    img = np.arange(16, dtype=np.float32).reshape(4, 4)
    result = P.rotate(img, delta_lat=0, delta_lon=90)
    assert not np.array_equal(result, img)
    map_x, map_y = log["remap"][0]
    assert map_x.shape == img.shape
    assert map_y.shape == img.shape
    # top-left pixel expected mapping
    import math

    lat = -90.0
    lon = -180.0
    rot_lon = math.radians(90)
    x = math.cos(math.radians(lat)) * math.cos(math.radians(lon))
    y = math.cos(math.radians(lat)) * math.sin(math.radians(lon))
    z = math.sin(math.radians(lat))
    x_r = math.cos(rot_lon) * x - math.sin(rot_lon) * y
    y_r = math.sin(rot_lon) * x + math.cos(rot_lon) * y
    lat_rot = math.degrees(math.asin(z))
    lon_rot = math.degrees(math.atan2(y_r, x_r))
    lon_rot = (lon_rot + 180) % 360 - 180
    expected_x = (lon_rot + 180) / 360 * img.shape[1]
    expected_y = (lat_rot + 90) / 180 * img.shape[0]
    assert np.isclose(map_x[0, 0], expected_x)
    assert np.isclose(map_y[0, 0], expected_y)


def test_preprocess_sequence(monkeypatch, prep_module):
    module, _ = prep_module
    P = module.PreprocessEquirectangularImage
    call_order = []

    def stub_extend(img, angle):
        call_order.append("extend")
        return np.vstack((img, np.zeros_like(img[:1])))

    def stub_rotate(img, dlat, dlon, interpolation=None):
        call_order.append("rotate")
        return img + 1

    def stub_resize(self, img):
        call_order.append("resize")
        return img * 2

    monkeypatch.setattr(P, "extend_height", stub_extend)
    monkeypatch.setattr(P, "rotate", stub_rotate)
    monkeypatch.setattr(module.ImageResizer, "resize_image", stub_resize)

    img = np.ones((2, 2), dtype=np.float32)
    out = P.preprocess(
        img,
        shadow_angle=10,
        delta_lat=0,
        delta_lon=0,
        resize_factor=0.5,
        resize_method="skimage",
    )
    assert call_order == ["extend", "rotate", "resize"]
    expected = stub_resize(None, stub_rotate(stub_extend(img, 10), 0, 0))
    assert np.array_equal(out, expected)
