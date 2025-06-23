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


class DummyGnomonicFace:
    def __init__(self, data, lat=0.0, lon=0.0, fov=90.0):
        self._data = np.asarray(data)
        self.lat = lat
        self.lon = lon
        self.fov = fov

    def __array__(self, dtype=None, copy=True):
        return np.array(self._data, dtype=dtype, copy=copy)


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # minimal open3d stub
    o3d = ModuleType("open3d")
    geo = ModuleType("open3d.geometry")
    util = ModuleType("open3d.utility")

    class DummyPointCloud:
        def __init__(self):
            self.points = None
            self.colors = None

    class DummyMesh:
        @staticmethod
        def create_arrow(**kwargs):
            return DummyMesh()

        def rotate(self, *a, **k):
            pass

        def translate(self, *a, **k):
            pass

        def paint_uniform_color(self, *a, **k):
            pass

    geo.PointCloud = DummyPointCloud
    geo.TriangleMesh = DummyMesh
    geo.get_rotation_matrix_from_xyz = lambda a: a

    util.Vector3dVector = lambda a: a

    o3d.geometry = geo
    o3d.utility = util
    monkeypatch.setitem(sys.modules, "open3d", o3d)
    monkeypatch.setitem(sys.modules, "open3d.geometry", geo)
    monkeypatch.setitem(sys.modules, "open3d.utility", util)

    # cv2 stub
    cv2_stub = ModuleType("cv2")
    cv2_stub.CV_64F = 0

    def sobel(img, ddepth, dx, dy, ksize=3):
        if dx == 1:
            return np.gradient(img, axis=1)
        elif dy == 1:
            return np.gradient(img, axis=0)
        return np.zeros_like(img)

    cv2_stub.Sobel = sobel
    monkeypatch.setitem(sys.modules, "cv2", cv2_stub)

    # panorai.data.GnomonicFace stub
    data_pkg = ModuleType("panorai.data")
    data_pkg.GnomonicFace = DummyGnomonicFace
    monkeypatch.setitem(sys.modules, "panorai.data", data_pkg)

    yield


@pytest.fixture
def PCD(patch_dependencies):
    module = _load_module("panorai.pcd.data", "panorai/pcd/data.py")
    return module.PCD


@pytest.fixture
def PCDHandler(PCD):
    module = _load_module("panorai.pcd.handler", "panorai/pcd/handler.py")
    return module.PCDHandler


@pytest.fixture
def BaseBlender(PCD):
    module = _load_module(
        "panorai.pcd.blender.base_blender", "panorai/pcd/blender/base_blender.py"
    )
    return module.BaseBlender


def test_pcd_basic_properties(PCD):
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    colors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=float)
    pcd = PCD(points, colors)
    assert np.array_equal(pcd.points, points)
    assert np.array_equal(pcd.colors, colors)
    assert pcd.radius_image is None
    assert pcd.o3d.points is points
    assert pcd.o3d.colors is colors


def test_pcd_hook_bias_adjustment(PCD):
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    r_img = DummyGnomonicFace([[1.0, 2.0], [3.0, 4.0]], lat=10, lon=20, fov=30)
    pcd = PCD(points, colors, radius_image=r_img)

    def scale(r):
        return r * 2

    new_pcd = pcd.hook_bias_adjustment(scale)
    expected_points = points * 2
    assert np.allclose(new_pcd.points, expected_points)
    assert np.allclose(np.array(new_pcd.radius_image), scale(np.array(r_img)))
    assert new_pcd.radius_image.lat == r_img.lat
    assert new_pcd.radius_image.lon == r_img.lon


def test_pcd_hook_bias_adjustment_zero_radius(PCD):
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    colors = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    r_img = DummyGnomonicFace([[1.0, 2.0], [3.0, 4.0]], lat=10, lon=20, fov=30)
    pcd = PCD(points, colors, radius_image=r_img)

    def scale(r):
        return r * 2

    new_pcd = pcd.hook_bias_adjustment(scale)
    expected_points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    assert np.allclose(new_pcd.points, expected_points)


def test_mask_high_gradient(PCDHandler):
    depth = np.zeros((3, 3), dtype=float)
    depth[1, 1] = 1.0
    mask = PCDHandler.mask_high_gradient(depth, threshold=0.1)
    assert mask.shape == depth.shape
    assert not mask[1, 1]
    assert mask[0, 0]


def test_to_xyz_and_rotations(PCDHandler):
    x, y, z = PCDHandler.to_xyz(0, 0, 1)
    assert pytest.approx(x) == 1.0
    assert pytest.approx(y) == 0.0
    assert pytest.approx(z) == 0.0

    R = PCDHandler.compute_rotation_matrices(0, 0)
    expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    assert np.allclose(R, expected)

    pts = np.array([[1.0, 0.0, 0.0]])
    rotated = PCDHandler.rotate_ccs_to_wcs(pts, 0, 0)
    assert np.allclose(rotated, pts @ expected)


def test_create_axis_arrows(PCDHandler):
    arrows = PCDHandler.create_axis_arrows(scale=1.0, shift=1.0)
    assert len(arrows) == 3


def test_base_blender_compute_radius(BaseBlender):
    depth = np.ones((2, 2), dtype=float)
    u, v = np.meshgrid(np.linspace(-1, 1, 2), np.linspace(-1, 1, 2), indexing="xy")
    radius = BaseBlender.compute_radius(depth, u, v)
    expected = np.sqrt((depth * u) ** 2 + (depth * v) ** 2 + depth ** 2)
    assert np.allclose(radius, expected)


def test_base_blender_to_pcd(BaseBlender):
    class Dummy(BaseBlender):
        def process_faceset(self, *a, **k):
            pass

    blender = Dummy(min_radius=0.5, max_radius=2.0)
    radius = np.array([[1.0, 0.2], [1.5, 3.0]], dtype=float)
    colors = np.zeros((2, 2, 3), dtype=float)
    pcd = blender._to_pcd(radius, colors)
    assert isinstance(pcd.points, np.ndarray)
    assert pcd.points.shape[0] == 2
    assert pcd.colors.shape[0] == 2
    assert np.array_equal(pcd.radius_image, radius)
