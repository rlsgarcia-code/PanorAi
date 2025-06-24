import importlib.util
import sys
from types import ModuleType, SimpleNamespace
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
    numpy_stub = ModuleType("numpy")
    numpy_stub.ndarray = list
    numpy_stub.float32 = "float32"
    numpy_stub.flip = lambda arr, axis=0: arr  # simple flip stub
    monkeypatch.setitem(sys.modules, "numpy", numpy_stub)

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
    cv2_stub.remap = lambda *a, **k: a[0]  # return input image
    monkeypatch.setitem(sys.modules, "cv2", cv2_stub)

    skimage_stub = ModuleType("skimage")
    transform_stub = ModuleType("skimage.transform")
    transform_stub.resize = lambda img, *a, **k: img
    skimage_stub.transform = transform_stub
    monkeypatch.setitem(sys.modules, "skimage", skimage_stub)
    monkeypatch.setitem(sys.modules, "skimage.transform", transform_stub)

    preproc_pkg = ModuleType("panorai.preprocessing")
    trans_mod = ModuleType("panorai.preprocessing.transformations")
    trans_mod.PreprocessEquirectangularImage = lambda x: x
    preproc_pkg.transformations = trans_mod
    monkeypatch.setitem(sys.modules, "panorai.preprocessing", preproc_pkg)
    monkeypatch.setitem(sys.modules, "panorai.preprocessing.transformations", trans_mod)
    yield
    for mod in [
        "numpy",
        "cv2",
        "skimage",
        "skimage.transform",
        "panorai.preprocessing",
        "panorai.preprocessing.transformations",
    ]:
        sys.modules.pop(mod, None)


@pytest.fixture
def modules(patch_dependencies):
    panorai_pkg = ModuleType("panorai")
    panorai_pkg.__path__ = [str(ROOT / "panorai")]
    sys.modules.setdefault("panorai", panorai_pkg)

    proj_pkg = ModuleType("panorai.projections")
    proj_pkg.__path__ = [str(ROOT / "panorai" / "projections")]
    sys.modules.setdefault("panorai.projections", proj_pkg)

    gn_pkg = ModuleType("panorai.projections.gnomonic")
    gn_pkg.__path__ = [str(ROOT / "panorai" / "projections" / "gnomonic")]
    sys.modules.setdefault("panorai.projections.gnomonic", gn_pkg)

    base_pkg = ModuleType("panorai.projections.base")
    base_pkg.__path__ = [str(ROOT / "panorai" / "projections" / "base")]
    sys.modules.setdefault("panorai.projections.base", base_pkg)

    cfg_registry_mod = _load_module("panorai.config.registry", "panorai/config/registry.py")
    cfg_registry_mod.ConfigRegistry._configs.pop("gnomonic_config", None)

    registry_mod = _load_module("panorai.projections.registry", "panorai/projections/registry.py")
    registry_mod.ProjectionRegistry._projections.pop("gnomonic", None)

    cfg_mod = _load_module("panorai.projections.gnomonic.config", "panorai/projections/gnomonic/config.py")
    grid_mod = _load_module("panorai.projections.gnomonic.grid", "panorai/projections/gnomonic/grid.py")
    strat_mod = _load_module("panorai.projections.gnomonic.strategy", "panorai/projections/gnomonic/strategy.py")
    trans_mod = _load_module("panorai.projections.gnomonic.transform", "panorai/projections/gnomonic/transform.py")
    interp_mod = _load_module("panorai.projections.base.interpolation", "panorai/projections/base/interpolation.py")
    proj_mod = _load_module("panorai.projections.gnomonic_projection", "panorai/projections/gnomonic_projection.py")

    return SimpleNamespace(
        GnomonicConfig=cfg_mod.GnomonicConfig,
        GnomonicGridGeneration=grid_mod.GnomonicGridGeneration,
        GnomonicProjectionStrategy=strat_mod.GnomonicProjectionStrategy,
        GnomonicTransformer=trans_mod.GnomonicTransformer,
        BaseInterpolation=interp_mod.BaseInterpolation,
        GnomonicProjection=proj_mod.GnomonicProjection,
        ProjectionRegistry=registry_mod.ProjectionRegistry,
    )


def test_gnomonic_config_attributes(modules):
    cfg = modules.GnomonicConfig(phi1_deg=10, lam0_deg=20, fov_deg=60)
    assert cfg.phi1_deg == 10
    assert cfg["lam0_deg"] == 20
    cfg.update(fov_deg=45)
    assert cfg.fov_deg == 45


def test_projection_registered(modules):
    assert "gnomonic" in modules.ProjectionRegistry.available_projections()


def test_project_pipeline(monkeypatch, modules):
    proj = modules.GnomonicProjection(config=modules.GnomonicConfig())
    monkeypatch.setattr(proj.grid_generator, "projection_grid", lambda: ("gx", "gy"))
    monkeypatch.setattr(proj.strategy, "from_projection_to_spherical", lambda x, y: ("lat", "lon"))
    monkeypatch.setattr(proj.transformer, "spherical_to_image_coords", lambda lat, lon, shape: ("mx", "my"))
    monkeypatch.setattr(proj.interpolation, "interpolate", lambda img, mx, my, mask=None: "result")
    class Img(list):
        @property
        def shape(self):
            return (1, 1)

    out = proj.project(Img([0]))
    assert out == "result"


def test_back_project_updates_config(monkeypatch, modules):
    proj = modules.GnomonicProjection(config=modules.GnomonicConfig())
    monkeypatch.setattr(proj.grid_generator, "spherical_grid", lambda: ("lon", "lat"))
    monkeypatch.setattr(proj.strategy, "from_spherical_to_projection", lambda lat, lon: ("x", "y", "mask"))
    monkeypatch.setattr(proj.transformer, "projection_to_image_coords", lambda x, y, c: ("mx", "my"))
    monkeypatch.setattr(proj.interpolation, "interpolate", lambda img, mx, my, mask=None: [[1]])
    class Img(list):
        @property
        def shape(self):
            return (1, 1)

    out = proj.back_project(Img([0]), (3, 4))
    assert proj.config.lat_points == 3
    assert proj.config.lon_points == 4
    assert out == [[1]]
