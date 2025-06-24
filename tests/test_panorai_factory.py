import importlib.util
import sys
from types import ModuleType
from pathlib import Path
import os
import pytest

ROOT = Path(__file__).resolve().parents[1]


class DummyEQ:
    def __init__(self, data):
        self.data = data


class DummyGF:
    def __init__(self, data, lat=0.0, lon=0.0, fov=90.0):
        self.data = data
        self.projection = None

    def attach_projection(self, name, **kwargs):
        self.projection = name


class DataFactoryStub:
    @classmethod
    def from_file(cls, path: str, data_type: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if data_type == "equirectangular":
            return DummyEQ(path)
        if data_type == "gnomonic_face":
            return DummyGF(path)
        raise ValueError("invalid type")

    @classmethod
    def from_array(cls, arr, data_type: str, **kwargs):
        if data_type == "equirectangular":
            return DummyEQ(arr)
        if data_type == "gnomonic_face":
            return DummyGF(arr)
        raise ValueError("invalid type")


class SamplerRegistryStub:
    names = ["sampler"]

    @classmethod
    def available_samplers(cls):
        return cls.names

    @classmethod
    def create(cls, name: str, **kwargs):
        return (name, kwargs)


class BlenderRegistryStub:
    names = ["blender"]

    @classmethod
    def available_blenders(cls):
        return cls.names

    @classmethod
    def create(cls, name: str, **kwargs):
        return (name, kwargs)


class ProjectionRegistryStub:
    names = ["gnomonic"]

    @classmethod
    def available_projections(cls):
        return cls.names

    @classmethod
    def create(cls, name: str, **kwargs):
        return (name, kwargs)


def _load_module(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


@pytest.fixture(autouse=True)
def patch_environment(monkeypatch):
    panorai_pkg = ModuleType("panorai")
    panorai_pkg.__path__ = [str(ROOT / "panorai")]
    monkeypatch.setitem(sys.modules, "panorai", panorai_pkg)

    data_pkg = ModuleType("panorai.data")
    data_pkg.__path__ = [str(ROOT / "panorai" / "data")]
    monkeypatch.setitem(sys.modules, "panorai.data", data_pkg)

    eq_mod = ModuleType("panorai.data.equirectangular_image")
    eq_mod.EquirectangularImage = DummyEQ
    monkeypatch.setitem(sys.modules, "panorai.data.equirectangular_image", eq_mod)

    gf_mod = ModuleType("panorai.data.gnomonic_image")
    gf_mod.GnomonicFace = DummyGF
    monkeypatch.setitem(sys.modules, "panorai.data.gnomonic_image", gf_mod)

    factory_mod = ModuleType("panorai.data.factory")
    factory_mod.DataFactory = DataFactoryStub
    monkeypatch.setitem(sys.modules, "panorai.data.factory", factory_mod)

    numpy_stub = ModuleType("numpy")
    numpy_stub.asarray = lambda x: x
    numpy_stub.ndarray = list
    monkeypatch.setitem(sys.modules, "numpy", numpy_stub)

    pil_mod = ModuleType("PIL")
    image_mod = ModuleType("PIL.Image")
    class DummyImage:
        pass
    pil_mod.Image = image_mod
    image_mod.Image = DummyImage
    monkeypatch.setitem(sys.modules, "PIL", pil_mod)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_mod)

    config_pkg = ModuleType("panorai.config")
    config_pkg.__path__ = [str(ROOT / "panorai" / "config")]
    monkeypatch.setitem(sys.modules, "panorai.config", config_pkg)
    config_mod = ModuleType("panorai.config.config_manager")
    class ConfigManagerStub:
        @classmethod
        def modify_config(cls, name, **kwargs):
            pass
        @classmethod
        def describe_config(cls, name):
            pass
        @classmethod
        def available_configs(cls):
            return []
        @classmethod
        def reset(cls):
            pass
    config_mod.ConfigManager = ConfigManagerStub
    monkeypatch.setitem(sys.modules, "panorai.config.config_manager", config_mod)

    samplers_pkg = ModuleType("panorai.samplers")
    samplers_pkg.__path__ = [str(ROOT / "panorai" / "samplers")]
    monkeypatch.setitem(sys.modules, "panorai.samplers", samplers_pkg)
    sr_mod = ModuleType("panorai.samplers.registry")
    sr_mod.SamplerRegistry = SamplerRegistryStub
    monkeypatch.setitem(sys.modules, "panorai.samplers.registry", sr_mod)

    blenders_pkg = ModuleType("panorai.blenders")
    blenders_pkg.__path__ = [str(ROOT / "panorai" / "blenders")]
    monkeypatch.setitem(sys.modules, "panorai.blenders", blenders_pkg)
    br_mod = ModuleType("panorai.blenders.registry")
    br_mod.BlenderRegistry = BlenderRegistryStub
    monkeypatch.setitem(sys.modules, "panorai.blenders.registry", br_mod)

    projections_pkg = ModuleType("panorai.projections")
    projections_pkg.__path__ = [str(ROOT / "panorai" / "projections")]
    monkeypatch.setitem(sys.modules, "panorai.projections", projections_pkg)
    pr_mod = ModuleType("panorai.projections.registry")
    pr_mod.ProjectionRegistry = ProjectionRegistryStub
    monkeypatch.setitem(sys.modules, "panorai.projections.registry", pr_mod)

    yield
    sys.modules.pop("panorai.factory.panorai_factory", None)


@pytest.fixture
def factory_module(patch_environment):
    return _load_module("panorai.factory.panorai_factory", "panorai/factory/panorai_factory.py")


def test_load_image_and_create_from_array(factory_module, tmp_path):
    PanoraiFactory = factory_module.PanoraiFactory

    img_path = tmp_path / "img.png"
    img_path.write_bytes(b"x")

    eq = PanoraiFactory.load_image(str(img_path), "equirectangular")
    assert isinstance(eq, DummyEQ)

    gf = PanoraiFactory.load_image(str(img_path), "gnomonic_face")
    assert isinstance(gf, DummyGF)
    assert gf.projection == "gnomonic"

    arr = [1, 2, 3]
    eq2 = PanoraiFactory.create_data_from_array(arr, "equirectangular")
    assert isinstance(eq2, DummyEQ)

    gf2 = PanoraiFactory.create_data_from_array(arr, "gnomonic_face")
    assert isinstance(gf2, DummyGF)
    assert gf2.projection == "gnomonic"


def test_load_image_invalid_paths(factory_module, tmp_path):
    PanoraiFactory = factory_module.PanoraiFactory
    with pytest.raises(ValueError):
        PanoraiFactory.load_image("not_an_image.txt")

    missing = tmp_path / "missing.png"
    with pytest.raises(FileNotFoundError):
        PanoraiFactory.load_image(str(missing))


def test_registry_not_found_errors(factory_module):
    PanoraiFactory = factory_module.PanoraiFactory
    with pytest.raises(factory_module.SamplerNotFoundError):
        PanoraiFactory.get_sampler("unknown")

    with pytest.raises(factory_module.BlenderNotFoundError):
        PanoraiFactory.get_blender("unknown")

    with pytest.raises(factory_module.ProjectionNotFoundError):
        PanoraiFactory.get_projection("unknown", lat=0.0, lon=0.0, fov=90.0)

