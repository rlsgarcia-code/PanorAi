import importlib.util
import sys
from types import ModuleType
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

panorai_pkg = ModuleType("panorai")
panorai_pkg.__path__ = [str(ROOT / "panorai")]
sys.modules.setdefault("panorai", panorai_pkg)

utils_pkg = ModuleType("panorai.utils")
utils_pkg.__path__ = [str(ROOT / "panorai" / "utils")]
sys.modules.setdefault("panorai.utils", utils_pkg)

samplers_pkg = ModuleType("panorai.samplers")
samplers_pkg.__path__ = [str(ROOT / "panorai" / "samplers")]
sys.modules.setdefault("panorai.samplers", samplers_pkg)

blenders_pkg = ModuleType("panorai.blenders")
blenders_pkg.__path__ = [str(ROOT / "panorai" / "blenders")]
sys.modules.setdefault("panorai.blenders", blenders_pkg)

projections_pkg = ModuleType("panorai.projections")
projections_pkg.__path__ = [str(ROOT / "panorai" / "projections")]
sys.modules.setdefault("panorai.projections", projections_pkg)


def _load_module(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module

registry_module = _load_module(
    "panorai.utils.registry", "panorai/utils/registry.py"
)


class DummySampler:
    def __init__(self, **kw):
        self.kw = kw


class DummyBlender:
    def __init__(self, **kw):
        self.kw = kw


class DummyProjection:
    def __init__(self, **kw):
        self.kw = kw


class SamplerError(Exception):
    pass


class SamplerRegistryStub:
    names = ["sampler"]

    @classmethod
    def available_samplers(cls):
        return cls.names

    @classmethod
    def create(cls, name: str, **kw):
        if name not in cls.names:
            raise SamplerError(name)
        return DummySampler(**kw)


class BlenderError(ValueError):
    pass


class BlenderRegistryStub:
    names = ["blender"]

    @classmethod
    def available_blenders(cls):
        return cls.names

    @classmethod
    def create(cls, name: str, **kw):
        if name not in cls.names:
            raise BlenderError(name)
        return DummyBlender(**kw)


class ProjectionRegistryStub:
    names = ["projection"]

    @classmethod
    def available_projections(cls):
        return cls.names

    @classmethod
    def create(cls, name: str, **kw):
        if name not in cls.names:
            raise KeyError(name)
        return DummyProjection(**kw)


@pytest.fixture(autouse=True)
def patch_registry(monkeypatch):
    monkeypatch.setattr(registry_module, "SamplerRegistry", SamplerRegistryStub)
    monkeypatch.setattr(registry_module, "BlenderRegistry", BlenderRegistryStub)
    monkeypatch.setattr(registry_module, "ProjectionRegistry", ProjectionRegistryStub)
    yield


def test_available_and_create():
    assert registry_module.PanoraiRegistry.available_samplers() == ["sampler"]
    assert registry_module.PanoraiRegistry.available_blenders() == ["blender"]
    assert registry_module.PanoraiRegistry.available_projections() == ["projection"]

    sampler = registry_module.PanoraiRegistry.create_sampler("sampler", a=1)
    assert isinstance(sampler, DummySampler)
    assert sampler.kw["a"] == 1

    blender = registry_module.PanoraiRegistry.create_blender("blender", b=2)
    assert isinstance(blender, DummyBlender)
    assert blender.kw["b"] == 2

    projection = registry_module.PanoraiRegistry.create_projection("projection", c=3)
    assert isinstance(projection, DummyProjection)
    assert projection.kw["c"] == 3


def test_missing_names_raise():
    with pytest.raises(SamplerError):
        registry_module.PanoraiRegistry.create_sampler("missing")
    with pytest.raises(BlenderError):
        registry_module.PanoraiRegistry.create_blender("missing")
    with pytest.raises(KeyError):
        registry_module.PanoraiRegistry.create_projection("missing")
