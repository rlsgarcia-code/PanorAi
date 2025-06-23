import importlib.util
import sys
from types import ModuleType
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

panorai_pkg = ModuleType("panorai")
panorai_pkg.__path__ = [str(ROOT / "panorai")]
sys.modules.setdefault("panorai", panorai_pkg)

blender_pkg = ModuleType("panorai.blenders")
blender_pkg.__path__ = [str(ROOT / "panorai" / "blenders")]
sys.modules.setdefault("panorai.blenders", blender_pkg)

proj_pkg = ModuleType("panorai.projections")
proj_pkg.__path__ = [str(ROOT / "panorai" / "projections")]
sys.modules.setdefault("panorai.projections", proj_pkg)

def _load_module(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module

blender_module = _load_module("panorai.blenders.registry", "panorai/blenders/registry.py")
projection_module = _load_module("panorai.projections.registry", "panorai/projections/registry.py")

BlenderRegistry = blender_module.BlenderRegistry
ProjectionRegistry = projection_module.ProjectionRegistry


def test_blender_registry_register_and_create():
    name = "dummy_blender_test"
    class DummyBlender:
        def __init__(self, **kwargs):
            self.args = kwargs
    if name in BlenderRegistry._registry:
        del BlenderRegistry._registry[name]

    @BlenderRegistry.register(name)
    class _B(DummyBlender):
        pass

    try:
        assert name in BlenderRegistry.available_blenders()
        instance = BlenderRegistry.create(name, value=1)
        assert isinstance(instance, _B)
        assert instance.args["value"] == 1
    finally:
        BlenderRegistry._registry.pop(name, None)


def test_projection_registry_register_and_create():
    name = "dummy_projection_test"
    class DummyProj:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    if name in ProjectionRegistry._projections:
        del ProjectionRegistry._projections[name]

    @ProjectionRegistry.register(name)
    class _P(DummyProj):
        pass

    try:
        assert name in ProjectionRegistry.available_projections()
        inst = ProjectionRegistry.create(name, a=2)
        assert isinstance(inst, _P)
        assert inst.kwargs["a"] == 2
    finally:
        ProjectionRegistry._projections.pop(name, None)
