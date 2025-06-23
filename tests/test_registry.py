import importlib.util
import sys
from types import ModuleType
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Provide a minimal stub for the sampler configuration module used by
# ``base_samplers`` to avoid heavy dependencies during testing.
sampler_config_stub = ModuleType("panorai.samplers.config")

class SamplerConfig:
    def __init__(self, **kwargs):
        self._config = kwargs

    def update(self, **kwargs):
        self._config.update(kwargs)

sys.modules.setdefault("panorai.samplers.config", sampler_config_stub)
sampler_config_stub.SamplerConfig = SamplerConfig

def _load_module(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module

# Create minimal package structure without triggering panorai.__init__
panorai_pkg = ModuleType("panorai")
panorai_pkg.__path__ = [str(ROOT / "panorai")]
sys.modules.setdefault("panorai", panorai_pkg)

samplers_pkg = ModuleType("panorai.samplers")
samplers_pkg.__path__ = [str(ROOT / "panorai" / "samplers")]
sys.modules.setdefault("panorai.samplers", samplers_pkg)

base_samplers = _load_module("panorai.samplers.base_samplers", "panorai/samplers/base_samplers.py")
registry_module = _load_module("panorai.samplers.registry", "panorai/samplers/registry.py")

Sampler = base_samplers.Sampler
SamplerRegistry = registry_module.SamplerRegistry


def test_register_and_create_dummy_sampler():
    name = "dummy_test_sampler"
    # Ensure a clean state for the test
    if name in SamplerRegistry._registry:
        del SamplerRegistry._registry[name]

    @SamplerRegistry.register(name)
    class DummySampler(Sampler):
        def get_tangent_points(self):
            return [(0, 0)]

    try:
        assert name in SamplerRegistry.available_samplers()
        instance = SamplerRegistry.create(name)
        assert isinstance(instance, DummySampler)
        assert instance.get_tangent_points() == [(0, 0)]
    finally:
        # Clean up to avoid side effects on other tests
        if name in SamplerRegistry._registry:
            del SamplerRegistry._registry[name]
