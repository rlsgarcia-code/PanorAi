import importlib.util
import sys
from types import ModuleType
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Stub SamplerConfig and minimal package structure
# ---------------------------------------------------------------------------

sampler_config_stub = ModuleType("panorai.samplers.config")

class SamplerConfig:
    def __init__(self, **kwargs):
        self._config = dict(kwargs)

    def update(self, **kwargs):
        self._config.update(kwargs)

    def __getattr__(self, item):
        if item in self._config:
            return self._config[item]
        raise AttributeError(item)

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

Sampler = base_samplers.Sampler


def test_rotate_points_clamps_and_normalizes():
    class DummySampler(Sampler):
        def __init__(self, points, **kwargs):
            super().__init__(**kwargs)
            self.points = points

        def get_tangent_points(self):
            return self.points

    points = [(60, 170), (-70, -170)]
    cfg = SamplerConfig(rotations=[(90, 90), (-90, -180)])
    sampler = DummySampler(points, config=cfg)

    rotated = sampler._rotate_points(points)
    expected = [
        (60, 170),
        (-70, -170),
        (30, 80),
        (20, -80),
        (-30, -10),
        (-20, -170),
    ]
    assert rotated == expected
    for lat, lon in rotated:
        assert -90 <= lat <= 90
        assert -180 <= lon <= 180
