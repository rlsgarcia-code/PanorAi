import importlib.util
import sys
from types import ModuleType
from pathlib import Path
import math
import random
import pytest

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Lightweight NumPy stub implementations
# ---------------------------------------------------------------------------
class Array(list):
    def __add__(self, other):
        if isinstance(other, (list, tuple, Array)):
            return Array([a + b for a, b in zip(self, other)])
        return Array([a + other for a in self])

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, (list, tuple, Array)):
            return Array([a - b for a, b in zip(self, other)])
        return Array([a - other for a in self])

    def __rsub__(self, other):
        if isinstance(other, (list, tuple, Array)):
            return Array([b - a for a, b in zip(self, other)])
        return Array([other - a for a in self])

    def __mul__(self, other):
        if isinstance(other, (list, tuple, Array)):
            return Array([a * b for a, b in zip(self, other)])
        return Array([a * other for a in self])

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (list, tuple, Array)):
            return Array([a / b for a, b in zip(self, other)])
        return Array([a / other for a in self])

    def __mod__(self, other):
        if isinstance(other, (list, tuple, Array)):
            return Array([a % b for a, b in zip(self, other)])
        return Array([a % other for a in self])

    __rmod__ = __mod__

    def copy(self):
        return Array(self[:])


def array(obj):
    if isinstance(obj, Array):
        return Array(obj)
    if isinstance(obj, (list, tuple)):
        if obj and isinstance(obj[0], (list, tuple, Array)):
            return Array([Array(o) for o in obj])
        return Array(list(obj))
    return Array([obj])


def _map(func, x):
    if isinstance(x, (list, tuple, Array)):
        return Array([func(v) for v in x])
    return func(x)


def sin(x):
    return _map(math.sin, x)


def cos(x):
    return _map(math.cos, x)


def arcsin(x):
    return _map(math.asin, x)


def arccos(x):
    return _map(math.acos, x)


def arctan2(y, x):
    if isinstance(y, (list, tuple, Array)):
        return Array([math.atan2(yy, xx) for yy, xx in zip(y, x)])
    return math.atan2(y, x)


def sqrt(x):
    return math.sqrt(x)


def linspace(start, stop, num):
    if num == 1:
        return Array([start])
    step = (stop - start) / (num - 1)
    return Array([start + step * i for i in range(num)])


def degrees(x):
    return _map(lambda v: v * 180 / math.pi, x)


def arange(start, stop=None, step=1):
    if stop is None:
        stop = start
        start = 0
    result = []
    val = start
    if step > 0:
        while val < stop:
            result.append(val)
            val += step
    else:
        while val > stop:
            result.append(val)
            val += step
    return Array(result)


def vstack(items):
    result = []
    for it in items:
        if isinstance(it, (list, tuple, Array)) and it and isinstance(it[0], (list, tuple, Array)):
            result.extend(it)
        else:
            result.append(it)
    return Array(result)


def linalg_norm(vec):
    return math.sqrt(sum(v * v for v in vec))


def random_uniform(low, high, size=None):
    if size is None:
        return random.uniform(low, high)
    return Array([random.uniform(low, high) for _ in range(size)])


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _load_module(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    """Provide lightweight stubs for numpy and package structure."""
    numpy_stub = ModuleType("numpy")
    numpy_stub.sin = sin
    numpy_stub.cos = cos
    numpy_stub.arcsin = arcsin
    numpy_stub.arccos = arccos
    numpy_stub.arctan2 = arctan2
    numpy_stub.sqrt = sqrt
    numpy_stub.linspace = linspace
    numpy_stub.arange = arange
    numpy_stub.degrees = degrees
    numpy_stub.pi = math.pi
    numpy_stub.ndarray = Array
    numpy_stub.array = array
    numpy_stub.vstack = vstack
    linalg_mod = ModuleType("numpy.linalg")
    linalg_mod.norm = linalg_norm
    numpy_stub.linalg = linalg_mod
    random_mod = ModuleType("numpy.random")
    random_mod.uniform = random_uniform
    numpy_stub.random = random_mod
    monkeypatch.setitem(sys.modules, "numpy", numpy_stub)
    monkeypatch.setitem(sys.modules, "numpy.linalg", linalg_mod)
    monkeypatch.setitem(sys.modules, "numpy.random", random_mod)

    panorai_pkg = ModuleType("panorai")
    panorai_pkg.__path__ = [str(ROOT / "panorai")]
    monkeypatch.setitem(sys.modules, "panorai", panorai_pkg)

    samplers_pkg = ModuleType("panorai.samplers")
    samplers_pkg.__path__ = [str(ROOT / "panorai" / "samplers")]
    monkeypatch.setitem(sys.modules, "panorai.samplers", samplers_pkg)

    config_stub = ModuleType("panorai.samplers.config")

    class SamplerConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.extra = kwargs

        def update(self, **kwargs):
            self.__dict__.update(kwargs)
            self.extra.update(kwargs)

    config_stub.SamplerConfig = SamplerConfig
    monkeypatch.setitem(sys.modules, "panorai.samplers.config", config_stub)

    yield

    for mod in [
        "numpy",
        "numpy.linalg",
        "numpy.random",
        "panorai",
        "panorai.samplers",
        "panorai.samplers.config",
        "panorai.samplers.base_samplers",
        "panorai.samplers.registry",
        "panorai.samplers.default_samplers",
    ]:
        sys.modules.pop(mod, None)


@pytest.fixture
def default_samplers(patch_dependencies):
    _load_module("panorai.samplers.base_samplers", "panorai/samplers/base_samplers.py")
    registry = _load_module("panorai.samplers.registry", "panorai/samplers/registry.py")
    return _load_module("panorai.samplers.default_samplers", "panorai/samplers/default_samplers.py")


def test_cube_sampler_rotation(default_samplers):
    cfg_cls = sys.modules["panorai.samplers.config"].SamplerConfig
    cfg = cfg_cls(rotations=[(90, 0)])
    sampler = default_samplers.CubeSampler(config=cfg)
    pts = sampler.get_tangent_points()
    assert len(pts) == 12


def test_icosahedron_unique_vertices(default_samplers):
    sampler = default_samplers.IcosahedronSampler(subdivisions=0)
    pts = sampler.get_tangent_points()
    assert len(pts) == len(set(pts)) == 12


def test_fibonacci_sampler_count(default_samplers):
    sampler = default_samplers.FibonacciSampler(n_points=5)
    pts = sampler.get_tangent_points()
    assert len(pts) == 5


def test_spiral_sampler_count(default_samplers):
    sampler = default_samplers.SpiralSampler(n_points=7)
    pts = sampler.get_tangent_points()
    assert len(pts) == 7


def test_blue_noise_no_rotation(default_samplers):
    cfg_cls = sys.modules["panorai.samplers.config"].SamplerConfig
    cfg = cfg_cls(n_points=10, rotations=[(45, 0)])
    sampler = default_samplers.BlueNoiseSampler(config=cfg)
    pts = sampler.get_tangent_points()
    assert len(pts) <= 10
    assert len(pts) == len(set(pts))
