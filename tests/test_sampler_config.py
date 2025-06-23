import importlib.util
import sys
from types import ModuleType
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
    """Provide minimal stubs for external dependencies."""
    panorai_pkg = ModuleType("panorai")
    panorai_pkg.__path__ = [str(ROOT / "panorai")]
    monkeypatch.setitem(sys.modules, "panorai", panorai_pkg)

    config_pkg = ModuleType("panorai.config")
    config_pkg.__path__ = [str(ROOT / "panorai" / "config")]
    monkeypatch.setitem(sys.modules, "panorai.config", config_pkg)

    utils_pkg = ModuleType("panorai.utils")
    utils_pkg.__path__ = [str(ROOT / "panorai" / "utils")]
    monkeypatch.setitem(sys.modules, "panorai.utils", utils_pkg)

    for name in ("numpy", "cv2", "skimage"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))

    class BaseModel:
        def __init__(self, **kwargs):
            fields = {k: v for k, v in self.__class__.__dict__.items()
                      if not k.startswith('_') and not callable(v)}
            for fname, default in fields.items():
                if fname == "n_points":
                    val = kwargs.pop(fname, default)
                    if val is not None and not isinstance(val, int):
                        raise ValueError("n_points must be int")
                    setattr(self, fname, val)
                else:
                    setattr(self, fname, kwargs.pop(fname, default))
            for k, v in kwargs.items():
                setattr(self, k, v)

        def dict(self):
            return dict(self.__dict__)

        def model_copy(self, update=None):
            data = self.dict()
            if update:
                data.update(update)
            return self.__class__(**data)

        copy = model_copy

    def Field(*, default=None, default_factory=None):
        if default_factory is not None:
            return default_factory()
        return default

    pyd_stub = ModuleType("pydantic")
    pyd_stub.BaseModel = BaseModel
    pyd_stub.Field = Field
    monkeypatch.setitem(sys.modules, "pydantic", pyd_stub)

    yield
    sys.modules.pop("panorai.samplers.config", None)
    sys.modules.pop("panorai.config.registry", None)
    sys.modules.pop("panorai.utils.exceptions", None)


@pytest.fixture
def SamplerConfig_cls():
    registry_module = _load_module(
        "panorai.config.registry", "panorai/config/registry.py"
    )
    exceptions_module = _load_module(
        "panorai.utils.exceptions", "panorai/utils/exceptions.py"
    )
    cfg_module = _load_module(
        "panorai.samplers.config", "panorai/samplers/config.py"
    )
    return (
        cfg_module.SamplerConfig,
        exceptions_module.ConfigurationError,
        registry_module.ConfigRegistry,
    )


def test_initialization_and_update(SamplerConfig_cls):
    SamplerConfig, _, registry = SamplerConfig_cls
    cfg = SamplerConfig(n_points=10, rotations=[(0, 0)])
    assert cfg.n_points == 10
    assert cfg["n_points"] == 10
    cfg.update(n_points=20)
    assert cfg.n_points == 20
    assert list(cfg) == list(cfg._config.dict().keys())
    rep = repr(cfg)
    assert "20" in rep and "rotations" in rep
    registry._configs.pop("sampler_config", None)


def test_invalid_initialization_raises(SamplerConfig_cls):
    SamplerConfig, ConfigurationError, registry = SamplerConfig_cls
    with pytest.raises(ConfigurationError):
        SamplerConfig(n_points="bad")
    registry._configs.pop("sampler_config", None)
