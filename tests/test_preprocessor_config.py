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
def patch_packages(monkeypatch):
    panorai_pkg = ModuleType("panorai")
    panorai_pkg.__path__ = [str(ROOT / "panorai")]
    monkeypatch.setitem(sys.modules, "panorai", panorai_pkg)

    config_pkg = ModuleType("panorai.config")
    config_pkg.__path__ = [str(ROOT / "panorai" / "config")]
    monkeypatch.setitem(sys.modules, "panorai.config", config_pkg)

    utils_pkg = ModuleType("panorai.utils")
    utils_pkg.__path__ = [str(ROOT / "panorai" / "utils")]
    monkeypatch.setitem(sys.modules, "panorai.utils", utils_pkg)

    preprocessing_pkg = ModuleType("panorai.preprocessing")
    preprocessing_pkg.__path__ = [str(ROOT / "panorai" / "preprocessing")]
    monkeypatch.setitem(sys.modules, "panorai.preprocessing", preprocessing_pkg)

    yield

    sys.modules.pop("panorai.preprocessing.config", None)
    sys.modules.pop("panorai.config.registry", None)
    sys.modules.pop("panorai.utils.exceptions", None)


@pytest.fixture
def PreprocessorConfig_cls():
    registry_module = _load_module(
        "panorai.config.registry", "panorai/config/registry.py"
    )
    exceptions_module = _load_module(
        "panorai.utils.exceptions", "panorai/utils/exceptions.py"
    )
    cfg_module = _load_module(
        "panorai.preprocessing.config", "panorai/preprocessing/config.py"
    )
    return (
        cfg_module.PreprocessorConfig,
        exceptions_module.ConfigurationError,
        registry_module.ConfigRegistry,
    )


def test_initialization_and_update(PreprocessorConfig_cls):
    PreprocessorConfig, _, registry = PreprocessorConfig_cls
    cfg = PreprocessorConfig(
        shadow_angle=5.0,
        delta_lat=1.0,
        delta_lon=2.0,
        resize_factor=0.5,
        resize_method="skimage",
    )
    assert cfg.shadow_angle == 5.0
    assert cfg["shadow_angle"] == 5.0
    cfg.update(shadow_angle=10.0, resize_factor=1.0)
    assert cfg.shadow_angle == 10.0
    assert cfg.resize_factor == 1.0
    expected_keys = list(getattr(cfg._config, "model_dump", cfg._config.dict)().keys())
    assert list(cfg) == expected_keys
    registry._configs.pop("preprocessor_config", None)


def test_invalid_resize_factor_raises(PreprocessorConfig_cls):
    PreprocessorConfig, ConfigurationError, registry = PreprocessorConfig_cls
    with pytest.raises(ConfigurationError):
        PreprocessorConfig(resize_factor=-1)
    registry._configs.pop("preprocessor_config", None)

