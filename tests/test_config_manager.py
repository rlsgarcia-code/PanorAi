from types import ModuleType
import sys
import pytest

from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]

panorai_pkg = ModuleType("panorai")
panorai_pkg.__path__ = [str(ROOT / "panorai")]
sys.modules.setdefault("panorai", panorai_pkg)

config_pkg = ModuleType("panorai.config")
config_pkg.__path__ = [str(ROOT / "panorai" / "config")]
sys.modules.setdefault("panorai.config", config_pkg)

yaml_stub = ModuleType("yaml")
yaml_stub.dump = lambda *a, **k: ""
sys.modules.setdefault("yaml", yaml_stub)

def _load_module(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module

registry_module = _load_module("panorai.config.registry", "panorai/config/registry.py")
manager_module = _load_module("panorai.config.config_manager", "panorai/config/config_manager.py")

ConfigRegistry = registry_module.ConfigRegistry
ConfigManager = manager_module.ConfigManager

# Provide stub modules expected in _auto_discover_configs
sys.modules.setdefault("panorai.pipelines", ModuleType("panorai.pipelines"))
sys.modules.setdefault("panorai.pipelines.sampler", ModuleType("panorai.pipelines.sampler"))
sys.modules.setdefault("panorai.pipelines.sampler.config", ModuleType("panorai.pipelines.sampler.config"))

sys.modules.setdefault("panorai.preprocessing", ModuleType("panorai.preprocessing"))
sys.modules.setdefault("panorai.preprocessing.config", ModuleType("panorai.preprocessing.config"))

@ConfigRegistry.register("dummy_test_config")
class DummyConfig:
    def __init__(self, **kwargs):
        self.params = kwargs


def test_create_and_get_config():
    ConfigManager.reset()
    ConfigManager._loaded_configs = False
    cfg = ConfigManager.create("dummy_test_config", value=42)
    assert isinstance(cfg, DummyConfig)
    cfg_again = ConfigManager.get("dummy_test_config")
    assert cfg is cfg_again
    assert cfg.params["value"] == 42
    ConfigManager.reset()
    ConfigRegistry._configs.pop("dummy_test_config", None)


def test_modify_and_reset_config():
    @ConfigRegistry.register("mod_cfg")
    class ModConfig:
        def __init__(self, **kw):
            self.params = dict(kw)

        def update(self, params):
            self.params.update(params)

    ConfigManager.reset()
    cfg = ConfigManager.create("mod_cfg", a=1)
    ConfigManager.modify_config("mod_cfg", b=2)
    assert cfg.params == {"a": 1, "b": 2}
    ConfigManager.reset("mod_cfg")
    assert "mod_cfg" not in ConfigManager.get_all_configs()
    ConfigRegistry._configs.pop("mod_cfg", None)


def test_describe_and_get_all_configs(capsys):
    ConfigManager.reset()
    ConfigRegistry.register("desc_cfg")(DummyConfig)
    ConfigManager.create("desc_cfg", val=3)

    ConfigManager.describe_config("desc_cfg", output_format="json")
    out = capsys.readouterr().out
    assert '"val": 3' in out
    configs = ConfigManager.get_all_configs()
    assert "desc_cfg" in configs
    ConfigManager.reset()
    ConfigRegistry._configs.pop("desc_cfg", None)


def test_get_config_parameters_object():
    @ConfigRegistry.register("obj_cfg")
    class ObjCfg:
        def __init__(self, a=1):
            self.a = a
            self._private = 2

    ConfigManager.reset()
    ConfigManager.create("obj_cfg", a=5)
    params = ConfigManager.get_config_parameters("obj_cfg")
    assert params == {"a": 5}
    ConfigManager.reset()
    ConfigRegistry._configs.pop("obj_cfg", None)


def test_get_config_parameters_dict():
    @ConfigRegistry.register("dict_cfg")
    def dict_cfg(**kwargs):
        return dict(kwargs)

    ConfigManager.reset()
    ConfigManager.create("dict_cfg", x=1, y=2)
    params = ConfigManager.get_config_parameters("dict_cfg")
    assert params == {"x": 1, "y": 2}
    ConfigManager.reset()
    ConfigRegistry._configs.pop("dict_cfg", None)


def test_get_config_parameters_invalid():
    @ConfigRegistry.register("bad_cfg")
    def bad_cfg(**kw):
        return 42

    ConfigManager.reset()
    ConfigManager.create("bad_cfg")
    with pytest.raises(TypeError):
        ConfigManager.get_config_parameters("bad_cfg")
    ConfigManager.reset()
    ConfigRegistry._configs.pop("bad_cfg", None)
