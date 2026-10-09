from types import ModuleType
import sys
from pathlib import Path
import importlib.util
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]

panorai_pkg = ModuleType("panorai")
panorai_pkg.__path__ = [str(ROOT / "panorai")]

config_pkg = ModuleType("panorai.config")
config_pkg.__path__ = [str(ROOT / "panorai" / "config")]

yaml_stub = ModuleType("yaml")
yaml_stub.dump = lambda *a, **k: ""

# Install initial stubs only while loading this test's isolated module objects.
# Pytest imports every test module before fixtures run, so leaving these stubs
# installed until fixture teardown poisons collection of unrelated tests.
_module_names = (
    "panorai",
    "panorai.config",
    "panorai.config.registry",
    "panorai.config.config_manager",
    "yaml",
)
_original_modules = {name: sys.modules.get(name) for name in _module_names}
for _name, _mod in {
    "panorai": panorai_pkg,
    "panorai.config": config_pkg,
    "yaml": yaml_stub,
}.items():
    sys.modules[_name] = _mod

def _load_module(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module

registry_module = _load_module(
    "panorai.config.registry", "panorai/config/registry.py"
)
manager_module = _load_module(
    "panorai.config.config_manager", "panorai/config/config_manager.py"
)

ConfigRegistry = registry_module.ConfigRegistry
ConfigManager = manager_module.ConfigManager

# Collection must leave the interpreter exactly as it found it. The fixture
# below re-installs only the dependencies needed while each test executes.
for _name, _module in _original_modules.items():
    if _module is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _module


@pytest.fixture(autouse=True)
def patch_sys_modules(monkeypatch):
    """Stub optional modules used during configuration discovery."""
    modules = {
        "panorai": panorai_pkg,
        "panorai.config": config_pkg,
        "yaml": yaml_stub,
        "panorai.pipelines": ModuleType("panorai.pipelines"),
        "panorai.pipelines.sampler": ModuleType("panorai.pipelines.sampler"),
        "panorai.pipelines.sampler.config": ModuleType(
            "panorai.pipelines.sampler.config"
        ),
        "panorai.preprocessing": ModuleType("panorai.preprocessing"),
        "panorai.preprocessing.config": ModuleType("panorai.preprocessing.config"),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    yield


def test_collection_does_not_poison_panorai_root() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-W",
            "error",
            "tests/test_config_manager.py",
            "tests/test_geometry_api.py",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

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


def test_modify_unknown_config_raises_key_error():
    ConfigManager.reset()
    with pytest.raises(KeyError):
        ConfigManager.modify_config("unknown_cfg", x=1)


def test_modify_unmodifiable_config_raises_type_error():
    @ConfigRegistry.register("imm_cfg")
    class ImmConfig:
        def __init__(self, **kw):
            self.params = dict(kw)

    ConfigManager.reset()
    ConfigManager.create("imm_cfg", a=1)
    with pytest.raises(TypeError):
        ConfigManager.modify_config("imm_cfg", b=2)
    ConfigManager.reset()
    ConfigRegistry._configs.pop("imm_cfg", None)
