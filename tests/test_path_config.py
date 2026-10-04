import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def _load_path_config():
    panorai_pkg = ModuleType("panorai")
    panorai_pkg.__path__ = [str(ROOT / "panorai")]
    sys.modules.setdefault("panorai", panorai_pkg)

    previous_yaml = sys.modules.get("yaml")
    yaml_stub = ModuleType("yaml")
    yaml_stub.safe_load = lambda src: json.loads(
        src.read() if hasattr(src, "read") else src
    )
    yaml_stub.safe_dump = lambda obj: json.dumps(obj)
    sys.modules["yaml"] = yaml_stub

    spec = importlib.util.spec_from_file_location(
        "panorai.path_config", ROOT / "panorai" / "path_config.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["panorai.path_config"] = module
    if previous_yaml is None:
        del sys.modules["yaml"]
    else:
        sys.modules["yaml"] = previous_yaml
    return module


path_config = _load_path_config()


def test_get_path_nested(monkeypatch, tmp_path):
    data = {
        "datasets": {"train": "/tmp/train", "nested": {"sub": "value"}},
        "model": {"ckpts": {"best": "best.ckpt"}},
    }
    cfg_file = tmp_path / "paths.yaml"
    cfg_file.write_text(json.dumps(data))
    monkeypatch.setenv("PANORAI_PATHS", str(cfg_file))
    monkeypatch.setattr(path_config, "_paths_cache", None)

    assert path_config.get_path("datasets", "train") == "/tmp/train"
    assert path_config.get_path("model", "ckpts", "best") == "best.ckpt"
    assert path_config.get_path("datasets", "nested", "sub") == "value"
    assert path_config.get_path("missing", "key", default="default") == "default"
