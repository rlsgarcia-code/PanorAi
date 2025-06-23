import os
from pathlib import Path
import yaml

_CFG_ENV_VAR = "PANORAI_PATHS"
_DEFAULT_CFG = "paths.yaml"

_paths_cache = None


def _load_paths():
    global _paths_cache
    if _paths_cache is None:
        cfg_path = os.getenv(_CFG_ENV_VAR, _DEFAULT_CFG)
        cfg_file = Path(cfg_path)
        if cfg_file.is_file():
            with open(cfg_file, "r") as f:
                _paths_cache = yaml.safe_load(f) or {}
        else:
            _paths_cache = {}
    return _paths_cache


def get_path(*keys, default=None):
    data = _load_paths()
    for k in keys:
        if isinstance(data, dict) and k in data:
            data = data[k]
        else:
            return default
    return data
