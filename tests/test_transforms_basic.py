import importlib.util
import sys
from pathlib import Path
from types import ModuleType
import pytest

try:
    import numpy as np
except Exception as e:  # pragma: no cover - skip if numpy missing
    pytest.skip(f"Skipping transform tests because numpy import failed: {e}", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    'transforms', ROOT / 'panorai' / 'depth' / 'trainers' / 'transforms.py'
)
transforms = importlib.util.module_from_spec(spec)

# This test exercises only the NumPy ``NormalizeImage`` helper. The legacy
# research module imports Torch but does not use it at module scope; keep the
# core test matrix independent of that optional backend and restore every
# temporary module immediately after loading.
_torch_names = ("torch", "torch.nn", "torch.nn.functional")
_original_torch_modules = {name: sys.modules.get(name) for name in _torch_names}
_torch = ModuleType("torch")
_torch.__path__ = []
_torch_nn = ModuleType("torch.nn")
_torch_nn.__path__ = []
_torch_functional = ModuleType("torch.nn.functional")
_torch.nn = _torch_nn
_torch_nn.functional = _torch_functional
sys.modules.update(
    {
        "torch": _torch,
        "torch.nn": _torch_nn,
        "torch.nn.functional": _torch_functional,
    }
)
try:
    spec.loader.exec_module(transforms)
finally:
    for _name, _module in _original_torch_modules.items():
        if _module is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _module
NormalizeImage = transforms.NormalizeImage


def test_normalize_image_simple():
    norm = NormalizeImage(mean=0.5, std=0.5)
    sample = {'rgb_image': np.array([[0, 255]], dtype=np.float32)}
    result = norm(sample.copy())
    expected = (sample['rgb_image'] / 255.0 - 0.5) / 0.5
    assert np.allclose(result['rgb_image'], expected)
