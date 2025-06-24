import importlib.util
import sys
from pathlib import Path
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
spec.loader.exec_module(transforms)
NormalizeImage = transforms.NormalizeImage


def test_normalize_image_simple():
    norm = NormalizeImage(mean=0.5, std=0.5)
    sample = {'rgb_image': np.array([[0, 255]], dtype=np.float32)}
    result = norm(sample.copy())
    expected = (sample['rgb_image'] / 255.0 - 0.5) / 0.5
    assert np.allclose(result['rgb_image'], expected)

