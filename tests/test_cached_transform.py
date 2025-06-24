import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    'cached', ROOT / 'panorai' / 'depth' / 'trainers' / 'utils' / 'cached_transforms.py'
)
cached = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cached)
CachedTransform = cached.CachedTransform


def test_cached_transform_caches_results():
    calls = []
    def transform(x):
        calls.append(x)
        return x * 2
    ct = CachedTransform(transform, maxsize=4)
    data = {'a': 1, 'b': 2}
    ct.attach_data(data)
    assert ct('a') == 2
    assert ct('a') == 2
    assert calls == [1]
    assert ct('b') == 4
    assert calls == [1, 2]

