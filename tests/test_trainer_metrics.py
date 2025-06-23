import importlib.util
import sys
from types import ModuleType
from pathlib import Path
import pytest

try:
    import numpy as np
except Exception as e:  # pragma: no cover - skip if numpy missing
    pytest.skip(f"Skipping metrics tests because numpy import failed: {e}", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]

# Stub skimage.metrics.structural_similarity used in metrics module
skimage_stub = ModuleType('skimage')
metrics_stub = ModuleType('skimage.metrics')

def dummy_ssim(a, b, *, data_range=None):
    return 0.5
metrics_stub.structural_similarity = dummy_ssim
skimage_stub.metrics = metrics_stub
sys.modules.setdefault('skimage', skimage_stub)
sys.modules.setdefault('skimage.metrics', metrics_stub)

spec = importlib.util.spec_from_file_location(
    'metrics', ROOT / 'panorai_models' / 'trainers' / 'metrics.py'
)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)
MonocularDepthMetrics = metrics.MonocularDepthMetrics


def test_monocular_depth_metrics_basic():
    m = MonocularDepthMetrics()
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    gt = np.array([[1.0, 1.0], [2.0, 4.0]])
    mask = np.ones_like(pred)
    m.update(pred, gt, mask)
    res = m.compute()
    expected_mae = np.mean(np.abs(pred - gt))
    expected_rmse = np.sqrt(np.mean((pred - gt) ** 2))
    assert pytest.approx(res["MAE"], rel=1e-6) == expected_mae
    assert pytest.approx(res["RMSE"], rel=1e-6) == expected_rmse
    assert res["Structural Similarity (SSIM)"] == 0.5


def test_metrics_no_valid_pixels_returns_nan():
    m = MonocularDepthMetrics()
    pred = np.zeros((2, 2))
    gt = np.zeros((2, 2))
    mask = np.zeros((2, 2))
    m.update(pred, gt, mask)
    res = m.compute()
    assert np.isnan(res["MAE"])
    assert np.isnan(res["RMSE"])
