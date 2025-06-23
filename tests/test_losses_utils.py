import importlib.util
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
except Exception as e:  # pragma: no cover - skip when torch missing
    pytest.skip(f"Skipping loss utils tests because torch import failed: {e}", allow_module_level=True)

spec = importlib.util.spec_from_file_location(
    'losses', ROOT / 'panorai_models' / 'trainers' / 'losses.py'
)
losses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(losses)

SiLogLoss = losses.SiLogLoss
depth_to_points = losses.depth_to_points
get_uv_grid = losses.get_uv_grid


def test_depth_to_points_and_grid():
    depth = torch.ones(1, 1, 2, 2)
    pts = depth_to_points(depth)
    assert pts.shape == (1, 3, 2, 2)
    u, v = get_uv_grid(2, 2, depth.device)
    assert u.shape == (1, 1, 2, 2)
    assert v.shape == (1, 1, 2, 2)
    # check simple relation
    assert torch.allclose(pts[:, 0], depth * u)
    assert torch.allclose(pts[:, 1], depth * v)


def test_silog_loss_zero():
    loss_fn = SiLogLoss()
    p = torch.ones(1, 1, 2, 2)
    t = torch.ones_like(p)
    m = torch.ones_like(p)
    loss = loss_fn(p, t, m)
    assert loss.item() == pytest.approx(0.0)

