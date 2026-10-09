from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from benchmarks.spherical_monocular_depth.da3_spherical_attention import (
    sparse_cross_view_attention_correction,
)


def _identity_attention_block(channels: int = 4, heads: int = 2) -> object:
    qkv = torch.nn.Linear(channels, 3 * channels, bias=False)
    projection = torch.nn.Linear(channels, channels, bias=False)
    with torch.no_grad():
        qkv.weight.zero_()
        identity = torch.eye(channels)
        qkv.weight[:channels] = identity
        qkv.weight[channels : 2 * channels] = identity
        qkv.weight[2 * channels :] = identity
        projection.weight.copy_(identity)
    attention = SimpleNamespace(
        num_heads=heads,
        scale=(channels // heads) ** -0.5,
        qkv=qkv,
        q_norm=torch.nn.Identity(),
        k_norm=torch.nn.Identity(),
        proj=projection,
        proj_drop=torch.nn.Identity(),
    )
    return SimpleNamespace(attn=attention)


def test_self_only_sparse_attention_is_neutral() -> None:
    block = _identity_attention_block()
    target = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 0.0]])
    correction = sparse_cross_view_attention_correction(
        block, target, target[None], torch.ones((1, 2))
    )
    torch.testing.assert_close(correction, torch.zeros_like(target), rtol=0.0, atol=0.0)


def test_identical_overlapping_views_are_neutral() -> None:
    block = _identity_attention_block()
    target = torch.tensor([[1.0, 0.0, 0.5, -0.5], [0.0, 2.0, 1.0, 1.0]])
    sampled = target[None].expand(3, -1, -1).clone()
    weights = torch.tensor([[1.0, 1.0], [0.5, 0.25], [0.1, 0.75]])
    correction = sparse_cross_view_attention_correction(block, target, sampled, weights)
    torch.testing.assert_close(
        correction, torch.zeros_like(target), atol=1e-7, rtol=0.0
    )


def test_different_overlapping_view_produces_finite_correction() -> None:
    block = _identity_attention_block()
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    sampled = torch.tensor([[[1.0, 0.0, 0.0, 0.0]], [[1.0, 2.0, 0.0, 0.0]]])
    correction = sparse_cross_view_attention_correction(
        block, target, sampled, torch.ones((2, 1))
    )
    assert bool(torch.isfinite(correction).all())
    assert correction[0, 1] > 0.0


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (torch.tensor([[0.0], [0.0]]), "positive spherical support"),
        (torch.tensor([[1.0], [-1.0]]), "non-negative"),
        (torch.tensor([[1.0], [float("nan")]]), "finite"),
    ],
)
def test_sparse_attention_rejects_invalid_weights(
    weights: torch.Tensor, message: str
) -> None:
    block = _identity_attention_block()
    target = torch.ones((1, 4))
    sampled = target[None].expand(2, -1, -1)
    with pytest.raises(ValueError, match=message):
        sparse_cross_view_attention_correction(block, target, sampled, weights)
