from pathlib import Path

import numpy as np
import pytest

from benchmarks.spherical_monocular_depth.da3_spherical_gate import (
    GATE_INPUT_NAMES,
    MAXIMUM_GATE_MAGNITUDE,
    SphericalAttentionGate,
    gate_features_and_objective,
    gate_parameter_count,
    load_gate,
    normalized_gate_objective,
    save_gate,
)


torch = pytest.importorskip("torch")


def test_gate_has_the_frozen_6_16_8_1_architecture() -> None:
    gate = SphericalAttentionGate.create()

    assert len(GATE_INPUT_NAMES) == 6
    assert gate_parameter_count(gate) == 257
    assert gate(torch.zeros(4, 6)).shape == (4,)
    values, latent = gate.forward_with_latent(torch.zeros(4, 6))
    assert values.shape == (4,)
    assert latent.shape == (4, 8)
    assert torch.all(values.abs() <= MAXIMUM_GATE_MAGNITUDE)


def test_zero_gate_is_the_exact_normalized_identity_objective() -> None:
    objective = torch.tensor(
        [
            [2.0, -0.5, 4.0, 1.0, 1.0],
            [3.0, 0.25, 2.0, 0.5, 0.5],
        ]
    )

    value = normalized_gate_objective(torch.zeros(2), objective)

    assert value.item() == pytest.approx(1.0)


def test_closed_form_gate_minimizes_the_quadratic() -> None:
    objective = torch.tensor([[2.0, -1.0, 4.0, 1.0, 0.5]])
    optimum = -objective[:, 1] / ((1.0 + objective[:, 4]) * objective[:, 0])
    below = normalized_gate_objective(optimum - 0.1, objective)
    at = normalized_gate_objective(optimum, objective)
    above = normalized_gate_objective(optimum + 0.1, objective)

    assert at < below
    assert at < above


def test_feature_builder_returns_finite_per_patch_statistics() -> None:
    channels = 4
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    sampled = torch.stack((target, target + 0.1), dim=0).reshape(2, 1, 2, channels)
    correction = torch.full((2, channels), 0.05)
    probabilities = torch.full((2, 2, 2), 0.5)
    weights = torch.tensor([[[1.0, 1.0]], [[0.5, 0.5]]])
    grids = torch.tensor(
        [
            [[[-0.5, 0.0], [0.5, 0.0]]],
            [[[-0.4, 0.0], [0.4, 0.0]]],
        ]
    )

    features, objective = gate_features_and_objective(
        target,
        sampled,
        correction,
        probabilities,
        weights,
        grids,
        self_source_offset=0,
        gaussian_exponent=2.0,
        identity_regularization=1.0,
    )

    assert features.shape == (2, 6)
    assert objective.shape == (2, 5)
    assert torch.isfinite(features).all()
    assert torch.isfinite(objective).all()
    assert torch.all(objective[:, 3] > 0)


def test_gate_safetensors_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    mean = np.arange(6, dtype=np.float32)
    std = np.arange(1, 7, dtype=np.float32)
    gate = SphericalAttentionGate.create(mean, std)
    path = tmp_path / "gate.safetensors"

    save_gate(path, gate, metadata={"depth_used": False})
    restored, metadata = load_gate(path)

    assert metadata == {"depth_used": False}
    for expected, actual in zip(
        gate.state_dict().values(), restored.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(expected, actual)
