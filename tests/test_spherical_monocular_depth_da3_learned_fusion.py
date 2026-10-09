import pytest
import numpy as np

from benchmarks.spherical_monocular_depth.da3_learned_fusion import (
    LOCAL_INPUT_NAMES,
    TOKEN_LATENT_DIM,
    LatentConditionedDepthFusion,
    _sample_map,
    build_aligned_ray_bundle,
    fusion_parameter_count,
)
from benchmarks.spherical_monocular_depth.vit_tangent import NativeTangentPlan


torch = pytest.importorskip("torch")


def _inputs() -> tuple:
    depths = torch.tensor([[2.0, 4.0, 8.0]])
    local = torch.zeros(1, 3, len(LOCAL_INPUT_NAMES))
    latent = torch.zeros(1, 3, TOKEN_LATENT_DIM)
    gate = torch.ones(1, 3)
    gaussian = torch.tensor([[1.0, 2.0, 1.0]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    return depths, local, latent, gate, gaussian, valid


def test_zero_initialized_fusion_is_exactly_gaussian() -> None:
    fusion = LatentConditionedDepthFusion.create()
    depths, local, latent, gate, gaussian, valid = _inputs()

    fused, weights = fusion(depths, local, latent, gate, gaussian, valid)

    torch.testing.assert_close(weights, torch.tensor([[0.25, 0.5, 0.25]]))
    torch.testing.assert_close(fused, torch.tensor([4.5]))
    assert fusion_parameter_count(fusion) == 449


def test_fusion_is_convex_and_ignores_invalid_contributions() -> None:
    fusion = LatentConditionedDepthFusion.create()
    depths, local, latent, gate, gaussian, valid = _inputs()
    valid[:, 1] = False

    fused, weights = fusion(depths, local, latent, gate, gaussian, valid)

    torch.testing.assert_close(weights, torch.tensor([[0.5, 0.0, 0.5]]))
    assert depths[valid].min() <= fused.item() <= depths[valid].max()


def test_first_gate_closes_the_learned_fusion_residual() -> None:
    fusion = LatentConditionedDepthFusion.create()
    depths, local, latent, gate, gaussian, valid = _inputs()
    with torch.no_grad():
        fusion.output.bias.fill_(1.0)
        fusion.conditioner.bias.copy_(torch.linspace(-1.0, 1.0, 32))
    gate.zero_()

    _, weights = fusion(depths, local, latent, gate, gaussian, valid)

    torch.testing.assert_close(weights, gaussian / gaussian.sum(dim=-1, keepdim=True))


def test_film_latent_can_change_face_confidence() -> None:
    fusion = LatentConditionedDepthFusion.create()
    depths, local, latent, gate, gaussian, valid = _inputs()
    with torch.no_grad():
        fusion.local_encoder.bias.fill_(0.5)
        fusion.conditioner.weight.fill_(0.1)
        fusion.output.weight.fill_(0.1)
    latent[:, 1] = 2.0

    _, weights = fusion(depths, local, latent, gate, gaussian, valid)

    assert weights[0, 1] > 0.5


def test_unsupported_ray_returns_nan_and_zero_weights() -> None:
    fusion = LatentConditionedDepthFusion.create()
    depths, local, latent, gate, gaussian, valid = _inputs()
    valid.zero_()

    fused, weights = fusion(depths, local, latent, gate, gaussian, valid)

    assert torch.isnan(fused).all()
    assert torch.count_nonzero(weights) == 0


@pytest.mark.parametrize("contribution_count", [1, 2, 3])
def test_variable_contribution_count(contribution_count: int) -> None:
    fusion = LatentConditionedDepthFusion.create()
    depths, local, latent, gate, gaussian, valid = _inputs()
    sl = slice(0, contribution_count)

    fused, weights = fusion(
        depths[:, sl],
        local[:, sl],
        latent[:, sl],
        gate[:, sl],
        gaussian[:, sl],
        valid[:, sl],
    )

    assert fused.shape == (1,)
    assert weights.shape == (1, contribution_count)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1))
    if contribution_count == 1:
        torch.testing.assert_close(weights, torch.ones(1, 1))


def test_fusion_is_invariant_to_contribution_order() -> None:
    fusion = LatentConditionedDepthFusion.create()
    depths, local, latent, gate, gaussian, valid = _inputs()
    with torch.no_grad():
        fusion.output.bias.fill_(0.25)
        fusion.output.weight.fill_(0.05)
    order = torch.tensor([2, 0, 1])

    expected, expected_weights = fusion(depths, local, latent, gate, gaussian, valid)
    actual, actual_weights = fusion(
        depths[:, order],
        local[:, order],
        latent[:, order],
        gate[:, order],
        gaussian[:, order],
        valid[:, order],
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_weights, expected_weights[:, order])


def test_depth_gate_and_latent_are_sampled_from_the_same_erp_ray() -> None:
    plan = NativeTangentPlan(
        erp_shape_hw=(28, 56),
        view_shape_hw=(28, 56),
        focal_px=28.0,
        hfov_deg=90.0,
        vfov_deg=53.1301023542,
        overlap_fraction=0.0,
        minimum_latitude_deg=-26.5650511771,
        maximum_latitude_deg=26.5650511771,
        centers_lat_lon_deg=((0.0, 0.0),),
    )
    yy, xx = np.indices(plan.view_shape_hw, dtype=np.float32)
    depth = (2.0 + xx / 100.0 + yy / 1000.0)[None]
    gate_yy, gate_xx = np.indices((2, 4), dtype=np.float32)
    gate = ((gate_xx + 10.0 * gate_yy) / 100.0 - 0.05)[None]
    latent = np.stack(
        [gate[0] + float(channel) for channel in range(TOKEN_LATENT_DIM)], axis=-1
    )[None]
    flat_index = np.asarray([14 * plan.erp_shape_hw[1] + 28], dtype=np.int64)

    bundle = build_aligned_ray_bundle(
        plan, depth, gate, latent, flat_index, maximum_contributions=1
    )

    assert bundle.validity[0, 0]
    assert bundle.depths[0, 0] > 2.0
    assert bundle.local_features[0, 0, -1] == pytest.approx(
        bundle.token_gate_strength[0, 0] * np.sign(bundle.local_features[0, 0, -1]),
        abs=1e-5,
    )
    expected_latent_offsets = np.arange(TOKEN_LATENT_DIM, dtype=np.float32)
    assert np.allclose(
        bundle.token_latents[0, 0] - bundle.token_latents[0, 0, 0],
        expected_latent_offsets,
        atol=1e-5,
    )


def test_large_same_face_sampling_is_split_without_changing_values() -> None:
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    x = np.full(40_000, 1.0, dtype=np.float32)
    y = np.full(40_000, 2.0, dtype=np.float32)

    sampled = _sample_map(values, x, y)

    assert sampled.shape == (40_000,)
    assert np.all(sampled == values[2, 1])
