from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from panorai.experimental.deep_learning import (  # noqa: E402
    SPHERICAL_INTERNIMAGE_INTERFACE,
    InternImageTermsNotAcceptedError,
    SphericalConv2d,
    SphericalDCNv3,
    acquire_internimage_g,
    attention_pool_class_contributions,
    port_internimage_to_spherical,
    spherical_dcnv3_core,
)


def _center_mask(
    batch: int, height: int, width: int, groups: int, *, dtype: torch.dtype
) -> torch.Tensor:
    mask = torch.zeros(batch, height, width, groups, 9, dtype=dtype)
    mask[..., 4] = 1
    return mask.flatten(-2)


def test_spherical_dcnv3_center_sample_is_identity_including_poles() -> None:
    values = torch.randn(1, 5, 10, 4, dtype=torch.float64)
    offset = torch.zeros(1, 5, 10, 2 * 9 * 2, dtype=torch.float64)
    mask = _center_mask(1, 5, 10, 2, dtype=torch.float64)

    actual = spherical_dcnv3_core(
        values,
        offset,
        mask,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=2,
        group_channels=2,
        offset_scale=1,
        max_sampled_elements=500,
    )

    torch.testing.assert_close(actual, values, rtol=1e-12, atol=1e-12)


def test_spherical_dcnv3_preserves_constants_for_deformed_samples() -> None:
    values = torch.full((1, 6, 12, 4), 3.25, dtype=torch.float64)
    generator = torch.Generator().manual_seed(42)
    offset = 1.7 * torch.randn(
        1, 6, 12, 2 * 9 * 2, generator=generator, dtype=torch.float64
    )
    mask = torch.randn(1, 6, 12, 2, 9, generator=generator, dtype=torch.float64)
    mask = mask.softmax(dim=-1).flatten(-2)

    actual = spherical_dcnv3_core(
        values,
        offset,
        mask,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=2,
        group_channels=2,
        offset_scale=1,
        max_sampled_elements=1000,
    )

    torch.testing.assert_close(actual, values, rtol=1e-12, atol=1e-12)


def test_spherical_dcnv3_is_longitude_roll_equivariant() -> None:
    generator = torch.Generator().manual_seed(7)
    values = torch.randn(1, 6, 12, 4, generator=generator, dtype=torch.float64)
    offset = torch.randn(1, 6, 12, 2 * 9 * 2, generator=generator, dtype=torch.float64)
    mask = torch.randn(1, 6, 12, 2, 9, generator=generator, dtype=torch.float64)
    mask = mask.softmax(dim=-1).flatten(-2)

    def apply(
        image: torch.Tensor, offsets: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        return spherical_dcnv3_core(
            image,
            offsets,
            weights,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=2,
            group_channels=2,
            offset_scale=1,
            max_sampled_elements=1000,
        )

    expected = torch.roll(apply(values, offset, mask), 3, dims=2)
    actual = apply(
        torch.roll(values, 3, dims=2),
        torch.roll(offset, 3, dims=2),
        torch.roll(mask, 3, dims=2),
    )

    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)


def test_spherical_dcnv3_gradients_cover_values_offsets_and_masks() -> None:
    values = torch.randn(1, 3, 4, 2, dtype=torch.float64, requires_grad=True)
    # Avoid exact pixel centers: bilinear sampling is continuous but its derivative
    # is intentionally undefined where the interpolation cell changes.
    offset = (
        torch.randn(1, 3, 4, 9 * 2, dtype=torch.float64) * 0.07 + 0.03
    ).requires_grad_()
    mask_logits = torch.randn(1, 3, 4, 9, dtype=torch.float64, requires_grad=True)

    def apply(
        image: torch.Tensor, offsets: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor:
        return spherical_dcnv3_core(
            image,
            offsets,
            logits.softmax(dim=-1),
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            group_channels=2,
            offset_scale=1,
            max_sampled_elements=1000,
        )

    assert torch.autograd.gradcheck(
        apply,
        (values, offset, mask_logits),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
    )


class _ToChannelsLast(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values.permute(0, 2, 3, 1)


class DCNv3_pytorch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.channels = 4
        self.kernel_size = 3
        self.stride = 1
        self.pad = 1
        self.dilation = 1
        self.group = 2
        self.group_channels = 2
        self.offset_scale = 1.0
        self.remove_center = 0
        self.center_feature_scale = False
        self.dw_conv = nn.Sequential(
            nn.Conv2d(4, 4, 3, padding=1, groups=4),
            _ToChannelsLast(),
            nn.GELU(),
        )
        self.offset = nn.Linear(4, 2 * 9 * 2)
        self.mask = nn.Linear(4, 2 * 9)
        self.input_proj = nn.Linear(4, 4)
        self.output_proj = nn.Linear(4, 4)


def test_internimage_port_reuses_every_parameter_and_ports_nested_conv() -> None:
    model = nn.Sequential(DCNv3_pytorch(), nn.Conv2d(4, 4, 1)).eval()
    parameters = {id(parameter) for parameter in model.parameters()}

    report = port_internimage_to_spherical(model, max_sampled_elements=1000)

    assert report.interface == SPHERICAL_INTERNIMAGE_INTERFACE
    assert len(report.dcnv3_layers) == 1
    assert report.dcnv3_layers[0].path == "0"
    assert report.dcnv3_layers[0].parameter_identity_preserved is True
    assert report.parameter_identity_preserved is True
    assert report.remaining_dcnv3_layers == ()
    assert report.spatial_layers.remaining_planar_spatial_layers == ()
    assert isinstance(model[0], SphericalDCNv3)
    assert isinstance(model[0].dw_conv[0], SphericalConv2d)
    assert isinstance(model[1], SphericalConv2d)
    assert {id(parameter) for parameter in model.parameters()} == parameters
    output = model[0](torch.randn(1, 5, 10, 4))
    assert output.shape == (1, 5, 10, 4)
    assert torch.isfinite(output).all()


class _CrossAttention(nn.Module):
    def __init__(self, channels: int, heads: int, output_channels: int) -> None:
        super().__init__()
        self.num_heads = heads
        self.scale = (channels // heads) ** -0.5
        self.q = nn.Linear(channels, channels, bias=False)
        self.k = nn.Linear(channels, channels, bias=False)
        self.v = nn.Linear(channels, channels, bias=False)
        self.q_bias = nn.Parameter(torch.randn(channels))
        self.k_bias = nn.Parameter(torch.randn(channels))
        self.v_bias = nn.Parameter(torch.randn(channels))
        self.attn_drop = nn.Dropout(0)
        self.proj = nn.Linear(channels, output_channels)


class _AttentionProjector(nn.Module):
    def __init__(self, channels: int, heads: int, output_channels: int) -> None:
        super().__init__()
        self.norm1_q = nn.LayerNorm(channels)
        self.norm1_k = nn.LayerNorm(channels)
        self.norm1_v = nn.LayerNorm(channels)
        self.cross_dcn = _CrossAttention(channels, heads, output_channels)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        cross = self.cross_dcn
        query = self.norm1_q(tokens.mean(dim=1, keepdim=True))
        keys = self.norm1_k(tokens)
        values = self.norm1_v(tokens)
        batch, token_count, channels = tokens.shape
        heads = cross.num_heads
        head_channels = channels // heads
        query = F.linear(query, cross.q.weight, cross.q_bias)
        keys = F.linear(keys, cross.k.weight, cross.k_bias)
        values = F.linear(values, cross.v.weight, cross.v_bias)
        query = query.reshape(batch, 1, heads, head_channels).permute(0, 2, 1, 3)
        keys = keys.reshape(batch, token_count, heads, head_channels).permute(
            0, 2, 1, 3
        )
        values = values.reshape(batch, token_count, heads, head_channels).permute(
            0, 2, 1, 3
        )
        attention = ((query * cross.scale) @ keys.transpose(-2, -1)).softmax(-1)
        pooled = (attention @ values).transpose(1, 2).reshape(batch, 1, channels)
        return cross.proj(pooled).squeeze(1)


def test_attention_contribution_map_has_exact_classifier_mean() -> None:
    torch.manual_seed(8)
    projector = _AttentionProjector(8, 2, 6).double().eval()
    fc_norm = nn.Sequential(nn.LayerNorm(6)).double().eval()
    head = nn.Linear(6, 5).double().eval()
    tokens = torch.randn(2, 12, 8, dtype=torch.float64, requires_grad=True)

    dense, global_logits, attention = attention_pool_class_contributions(
        tokens, projector, fc_norm, head
    )
    official = head(fc_norm(projector(tokens)))

    assert dense.shape == (2, 12, 5)
    assert attention.shape == (2, 2, 1, 12)
    torch.testing.assert_close(dense.mean(dim=1), official, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(global_logits, official, rtol=1e-12, atol=1e-12)
    global_logits.square().sum().backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()


def test_internimage_acquisition_requires_explicit_terms_before_network() -> None:
    with pytest.raises(InternImageTermsNotAcceptedError, match="terms"):
        acquire_internimage_g(accept_upstream_terms=False)


def test_spherical_dcnv3_rejects_unsupported_stride_semantics() -> None:
    values = torch.ones(1, 3, 4, 2)
    offset = torch.zeros(1, 3, 4, 18)
    mask = torch.full((1, 3, 4, 9), 1 / 9)
    with pytest.raises(ValueError, match="stride-one"):
        spherical_dcnv3_core(
            values,
            offset,
            mask,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            groups=1,
            group_channels=2,
            offset_scale=1,
        )
