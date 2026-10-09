from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from panorai.experimental.deep_learning import (  # noqa: E402
    OPENCLIP_REPOSITORY_REVISION,
    OPENCLIP_RN50_OPENAI,
    PLACES365_CATEGORIES,
    PLACES365_RESNET18,
    SPHERICAL_CLASSIFICATION_INTERFACE,
    ClassificationArtifactSpec,
    ClassificationTermsNotAcceptedError,
    OpenCLIPRN50Dense,
    Places365ResNet18FCN,
    SphericalAvgPool2d,
    acquire_openclip_rn50,
    acquire_places365_resnet18,
    encode_openclip_prompts,
    port_module_with_report,
)


def test_classifier_artifacts_are_fully_pinned_and_external() -> None:
    assert SPHERICAL_CLASSIFICATION_INTERFACE == "panorai-spherical-classification/v1"
    assert len(OPENCLIP_REPOSITORY_REVISION) == 40
    for spec in (PLACES365_RESNET18, PLACES365_CATEGORIES, OPENCLIP_RN50_OPENAI):
        assert len(spec.sha256) == 64
        assert spec.size_bytes > 0
        assert spec.filename not in spec.url or spec.url.endswith(spec.filename)


def test_classifier_api_is_exposed_only_from_experimental_deep_learning() -> None:
    import panorai
    import panorai.experimental as experimental
    import panorai.experimental.deep_learning as deep_learning

    names = (
        "ImageNetFCN",
        "Places365ResNet18FCN",
        "OpenCLIPRN50Dense",
        "load_pretrained_imagenet_model",
        "load_places365_resnet18",
        "load_openclip_rn50",
    )
    for name in names:
        assert hasattr(deep_learning, name)
        assert not hasattr(experimental, name)
        assert not hasattr(panorai, name)


def test_classifier_acquisition_requires_explicit_terms(tmp_path: Path) -> None:
    with pytest.raises(ClassificationTermsNotAcceptedError):
        acquire_places365_resnet18(
            accept_upstream_terms=False,
            cache_dir=tmp_path,
        )
    with pytest.raises(ClassificationTermsNotAcceptedError):
        acquire_openclip_rn50(
            accept_upstream_terms=False,
            cache_dir=tmp_path,
        )


def test_classifier_acquisition_verifies_hashes_and_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from panorai.experimental.deep_learning import classification

    payloads = {
        "https://example.test/places.pth": b"places checkpoint",
        "https://example.test/categories.txt": b"category list",
        "https://example.test/openclip.safetensors": b"openclip checkpoint",
    }

    def spec(name: str, filename: str, url: str) -> ClassificationArtifactSpec:
        payload = payloads[url]
        return ClassificationArtifactSpec(
            name=name,
            filename=filename,
            url=url,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            license="test-only",
            provenance="independent synthetic acquisition fixture",
        )

    monkeypatch.setattr(
        classification,
        "PLACES365_RESNET18",
        spec("places", "places.pth", "https://example.test/places.pth"),
    )
    monkeypatch.setattr(
        classification,
        "PLACES365_CATEGORIES",
        spec(
            "categories",
            "categories.txt",
            "https://example.test/categories.txt",
        ),
    )
    monkeypatch.setattr(
        classification,
        "OPENCLIP_RN50_OPENAI",
        spec(
            "openclip",
            "openclip.safetensors",
            "https://example.test/openclip.safetensors",
        ),
    )

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    calls: list[str] = []

    def fake_urlopen(request, *, timeout):
        assert timeout == 3.0
        calls.append(request.full_url)
        return Response(payloads[request.full_url])

    monkeypatch.setattr(classification, "urlopen", fake_urlopen)

    places = acquire_places365_resnet18(
        accept_upstream_terms=True,
        cache_dir=tmp_path,
        timeout_seconds=3.0,
    )
    openclip = acquire_openclip_rn50(
        accept_upstream_terms=True,
        cache_dir=tmp_path,
        timeout_seconds=3.0,
    )
    assert places.checkpoint.previously_cached is False
    assert places.categories.previously_cached is False
    assert openclip.checkpoint.previously_cached is False
    assert len(calls) == 3

    places_cached = acquire_places365_resnet18(
        accept_upstream_terms=True,
        cache_dir=tmp_path,
        timeout_seconds=3.0,
    )
    openclip_cached = acquire_openclip_rn50(
        accept_upstream_terms=True,
        cache_dir=tmp_path,
        timeout_seconds=3.0,
    )
    assert places_cached.checkpoint.previously_cached is True
    assert places_cached.categories.previously_cached is True
    assert openclip_cached.checkpoint.previously_cached is True
    assert len(calls) == 3
    manifest = json.loads(Path(openclip.manifest_path).read_text(encoding="utf-8"))
    assert "not part of PanorAi" in manifest["distribution_boundary"]

    Path(openclip.checkpoint.path).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="size mismatch"):
        acquire_openclip_rn50(
            accept_upstream_terms=True,
            cache_dir=tmp_path,
            timeout_seconds=3.0,
        )


def test_places365_resnet18_dense_conversion_matches_planar_classifier() -> None:
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.resnet18(weights=None, num_classes=365).eval()
    original = __import__("copy").deepcopy(model)
    values = torch.randn(1, 3, 224, 224)
    dense_model = Places365ResNet18FCN(model).eval()

    with torch.inference_mode():
        expected = original(values)
        dense = dense_model.forward_dense(values)
        actual = dense_model(values, spherical_average=False)

    assert dense.features.shape == (1, 512, 7, 7)
    assert dense.logits.shape == (1, 365, 7, 7)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_spherical_average_pool_is_roll_equivariant_and_ported() -> None:
    source = nn.AvgPool2d(kernel_size=2, stride=2)
    adapter = SphericalAvgPool2d(source)
    values = torch.randn(1, 3, 8, 16, dtype=torch.float64, requires_grad=True)

    reference = adapter(values)
    shifted = adapter(torch.roll(values, shifts=4, dims=-1))
    torch.testing.assert_close(
        shifted,
        torch.roll(reference, shifts=2, dims=-1),
        rtol=1e-11,
        atol=1e-11,
    )
    assert torch.isfinite(reference).all()
    reference.sum().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()

    constant = torch.full((1, 2, 8, 16), 3.25, dtype=torch.float64)
    torch.testing.assert_close(
        adapter(constant),
        torch.full((1, 2, 4, 8), 3.25, dtype=torch.float64),
    )

    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), source)
    report = port_module_with_report(model)
    assert [layer.source_type for layer in report.layers] == ["Conv2d", "AvgPool2d"]
    assert report.remaining_planar_spatial_layers == ()
    assert isinstance(model[1], SphericalAvgPool2d)


class _TinyAttentionPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.v_proj = nn.Linear(4, 4)
        self.c_proj = nn.Linear(4, 3)


class _TinyVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(4)
        self.act3 = nn.ReLU()
        self.avgpool = nn.AvgPool2d(2)
        self.layer1 = nn.Identity()
        self.layer2 = nn.Identity()
        self.layer3 = nn.Identity()
        self.layer4 = nn.Identity()
        self.attnpool = _TinyAttentionPool()


class _TinyCLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _TinyVisual()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))


def test_openclip_dense_head_reuses_exact_projection_parameters() -> None:
    clip = _TinyCLIP().eval()
    dense_model = OpenCLIPRN50Dense(clip).eval()
    values = torch.randn(2, 3, 16, 32)
    text = F.normalize(torch.randn(5, 3), dim=1)

    with torch.inference_mode():
        features = dense_model.forward_features(values)
        actual_embeddings = dense_model.project_features(features)
        tokens = features.permute(0, 2, 3, 1)
        expected_embeddings = F.linear(
            F.linear(
                tokens,
                clip.visual.attnpool.v_proj.weight,
                clip.visual.attnpool.v_proj.bias,
            ),
            clip.visual.attnpool.c_proj.weight,
            clip.visual.attnpool.c_proj.bias,
        ).permute(0, 3, 1, 2)
        output = dense_model.forward_dense(values, text)

    assert dense_model.value_projection is clip.visual.attnpool.v_proj
    assert dense_model.output_projection is clip.visual.attnpool.c_proj
    assert dense_model.logit_scale is clip.logit_scale
    torch.testing.assert_close(actual_embeddings, expected_embeddings)
    assert output.logits.shape == (2, 5, 4, 8)
    assert torch.isfinite(output.logits).all()


def test_openclip_visual_backbone_ports_every_conv_and_average_pool() -> None:
    dense_model = OpenCLIPRN50Dense(_TinyCLIP().eval()).eval()
    parameters = tuple(dense_model.parameters())

    report = port_module_with_report(dense_model)

    assert report.remaining_planar_spatial_layers == ()
    assert tuple(dense_model.parameters()) == parameters
    assert {layer.source_type for layer in report.layers} == {"Conv2d", "AvgPool2d"}
    assert all(
        layer.parameter_identity_preserved is not False for layer in report.layers
    )


def test_openclip_prompt_encoder_validates_and_normalizes() -> None:
    class Model:
        def encode_text(self, tokens, *, normalize):
            assert normalize is True
            return F.normalize(tokens.float(), dim=1)

    tokenizer = lambda prompts: torch.tensor(  # noqa: E731
        [[len(prompt), 1.0, 2.0] for prompt in prompts]
    )
    result = encode_openclip_prompts(Model(), tokenizer, ("a chair", "a table"))
    torch.testing.assert_close(result.norm(dim=1), torch.ones(2))
    with pytest.raises(ValueError, match="non-empty"):
        encode_openclip_prompts(Model(), tokenizer, ())
