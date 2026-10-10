from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from panorai.data import EquirectangularImage  # noqa: E402
from panorai.experimental.deep_learning import (  # noqa: E402
    DA3_CANONICAL_FOCAL_PX,
    DA3_CHECKPOINT_REVISION,
    DA3_MODEL_ID,
    DA3_SOURCE,
    DA3_SOURCE_COMMIT,
    DA3METRIC_LARGE,
    DA3MetricTangent,
    DA3TermsNotAcceptedError,
    SPHERICAL_DA3_METRIC_DEPTH_INTERFACE,
    acquire_da3metric_large,
)
from panorai.experimental.deep_learning import da3 as da3_module  # noqa: E402
from panorai.geometry import GnomonicSpec, gnomonic_intrinsics  # noqa: E402


class _ConstantDA3:
    def __init__(self, value: float) -> None:
        self.value = value
        self.received_shape = None

    def __call__(self, tensor):
        self.received_shape = tuple(tensor.shape)
        _, views, _, height, width = tensor.shape
        return {
            "depth": tensor.new_full((1, views, height, width), self.value),
        }


def test_da3_identity_and_external_apache_boundary_are_explicit() -> None:
    assert SPHERICAL_DA3_METRIC_DEPTH_INTERFACE.endswith("/v1-experimental")
    assert DA3_SOURCE_COMMIT == "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
    assert DA3_CHECKPOINT_REVISION == "4010e39f3634a45bc60553321fb49fb760bd594e"
    assert DA3_MODEL_ID == "depth-anything/DA3METRIC-LARGE"
    assert DA3_SOURCE.sha256 == (
        "98aa2dd53ab44b96cef5190ae4841c6ae51797d099b784e08e12a1a55bd3a69b"
    )
    assert DA3METRIC_LARGE.sha256 == (
        "bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776"
    )
    assert "Apache-2.0" in DA3_SOURCE.license_statement
    assert "Apache-2.0" in DA3METRIC_LARGE.license_statement
    assert "external-only" in DA3METRIC_LARGE.redistribution


def test_da3_acquisition_requires_opt_in_before_touching_cache(tmp_path: Path) -> None:
    cache = tmp_path / "absent"
    with pytest.raises(DA3TermsNotAcceptedError, match="Apache-2.0"):
        acquire_da3metric_large(
            accept_upstream_terms=False,
            cache_dir=cache,
        )
    assert not cache.exists()


def test_da3_metric_rule_and_rectangular_axial_to_radial_conversion() -> None:
    spec = GnomonicSpec(
        hfov_deg=100.0,
        vfov_deg=60.0,
        output_shape_hw=(28, 42),
    )
    model = _ConstantDA3(DA3_CANONICAL_FOCAL_PX / 1000.0)
    adapter = DA3MetricTangent(model, spec)

    radial, valid = adapter(np.full((28, 42, 3), 127, dtype=np.uint8))

    intrinsics = np.asarray(gnomonic_intrinsics(spec))
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    axial_m = ((fx + fy) / 2.0) / 1000.0
    x = (0.5 - 42.0 / 2.0) / fx
    y = (0.5 - 28.0 / 2.0) / fy
    assert radial[0, 0] == pytest.approx(
        axial_m * math.sqrt(1.0 + x * x + y * y), rel=1e-6
    )
    assert radial.dtype == np.float32
    assert valid.dtype == np.bool_
    assert valid.all()
    assert model.received_shape == (1, 1, 3, 28, 42)


def test_da3_tangent_normalizes_imagenet_rgb_without_resizing() -> None:
    class InspectModel:
        def __call__(self, tensor):
            assert tuple(tensor.shape) == (1, 1, 3, 28, 28)
            expected = torch.tensor(
                [
                    (1.0 - 0.485) / 0.229,
                    (1.0 - 0.456) / 0.224,
                    (1.0 - 0.406) / 0.225,
                ]
            )
            torch.testing.assert_close(tensor[0, 0, :, 0, 0], expected)
            return {"depth": tensor.new_ones((1, 1, 28, 28))}

    adapter = DA3MetricTangent(InspectModel(), GnomonicSpec(output_shape_hw=(28, 28)))
    depth, valid = adapter(np.full((28, 28, 3), 255, dtype=np.uint8))

    assert depth.shape == (28, 28)
    assert valid.all()


def test_da3_tangent_does_not_clip_or_smooth_large_finite_prediction() -> None:
    spec = GnomonicSpec(output_shape_hw=(28, 28), hfov_deg=40.0, vfov_deg=40.0)
    adapter = DA3MetricTangent(_ConstantDA3(10_000.0), spec)

    depth, valid = adapter(np.zeros((28, 28, 3), dtype=np.uint8))

    assert valid.all()
    assert float(depth.min()) > 200.0
    assert np.unique(depth).size > 1  # radial conversion, not a constant clamp


def test_da3_tangent_keeps_numerical_validity_separate() -> None:
    class InvalidModel:
        def __call__(self, tensor):
            result = tensor.new_ones((1, 1, 28, 28))
            result[0, 0, 0, 0] = float("nan")
            result[0, 0, 0, 1] = -1.0
            return {"depth": result}

    adapter = DA3MetricTangent(InvalidModel(), GnomonicSpec(output_shape_hw=(28, 28)))
    depth, valid = adapter(np.zeros((28, 28, 3), dtype=np.uint8))

    assert not valid[0, 0]
    assert not valid[0, 1]
    assert np.isnan(depth[0, 0])
    assert np.isnan(depth[0, 1])
    assert valid[1:, :].all()


@pytest.mark.parametrize(
    ("values", "exception", "message"),
    [
        (np.zeros((29, 28, 3), dtype=np.uint8), ValueError, "shape"),
        (np.zeros((28, 28, 3), dtype=np.float64), TypeError, "uint8 or float32"),
        (np.full((28, 28, 3), 256.0, dtype=np.float32), ValueError, "range"),
    ],
)
def test_da3_tangent_rejects_implicit_input_changes(
    values: np.ndarray, exception: type[Exception], message: str
) -> None:
    adapter = DA3MetricTangent(
        _ConstantDA3(1.0), GnomonicSpec(output_shape_hw=(28, 28))
    )
    with pytest.raises(exception, match=message):
        adapter(values)


def test_da3_tangent_rejects_non_patch_aligned_geometry() -> None:
    with pytest.raises(ValueError, match="patch size 14"):
        DA3MetricTangent(
            _ConstantDA3(1.0),
            GnomonicSpec(output_shape_hw=(29, 28)),
        )


def test_da3_adapter_integrates_with_real_view_map_and_reconstruction() -> None:
    panorama = EquirectangularImage(np.full((28, 56, 3), 127, dtype=np.uint8))
    views = panorama.views("cube", size=(28, 28), fov=90.0)
    adapter = DA3MetricTangent(_ConstantDA3(1.0), views[0].spec)

    predicted = views.map(adapter, input="image", output="depth", units="m")
    reconstructed = predicted.reconstruct(
        blend={"depth": "gaussian"}, modalities=("depth",)
    )

    assert reconstructed.depth.shape == (28, 56)
    assert reconstructed.validity("depth").all()
    assert np.isfinite(reconstructed.depth).all()


def test_da3_loader_uses_strict_safetensors_and_freezes_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    safetensors_torch = pytest.importorskip("safetensors.torch")

    class Envelope(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Linear(2, 1)

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"test-only")
    assets = SimpleNamespace(
        source_root=str(tmp_path / "source"),
        checkpoint=SimpleNamespace(path=str(checkpoint)),
    )
    observed = {}

    def fake_load_model(model, path, *, strict, device):
        observed.update(model=model, path=path, strict=strict, device=device)
        return [], []

    monkeypatch.setattr(da3_module, "acquire_da3metric_large", lambda **kwargs: assets)
    monkeypatch.setattr(
        da3_module, "_construct_official_da3metric_large", lambda source: Envelope()
    )
    monkeypatch.setattr(safetensors_torch, "load_model", fake_load_model)

    loaded = da3_module.load_da3metric_large(
        accept_upstream_terms=True,
        cache_dir=tmp_path,
    )

    assert observed["path"] == str(checkpoint)
    assert observed["strict"] is True
    assert observed["device"] == "cpu"
    assert loaded.parameter_count == 3
    assert not loaded.model.training
    assert all(not parameter.requires_grad for parameter in loaded.model.parameters())
