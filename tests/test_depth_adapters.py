from __future__ import annotations

import importlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from panorai.depth import (
    DepthAdapterUnavailableError,
    ModelRegistry,
    load_dav2_model,
    load_dust3r_model,
    load_m3dv2_model,
    load_zoe_model,
)
import panorai.depth._adapters as adapters


EXPECTED_LOADERS = {
    "dav2": load_dav2_model,
    "dust3r": load_dust3r_model,
    "m3dv2": load_m3dv2_model,
    "zoe": load_zoe_model,
}


def test_depth_import_is_lightweight_and_all_adapters_are_discoverable() -> None:
    code = r"""
import sys
import panorai.depth as depth

forbidden = {"torch", "open3d", "transformers", "mmcv", "mmengine"}
loaded = {name.split(".")[0] for name in sys.modules}
assert forbidden.isdisjoint(loaded), forbidden & loaded
assert set(depth.ModelRegistry.list_models()) == {"dav2", "dust3r", "m3dv2", "zoe"}
for name in (
    "load_dav2_model",
    "load_dust3r_model",
    "load_m3dv2_model",
    "load_zoe_model",
):
    assert callable(getattr(depth, name))
from panorai.depth.zoe import load_zoe_model
assert load_zoe_model is depth.load_zoe_model
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_registry_preserves_loader_keys_defaults_and_identity() -> None:
    assert set(ModelRegistry.list_models()) == set(EXPECTED_LOADERS)
    for key, loader in EXPECTED_LOADERS.items():
        assert ModelRegistry.get_loader(key) is loader
        config = ModelRegistry.get_config(key)
        assert isinstance(config, dict)
        config["mutated-by-caller"] = True
        assert "mutated-by-caller" not in ModelRegistry.get_config(key)

    with pytest.raises(ValueError, match="not registered"):
        ModelRegistry.get_loader("missing")


@pytest.mark.parametrize(
    ("name", "loader", "expected_upstream", "expected_license"),
    [
        ("dav2", load_dav2_model, "Depth Anything V2", "Apache-2.0"),
        ("m3dv2", load_m3dv2_model, "Metric3D", "BSD-2-Clause"),
        ("dust3r", load_dust3r_model, "DUSt3R", "CC BY-NC-SA 4.0"),
        ("zoe", load_zoe_model, "ZoeDepth", "Apache-2.0"),
    ],
)
def test_missing_upstream_errors_are_actionable_and_license_explicit(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    loader,
    expected_upstream: str,
    expected_license: str,
) -> None:
    def missing(module: str):
        raise ModuleNotFoundError(module)

    monkeypatch.setattr(adapters, "_import_module", missing)

    with pytest.raises(DepthAdapterUnavailableError) as caught:
        loader()

    message = str(caught.value)
    assert name in message
    assert expected_upstream in message
    assert expected_license in message
    assert "adapter-only" in message
    assert "pip install \"panorai[depth]\"" in message
    assert "https://" in message


def test_registry_load_delegates_to_the_registered_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, object] = {}

    def fake_loader(**kwargs):
        received.update(kwargs)
        return "loaded"

    original = ModelRegistry._registry["dav2"]["loader_func"]
    monkeypatch.setitem(ModelRegistry._registry["dav2"], "loader_func", fake_loader)
    assert ModelRegistry.load("dav2", device="cpu", max_depth=12) == "loaded"
    assert received["device"] == "cpu"
    assert received["max_depth"] == 12
    assert original is load_dav2_model


def test_incompatible_upstream_import_keeps_adapter_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def incompatible(module: str):
        raise RuntimeError(f"incompatible ABI while importing {module}")

    monkeypatch.setattr(adapters, "_import_module", incompatible)

    with pytest.raises(DepthAdapterUnavailableError) as caught:
        load_dav2_model()

    message = str(caught.value)
    assert "incompatible ABI" in message
    assert "Depth Anything V2" in message
    assert "Apache-2.0" in message
    assert "https://" in message


@pytest.mark.parametrize(
    ("dataset", "expected_name"),
    [
        (None, "depth_anything_v2_vits.pth"),
        ("vkitti", "depth_anything_v2_metric_vkitti_vits.pth"),
    ],
)
def test_dav2_default_checkpoint_names_preserve_legacy_convention(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    dataset: str | None,
    expected_name: str,
) -> None:
    loaded: dict[str, object] = {}

    class FakeModel:
        def __init__(self, **config):
            loaded["config"] = config

        def load_state_dict(self, state):
            loaded["state"] = state

        def to(self, device):
            loaded["device"] = device
            return self

        def eval(self):
            return self

    fake_torch = SimpleNamespace(
        load=lambda path, map_location: loaded.update(
            path=path, map_location=map_location
        )
        or {"weight": 1}
    )
    fake_dpt = SimpleNamespace(__name__="depth_anything_v2.dpt", DepthAnythingV2=FakeModel)

    def require(key: str, module: str):
        assert key == "dav2"
        return fake_torch if module == "torch" else fake_dpt

    monkeypatch.setattr(adapters, "_require_module", require)
    runtime_path_config = importlib.import_module("panorai.path_config")
    monkeypatch.setattr(runtime_path_config, "get_path", lambda *_: tmp_path)

    model = load_dav2_model(dataset=dataset, device="cpu", return_model=True)

    assert isinstance(model, FakeModel)
    assert loaded["path"] == str(tmp_path / expected_name)
