from __future__ import annotations

import hashlib
import json
import math

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from panorai import EquirectangularImage  # noqa: E402
from panorai.ops.spherical import nn as spherical_nn  # noqa: E402
from panorai.pipeline import (  # noqa: E402
    OnSphere,
    Pipeline,
    PipelineCacheError,
    PipelineSerializationError,
    register_torch_module,
    registered_torch_modules,
)


def _model() -> torch.nn.Sequential:
    model = torch.nn.Sequential(
        spherical_nn.Conv2d(3, 4, 3, padding=1, dtype=torch.float64),
        torch.nn.ReLU(),
        spherical_nn.MaxPool2d(2),
    )
    with torch.no_grad():
        model[0].weight.copy_(
            torch.linspace(
                -0.25,
                0.75,
                model[0].weight.numel(),
                dtype=torch.float64,
            ).reshape_as(model[0].weight)
        )
        model[0].bias.copy_(torch.asarray((0.1, -0.2, 0.3, -0.4), dtype=torch.float64))
    return model.eval()


def _pipeline(model=None, *, memory=None) -> Pipeline:
    return Pipeline(
        OnSphere(model if model is not None else _model(), output="features"),
        memory=memory,
    )


def _input() -> EquirectangularImage:
    values = torch.linspace(0, 1, 3 * 8 * 16, dtype=torch.float64).reshape(3, 8, 16)
    return EquirectangularImage(values)


def test_registered_spherical_sequential_round_trips_with_safetensors(tmp_path) -> None:
    pipeline = _pipeline()
    expected = pipeline(_input())
    archive = pipeline.save(tmp_path / "archive")
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["weights_file"] == "weights.safetensors"
    assert len(manifest["registered_torch_modules"]) == 4
    assert not list(archive.glob("*.pkl"))
    assert "torch.nn.Sequential" in registered_torch_modules()
    with pytest.raises(
        PipelineSerializationError, match="safetensors pipeline archive"
    ):
        Pipeline.from_config(pipeline.to_config())

    restored = Pipeline.load(archive)
    actual = restored(_input())
    torch.testing.assert_close(actual.values, expected.values, rtol=0, atol=0)
    restored_model = restored.named_steps["on_sphere"].operation
    assert restored_model.training is False
    for expected_item, actual_item in zip(
        pipeline.named_steps["on_sphere"].operation.state_dict().values(),
        restored_model.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(actual_item, expected_item, rtol=0, atol=0)


def test_registered_spherical_rotation_round_trips_in_pipeline_archive(
    tmp_path,
) -> None:
    angle = math.pi / 8
    rotation = (
        (math.cos(angle), 0.0, math.sin(angle)),
        (0.0, 1.0, 0.0),
        (-math.sin(angle), 0.0, math.cos(angle)),
    )
    model = torch.nn.Sequential(
        spherical_nn.Conv2d(3, 3, 1, dtype=torch.float64),
        spherical_nn.Rotate(rotation, interpolation="bilinear"),
    ).eval()
    pipeline = _pipeline(model)
    expected = pipeline(_input())

    restored = Pipeline.load(pipeline.save(tmp_path / "rotation"))
    actual = restored(_input())

    torch.testing.assert_close(actual.values, expected.values, rtol=0, atol=0)
    restored_rotate = restored.named_steps["on_sphere"].operation[1]
    assert isinstance(restored_rotate, spherical_nn.Rotate)
    torch.testing.assert_close(
        restored_rotate.rotation_matrix,
        torch.as_tensor(rotation, dtype=torch.float64),
        rtol=0,
        atol=0,
    )


def test_weight_tampering_and_file_selected_module_ids_fail_closed(tmp_path) -> None:
    archive = _pipeline().save(tmp_path / "archive")
    weights = archive / "weights.safetensors"
    payload = bytearray(weights.read_bytes())
    payload[-1] ^= 1
    weights.write_bytes(payload)
    with pytest.raises(PipelineSerializationError, match="checksum mismatch"):
        Pipeline.load(archive)

    archive = _pipeline().save(tmp_path / "unknown", overwrite=False)
    config_path = archive / "pipeline.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    module = config["steps"][0]["stage"]["parameters"]["operation"]
    module["id"] = "malicious.file.SelectedClass"
    config_payload = (
        json.dumps(
            config,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    config_path.write_bytes(config_payload)
    manifest_path = archive / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config_sha256"] = hashlib.sha256(config_payload).hexdigest()
    manifest["registered_torch_modules"][0]["identifier"] = module["id"]
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PipelineSerializationError, match="unknown registered"):
        Pipeline.load(archive)


def test_cache_requires_eval_and_keys_registered_weights(tmp_path) -> None:
    first_model = _model().requires_grad_(False)
    first = _pipeline(first_model, memory=tmp_path / "cache")
    second_model = _model()
    with torch.no_grad():
        second_model[0].weight.add_(1.0)
    second_model.requires_grad_(False)
    second = _pipeline(second_model, memory=tmp_path / "cache")

    cold = first.run(_input())
    warm = first.run(_input())
    changed = second.run(_input())

    assert cold.manifest()["cache"]["hit"] is False
    assert warm.manifest()["cache"]["hit"] is True
    assert changed.manifest()["cache"]["hit"] is False
    assert (
        cold.manifest()["cache"]["torch_state_sha256"]
        != changed.manifest()["cache"]["torch_state_sha256"]
    )

    training = _model().train()
    with pytest.raises(PipelineSerializationError, match="eval mode"):
        _pipeline(training).save(tmp_path / "training")
    with pytest.raises(PipelineCacheError, match="eval mode"):
        _pipeline(training, memory=tmp_path / "training-cache").run(_input())


def test_trusted_process_local_custom_registration_round_trips(tmp_path) -> None:
    class Scale(torch.nn.Conv2d):
        def __init__(self, channels: int) -> None:
            super().__init__(
                channels,
                channels,
                kernel_size=1,
                groups=channels,
                bias=False,
                dtype=torch.float64,
            )
            self.channels = channels
            with torch.no_grad():
                self.weight.fill_(1.0)

    register_torch_module(
        "tests.Scale.v1",
        Scale,
        encode=lambda module: {"channels": module.channels},
        decode=lambda values: Scale(**values),
        spherical_semantics="pointwise",
    )
    model = Scale(3).eval()
    with torch.no_grad():
        model.weight.copy_(
            torch.asarray((0.5, 1.0, 1.5), dtype=model.weight.dtype)[
                :, None, None, None
            ]
        )
    pipeline = _pipeline(model)
    expected = pipeline(_input())

    restored = Pipeline.load(pipeline.save(tmp_path / "custom"))
    actual = restored(_input())

    torch.testing.assert_close(actual.values, expected.values, rtol=0, atol=0)
