#!/usr/bin/env python3
"""Installed-wheel consumer for mixed Pipeline, persistence, cache, and Torch."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import tempfile

import numpy as np


def _assert_installed_origin(package_file: str, source_root: Path) -> Path:
    origin = Path(package_file).resolve()
    try:
        origin.relative_to(source_root.resolve())
    except ValueError:
        return origin
    raise AssertionError(f"panorai resolved inside the source checkout: {origin}")


class BatchedFeatures:
    supports_batch = True

    def __init__(self) -> None:
        self.chunks: list[int] = []

    def __call__(self, values: np.ndarray) -> np.ndarray:
        self.chunks.append(int(values.shape[0]))
        return values * np.float32(0.75)


class BatchedEmbeddings:
    supports_batch = True

    def __call__(self, values: np.ndarray) -> np.ndarray:
        return values.mean(axis=(1, 2, 3))[:, None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()

    import panorai
    from panorai.ops.spherical import GaussianBlur
    from panorai.pipeline import (
        EmbeddingResult,
        OnSphere,
        Output,
        PerView,
        Pipeline,
        RasterField,
        ReconstructERP,
        ResultBundle,
        SampleViews,
    )

    origin = _assert_installed_origin(panorai.__file__, args.source_root)
    values = np.linspace(0, 1, 16 * 32 * 3, dtype=np.float32).reshape(16, 32, 3)
    panorama = panorai.EquirectangularImage(values)
    features = BatchedFeatures()
    mixed = Pipeline(
        SampleViews("cube", size=8),
        PerView.parallel(
            batch=4,
            features=Output(features, kind="features"),
            embedding=Output(BatchedEmbeddings(), kind="embedding"),
            detections=Output(
                lambda batch: [
                    {"points_xy": np.asarray(((3.0, 3.0),))}
                    for _ in range(batch.shape[0])
                ],
                kind="detections",
            ),
        ),
        ReconstructERP(features="gaussian"),
    )
    bundle = mixed(panorama)
    assert isinstance(bundle, ResultBundle)
    assert isinstance(bundle["embedding"], EmbeddingResult)
    assert len(bundle["embedding"]) == 6
    assert len(bundle["detections"]) == 6
    assert features.chunks == [4, 2]
    assert bundle["erp"]._workflow_data()["features"].shape == values.shape

    registered = Pipeline(
        OnSphere(
            Output(GaussianBlur(3, 0.8), kind="features"),
            output="features",
        )
    )
    expected = registered(panorama)
    assert isinstance(expected, RasterField)
    with tempfile.TemporaryDirectory(prefix="panorai-composed-") as directory:
        root = Path(directory)
        archive = registered.save(root / "pipeline")
        restored = Pipeline.load(archive)
        actual = restored(panorama)
        np.testing.assert_allclose(actual.values, expected.values, rtol=0, atol=0)

        cached = Pipeline(
            OnSphere(
                Output(GaussianBlur(3, 0.8), kind="features"),
                output="features",
            ),
            memory=root / "cache",
        )
        cold = cached.run(panorama)
        warm = cached.run(panorama)
        assert cold.manifest()["cache"]["hit"] is False
        assert warm.manifest()["cache"]["hit"] is True
        np.testing.assert_allclose(
            warm.output.values, cold.output.values, rtol=0, atol=0
        )

    import torch
    from panorai.ops.spherical import nn as spherical_nn

    tensor = torch.randn(3, 8, 16, dtype=torch.float64, requires_grad=True)
    network = torch.nn.Sequential(
        spherical_nn.Conv2d(3, 4, 3, padding=1, dtype=torch.float64),
        torch.nn.ReLU(),
        spherical_nn.Closing(8.0, max_sampled_elements=2_000),
        spherical_nn.MaxPool2d(2),
    ).eval()
    torch_pipeline = Pipeline(OnSphere(network, output="features"))
    torch_result = torch_pipeline(panorai.EquirectangularImage(tensor))
    torch_result.values.sum().backward()
    assert tensor.grad is not None
    assert tuple(torch_result.values.shape) == (4, 4, 8)
    assert torch_result.grid.shape_hw == (4, 8)
    with tempfile.TemporaryDirectory(prefix="panorai-torch-pipeline-") as directory:
        restored_torch = Pipeline.load(torch_pipeline.save(Path(directory) / "model"))
        restored_result = restored_torch(panorai.EquirectangularImage(tensor.detach()))
        torch.testing.assert_close(
            restored_result.values,
            torch_result.values.detach(),
            rtol=0,
            atol=0,
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "consumer": "composed-spherical-api",
                "version": version("panorai"),
                "origin": str(origin),
                "mixed_fields": sorted(bundle),
                "batch_chunks": features.chunks,
                "cache_hit": True,
                "torch_shape": list(torch_result.values.shape),
                "autograd": True,
                "torch_morphology": True,
                "torch_safetensors_round_trip": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
