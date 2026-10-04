#!/usr/bin/env python3
"""Installed-wheel consumer for arbitrary-N multimodal reconstruction."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np


COUNT = 14
HEIGHT = 48
WIDTH = 96
VIEW_SHAPE = (20, 24)
FOV = (110.0, 82.0)
BLENDS = {"image": "gaussian", "depth": "average", "labels": "closest"}


def _assert_installed_origin(package_file: str, source_root: Path) -> Path:
    origin = Path(package_file).resolve()
    try:
        origin.relative_to(source_root.resolve())
    except ValueError:
        return origin
    raise AssertionError(
        f"panorai resolved inside the source checkout: {origin} (root={source_root})"
    )


def _numpy_bundle():
    y, x = np.indices((HEIGHT, WIDTH), dtype=np.float32)
    image = np.stack(
        (x / WIDTH, y / HEIGHT, (x + 2.0 * y) / (WIDTH + 2.0 * HEIGHT)),
        axis=-1,
    ).astype(np.float32)
    depth = (1.0 + x / WIDTH + y / HEIGHT).astype(np.float32)
    valid = np.ones((HEIGHT, WIDTH), dtype=bool)
    valid[9:16, 31:42] = False
    depth[~valid] = np.nan
    labels = ((x.astype(np.int16) // 9 + y.astype(np.int16) // 7) % 7).astype(
        np.int16
    )
    return image, depth, valid, labels


def _torch_bundle(torch):
    y, x = torch.meshgrid(
        torch.arange(HEIGHT, dtype=torch.float32),
        torch.arange(WIDTH, dtype=torch.float32),
        indexing="ij",
    )
    image_one = torch.stack(
        (x / WIDTH, y / HEIGHT, (x + 2.0 * y) / (WIDTH + 2.0 * HEIGHT)),
        dim=0,
    )
    image = torch.stack((image_one, torch.flip(image_one, dims=(-1,))), dim=0)
    depth_one = (1.0 + x / WIDTH + y / HEIGHT).unsqueeze(0)
    depth = torch.stack((depth_one, depth_one + 0.25), dim=0)
    valid = torch.ones((2, HEIGHT, WIDTH), dtype=torch.bool)
    valid[:, 9:16, 31:42] = False
    depth = depth.masked_fill(~valid.unsqueeze(1), float("nan"))
    labels_one = ((x.to(torch.int16) // 9 + y.to(torch.int16) // 7) % 7).unsqueeze(0)
    labels = torch.stack((labels_one, torch.flip(labels_one, dims=(-1,))), dim=0)
    return image, depth, valid, labels


def _clone(value):
    return value.clone() if type(value).__module__.startswith("torch") else value.copy()


def _assert_equal(left, right) -> None:
    if type(left).__module__.startswith("torch"):
        import torch

        if left.dtype.is_floating_point:
            torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
        else:
            assert torch.equal(left, right)
        return
    if np.issubdtype(left.dtype, np.floating):
        np.testing.assert_allclose(left, right, rtol=0, atol=0, equal_nan=True)
    else:
        np.testing.assert_array_equal(left, right)


def _assert_finite_where_valid(value, valid) -> None:
    if type(value).__module__.startswith("torch"):
        import torch

        expanded = valid
        if value.ndim == 4:
            expanded = valid.unsqueeze(1).expand_as(value)
        elif value.ndim == 3 and valid.ndim == 2:
            expanded = valid.unsqueeze(0).expand_as(value)
        assert torch.isfinite(value[expanded]).all()
        return
    expanded = (
        np.broadcast_to(valid[..., None], value.shape)
        if value.ndim == 3 and valid.ndim == 2
        else valid
    )
    assert np.isfinite(value[expanded]).all()


def _run(backend: str) -> dict:
    import panorai

    if backend == "numpy":
        image, depth, valid, labels = _numpy_bundle()
    else:
        import torch

        image, depth, valid, labels = _torch_bundle(torch)

    source = {
        "image": _clone(image),
        "depth": _clone(depth),
        "valid": _clone(valid),
        "labels": _clone(labels),
    }
    panorama = (
        panorai.EquirectangularImage(image)
        .with_depth(depth, valid=valid, units="m")
        .with_labels(labels)
    )

    def model(value):
        return value * 0.75 + 0.125

    view_kwargs = {
        "layout": "fibonacci",
        "count": COUNT,
        "size": VIEW_SHAPE,
        "fov": FOV,
        "depth_policy": "renormalize",
        "min_valid_weight": 0.5,
    }
    views = panorama.views(**view_kwargs)
    description = views.describe()
    assert len(views) == COUNT
    assert description["contract"] == "geometry-v1"
    assert description["interface"] == "panorai-object-workflow/v1"
    assert description["stability"] == "stable"
    assert description["layout"] == "fibonacci"
    assert description["view_count"] == COUNT
    assert tuple(description["view_shape_hw"]) == VIEW_SHAPE
    assert set(description["modalities"]) == {"image", "depth", "labels"}
    expected_layout = "HWC" if backend == "numpy" else "NCHW"
    assert description["modalities"]["image"]["layout"] == expected_layout
    assert description["modalities"]["depth"]["units"] == "m"
    assert description["modalities"]["labels"]["interpolation"] == "nearest"

    expanded = views.map(model, input="image").reconstruct(blend=BLENDS)
    shortcut = panorama.process_views(
        model,
        input="image",
        blend=BLENDS,
        **view_kwargs,
    )

    for name in ("image", "depth", "labels"):
        _assert_equal(getattr(expanded, name), getattr(shortcut, name))
        _assert_equal(expanded.validity(name), shortcut.validity(name))
        _assert_finite_where_valid(getattr(expanded, name), expanded.validity(name))

    _assert_equal(panorama.image, source["image"])
    _assert_equal(panorama.depth, source["depth"])
    _assert_equal(panorama.validity("depth"), source["valid"])
    _assert_equal(panorama.labels, source["labels"])

    if backend == "numpy":
        assert expanded.image.shape == (HEIGHT, WIDTH, 3)
        assert expanded.depth.shape == (HEIGHT, WIDTH)
        assert expanded.labels.shape == (HEIGHT, WIDTH)
        assert set(np.unique(expanded.labels)).issubset(set(np.unique(labels)))
        valid_counts = {
            name: int(expanded.validity(name).sum())
            for name in ("image", "depth", "labels")
        }
    else:
        import torch

        assert tuple(expanded.image.shape) == (2, 3, HEIGHT, WIDTH)
        assert tuple(expanded.depth.shape) == (2, 1, HEIGHT, WIDTH)
        assert tuple(expanded.labels.shape) == (2, 1, HEIGHT, WIDTH)
        assert set(torch.unique(expanded.labels).tolist()).issubset(
            set(torch.unique(labels).tolist())
        )
        valid_counts = {
            name: int(expanded.validity(name).sum().item())
            for name in ("image", "depth", "labels")
        }

    return {
        "status": "ok",
        "consumer": "multimodal-arbitrary-n",
        "backend": backend,
        "version": version("panorai"),
        "view_count": COUNT,
        "view_shape_hw": list(VIEW_SHAPE),
        "valid_pixels": valid_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("numpy", "torch"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()

    import panorai

    origin = _assert_installed_origin(panorai.__file__, args.source_root)
    result = _run(args.backend)
    result["origin"] = str(origin)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
