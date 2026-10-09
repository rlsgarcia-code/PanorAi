# Tutorial: resize-free spherical monocular depth

This tutorial exposes the CNN stage of PanorAi's monocular-depth portability
work as an **Experimental**, opt-in API. It loads the exact external
Metric3D-v1 ConvNeXt-Tiny/Hourglass model evaluated by the P74 experiment,
ports its learned spatial layers to PanorAi spherical convolution, and returns
radial range on the input ERP lattice.

The interface is
`panorai-spherical-metric-depth/v1-experimental`. It is not part of stable
geometry, does not train a model, and does not bundle Metric3D code or weights.

See the {ref}`capability-map-image-processing` theme for the distinction
between sphere-native learned sampling and a planar model evaluated through
projected views.

**PanorAi-specific:** exact artifact identity, explicit acquisition consent,
complete layer/parameter provenance, differentiable longitude-wrapped tangent
sampling, bounded-memory row chunking, resize-free ERP input, and radial-range
semantics in the canonical panorama frame.

## 1. Install the optional runtime

```bash
pip install "panorai[deep-learning-depth]"
```

The ordinary `panorai`, `panorai.geometry`, and `panorai.experimental`
imports remain independent of Torch and Metric3D. Merely importing the
deep-learning module never contacts the network. Acquisition occurs only in
the explicit function below.

## 2. Review the external boundary

Only one model is supported in this first version:

| component | pinned identity | SHA-256 | terms |
| --- | --- | --- | --- |
| Metric3D source | commit `fd90d0dad05d913c4f45ed7a9789b8f1dfde22e0` | `929432fd1f7d1f2c45f51407e4ef30ca5758eb207b15e51c9b480048176ee365` | BSD-2-Clause source |
| ConvNeXt-Tiny/Hourglass v1 checkpoint | Hugging Face revision `80d2d1410afb4b23cd9d18c6be9144483d4b70b6` | `bc41f5f919bb0388bbc88fe1d9e60b49b826c620b4acc1b6c10f473f6d4741a5` | no separate checkpoint/model-card license found |

The missing checkpoint terms are a redistribution blocker. Setting
`accept_upstream_terms=True` records that the caller reviewed this boundary;
it is not a license grant from PanorAi. Source, checkpoint, extraction,
manifest, and predictions stay in the external cache selected by the caller.

## 3. Acquire and port the model

```python
from pathlib import Path

from panorai.experimental.deep_learning import (
    load_spherical_metric3d_convnext_tiny_v1,
)

loaded = load_spherical_metric3d_convnext_tiny_v1(
    accept_upstream_terms=True,
    cache_dir=Path("/data/model-cache/panorai-metric3d"),
    device="cpu",
    max_sampled_elements=100_000_000,
)

print(loaded.assets.to_dict())
print(loaded.port_report.to_dict())
```

Acquisition verifies full file size and SHA-256 before extraction or
deserialization. The checkpoint is read with PyTorch's `weights_only=True`.
A corrupt cached file is rejected and is not silently deleted or replaced.

For this exact model, the port report should identify 45 `Conv2d` layers,
three `ConvTranspose2d` layers, and four reflection pads absorbed into the
following spherical neighbourhood. Every learned weight and bias remains the
same `Parameter` object. A non-empty `remaining_planar_spatial_layers` is a
hard failure.

`max_sampled_elements` bounds the temporary tangent-neighbour tensor by
processing output rows in chunks. It does not resize the ERP, change kernel
weights, or alter the global output lattice.

## 4. Infer without resizing

```python
import numpy as np
from PIL import Image
import torch

rgb = np.asarray(Image.open("panorama.png").convert("RGB"), dtype=np.float32)
height, width = rgb.shape[:2]
assert width == 2 * height
assert height % 32 == 0 and width % 32 == 0

values = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
with torch.inference_mode():
    radial_range_m = loaded.model(values)

assert tuple(radial_range_m.shape) == (1, 1, height, width)
```

Input is `torch.float32` NCHW RGB in the explicit range `[0, 255]` on the same
device as the model. The wrapper does
not resize, crop, pad, or prefilter. This is intentional: resizing would add a
minification/aliasing variable that the current experiment has not controlled
well enough. If the native ERP dimensions are not divisible by 32, the caller
must choose and document a dataset-side sampling policy; the loader refuses to
invent one.

The model's canonical output is converted with effective ERP focal length
`width / (2π)` and clamped to its declared `0.3–150 m` domain. The result is
**radial range**, so a 3D point is `X = range * ray` in PanorAi's canonical
`+X` right, `+Y` up, `+Z` forward frame. It is not camera z-depth.

Model prediction coverage is not data validity. A scanner blind region,
missing RGB, or missing reference range still requires a separate explicit
boolean mask; neither black pixels nor the `0.3 m` prediction floor imply
invalidity.

## 5. What spherical porting means

At each output ray, PanorAi lays the pretrained raster kernel in the local
east/north tangent basis, maps taps to the unit sphere with the exponential
map, and samples the ERP differentiably with longitude wrapping. Transposed
convolutions first insert zeros on the source lattice and then use the same
spherical neighbourhood while preserving the planar layer's output shape.

This procedure preserves learned numbers; it does not prove that a
perspective-trained feature is rotation invariant or semantically calibrated
on panoramas. Resolution also changes angular tap spacing and effective
receptive field. A native-resolution run is therefore a distinct experiment,
not merely a higher-quality rendering.

## 6. Current evidence and limits

The development P74 study compares this direct spherical CNN with the same
checkpoint evaluated on six cube faces. Its source data, predictions, PLYs,
and metrics remain outside the package. The evaluated checkpoint is described
upstream as outdoor-only, while P74 contains industrial interiors. Results are
diagnostic and cannot establish general depth accuracy.

The first public adapter deliberately excludes:

- hidden resize or antialiasing policy;
- dataset loading, training, fine-tuning, or checkpoint redistribution;
- a tangent-view oracle;
- ViT attention or positional-embedding conversion;
- calibrated uncertainty or automatic validity masks.

The next separately documented stage will compare CNN and ViT portability
under the same radial-range, explicit-mask, resolution, cubemap, and 3D
structure protocol.

## Primary references

- [Metric3D: Towards Zero-shot Metric 3D Prediction from a Single Image](https://arxiv.org/abs/2307.10984)
- [Metric3D v2](https://arxiv.org/abs/2404.15506)
- [Metric3D source and checkpoints](https://github.com/YvanYin/Metric3D)
- [PanorAi pretrained spherical FCN/CAM tutorial](09_spherical_fcn_cam.md)
- [PanorAi geometry contract](../geometry-v1.md)
