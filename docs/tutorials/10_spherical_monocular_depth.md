# Tutorial: resize-free spherical monocular depth

This tutorial exposes two complementary stages of PanorAi's monocular-depth
portability work as **Experimental**, opt-in APIs. The CNN path loads the exact
external Metric3D-v1 ConvNeXt-Tiny/Hourglass model, ports its learned spatial
layers to PanorAi spherical convolution, and returns radial range on the input
ERP lattice. The ViT control keeps the official DA3Metric-Large network
unchanged, evaluates native-density gnomonic views without resizing, converts
its axial output to radial range, and reconstructs the predictions in ERP.

The interfaces are `panorai-spherical-metric-depth/v1-experimental` and
`panorai-depth-anything-3-metric-tangent/v1-experimental`. Neither is part of
stable geometry. They do not train models or bundle upstream code or weights.

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

No private evaluation dataset, prediction, metric, or result is included in
this distribution or used as a package claim. Public accuracy evidence remains
an open requirement for this Experimental adapter.

The loader and port report therefore prove only mechanical availability,
complete learned-layer replacement, and identity preservation of the original
weights. They do **not** prove that those outdoor-trained weights retain useful
semantics on an ERP, that the spherical and planar/cubemap computations are
numerically equivalent, or that the model is suitable for a particular domain.

The first public adapter deliberately excludes:

- hidden resize or antialiasing policy;
- dataset loading, training, fine-tuning, or checkpoint redistribution;
- a tangent-view oracle;
- ViT attention or positional-embedding conversion;
- calibrated uncertainty or automatic validity masks.

The DA3 control below makes the CNN and ViT portability comparison reproducible
under the same radial-range, explicit-mask, native-resolution, view-projection,
and 3D structure protocol. It does not turn the ViT into a sphere-native
network.

## 7. Depth Anything 3 Metric Large tangent control

DA3Metric-Large is available as a separate reproducibility adapter. Its
artifact boundary is fully explicit:

| component | pinned identity | SHA-256 | terms |
| --- | --- | --- | --- |
| Depth Anything 3 source | commit `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` | `98aa2dd53ab44b96cef5190ae4841c6ae51797d099b784e08e12a1a55bd3a69b` | Apache-2.0 |
| DA3Metric-Large checkpoint | Hugging Face revision `4010e39f3634a45bc60553321fb49fb760bd594e` | `bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776` | Apache-2.0 model card |

Source and checkpoint remain outside the source tree, wheel, and sdist. The
adapter builds the architecture declared by the pinned
`da3metric-large.yaml`, loads the safetensors checkpoint strictly, freezes the
network, and bypasses the official high-level resize path.

```python
from pathlib import Path

import numpy as np
from PIL import Image

from panorai.data import EquirectangularImage
from panorai.experimental.deep_learning import load_da3metric_large

rgb = np.asarray(Image.open("panorama.png").convert("RGB"), dtype=np.uint8)
panorama = EquirectangularImage(rgb)

# 42 overlapping views. H and W must each be divisible by DA3's patch size 14.
views = panorama.views(
    "icosahedron",
    subdivisions=1,
    size=(812, 1400),
    fov=(100.0, 80.0),
)

loaded_da3 = load_da3metric_large(
    accept_upstream_terms=True,
    cache_dir=Path("/data/model-cache/panorai-da3"),
    device="cuda",
)

# All views above share shape and FOV, hence the same intrinsic matrix.
adapter = loaded_da3.tangent(views[0].spec)
predicted = views.map(adapter, input="image", output="depth", units="m")
erp_depth = predicted.reconstruct(
    blend={"depth": "gaussian"},
    modalities=("depth",),
)

radial_range_m = erp_depth.depth
radial_valid = erp_depth.validity("depth")
```

The adapter accepts only HWC `uint8` or `float32` RGB in `[0, 255]` at the
declared tangent shape. It does not resize, crop, pad, clamp, prefilter, or
smooth. The exact conversion for pixel centre `(u, v)` is

```text
z_m = network_depth * ((fx + fy) / 2) / 300
range_m = z_m * sqrt(1 + ((u + 0.5 - W/2) / fx)^2
                           + ((v + 0.5 - H/2) / fy)^2)
```

Here `z_m` is axial depth along the virtual-camera forward axis and
`range_m` is the radial quantity required by PanorAi's geometry contract.
Finite positive model outputs define numerical prediction validity; the
workflow intersects that mask with each view's geometric support. It never
infers validity from black RGB or from a zero-filled missing region.

The final Gaussian blend in this example is intentionally a fixed baseline.
It does not claim that view boundaries are solved. A learned overlap fusion
conditioned by corresponding view tokens remains a separate research module,
not part of this adapter or the projection core.

## Primary references

- [Metric3D: Towards Zero-shot Metric 3D Prediction from a Single Image](https://arxiv.org/abs/2307.10984)
- [Metric3D v2](https://arxiv.org/abs/2404.15506)
- [Metric3D source and checkpoints](https://github.com/YvanYin/Metric3D)
- [Depth Anything 3 source](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [DA3Metric-Large model card](https://huggingface.co/depth-anything/DA3METRIC-LARGE)
- [PanorAi pretrained spherical FCN/CAM tutorial](09_spherical_fcn_cam.md)
- [PanorAi geometry contract](../geometry-v1.md)
