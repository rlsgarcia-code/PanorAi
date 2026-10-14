# PanorAi

[![PyPI](https://img.shields.io/pypi/v/panorai.svg)](https://pypi.org/project/panorai/)
[![Python](https://img.shields.io/pypi/pyversions/panorai.svg)](https://pypi.org/project/panorai/)
[![CI](https://github.com/rlsgarcia-code/PanorAi/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rlsgarcia-code/PanorAi/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-online-2563eb.svg)](https://rlsgarcia-code.github.io/PanorAi/)
[![License: MIT](https://img.shields.io/badge/license-MIT-059669.svg)](LICENSE)

**Build computer-vision methods for spherical images.** PanorAi provides two
composable routes: process sampled pinhole views and fuse them, or operate
directly on the spherical ERP lattice.

## Install

```bash
pip install panorai==3.7.0
```

NumPy is required. Torch is optional: `pip install "panorai[torch]"`.

## The three objects

| Object | Meaning |
| --- | --- |
| `EquirectangularImage` | The complete spherical raster and its typed fields. |
| `GnomonicFace` | One pinhole view with projection spec, support and validity. |
| `GnomonicFaceSet` | A set of views that can be processed, lifted or fused. |

## 1. Multi-view: sample, process, fuse

```text
EquirectangularImage
        │ .views(...)
        ▼
GnomonicFaceSet ── contains ──► GnomonicFace × N
        │ .map(pinhole_operation)
        ▼
GnomonicFaceSet
        │ .reconstruct(...)
        ▼
EquirectangularImage
```

```python
import cv2
import numpy as np
from panorai import EquirectangularImage

rgb = np.zeros((64, 128, 3), dtype=np.uint8)
panorama = EquirectangularImage(rgb)

views = panorama.views("icosahedron", size=64, fov=100)
blurred_views = views.map(lambda image: cv2.GaussianBlur(image, (5, 5), 1.0))
result = blurred_views.reconstruct(blend="gaussian")

# The same complete raster workflow in one call.
blurred = panorama.process_views(
    lambda image: cv2.GaussianBlur(image, (5, 5), 1.0),
    layout="icosahedron",
    size=64,
    fov=100,
    blend="gaussian",
)

# Swap in another pinhole operation; keep discrete edges discrete at fusion.
def canny(image):
    image_u8 = np.clip(image, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, 40, 120)

edges = panorama.process_views(
    canny, layout="cube", size=64, fov=100, blend="closest"
)
```

![ERP sampled into overlapping gnomonic views, processed and fused](docs/_static/tutorials/pinhole-workflow-sundowner.jpg)

Sampling selects face centres (`layout`), pinhole resolution (`size`) and
angular coverage (`fov`). Overlap lets the same spherical direction be seen by
multiple faces. Each processed pixel is then back-projected through its
`face.spec`; fusion combines only supported, valid contributors.

Use `average` for a uniform continuous mean, `gaussian` to favour face centres,
and `closest` to preserve one discrete value. PanorAi provides cube, Fibonacci,
icosahedron and spiral layouts. Keep the explicit three stages when inspecting
or batching faces; use `process_views()` for a single raster operation.
[Compare sampling and fusion methods →](https://rlsgarcia-code.github.io/PanorAi/reference/samplers_blenders.html)

## 2. Native spherical operations

```text
ERP ─┬─► spherical blur σ₁ ─┐
     └─► spherical blur σ₂ ─┴─ subtract ─► DoG octave 0
                    └─► SphericalResize(0.5) ─► DoG octave 1
```

```python
from panorai.core.numpy import (
    SphericalGaussianBlur2d,
    SphericalResize,
)

gray = rgb.mean(axis=-1, dtype=np.float32)[None, None] / 255.0
blur_small = SphericalGaussianBlur2d(sigma=1.0)
blur_large = SphericalGaussianBlur2d(sigma=2.0)

smooth_0 = blur_small(gray)
dog_0 = blur_large(gray) - smooth_0

octave_1 = SphericalResize(scale_factor=0.5)(smooth_0)
dog_1 = blur_large(octave_1) - blur_small(octave_1)
```

![Sphere-native multiscale DoG detections and tangent descriptors](docs/_static/tutorials/spherical-dog-sift.jpg)

The two response tensors stay on ERP lattices: `dog_1` has half the height and
width of `dog_0`. No faces are created or fused. The same core also provides
`SphericalConv2d`, pooling, gradients, reductions and differentiable Torch
layers.
[See the NumPy/C++ and Torch operations →](https://rlsgarcia-code.github.io/PanorAi/reference/core_operations.html)

### Pretrained CNNs on the sphere

![Pretrained planar layers converted to direct spherical depth and CAM inference](docs/_static/tutorials/spherical-cnn-porting.svg)

The port keeps learned parameters and replaces spatial sampling:
`Conv2d → SphericalConv2d`, `ConvTranspose2d → SphericalConvTranspose2d`, and
`MaxPool2d → SphericalMaxPool2d`. These adapters are Experimental and use
external weights; inference consumes the ERP directly, without multiface
projection or fusion.

Metric3D depth returns radial range on the input ERP lattice:

```python
from panorai.experimental.deep_learning import (
    load_spherical_metric3d_convnext_tiny_v1,
)

depth = load_spherical_metric3d_convnext_tiny_v1(
    accept_upstream_terms=True,
    device="cpu",
)
radial_range_m = depth.model(rgb_erp_nchw_float32_0_255)
assert not depth.port_report.remaining_planar_spatial_layers
```

[Metric depth tutorial →](https://rlsgarcia-code.github.io/PanorAi/tutorials/10_spherical_monocular_depth.html)

A pretrained ImageNet classifier becomes a fully convolutional spherical
network before producing a dense class-activation map:

```python
from panorai.experimental.deep_learning import (
    ImageNetFCN,
    class_activation_map,
    load_pretrained_imagenet_model,
    port_module_with_report,
)

loaded = load_pretrained_imagenet_model("resnet18")
classifier = ImageNetFCN(loaded.model, "resnet18").eval()
report = port_module_with_report(classifier)  # converts the model in place
dense = classifier.forward_dense(normalized_erp_nchw)
cam = class_activation_map(dense.logits, class_index=743)
assert not report.remaining_planar_spatial_layers
```

![Direct spherical ImageNet CAM over a complete ERP](docs/_static/tutorials/spherical-cam-imagenet-lakeside.jpg)

[Spherical FCN and CAM tutorial →](https://rlsgarcia-code.github.io/PanorAi/tutorials/09_spherical_fcn_cam.html)

## 3. What a view operation can return

| Output from each `GnomonicFace` | Face-set operation | Spherical result |
| --- | --- | --- |
| Dense raster: image, depth, label, logits | `map()` then `reconstruct()` | `EquirectangularImage` |
| Points: keypoints or detections | `map_points()` | `SphericalPointSet` |
| Regions: boxes, polygons or masks | `map_regions()` | `SphericalRegionSet` |
| Whole-view scores or embeddings | `map_views()` | `SphericalViewSet` |

The output type determines the return path: rasters are fused; observations are
lifted to spherical geometry and can then be deduplicated or aggregated.
[Read the typed observation workflow →](https://rlsgarcia-code.github.io/PanorAi/explanation/workflow-evolution.html)

## Go deeper

- [Quick start](https://rlsgarcia-code.github.io/PanorAi/tutorials/00_quick_start.html)
- [Geometry and coordinate contract](https://rlsgarcia-code.github.io/PanorAi/geometry-v1.html)
- [Stability matrix](https://rlsgarcia-code.github.io/PanorAi/reference/stability.html)
- [Changelog](CHANGELOG.md) · [License](LICENSE)
