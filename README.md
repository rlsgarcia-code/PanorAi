# PanorAi

[![PyPI](https://img.shields.io/pypi/v/panorai.svg)](https://pypi.org/project/panorai/)
[![Python](https://img.shields.io/pypi/pyversions/panorai.svg)](https://pypi.org/project/panorai/)
[![CI](https://github.com/rlsgarcia-code/PanorAi/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rlsgarcia-code/PanorAi/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-online-2563eb.svg)](https://rlsgarcia-code.github.io/PanorAi/)
[![License: MIT](https://img.shields.io/badge/license-MIT-059669.svg)](LICENSE)

**Spherical image processing and projection with explicit geometry, support,
validity and units.** Use ready-made applications, compose a pipeline, call
sphere-native operations, or keep using the Stable object workflow directly.

```bash
pip install panorai==3.7.0
```

NumPy is required. Torch and model integrations are optional extras. Models,
datasets and checkpoints are never bundled into the wheel.

## Stable foundation: three objects

| Object | Meaning |
| --- | --- |
| `EquirectangularImage` | Complete spherical raster with typed fields. |
| `GnomonicFace` | One pinhole view with projection, support and validity. |
| `GnomonicFaceSet` | Views processed, lifted or fused together. |

```python
import numpy as np
from panorai import EquirectangularImage

rgb = np.zeros((64, 128, 3), dtype=np.float32)
panorama = EquirectangularImage(rgb)
views = panorama.views("fibonacci", count=20, size=64, fov=100)
result = views.map(lambda image: image * 0.8).reconstruct(blend="gaussian")
```

Projection follows `geometry-v1`: pixel centres, top-left image origin,
`+X` right, `+Y` up, `+Z` forward, longitude `[-π, π)`, and latitude north to
south. Depth means radial range unless another quantity is named explicitly.

## 1. Ready-made applications

The application wrappers are the shortest entry point:

```python
from panorai.applications import (
    SphericalDoGFeatures,
    SphericalSLIC,
    SphericalStereoDepth,
)

labels = SphericalSLIC(n_segments=100)(rgb)
features = SphericalDoGFeatures()(rgb, panorama_id="frame-0001")
depth = SphericalStereoDepth()(reference, source, R_source_from_reference, t)
```

| Application | Input | Result | Implementation reused |
| --- | --- | --- | --- |
| `SphericalSLIC` | ERP RGB | typed label `RasterField` | intrinsic spherical SLIC |
| `SphericalDoGFeatures` | ERP RGB | bearings and descriptors | spherical DoG + tangent SIFT |
| `SphericalStereoDepth` | two ERPs and metric pose | radial range, confidence, validity | direct spherical plane sweep |
| `SphericalMaskRCNNPaper` | ERP plus external adapters | proposals, instances and masks | CAM/SAM/heuristic composition |

Every wrapper supports `run()`, `__call__()`, `get_params()`, immutable
`with_params()`, `describe()`, and `as_pipeline()`. Wrappers adapt the existing
algorithms; they do not fork their mathematics.

These wrappers are currently **Experimental**. Their source implementation is
available, but each one remains blocked from Stable until its installed-wheel,
license, performance and prospective scientific gates pass.

## 2. Compose the lower-level API

Use `Pipeline` when the ready-made application is close, but its stages need to
change:

The generic `panorai-spherical-pipeline/v1` contract is **Stable**. CAM/SAM
proposal and instance-fusion stages remain Experimental and do not inherit
that promotion merely because they share the namespace.

```python
from panorai.pipeline import (
    Output,
    PerView,
    Pipeline,
    ReconstructERP,
    SampleViews,
)

pipeline = Pipeline(
    SampleViews("fibonacci", count=100, size=256, fov=100),
    PerView.parallel(
        batch="auto",
        labels=Output(segmenter, kind="labels"),
        depth=Output(depth_model, kind="depth", units="m"),
        detections=Output(detector, kind="detections"),
    ),
    ReconstructERP(labels="closest", depth="gaussian"),
)
result = pipeline(panorama)
```

The core stages are:

- `OnSphere`: apply a direct ERP operation and transform its grid;
- `SampleViews`: create canonical gnomonic views;
- `PerView` and `PerView.parallel`: apply one or several view-local heads;
- `ReconstructERP`: fuse raster branches and retain non-raster branches;
- `LiftToBearings` and `AggregateViews`: convert or aggregate view evidence;
- `Parallel`, `Select`, and `Attach`: branch and route named results.

`PerView.parallel` is logical branching, not a promise of threads. `batch=None`
is per-view execution, an integer fixes chunk size, and `batch="auto"` is only
available to adapters that declare batch support. There is no silent retry on
out-of-memory errors.

Pipeline outputs preserve their meaning:

| Output family | Public result |
| --- | --- |
| image, depth, labels, mask, logits, features | `RasterField` with `RasterGrid` |
| classification | `ClassificationResult` |
| embedding | `EmbeddingResult` |
| keypoints or detections | per-view or `SphericalObservations` |
| multiple branches | `ResultBundle` |

Third-party stages implement the public `Stage` protocol and return
`StageResult`. A `Pipeline` also implements `Stage`, so pipelines can be nested.

## 3. Sphere-native operations and Torch layers

Functions and immutable operation objects live in `panorai.ops.spherical`:

```python
from panorai.ops.spherical import GaussianBlur, SLIC, sobel
from panorai.pipeline import OnSphere, Pipeline

smooth = Pipeline(OnSphere(GaussianBlur(ksize=5, sigma=1.2)))(panorama)
labels = Pipeline(OnSphere(SLIC(n_segments=100), output="labels"))(panorama)
east_gradient = sobel(rgb.mean(axis=-1), direction="east")
```

The library includes resampling, rotation, filters, Gaussian/box/median/
bilateral blur, Sobel, gradient, Laplacian, Canny, solid-angle statistics,
SLIC, DoG, spherical NMS, and Gaussian/Laplacian pyramids. Historical imports
from `panorai.image_processing` remain valid.

The general NumPy operations form the Stable `panorai-spherical-ops/v1`
contract. `SLIC`, `DoGDetector`, the mixed `algorithms` gateway, and the
optional Torch layer namespace remain Experimental while their separate gates
are open.

Optional Torch layers use the same spherical grid contract:

```python
import torch
from panorai.ops.spherical import nn as spherical_nn

network = torch.nn.Sequential(
    spherical_nn.Conv2d(3, 16, 3, padding=1),
    torch.nn.ReLU(),
    spherical_nn.MaxPool2d(2),
)
report = spherical_nn.validate(network, input_shape=(256, 512))
```

Convolution, transposed convolution and pooling update lattice anchor, stride
and resolution. A reduced lattice cannot masquerade as canonical ERP. Learned
convolutions propagate validity strictly; only normalized average/Gaussian
operations define validity renormalization.

## 4. Mask R-CNN taxonomy and the paper method

The executable paper method keeps Mask R-CNN responsibilities visible without
requiring users to assemble every head:

```python
from panorai.experimental.deep_learning import mask_rcnn

method = mask_rcnn.paper(
    proposal_mode="cam",
    proposal=cam_adapter,
    mask=sam_adapter,
    heuristic=instance_heuristic,
    concepts=("pipe", "equipment"),
    num_classes=3,
)
instances = method(rgb)
architecture = method.architecture
```

- In CAM mode, CAM is the proposal generator and SAM refines region masks.
- In SAM mode, SAM proposes masks and can reuse them with
  `reuse_proposal_masks=True`.
- The instance heuristic fills the `BoxHead` classification responsibility,
  can abstain, and is not presented as a learned classifier model.
- `.architecture` exposes Backbone, ProposalGenerator, RoIAlign, BoxHead,
  MaskHead and RoIHeads.

`SphericalMaskRCNN` is different: it represents the conventional future
trainable Backbone → RPN → RoIAlign → Box/Mask topology, but deliberately
raises `UnboundMaskRCNNError`. PanorAi does not add a trainer, dataset framework
or checkpoints to its projection core.

## Reproducibility and safety

Pipeline topology uses canonical JSON and an explicit PanorAi allowlist; a file
cannot request an arbitrary Python import. Cache entries use safe NumPy
archives, atomic writes and checksums—never pickle. Cache identity covers
values, support, validity, grid, dtype, layout, backend, device, version and
configuration. Registered Torch architectures store weights with safetensors;
cache identity also covers their weights and requires eval mode. Tensors with
`requires_grad=True` fail before caching so autograd is never disconnected
silently.

## Go deeper

- [Composed API](https://rlsgarcia-code.github.io/PanorAi/guides/composed_spherical_api.html)
- [Geometry contract](https://rlsgarcia-code.github.io/PanorAi/geometry-v1.html)
- [Sampling and fusion](https://rlsgarcia-code.github.io/PanorAi/reference/samplers_blenders.html)
- [Spherical operations](https://rlsgarcia-code.github.io/PanorAi/reference/spherical_ops.html)
- [Stability matrix](https://rlsgarcia-code.github.io/PanorAi/reference/stability.html)
- [Changelog](CHANGELOG.md) · [License](LICENSE)
