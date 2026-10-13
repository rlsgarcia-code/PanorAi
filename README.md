# PanorAi

[![PyPI](https://img.shields.io/pypi/v/panorai.svg)](https://pypi.org/project/panorai/)
[![Python](https://img.shields.io/pypi/pyversions/panorai.svg)](https://pypi.org/project/panorai/)
[![CI](https://github.com/rlsgarcia-code/PanorAi/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rlsgarcia-code/PanorAi/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-online-2563eb.svg)](https://rlsgarcia-code.github.io/PanorAi/)
[![License: MIT](https://img.shields.io/badge/license-MIT-059669.svg)](LICENSE)

**Geometry-safe spherical image processing and projection for computer-vision
applications.**

## How PanorAi organizes spherical computer vision

A spherical image is not a pinhole image. PanorAi exposes two explicit ways to
use computer-vision methods without hiding that difference:

1. **Transfer through projections.** Sample the sphere into tangent pinhole
   views, run an existing planar method on each view, map its outputs back, and
   fuse them into an ERP or a set of spherical observations.
2. **Use a sphere-aware route.** Run an operator directly on the ERP/bearings,
   or use a hybrid pipeline whose local evidence is planar but whose matching,
   geometry, fusion, or association is spherical.

The first route is the essential reusable mechanism:

```text
ERP → sample views → run the pinhole method per view
    → back-project raster outputs or lift point outputs to bearings
    → fuse/deduplicate → final ERP, features, geometry, or hypotheses
```

PanorAi keeps sampling support, numerical validity, interpolation, face
provenance, and fusion policy in the result. It never infers validity from a
non-zero pixel value.

## Install

```bash
pip install panorai==3.7.0
```

NumPy and OpenCV cover the projection and feature route. Install
`panorai[torch]`, `panorai[deep-learning]`, `panorai[pycolmap]`,
`panorai[slam]`, or `panorai[pcd]` only for the corresponding optional route.

## 1. Transfer an existing pinhole method through projections

### Step 1 — sample the sphere

Create the view set once. Every face retains its projection specification,
support mask, source ERP shape, modality metadata, and stable face ID.

```python
import cv2
import numpy as np
from panorai import EquirectangularImage

erp = np.zeros((128, 256, 3), dtype=np.uint8)  # HWC, 2:1 ERP
panorama = EquirectangularImage(erp)
views = panorama.views(
    "icosahedron",
    subdivisions=0,
    size=(64, 64),
    fov=100,
)
print(len(views), views[0].face_id, views[0].spec)
```

Choose the sampling pattern according to the method's receptive field and the
required overlap:

| Sampling method | Call | Use when |
| --- | --- | --- |
| [Cube](docs/reference/samplers_blenders.rst#cube) | `views("cube", ...)` | six fixed orthogonal faces are sufficient |
| [Icosahedron](docs/reference/samplers_blenders.rst#icosahedron) | `views("icosahedron", subdivisions=n, ...)` | progressively denser, symmetric coverage is useful |
| [Fibonacci](docs/reference/samplers_blenders.rst#fibonacci) | `views("fibonacci", count=n, ...)` | a requested number of nearly uniform directions is needed |
| [Spiral](docs/reference/samplers_blenders.rst#spiral) | `views("spiral", count=n, ...)` | ordered coverage along a spherical path is useful |
| [Custom sampler](docs/reference/samplers_blenders.rst#custom-sampler-objects) | `views(sampler_object, ...)` | the application owns the tangent directions |

`size` controls pinhole resolution; `fov` controls angular coverage. Increasing
face count without sufficient FOV does not close gaps, and increasing FOV also
increases gnomonic distortion toward face boundaries.

### Step 2 — run the face-local application

#### Process each face

`GnomonicFaceSet.map()` applies a planar array-to-array function without
changing the original views. Raster outputs can be reconstructed directly.

```python
processed_views = views.map(
    lambda image: cv2.GaussianBlur(image, (5, 5), 1.2),
    input="image",
)
processed_erp = processed_views.reconstruct(
    modalities="image",
    blend="gaussian",
)
```

Use the same pattern for enhancement, restoration, edge maps, dense features,
or any planar network that preserves the face's spatial shape.

#### Detect points or objects

Detections are not raster images. Run the detector per face, convert face-local
pixels to panorama-frame bearings, then deduplicate or associate the spherical
observations. This example uses a point detector; box centres and corners use
the same conversion.

```python
from panorai.geometry import gnomonic_pixels_to_rays

detections_by_face = []
for face in views:
    gray = cv2.cvtColor(face.image, cv2.COLOR_RGB2GRAY)
    points = cv2.goodFeaturesToTrack(gray, 20, 0.01, 4)
    pixels_xy = (
        np.empty((0, 2), dtype=np.float64)
        if points is None
        else points[:, 0, :].astype(np.float64)
    )
    projected = gnomonic_pixels_to_rays(pixels_xy, face.spec)
    detections_by_face.append(
        (face.face_id, projected.rays_xyz[projected.valid])
    )
```

For general object boxes, the detector still owns class scores and box NMS;
PanorAi owns the face geometry and spherical coordinates. For semantic regions
and registered multiview hypotheses, use `panorai.object_localization`.

#### Extract local features

The high-level feature pipeline encapsulates the same projection route because
feature fusion requires face provenance, edge rejection, spherical duplicate
suppression, and descriptor-row bookkeeping—not pixel blending.

```python
from panorai.features import SphericalFeaturePipeline

feature_pipeline = SphericalFeaturePipeline.from_preset(
    "sift-flann",
    face_sampler="icosahedron",
    face_shape_hw=(64, 64),
    face_fov_deg=100,
    edge_margin_px=4,
    max_features=100,
)
features = feature_pipeline.extract(erp, panorama_id="p0")
print(features.source_erp_xy.shape, features.bearings.shape)
```

Use `extract_and_match()` for two ERPs. The result contains ERP coordinates,
unit bearings, descriptors, matches, and duplicate provenance. See
[features and matching](docs/tutorials/03_features_and_matching.md).

#### Segment each face

Dense class IDs are mapped as the `labels` modality. The nearest/closest route
preserves discrete IDs and never interpolates them bilinearly. The callback
below demonstrates the required planar segmenter contract.

```python
def planar_segmenter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return (gray >= 128).astype(np.int16)


label_views = views.map(
    planar_segmenter,
    input="image",
    output="labels",
)
segmentation_erp = label_views.reconstruct(
    modalities="labels",
    blend="closest",
)
```

Replace the callback with the planar semantic model of choice. If the model
returns masks, keep their explicit validity/support instead of inferring it
from class ID zero.

#### Estimate monocular depth per face

Depth callbacks return axial or radial values plus a boolean validity mask. A
model adapter must declare the conversion to PanorAi's canonical radial range.
The simple callback below only demonstrates the workflow contract; the
published DA3 adapter performs the calibrated conversion.

```python
def planar_radial_range(image):
    value = 1.0 + image.mean(axis=-1, dtype=np.float32) / 255.0
    return value.astype(np.float32), np.isfinite(value)


depth_views = views.map(
    planar_radial_range,
    input="image",
    output="depth",
    units="m",
)
range_erp = depth_views.reconstruct(
    modalities="depth",
    blend="gaussian",
)
radial_range_m = range_erp.depth
radial_valid = range_erp.validity("depth")
```

For real inference, see the checksum-pinned
[spherical monocular-depth adapter](docs/tutorials/10_spherical_monocular_depth.md).

### Step 3 — fuse the face results

Raster fusion always uses explicit per-face validity masks. Select the policy
by output semantics:

| Fusion method | Use for | Behavior |
| --- | --- | --- |
| [Average](docs/reference/samplers_blenders.rst#average) | continuous values with comparable confidence | uniform mean of valid overlaps |
| [Gaussian](docs/reference/samplers_blenders.rst#gaussian) | RGB, logits, or continuous depth | favours face centres over distorted borders |
| [Closest](docs/reference/samplers_blenders.rst#closest) | class IDs and other discrete labels | selects one valid contributor; never averages IDs |
| [Feathering](docs/reference/samplers_blenders.rst#feathering) | NumPy continuous rasters | distance-to-support-boundary weighting |
| [Huber](docs/reference/samplers_blenders.rst#huber) | NumPy overlaps with outliers | robust per-value location estimate |
| [Diagnostics](docs/reference/samplers_blenders.rst#diagnostic-blenders) | audit only | overlap count or dispersion, not a fused prediction |

`average`, `gaussian`, and `closest` are first-class named policies in
`reconstruct()`. Additional NumPy blenders are passed as blender objects.
Bundles choose per modality, for example
`blend={"image": "gaussian", "labels": "closest", "depth": "gaussian"}`.

`process_views()` is the one-call form of exactly
`views().map().reconstruct()`:

```python
shortcut = panorama.process_views(
    lambda image: cv2.GaussianBlur(image, (5, 5), 1.2),
    layout="cube",
    size=64,
    fov=95,
    blend="gaussian",
)
```

For point/vector outputs, “fusion” means mapping to bearings followed by the
appropriate spherical deduplication, matching, NMS, or association—not a
raster blender.

## 2. Use routes designed for the sphere

Projection transfer is not mandatory. PanorAi also provides algorithms whose
neighbourhoods, residuals, search, or learned operations are defined on the
sphere, plus hybrid pipelines that combine local pinhole evidence with
spherical geometry.

### Spherical-only routes

| Application | Public entry point | Final representation |
| --- | --- | --- |
| filtering, gradients, Canny, pyramids | `panorai.image_processing` | ERP-aligned raster |
| direct spherical DoG proposals | `SphericalDoGDetector` | ERP pixels and unit bearings |
| relative pose | `SphericalRelativePoseEstimator` | rotation, translation direction, inliers, quality report |
| dense two-view range | `estimate_spherical_range` | radial range, validity, confidence, consistency |
| pretrained dense CAM evidence | `panorai.experimental.deep_learning` | class channels on a spherical evidence lattice |

Example—no faces and no reconstruction:

```python
from panorai.image_processing import spherical_canny, spherical_gaussian_blur

direct_blur = spherical_gaussian_blur(erp, ksize=5, sigma=1.2)
direct_edges = spherical_canny(
    erp[..., 0],
    threshold_low=20,
    threshold_high=60,
)
```

### Hybrid routes

| Application | Local stage | Spherical stage |
| --- | --- | --- |
| `SphericalFeaturePipeline` | planar detector/descriptor on overlapping faces | map to bearings and suppress duplicates |
| `SphericalGlobalMapper` and SLAM | local feature evidence | pose graph, bearing tracks, triangulation, spherical BA |
| `SphericalSemanticSegmenter` | transient SAM gnomonic prompts | spherical CAM seeds, expansion, mask fusion |
| `SphericalObjectHypothesisMethod` | optional per-view CAM/embedding regions | registered-view association and spatial hypotheses |

Hybrid does not mean “convert everything back to an ERP.” The final object may
be a fused raster, a set of bearings, camera poses, sparse 3D points, or an
auditable semantic hypothesis. Follow the contract of that result:

- [sphere-native image processing](docs/tutorials/spherical_image_processing.md)
- [two-view geometry](docs/tutorials/04_two_view_geometry.md)
- [dense spherical stereo](docs/tutorials/06_spherical_dense_stereo.md)
- [multiview reconstruction](docs/tutorials/05_multiview_reconstruction.md)
- [spherical FCN/CAM](docs/tutorials/09_spherical_fcn_cam.md)
- [semantic object hypotheses](docs/tutorials/11_cam_guided_object_hypotheses.md)

## Contracts that apply to both routes

- Coordinates are pixel-centre based, top-left origin, `+X` right, `+Y` up,
  `+Z` forward, longitude `[-π, π)`, and north-to-south latitude.
- Depth means radial range unless an API explicitly names another quantity.
- Geometric support, numerical validity, and value are distinct. Zero is data;
  it is never an implicit invalid marker.
- Labels and masks require nearest-neighbour sampling. Bilinear interpolation
  of integer or boolean data fails explicitly.
- NumPy is required; Torch, Open3D, PyCOLMAP, learned checkpoints, and datasets
  remain optional and are not imported by the canonical geometry API.
- Check the [stability matrix](docs/reference/stability.rst) before depending
  on an Experimental surface.

## Reference

- [Projection foundations](docs/tutorials/02_projection_foundations.md)
- [Sampling and fusion methods](docs/reference/samplers_blenders.rst)
- [Execution-strategy capability map](docs/tutorials/spherical_capability_map.md)
- [API reference](docs/reference/index.rst)
- [Geometry v1 contract](docs/geometry-v1.md)
- [Evidence and benchmarks](docs/benchmarks.rst)
- [Changelog](CHANGELOG.md)
- [License](LICENSE)
