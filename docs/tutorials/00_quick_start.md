# Quick start

```console
pip install panorai
```

`EquirectangularImage` is the ERP, `GnomonicFace` is one pinhole view, and
`GnomonicFaceSet` is the collection that processes and returns the views.

## 1. Multi-view: sample, process, fuse

```python
import cv2
import numpy as np
from panorai import EquirectangularImage

rgb = np.zeros((64, 128, 3), dtype=np.uint8)
panorama = EquirectangularImage(rgb)
views = panorama.views("icosahedron", size=64, fov=100)
blurred = views.map(lambda image: cv2.GaussianBlur(image, (5, 5), 1.0))
result = blurred.reconstruct(blend="gaussian")
```

```{image} ../_static/tutorials/pinhole-workflow-sundowner.jpg
:alt: ERP, views before processing, blurred views and fused ERP
```

- `layout`, `size`, `fov`: where and how densely to sample.
- `face.spec`: the exact path from each processed pixel back to the sphere.
- `average`/`gaussian`: continuous fusion; `closest`: labels and masks.
- {doc}`../reference/samplers_blenders`: all sampling and fusion policies.

Compose planar operations and let `process_views()` perform the complete route:

```python
def canny(image):
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.0), 40, 120)

edges = panorama.process_views(
    canny, layout="cube", size=64, fov=100, blend="closest"
)
```

The same `views.map(...).reconstruct(...)` route runs
{doc}`Depth Anything <../applications/pose_range/depth_anything_views>` on
tangent views and fuses radial range.

## 2. Native spherical operations

A native operation produces the ERP result directly:

```python
from panorai.core.numpy import SphericalGradient

gray = rgb.mean(axis=-1, dtype=np.float32)[None, None] / 255.0
gradient = SphericalGradient(operator="scharr")(gray)
magnitude = np.hypot(gradient[:, 0::2], gradient[:, 1::2])
```

```{image} ../_static/tutorials/spherical-image-processing.jpg
:alt: Direct spherical blur, Scharr tangent gradient and geodesic Canny
```

Two 256×512 installed-wheel trials measured 1.89–1.91 ms for native C++ Scharr
and 22.61–27.71 ms for cube → OpenCV → fusion (11.9–14.5×); yaw-equivariance
MAE was 0.00405 versus 0.00515. See {ref}`image-processing-routes`.

Compose the same primitives into a DoG pyramid:

```python
from panorai.core.numpy import SphericalGaussianBlur2d, SphericalResize

small = SphericalGaussianBlur2d(sigma=1.0)
large = SphericalGaussianBlur2d(sigma=2.0)
smooth_0 = small(gray)
dog_0 = large(gray) - smooth_0
octave_1 = SphericalResize(scale_factor=0.5)(smooth_0)
dog_1 = large(octave_1) - small(octave_1)
```

```{image} ../_static/tutorials/spherical-dog-pyramid.jpg
:alt: Input ERP and three sphere-native DoG octaves
```

The core also exposes `SphericalConv2d` and `SphericalMaxPool2d`. Experimental
ports replace pretrained CNN spatial layers with these ERP-native operations:
{doc}`Metric3D depth <10_spherical_monocular_depth>` and
{doc}`ImageNet CAM <09_spherical_fcn_cam>`.

## 3. View output types

| Face output | Call | Spherical result |
| --- | --- | --- |
| Dense C-channel logits, depth or labels | `map()` + `reconstruct()` | `EquirectangularImage` |
| Point detections or keypoints | `map_points()` + angular NMS | `SphericalPointSet` |
| Boxes, polygons or masks | `map_regions()` + solid-angle NMS | `SphericalRegionSet` |
| Whole-view scores or embeddings | `map_views()` + aggregation | `SphericalViewSet` |

```python
from panorai.data import RasterFieldSpec

logit_views = views.map(
    predict_logits,
    output="logits",
    spec=RasterFieldSpec.continuous(component_ids=class_names),
)
logits = logit_views.reconstruct(modalities="logits", blend="gaussian")
points = views.map_points(detect_points, source_id="detector")
regions = views.map_regions(detect_regions, source_id="detector")
evidence = views.map_views(classify_or_embed, source_id="encoder")
```

Dense channels are fused independently. Points become bearings for detection
or feature matching; regions use spherical overlap; embeddings use
coverage-aware aggregation. See {doc}`../explanation/workflow-evolution`.

## Next

- {doc}`02_projection_foundations` — exact projection geometry
- {doc}`../reference/core_operations` — NumPy/C++ and Torch primitives
- {doc}`../reference/stability` — Stable and Experimental surfaces
- {ref}`capability-map-quick-start` — complete capability map
