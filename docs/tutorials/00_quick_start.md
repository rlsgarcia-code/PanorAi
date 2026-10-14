# Quick start

```console
pip install panorai
```

## Three objects

| Object | Role |
| --- | --- |
| `EquirectangularImage` | Complete spherical raster |
| `GnomonicFace` | One projected pinhole view |
| `GnomonicFaceSet` | Views processed and returned together |

## 1. Multi-view: sample, process, fuse

```text
EquirectangularImage
  └─ views(...) → GnomonicFaceSet[GnomonicFace × N]
       └─ map(operation) → processed views
            └─ reconstruct(...) → EquirectangularImage
```

```python
import cv2
import numpy as np
from panorai import EquirectangularImage

rgb = np.zeros((64, 128, 3), dtype=np.uint8)
panorama = EquirectangularImage(rgb)
views = panorama.views("icosahedron", size=64, fov=100)
views = views.map(lambda image: cv2.GaussianBlur(image, (5, 5), 1.0))
result = views.reconstruct(blend="gaussian")

# Equivalent shortcut: sample → process every face → fuse → ERP.
blurred = panorama.process_views(
    lambda image: cv2.GaussianBlur(image, (5, 5), 1.0),
    layout="icosahedron",
    size=64,
    fov=100,
    blend="gaussian",
)

def canny(image):
    image_u8 = np.clip(image, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, 40, 120)

edges = panorama.process_views(
    canny, layout="cube", size=64, fov=100, blend="closest"
)
```

Use `process_views()` when the face operation returns one raster.

```{image} ../_static/tutorials/pinhole-workflow-sundowner.jpg
:alt: ERP sampled into gnomonic views, blurred and fused into an ERP
```

- Sampling: `layout` places face centres, `size` sets pinhole resolution, and
  `fov` sets angular coverage and overlap.
- Return: each face result is back-projected with its own `face.spec`; only
  supported, valid pixels contribute.
- Fusion: `average` uniformly blends continuous values, `gaussian` favours
  face centres, and `closest` preserves one discrete value.
- {doc}`../reference/samplers_blenders`

## 2. Native spherical operations

```text
ERP → SphericalConv2d → SphericalResize → SphericalMaxPool2d → ERP
```

```python
from panorai.core.numpy import (
    SphericalConv2d,
    SphericalMaxPool2d,
    SphericalResize,
)

gray = rgb.mean(axis=-1, dtype=np.float32)[None, None] / 255.0
kernel = np.ones((1, 1, 3, 3), dtype=np.float32) / 9.0
x = SphericalConv2d(kernel, padding=1)(gray)
x = SphericalResize(scale_factor=0.5)(x)
x = SphericalMaxPool2d(kernel_size=2)(x)
```

No faces are created. NumPy can use the native C++ backend; the equivalent
Torch layers preserve autograd. See {doc}`../reference/core_operations`.

## 3. View output types

| Output | Operation | Result |
| --- | --- | --- |
| Dense raster | `map()` + `reconstruct()` | `EquirectangularImage` |
| Points | `map_points()` | `SphericalPointSet` |
| Regions | `map_regions()` | `SphericalRegionSet` |
| View score or embedding | `map_views()` | `SphericalViewSet` |

Rasters are fused. Points, regions and view evidence are lifted into spherical
observations. See {doc}`../explanation/workflow-evolution`.

## Next

- {doc}`02_projection_foundations` — projection geometry
- {doc}`../reference/stability` — Stable and Experimental surfaces
- {ref}`capability-map-quick-start` — complete capability map
