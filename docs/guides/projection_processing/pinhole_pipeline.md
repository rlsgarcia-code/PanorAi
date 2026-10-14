# Apply a pinhole method over the sphere

**Level:** Start · **Strategy:** Projection-domain · **Stability:** Stable
object workflow

Use this route when the operation already consumes an ordinary perspective
image. PanorAi owns the spherical sampling and reconstruction; the face
function remains caller-owned.

## What the workflow guarantees

The callback receives a normal HWC image, but it is not an arbitrary crop. Its
pixels are sampled along the rays described by `face.spec`; `face.support`
identifies directions that really belong to the source panorama. The inverse
mapping uses the same geometry, so overlap is resolved on the sphere rather
than by pasting rectangular faces.

```python
import cv2
import numpy as np
from panorai import EquirectangularImage

rgb = np.zeros((128, 256, 3), dtype=np.uint8)
panorama = EquirectangularImage(rgb)
views = panorama.views("icosahedron", size=128, fov=100)
processed = views.map(lambda face: cv2.GaussianBlur(face, (5, 5), 1.0))
result = processed.reconstruct(blend="gaussian")
```

The result is another `EquirectangularImage`. Access the reconstructed RGB with
`result.image` and retain `result.support`; black pixels are not an implicit
invalid mask.

## When to use the explicit stages

Keep `views`, `processed` and `result` separate when you need to inspect a
failed face, batch a model, or return something other than a raster. For a
single raster callback, `panorama.process_views(...)` executes the same
`views().map().reconstruct()` composition.

Typical face-local functions include OpenCV filters, a detector, a semantic
segmenter, or a monocular depth model. Their outputs do **not** all use the
same return path: continuous rasters are blended, labels use nearest/closest,
and points or regions are lifted as observations.

## Failure checks

- If seams appear, inspect overlap, border support and the selected blender.
- If integer labels acquire new IDs, bilinear sampling was used incorrectly.
- If a method fails near face edges, increase overlap or reject an edge margin.
- If a face needs a camera matrix, derive it from `face.spec`; do not reuse an
  ERP width as a pinhole focal length.

Next: [choose sampling and fusion](sampling_and_fusion.md), or read the complete
[projection tutorial](../tutorials/02_projection_foundations.md).
