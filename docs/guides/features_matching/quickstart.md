# Extract and match spherical features

**Level:** Start · **Strategy:** Projection-domain · **Stability:** Stable
face-based feature route

`SphericalFeaturePipeline` samples views, runs the planar detector, lifts every
accepted point to a unit bearing and suppresses overlap duplicates.

```python
from panorai.features import SphericalFeaturePipeline

pipeline = SphericalFeaturePipeline.from_preset(
    "sift-flann",
    face_sampler="icosahedron",
    face_shape_hw=(256, 256),
)
features_a = pipeline.extract(erp_a, panorama_id="a")
features_b = pipeline.extract(erp_b, panorama_id="b")
matches = pipeline.match(features_a, features_b)
bearings = matches.to_bearing_correspondences()
```

The result retains face pixels, ERP pixels, bearings, descriptors and
provenance. Descriptor matches are candidates; pose estimation performs the
geometric verification.

## Inputs and outputs

`erp_a` and `erp_b` are NumPy ERPs. If either image has unobserved or invalid
directions, pass its explicit support/validity mask through the documented
pipeline configuration; do not encode invalidity as black RGB. Each extracted
feature contains its unit bearing and origin face, which is the representation
consumed by spherical geometry.

`matches` records descriptor indices and scores. Calling
`to_bearing_correspondences()` produces paired unit vectors, not verified 3-D
points and not metric motion.

## First diagnostics

Before pose estimation, plot accepted bearings over both ERPs and inspect:

- angular coverage instead of raw match count;
- concentration at face edges or the ERP seam;
- repeated structures that survive the ratio test;
- the number removed by reciprocal matching and angular NMS.

If the scene is mostly rotational, distant or textureless, descriptor matches
may look plausible while translation remains unobservable.

Next: [understand projected features](projected_features.md).
