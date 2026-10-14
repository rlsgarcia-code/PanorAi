# How projected features become spherical observations

**Level:** Compose · **Strategy:** Projection-domain

The face-based route is:

```text
ERP -> overlapping gnomonic faces -> detector/descriptor
    -> supported face points -> unit bearings -> angular NMS
```

Overlap prevents an object from being lost at a face boundary, but it also
creates duplicate detections. Deduplication therefore uses angular distance,
not ERP pixel distance. Each survivor retains the suppressed observations as
provenance.

Use a caller mask for invalid image regions and `edge_margin_px` for unreliable
face borders. Never infer support from black pixels.

## What is lifted

For a face pixel \(u_f\), the gnomonic specification defines a unit ray
\(b_f\). PanorAi rotates that ray into the panorama frame and also computes the
corresponding ERP pixel. The descriptor remains unchanged; only its geometric
location is lifted. Invalid pixels and pixels outside support are retained in
the audit record but excluded from the accepted set.

The generic route is available when the detector is not wrapped by
`SphericalFeaturePipeline`:

```python
from panorai.data import FacePointBatch

points = views.map_points(
    lambda face: FacePointBatch(
        pixels_xy=detect_pixels(face.image),
        scores=score_pixels(face.image),
    ),
    source_id="my-detector/v1",
)
unique = points.angular_nms(max_angle_deg=0.25)
```

The callback may attach aligned class IDs and attributes. Angular NMS can be
class-aware; its report exposes kept indices, suppressed indices and groups so
deduplication is reproducible.

## Why ERP-pixel NMS is wrong

One horizontal pixel represents a fixed longitude increment, but its physical
angular neighbourhood contracts toward the poles. The seam also separates
adjacent directions into opposite image columns. Unit-vector angles avoid both
problems and make the threshold independent of ERP width.

Read the [features tutorial](../../tutorials/03_features_and_matching.md) for
SIFT/ORB/AKAZE configuration and the
[spherical-observation reference](https://rlsgarcia-code.github.io/PanorAi/reference/spherical_observations.html)
for the generic lifting contract. Next:
[detect features directly on the sphere](sphere_native_features.md).
