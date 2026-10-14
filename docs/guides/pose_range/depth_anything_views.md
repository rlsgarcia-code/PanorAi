# Run Depth Anything over tangent views

**Level:** Extend · **Strategy:** Projection-domain pretrained model ·
**Stability:** Experimental

Depth Anything 3 Metric Large is not converted to a sphere-native network in
the current adapter. Its attention, patch embedding and positional treatment
remain unchanged.

```python
from panorai.experimental.deep_learning import load_da3metric_large

views = panorama.views(
    "icosahedron", subdivisions=1, size=(812, 1400), fov=(100.0, 80.0)
)
loaded = load_da3metric_large(
    accept_upstream_terms=True,
    cache_dir=model_cache,
    device="cpu",
)
adapter = loaded.tangent(views[0].spec)
predicted = views.map(adapter, input="image", output="depth", units="m")
radial = predicted.reconstruct(blend={"depth": "gaussian"}, modalities=("depth",))
```

PanorAi owns view geometry, virtual intrinsics, axial-to-radial conversion,
support, validity, fusion and provenance. The upstream model owns the local
prediction. Fusion quality at view boundaries remains part of the evaluation,
not a solved property implied by model accuracy.

## Why conversion is required

A pinhole model may return axial depth $z$ along its optical axis. For a face
ray $d=(d_x,d_y,d_z)$ in face coordinates, the corresponding radial range is

$$\rho = z / d_z,$$

after applying the exact ray normalization and sign convention declared by
the adapter. Treating $z$ as $\rho$ stretches geometry away from the face
centre. The conversion is performed before spherical fusion.

## What must be fused

Each face contributes radial range only where source support, model validity
and finite conversion all agree. Overlap weights must not revive invalid
pixels. Record the view specification and virtual intrinsics with each
prediction so a seam artifact can be traced back to its source face.

## Evaluation boundary

The route is an unchanged planar-model control, not a claim that the model is
spherical. Evaluate view-centre versus boundary error, overlap disagreement,
seam continuity, polar coverage, runtime and peak memory. Compare those
results with the direct spherical CNN under the same ERP and range reference.

See the [monocular-depth tutorial](../tutorials/10_spherical_monocular_depth.md)
for the pinned artifact, terms and complete conversion contract. Next:
[validate the range](validate_range.md).
