# Port Metric3D directly to the sphere

**Level:** Extend · **Strategy:** Sphere-native pretrained CNN ·
**Stability:** Experimental

```python
from panorai.experimental.deep_learning import (
    load_spherical_metric3d_convnext_tiny_v1,
)

loaded = load_spherical_metric3d_convnext_tiny_v1(
    accept_upstream_terms=True,
    cache_dir=model_cache,
    device="cpu",
)
radial_range_m = loaded.model(erp_nchw)
```

The adapter keeps the pretrained parameter objects and replaces supported
`Conv2d`, `ConvTranspose2d`, pooling and reflection-padding neighbourhoods with
spherical equivalents. Pointwise activation and normalization remain ordinary
Torch operations. Autograd is preserved, so the same layers can participate in
training or fine-tuning outside the library core.

Porting proves mechanical layer coverage and parameter identity; it does not
prove that perspective-trained features remain accurate on panoramas. Inspect
`loaded.port_report`, input normalization, output units and validity.

## What is preserved and what changes

Parameter tensors, module ordering, pointwise nonlinearities and normalization
remain those of the acquired model. The sampling lattice of each spatial
operator changes from planar offsets to sphere-aware neighbourhoods. That is
the desired domain transfer, but it means equality with the planar network is
not expected away from controlled local cases.

The loader pins upstream identity and records acquisition metadata; no
checkpoint is distributed in the PanorAi wheel or sdist. The caller must
explicitly accept upstream terms and choose a cache directory.

## Acceptance checks

1. `remaining_planar_spatial_layers` is empty for the supported profile.
2. Input layout, channel order, normalization and ERP geometry match the
   adapter contract.
3. The output is identified as radial range in metres before 3-D conversion.
4. Seam, poles and unsupported directions are inspected independently.
5. A controlled planar or tangent-view comparison is reported when discussing
   pretrained-semantic retention.

See the [monocular-depth tutorial](../../tutorials/10_spherical_monocular_depth.md)
for acquisition and inference and
[composing spherical models](https://rlsgarcia-code.github.io/PanorAi/tutorials/12_composing_spherical_models.html)
for the core porting mechanism. Next: [validate the range](validate_range.md).
