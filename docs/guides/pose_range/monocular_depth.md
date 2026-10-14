# Choose a monocular range strategy

**Level:** Start · **Input:** one ERP · **Output:** model-dependent radial range

PanorAi currently exposes two complementary routes:

| Route | Execution | Appropriate when |
| --- | --- | --- |
| Metric3D ConvNeXt | port learned spatial layers to the sphere | the model is CNN-based and direct ERP inference is required |
| Depth Anything 3 Metric Large | run the unchanged model on tangent views and fuse | the architecture is not covered by spherical layer porting |

Both are Experimental and use external checkpoints with explicit acquisition,
identity and terms. Neither route makes an uncalibrated model prediction into
ground truth.

## Decision rule

Choose the direct route only when every spatial operator used by the model has
a supported spherical equivalent and the port report is clean. It avoids face
fusion and preserves one ERP lattice, but changing neighbourhood geometry can
change pretrained behaviour.

Choose tangent views when the model architecture cannot be ported faithfully
or when an unchanged upstream inference path is the required control. This
preserves the planar model but introduces view selection, virtual intrinsics,
boundary support and fusion as explicit experimental variables.

## Common contract

Both routes must state whether the upstream prediction is metric or relative,
whether it is pinhole axial depth or radial range, its units, support,
validity, checkpoint identity and any resizing. The useful PanorAi product is
not a coloured depth image; it is radial range plus those contracts and
provenance.

Continue with [sphere-native Metric3D](metric3d_sphere_native.md) or
[Depth Anything over tangent views](depth_anything_views.md). Validate either
output with the [range validation guide](validate_range.md).
