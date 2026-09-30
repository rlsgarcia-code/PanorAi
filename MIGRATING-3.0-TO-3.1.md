# Migrating from PanorAi 3.0.19 to 3.1.0

PanorAi 3.1 keeps the public 3.0 imports for containers, projections, samplers,
blenders, factories, depth, and point-cloud adapters. The new API is opt-in so
existing code does not silently change coordinate conventions.

## Use the canonical API in new code

```python
from panorai.geometry import GnomonicProjector, GnomonicSpec

spec = GnomonicSpec(
    center_lat_deg=10,
    center_lon_deg=-35,
    hfov_deg=100,
    vfov_deg=60,
    roll_deg=5,
    output_shape_hw=(320, 640),
)
result = GnomonicProjector(spec).project(erp)
view, support = result.data, result.support_mask
```

The canonical frame is `+X` right, `+Y` up, `+Z` forward. ERP coordinates are
pixel-center coordinates with a top-left origin, circular longitude in
`[-pi, pi)`, and latitude decreasing from north to south. Depth is radial
range, not implicit z-depth.

Canonical specs validate eagerly: longitude and roll are normalized to
`[-180, 180)`, all angles must be finite real numbers, FOV values remain in
`(0, 180)`, and dimensions must be positive integers. Values that older
internal helpers silently truncated, such as a floating image dimension, now
raise an explicit error in the new canonical API.

Projector results compose directly:

```python
view = projector.project(erp)
restored = projector.back_project(view, output_shape_hw=erp.shape[:2])
```

Extracting `view.data` first remains supported. The wheel is PEP 561 typed and
keeps Torch optional at import time.

## Container compatibility

Legacy calls remain valid:

```python
face = image.to_gnomonic(lat=10, lon=-35, fov=90, x_points=640, y_points=320)
```

The resolution arguments now produce exactly `(320, 640)`. The legacy inverse
gnomonic calculation also defines its zero-radius sample analytically as the
requested projection center. In 3.0.19 that sample was `NaN` before OpenCV
remapping and could become a different pixel or border fill on different
platforms. All other recorded legacy samples retain their 3.0 values.

To opt into the canonical convention through a container, pass only `spec`:

```python
face = image.to_gnomonic(spec=spec)
```

`spec` cannot be combined with `lat`, `lon`, `fov`, or legacy projection
keywords.

## Interpolation and optional dependencies

- Use `bilinear` for floating RGB, features, and continuous depth.
- Use `nearest` for integer labels and masks. Bilinear integer input is an
  explicit error.
- Bilinear defaults to `invalid_policy="propagate"`, preserving strict IEEE
  `NaN` propagation. The opt-in `renormalize` mode requires an explicit
  boolean validity mask and `min_valid_weight`; it reports output validity and
  contributing weight separately from geometric support.
- Core: `pip install panorai`
- Torch: `pip install "panorai[torch]"`
- Point clouds: `pip install "panorai[pcd]"`
- Lightweight depth-adapter dependencies: `pip install "panorai[depth]"`

Neither Torch nor Open3D is imported by `import panorai`.

For sparse floating depth, for example:

```python
projector = GnomonicProjector(
    spec,
    invalid_policy="renormalize",
    min_valid_weight=0.5,
)
view = projector.project(radial_range_m, validity_mask=depth_is_valid)
usable = view.support_mask & view.validity_mask
```

Choose the threshold from the measurement contract; PanorAi does not infer
validity from zero or choose a universal threshold.

## Depth distribution boundary

PanorAi 3.0 artifacts exposed deep copies of Depth Anything V2, DUSt3R/CroCo,
Metric3D, legacy ZoeDepth, and research training/data helpers. Those trees are
experimental/internal rather than stable PanorAi API and are excluded from the
3.1 wheel and sdist. This prevents an MIT-only PanorAi artifact from silently
redistributing code with additional terms, including DUSt3R/CroCo's
CC BY-NC-SA 4.0 license.

The supported 3.x compatibility surface remains:

- `panorai.depth.ModelRegistry`;
- `load_dav2_model`, `load_m3dv2_model`, `load_dust3r_model`, and
  `load_zoe_model`;
- registry keys `dav2`, `m3dv2`, `dust3r`, and `zoe`;
- `panorai.pcd` modules and container `to_pcd` methods.

Loaders now import separately installed upstream projects only when invoked.
If an implementation is absent, they raise `DepthAdapterUnavailableError`
with the upstream location and license instead of becoming `None` at import
time. Installing `panorai[depth]` supplies common adapter dependencies; it does
not install or license the upstream source or model weights. Code importing a
deep path such as `panorai.depth.Dust3r.*`,
`panorai.depth.DepthAnythingV2.*`, or `panorai.depth.Metric3D.*` must install
and import the corresponding upstream project directly.

## Root imports are lazy

The five public 3.0 root names remain available and unchanged:

```python
from panorai import (
    ConfigManager,
    EquirectangularImage,
    GnomonicFace,
    GnomonicFaceSet,
    PanoraiFactory,
)
```

PanorAi 3.1 resolves these compatibility objects only when code accesses them.
Consequently, `import panorai.geometry` no longer imports containers,
factories, registries, OpenCV, Pillow, Pydantic, SciPy, scikit-image, or YAML.
This changes import timing, not the identity or import path of any recorded
public 3.0 root object. Import subpackages explicitly rather than relying on an
undocumented module attribute created as a side effect of `import panorai`.

## Blender masks and output shapes

Blender masks are now authoritative and independent of values. A valid black
RGB pixel or zero-valued scalar participates when its mask is true; an
unsupported non-zero value does not. Valid non-finite floating samples raise an
error instead of silently becoming zero.

The supported fusion blenders (`average`, `closest`, `feathering`, `gaussian`,
and `huber`) preserve the input image shape. `HuberBlender` therefore returns
`(H, W)` for scalar inputs and `(H, W, C)` for channel inputs instead of always
forcing three channels. Pass `return_mask=True` to any built-in blender to get
`(data, support_mask)`. `GnomonicFaceSet.to_equirectangular` now obtains real
back-projection masks and stores the fused mask on the returned
`EquirectangularImage.support_mask`.

The overlap count/standard-deviation blenders are diagnostic. The
`huber_no_confidence`, `huber_spatial`, and `bundle_adjustment` strategies are
experimental and are not part of the stable projection path.
