# Work with a spherical image object

Use `EquirectangularImage` at the boundary of an application instead of
passing anonymous arrays through every step. It owns the ERP raster and keeps
support, validity, modality and geometry available to later operations.

## Open or create an ERP

For a real panorama, the usual entry point is one line:

```python
from panorai import EquirectangularImage

panorama = EquirectangularImage.open("room_360.jpg")
print(panorama.size, panorama.mode, panorama.backend)
```

`open()` decodes eagerly; the returned object does not depend on the source
file remaining open. Use `new()` for a generated constant ERP and
`fromarray()` at NumPy integration boundaries. For executable examples, use a
deterministic synthetic ERP without downloading an asset:

```python
sample = EquirectangularImage.sample("spherical-grid", size=(512, 256))
```

The available `spherical-grid`, `directions` and `checkerboard` patterns use
the canonical pixel-centre lattice. They are documentation fixtures, not a
dataset or evidence of scientific accuracy.

The complete self-contained flow below is executed in CI:

```{literalinclude} ../../../scripts/run_documentation_examples.py
:language: python
:start-after: DOCS_EQUIRECTANGULAR_IMAGE_START = None
:end-before: DOCS_EQUIRECTANGULAR_IMAGE_END = None
:dedent: 4
```

## Resize and thumbnail the sphere

`resize((width, height))` samples the pixel-centre bearings of a new canonical
ERP. Longitude wraps, support and validity are transformed separately, and
attached labels remain nearest-only. It is therefore not equivalent to calling
a generic planar resize on the storage rectangle.

`thumbnail(max_size)` performs the same spherical resampling but preserves the
aspect ratio, never enlarges the input and returns a new object. The original
Pillow method is intentionally not copied as an in-place mutation. If the
requested bounds would collapse either spherical dimension below two pixels,
the method fails instead of distorting the aspect ratio.

## Rotate an orientation, not a canvas

```python
turned = panorama.rotate(yaw_deg=30, pitch_deg=-5)
```

Yaw is positive from canonical `+Z` toward `+X`; pitch is positive toward
`+Y`; roll is positive clockwise in the image plane. Advanced callers may pass
a proper canonical 3×3 rotation matrix. A positional scalar such as
`rotate(45)` fails because it does not identify a 3D axis.

Use named reflections when reflection is the actual intent:

```python
mirrored = panorama.transpose(panorama.Transpose.FLIP_LONGITUDE)
antipodal = panorama.transpose(panorama.Transpose.ANTIPODAL)
```

No transpose operation silently swaps ERP height and width.

## Crop a tangent view

```python
from panorai.geometry import GnomonicSpec

face = panorama.crop(
    GnomonicSpec(
        center_lat_deg=10,
        center_lon_deg=25,
        hfov_deg=90,
        vfov_deg=70,
        output_shape_hw=(256, 320),
    )
)
```

The result is a `GnomonicFace`, not a smaller `EquirectangularImage`: it is a
pinhole chart over finite angular support and carries the exact projection
needed to lift or reproject results.

## Choose direct-sphere or projected-view processing

Apply a registered direct operation through the object:

```python
from panorai.ops.spherical import GaussianBlur

smooth = panorama.filter(GaussianBlur(ksize=5, sigma=1.2))
```

For a planar model, use `views().map().reconstruct()` instead. The two routes
share the same container and geometry contract; `filter()` does not disguise a
planar function as a spherical operation.

## Composite by reprojection

`paste(face)` and `composite(face)` require a typed `GnomonicFace`. They use
its `GnomonicSpec` to back-project the patch rather than copying pixels into an
ERP box. `blend="replace"` makes face support and validity authoritative.
`blend="alpha"` is available for continuous fields and requires an explicit
scalar alpha; categorical alpha blending fails.

## Save a viewable raster

```python
smooth.save("smoothed.png", value_range=(0, 255))
```

Image formats do not store PanorAi support, validity, grids, units or attached
depth/labels. `save()` therefore encodes one selected field and does not claim
to persist the full workflow. Integer images save at their native range. A
floating image outside `[0, 1]` requires an explicit `value_range`, preventing
scientific values from being silently normalized or clipped.

Read {doc}`../../concepts/equirectangular-image` for the design rationale and
{doc}`../../reference/equirectangular_image` for exact signatures.
