# Ergonomic workflow evolution

Status: public Stable `panorai-object-workflow/v1` contract. The stable
mathematical API remains `panorai.geometry`; no 3.0 public name is removed or
redefined.

## Purpose

The existing objects now express the common workflow without making users
assemble channels, projection plans, samplers and blenders first:

```python
import panorai as pa

pano = pa.EquirectangularImage(rgb)
result = pano.process_views(model, layout="cube")
```

This is a composition layer over `geometry-v1`, not a second projection
engine. Advanced functions, projectors, sampler objects, blender objects and
all legacy container methods remain available.

## Modalities

The initial array is the continuous `image` modality. Add radial range and
categorical data immutably:

```python
pano = (
    pa.EquirectangularImage(rgb)
    .with_depth(depth_m, valid=depth_is_valid, units="m")
    .with_labels(class_ids)
)

assert pano.image is not None
assert pano.depth is not None
assert pano.labels is not None
depth_is_valid_copy = pano.validity("depth")
```

All modalities in a bundle share backend, Torch device, batch, and spatial
shape, while dtype and channel count may differ. NumPy uses `HW`/`HWC`; Torch
uses `HW`/`CHW`/`NCHW`. Labels require integer or boolean dtype. Depth requires
floating point and always means radial range; `units=None` means the unit was
not declared.

Validity is explicit. Omitting `valid` means valid over the entire source
support. Non-finite values marked valid fail. Zero, black, a channel name, or a
dtype never implies invalidity.

Legacy dictionaries still work with the old methods. The ergonomic workflow
rejects them because their semantics cannot be inferred safely; start with an
image and use `with_depth()`/`with_labels()` instead.

## Views

```python
cube = pano.views()
fibonacci = pano.views("fibonacci", count=20)
icosphere = pano.views("icosahedron", subdivisions=1)
spiral = pano.views("spiral", count=20)
```

`cube` is the six-view default. The other presets are deterministic. An
integer `size` creates square views; `(height, width)` is explicit. When size
is absent, the view shape is `ceil(ERP height / 2)` by
`ceil(ERP width / 4)`, which is square for a 2:1 panorama. Scalar FOV applies
to both axes and `(horizontal, vertical)` creates a rectangular FOV.

Image and depth are projected independently with bilinear interpolation;
labels use nearest. Strict `depth_policy="propagate"` is the default.
`depth_policy="renormalize"` is opt-in and requires an explicit
`min_valid_weight` in `(0, 1]`, preserving the `geometry-v1` invalid-data
contract.

Every `GnomonicFace` records its immutable `GnomonicSpec`, geometric support,
validity, and modality policy. `lat`, `lon`, and `fov` remain compatibility
attributes. `GnomonicFaceSet` iteration is reentrant.

## Model mapping

```python
processed = cube.map(model)
depth_views = cube.map(model, output="depth", units="m")

processed_depth = pano.views().map(
    depth_model,
    input="image",
    output="depth",
    units="m",
)
```

With one modality, `input` may be omitted. A multimodal bundle requires it.
The model receives the raw NumPy array or Torch tensor and may return an array
or `(array, boolean_validity_mask)`. Backend, device, batch and spatial shape
must remain compatible; channel count may change.

An omitted `output` replaces the selected modality in the returned set. A
different output adds a modality. Existing output names require
`replace=True`. Array-only output is valid over geometric support, so a
non-finite value in that region fails. `map()` always creates a new face set.

## Reconstruction and one-call form

```python
result = processed.reconstruct()
image = processed.reconstruct(modalities="image", blend="gaussian")
all_modalities = processed.reconstruct(
    modalities="all",
    blend={"image": "average", "depth": "average", "labels": "closest"},
)

same_result = pano.process_views(model, layout="cube")
```

The original ERP shape is recorded by `views()` and inferred during
reconstruction. Manually constructed sets must provide `eq_shape`.
Image/depth default to masked average; labels default to closest-view
selection. A single selected modality accepts a blend string. Bundles use a
mapping so categorical data cannot accidentally receive a continuous blender.

`process_views()` is exactly `views().map().reconstruct()` and has no second
implementation. Tests require numerical identity between the two forms.

## Transparency

`views.describe()` returns a JSON-friendly dictionary with:

- `geometry-v1` mathematical contract, `panorai-object-workflow/v1` interface,
  and Stable support tier;
- layout, deterministic order, count, ERP/view shapes and every resolved spec;
- backend, device, dtype and layout for each modality;
- interpolation, depth policy, unit and resolved reconstruction blend.

`repr()` provides only a short interactive summary.

## Safe extension boundary

The simple surface accepts the four deterministic layout names and the three
typed modalities above. Advanced users may pass sampler and blender objects,
or a `GnomonicProjector` template whose spec and modality interpolation are
resolved for each view. The template's original `spec`, `interpolation`,
`invalid_policy`, and `min_valid_weight` are intentionally replaced by the
view and modality contract; other projector configuration, such as
`fill_value`, is preserved. Passing another projector type fails explicitly.
Blue-noise, HEALPix, arbitrary channel semantics and serialized projection
plans remain outside this first ergonomic contract.

Direct object composition must not load Torch for a NumPy workflow. Optional
components fail visibly rather than silently selecting another algorithm.

## Stability evidence

This surface is Stable for the 3.x line. ADOPT-001 exercised an installed
wheel with fourteen rectangular Fibonacci views, RGB, radial depth, labels,
explicit validity, NumPy HWC/HW, and Torch NCHW batch data. ADOPT-002 exercised
the same composition boundary through real OpenCV features, PanorAi pose
quality handling, and real PyCOLMAP export/read-back. Full 3.0 compatibility is
retained. PERF-009 separately records arbitrary-N runtime and peak-memory
evidence; those measurements do not turn a particular native backend into part
of the public contract.

PERF-012 adds optional first-party C++17 acceleration behind that same Stable
surface for compatible NumPy `float32`/`float64` face generation and Gaussian
reconstruction. There is no public backend switch: compatible installed wheels
select it automatically, while Torch, categorical nearest-neighbour data,
validity-normalized depth, custom projectors, and other blend modes retain the
existing paths. On the recorded Apple M3 Max source benchmark with 42 Fibonacci
faces, a 512×1024 ERP and 256×256 faces, generation improved 2.41×, reused
Gaussian reconstruction improved 17.17×, and warm-process end-to-end runs on
fresh face sets improved 3.60× while reducing peak memory by 18.6%. These are
reproducible reference measurements, not API guarantees or CI timing gates.

Stable means the documented methods, validation, provenance fields, modality
semantics, defaults, and failure behavior remain compatible through 3.x. It
does not stabilize arbitrary private attributes, custom model behavior, or
research algorithms consumed after `map()`.
