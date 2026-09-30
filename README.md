# PanorAi

**Spherical image processing and projection with explicit geometry, masks, and
NumPy/Torch parity.**

PanorAi converts equirectangular panoramas to gnomonic views or cubemaps and
back again. It supports both array-first pipelines and the object workflow
introduced in PanorAi 3.0:

```text
Panorama → sample views → process faces → reconstruct panorama
```

The stable 3.x mathematical API lives in `panorai.geometry`. Its conventions
are explicit and executable: pixel-center coordinates, top-left image origin,
`+X` right, `+Y` up, `+Z` forward, radial depth, horizontal seam wrapping, and
geometric support kept separate from data validity.

## Why PanorAi

- One geometry contract for NumPy and optional Torch tensors.
- Functional calls and reusable immutable projectors backed by the same engine.
- Explicit support, validity, and valid-weight outputs—zero is never treated as
  missing data implicitly.
- Correct modality rules: continuous data can use bilinear interpolation;
  masks and labels use nearest sampling.
- Familiar panorama, face, sampler, and blender abstractions retained for 3.x
  workflows.
- Optional Torch and Open3D backends do not load with the core geometry API.

## Installation

```bash
pip install panorai
```

Install only the optional backend you need:

```bash
pip install "panorai[torch]"  # differentiable Torch geometry
pip install "panorai[pcd]"    # Open3D compatibility surface
pip install "panorai[depth]"  # lightweight depth-adapter dependencies
```

NumPy is required. PanorAi supports Python 3.10–3.12.

## Quick start: canonical geometry

This self-contained example uses `HWC` floating RGB data and bilinear
interpolation. Angles are degrees and shapes are `(height, width)`.

```python
import numpy as np

from panorai.geometry import GnomonicProjector, GnomonicSpec

height, width = 16, 32
rgb = np.linspace(0.0, 1.0, height * width * 3, dtype=np.float32)
rgb = rgb.reshape(height, width, 3)

spec = GnomonicSpec(
    center_lat_deg=15.0,
    center_lon_deg=30.0,
    hfov_deg=90.0,
    vfov_deg=60.0,
    output_shape_hw=(8, 12),
)
projector = GnomonicProjector(spec, interpolation="bilinear")

view = projector.project(rgb)
restored = projector.back_project(view, output_shape_hw=(height, width))

assert view.data.shape == (8, 12, 3)
assert view.support_mask.shape == (8, 12)
assert restored.data.shape == rgb.shape
```

`ProjectionResult` travels naturally from projection to back-projection and
keeps masks beside the data they describe.

## Workflow API: panorama to faces and back

The 3.0 containers remain available as a compatibility workflow. Samplers
choose tangent points, projectors create views, and blenders reconstruct their
overlaps:

```python
import numpy as np

from panorai.data import EquirectangularImage

height, width = 16, 32
rgb = np.linspace(0.0, 1.0, height * width * 3, dtype=np.float32)
rgb = rgb.reshape(height, width, 3)

panorama = EquirectangularImage(rgb)
panorama.attach_sampler("cube")
faces = panorama.to_gnomonic_face_set(fov=90.0)

for face in faces:
    face.data = np.clip(face.data, 0.0, 1.0)  # your model or transform

reconstructed = faces.to_equirectangular(
    (height, width), blend_method="average"
)

assert len(faces) == 6
assert reconstructed.data.shape == rgb.shape
assert reconstructed.support_mask.all()
```

The workflow abstractions are useful and supported throughout 3.x, but new
geometry semantics are defined by `panorai.geometry` rather than mutable
container configuration.

## Choose the right surface

| Need | Recommended surface | Stability |
| --- | --- | --- |
| Array or tensor projection | `panorai.geometry` functions | Stable 3.x |
| Repeated projection settings | `GnomonicProjector`, `CubemapProjector` | Stable 3.x |
| Panorama → faces → processing → reconstruction | 3.0 data containers | Compatibility 3.x |
| Custom view placement | Sampler registry and sampler objects | Compatibility 3.x |
| Mask-aware overlap fusion | Supported blenders | Stable where documented |
| Point-cloud export | `panorai.pcd` | Optional compatibility |
| Third-party depth models | `panorai.depth` adapters | Optional compatibility |

No stable public 3.0 name is removed before 4.0. Compatibility does not mean
that legacy objects define the canonical coordinate or validity contract.

## Data modalities

Interpolation is part of the data contract:

| Data | Typical dtype | Interpolation | Validity |
| --- | --- | --- | --- |
| RGB/features | floating point | `bilinear` | explicit mask when needed |
| Labels/classes | integer | `nearest` | explicit mask |
| Boolean masks | boolean | `nearest` | the boolean values are data |
| Radial depth/range | floating point | `bilinear` or `nearest` | explicit validity mask |

Do not stack RGB, labels, masks, or depth and project them under one implicit
interpolation policy. Project each modality with its appropriate policy. For
invalid continuous samples, strict IEEE propagation is the default; explicitly
select `invalid_policy="renormalize"`, provide a validity mask, and set
`min_valid_weight` to opt into normalized valid-neighbor interpolation.

## Samplers, blenders, and extension points

Built-in samplers include `cube`, `icosahedron`, `fibonacci`, `spiral`, and
`blue_noise`. Supported fusion blenders include `average`, `gaussian`,
`feathering`, `closest`, and `huber`; diagnostic blenders expose overlap count
or variation. Experimental blenders are labeled separately in the API
stability reference.

Registries let existing 3.x applications attach named samplers, projections,
and blenders. A future typed workflow layer is being designed to preserve this
composition while making modality policies, configuration, plugins, batching,
and provenance explicit. It is a proposal, not a current API commitment; see
[Workflow evolution](docs/explanation/workflow-evolution.md).

## Compatibility and optional integrations

- See [Migration from 3.0 to 3.1](MIGRATING-3.0-TO-3.1.md) for intentional
  behavior corrections and retained names.
- Depth model implementations, checkpoints, datasets, and training pipelines
  are not bundled. Adapters require separately installed upstream projects and
  their licenses still apply.
- DUSt3R is not distributed inside PanorAi's MIT wheel or sdist because its
  upstream terms include CC BY-NC-SA 4.0.
- Open3D is required only for the optional PCD compatibility surface.

## Documentation

- [Geometry v1 contract](docs/geometry-v1.md)
- [Executable tutorials](docs/tutorials/index.rst)
- [Data modality guide](docs/how_to/data_modalities.rst)
- [API stability tiers](docs/reference/stability.rst)
- [Architecture](docs/explanation/architecture.rst)
- [Changelog](CHANGELOG.md)

The examples in the public documentation are executed in CI against both the
source checkout and an installed wheel. Development, test, build, release, and
audit procedures live in the documentation rather than the product overview.

## License

PanorAi's distributed source is MIT licensed. Optional upstream projects and
models may have different terms; installing an adapter does not relicense them.
See [LICENSE](LICENSE) and the integration-specific documentation before use.
