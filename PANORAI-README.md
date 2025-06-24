# PanorAi

PanorAi is a Python toolbox for spherical images. It simplifies common
operations such as converting an equirectangular panorama into gnomonic
faces and blending them back together. The library ships with registry
systems for samplers, projections and blenders so both beginners and
power users can tailor the processing pipeline.

## Quick start

```bash
pip install panorai[depth]
```

```python
from panorai.data import DataFactory

# Load a panorama as an `EquirectangularImage`
eq = DataFactory.from_file("panorama.jpg", data_type="equirectangular")

# Convert it to a single gnomonic face
face = eq.to_gnomonic(lat=0, lon=0, fov=90)

# Convert back to equirectangular
reproj = face.to_equirectangular(eq_shape=(512, 1024))
```

## Interfaces for everyday use

### `panorai.data`
- **EquirectangularImage** – container for a full panorama. Can attach a
  sampler or a projector to itself.
- **GnomonicFace** – one rectilinear view extracted from the panorama.
  Supports back-projection and custom projection parameters.
- **GnomonicFaceSet** – collection of faces that can be blended back into
  an equirectangular image.
- **DataFactory** – convenience class that creates the objects above from
  NumPy arrays, dictionaries, lists or image files.

### `panorai.factory.PanoraiFactory`
Provides helper methods to get preconfigured samplers, blenders and
projections. It also exposes utilities for loading images or building
objects from arrays.

```python
from panorai.factory.panorai_factory import PanoraiFactory

# Retrieve a sampler and a blender
sampler = PanoraiFactory.get_sampler("fibonacci", n_points=10)
blender = PanoraiFactory.get_blender("average")

# List available projection types
print(PanoraiFactory.get_projection("gnomonic", lat=0, lon=0, fov=90))
```

### `panorai.utils.PanoraiRegistry`
A single entry point to inspect which samplers, blenders and projections
are registered. Useful to discover available components.
Calling `PanoraiRegistry.available_samplers()`, `available_blenders()`, and `available_projections()` return the currently registered names (e.g. ['cube', 'fibonacci']).

```python
from panorai.utils.registry import PanoraiRegistry
print(PanoraiRegistry.available_samplers())
print(PanoraiRegistry.available_blenders())
print(PanoraiRegistry.available_projections())
```

## Customising projectors, samplers and blenders

Each data object can change its behaviour by attaching a different
component. Advanced users can swap these at any time:

```python
# Change sampler on an existing image
eq.attach_sampler("spiral", n_points=30)
faces = eq.to_gnomonic_face_set(fov=60)

# Change the projection used by a face
face.attach_projection("gnomonic", lat=30, lon=45, fov=60)

# Blend faces using another strategy
faces.attach_blender("gaussian", fov_deg=60)
result = faces.to_equirectangular(eq_shape=(512, 1024))
```

Configurations for these components are handled via
`panorai.config.ConfigManager` which lets you inspect or modify default
settings.

## Extra model dependencies
Some optional models under `panorai.depth` require additional packages.
Install them if you plan to train those models.

## Tests
Run `pytest` to execute the test suite.
