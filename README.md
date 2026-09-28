# **PanorAi: Spherical Image Processing & Projection**

**PanorAi** lets you work with **spherical (equirectangular) images** and efficiently transform them into **Gnomonic projections** and back to equirectangular format. The framework offers flexible **samplers** and **blenders** that optimize projection and reconstruction processes.

---

## Data Types

PanorAi organizes data into three main containers:

- **`EquirectangularImage`** – holds a full panorama and exposes methods such as `to_gnomonic` and `to_gnomonic_face_set`.
- **`GnomonicFace`** – represents a single rectilinear face with methods like `to_equirectangular`.
- **`GnomonicFaceSet`** – a collection of gnomonic faces that can be blended back into an equirectangular image.

Each container includes a convenient `show()` method that uses **PIL** to quickly preview the underlying image data.

`DataFactory` can create these objects from arrays, dictionaries or files, allowing the data type to drive the processing pipeline.


### Transformation Flow

The main data containers can transform into each other using the built‑in
projection helpers.
The diagram below illustrates the typical direction of
each conversion:

```
EquirectangularImage
    |     \-- to_gnomonic_face_set --> GnomonicFaceSet -- to_equirectangular -->
    |                                                ^
    |                                                |
    \-- to_gnomonic ----------> GnomonicFace -- to_equirectangular --/
```

Both **`GnomonicFace`** and **`GnomonicFaceSet`** can be retro‑projected back to
an equirectangular panorama.
This step often happens *after image processing* on
the faces has been performed.

### Attachable Components

Each container can **attach** three types of helpers that shape the projection
workflow:

- **Projector** – performs the geometric transformation between the
  equirectangular panorama and a rectilinear face. The same projector is used
  when creating the face and when mapping it back.
- **Sampler** – chooses the tangent points on the sphere from which faces are
  extracted. Built‑in samplers like `cube` or `fibonacci` provide different
  coverage strategies.
- **Blender** – combines multiple retro‑projected faces into a single panorama,
  controlling how overlaps are weighted.

This design lets you project faces, perform image‑level processing on them (for
instance with a neural network), and then retro‑project the results back onto
the panorama using the attached projector and blender.

---

## **🚀 Quick Start**

### **Installation**
```bash
pip install panorai
# Optional differentiable backend:
pip install "panorai[torch]"
```

### Optional depth adapters

PanorAi 3.1 does not distribute third-party depth-model implementations,
training code, datasets, or checkpoints. It keeps lightweight compatibility
loaders and registry keys without importing Torch during discovery:

```python
from panorai.depth import ModelRegistry, load_dav2_model

assert {"dav2", "m3dv2", "dust3r", "zoe"} <= set(
    ModelRegistry.list_models()
)
```

Install common adapter dependencies with `pip install "panorai[depth]"`, then
install the selected upstream project separately and review its license and
model-card terms. In particular, DUSt3R is CC BY-NC-SA 4.0 and is not bundled
in PanorAi's MIT artifacts. Missing upstream implementations raise an
actionable `DepthAdapterUnavailableError`; PanorAi does not silently replace a
model with numerically different code.

### Functional geometry API

Use pure functions when building pipelines from arrays:

```python
from panorai.geometry import GnomonicSpec, equirectangular_to_gnomonic

spec = GnomonicSpec(
    center_lat_deg=15,
    center_lon_deg=-30,
    hfov_deg=100,
    vfov_deg=60,
    roll_deg=5,
    output_shape_hw=(320, 640),
)
result = equirectangular_to_gnomonic(erp_array, spec)
view = result.data
valid_pixels = result.support_mask
```

### Reusable projector API

```python
from panorai.geometry import GnomonicProjector

projector = GnomonicProjector(spec, interpolation="bilinear")
view = projector.project(erp_array)
restored = projector.back_project(view.data, output_shape_hw=erp_array.shape[:2])
```

For labels and masks, select `nearest`; integer inputs intentionally reject
bilinear interpolation. Torch tensors use the same functions and projectors,
preserve `HW`, `CHW`, or `NCHW` layout, and support gradients with respect to
the input tensor.

### **1️⃣ Load an Equirectangular Image**
Convert an image to an **EquirectangularImage** object.
```python
from panorai.data import DataFactory

eq_image = DataFactory.from_file("path/to/image.png", data_type="equirectangular")
```

Other helpers load data from different sources:
```python
eq_image = DataFactory.from_array(ndarray, data_type="equirectangular")
eq_image = DataFactory.from_dict(my_dict, data_type="equirectangular")
eq_image = DataFactory.from_pil(pil_image, data_type="equirectangular")
face_set = DataFactory.from_list(list_of_faces)  # attaches default blender
```

Containers can also opt into canonical geometry without changing legacy code:

```python
face = eq_image.to_gnomonic(spec=spec)
```

The coordinate contract, cubemap order, and backend rules are documented in
[`docs/geometry-v1.md`](docs/geometry-v1.md). See
[`MIGRATING-3.0-TO-3.1.md`](MIGRATING-3.0-TO-3.1.md) for compatibility notes
and [`docs/release-3.1.0-checklist.md`](docs/release-3.1.0-checklist.md) for the
release gates.

---

## **📌 Core Functions**

### **2️⃣ Convert to Gnomonic Projection**
Extract a **rectilinear (Gnomonic) face** from the equirectangular image.
```python
face = eq_image.to_gnomonic(lat=45, lon=90, fov=60)
face.show()
```

### **3️⃣ Convert Back to Equirectangular**
Reproject a gnomonic face back to equirectangular.
```python
eq_reprojected = face.to_equirectangular(eq_shape=(512, 1024))
eq_reprojected.show()
```

### **4️⃣ Preprocess the Image**
You can apply the same preprocessing operations directly on the container.
```python
eq_image.preprocess(delta_lat=5.0, delta_lon=15.0, resize_factor=0.5)
```


---

## **🛠️ Advanced Usage**

### **5️⃣ Convert to Multiple Gnomonic Faces**
Use **sampling strategies** (e.g., `"cube"`, `"fibonacci"`) to extract multiple faces.
```python
face_set = eq_image.to_gnomonic_face_set(fov=60, sampling_method="cube")
face_set[0].show()  # View first face
```

### **6️⃣ Reconstruct Using a Blender**
Back-project multiple faces using different blending methods (`"closest"`, `"average"`).
```python
eq_reconstructed = face_set.to_equirectangular(eq_shape=(512, 1024), blend_method="closest")
eq_reconstructed.show()
```

### MultiChannelHandler

`MultiChannelHandler` helps when your data is stored in multiple channels
(for example an RGB image plus a depth or mask channel).
It can **stack** a
dictionary of arrays into a single `(H, W, C)` array, apply a projection to
all channels at once and then **unstack** the result back to the original
layout.

```python
from panorai.data.multi_handler import MultiChannelHandler
from panorai.projections.gnomonic_projection import GnomonicProjection
import numpy as np

data = {
    "rgb": rgb_array,      # shape (H, W, 3)
    "mask": mask_array     # shape (H, W, 1)
}

handler = MultiChannelHandler(data)
projector = GnomonicProjection(fov_deg=90)

# Project every channel together
handler.apply_projection(projector.project)
```

### Customizing With Attachables
Each data type can **attach** processing components at runtime:

```python
# Attach a sampler to control how multiple faces are sampled
eq_image.attach_sampler("fibonacci", n_points=8)

# Override the projection used by a gnomonic face
face.attach_projection("gnomonic", lat=30, lon=45, fov=75)

# Attach a blender to merge a set of faces
face_set.attach_blender("feathering")
```
## Preprocessing Without Containers

Alternatively, if you want to operate on raw NumPy arrays, the `Preprocessor.preprocess_eq` performs NumPy-based preprocessing on a panorama.
It
can extend the vertical field of view, rotate by latitude and longitude offsets
and optionally resize the image.
Parameters may be supplied directly or via a
`PreprocessorConfig` which stores defaults.

```python
from panorai.preprocessing.preprocessor import Preprocessor
from panorai.preprocessing.config import PreprocessorConfig

# define preprocessing defaults
cfg = PreprocessorConfig(
    shadow_angle=10.0,
    delta_lat=5.0,
    delta_lon=15.0,
    resize_factor=0.5,
)

processed = Preprocessor.preprocess_eq(
    eq_image.data,
    shadow_angle=cfg.shadow_angle,
    delta_lat=cfg.delta_lat,
    delta_lon=cfg.delta_lon,
    resize_factor=cfg.resize_factor,
    config=cfg,
)
```

The ``shadow_angle`` parameter represents the portion of the panorama a
3D scanner misses near the bottom of the sphere.
It is measured from
the South Pole upward and padding this region ensures that subsequent
projections cover any blind spots.

The returned array can be assigned back to the `EquirectangularImage`
for further steps.


## **🔧 Configuring Samplers & Blenders**
You can **fine-tune sampling & blending strategies** or modify the default projection configuration with `ConfigManager`.

### Set Custom Sampler
```python
from panorai.samplers.config import SamplerConfig

# Create a sampler configuration and attach it
custom_cfg = SamplerConfig(n_points=12, rotations=[(0, 45)])
eq_image.attach_sampler("fibonacci", config=custom_cfg)
```

### Override the Default Projection
```python
from panorai.config.config_manager import ConfigManager

# Update the global gnomonic config before attaching
cfg = ConfigManager.create("gnomonic_config", fov_deg=120, x_points=512, y_points=512)
eq_image.attach_projection("gnomonic", fov=cfg.fov_deg)
```

### Select Blender
```python
from panorai.blenders.registry import BlenderRegistry

blend = BlenderRegistry.create("gaussian", sig=1.2)
face_set.attach_blender("gaussian", sig=1.2)
```

### Component Attachment & Configuration Flow
Data containers such as `EquirectangularImage` and `GnomonicFace` expose
`attach_sampler`, `attach_projection`, and `attach_blender` helpers.
These
simply call **`PanoraiFactory`** which in turn pulls the requested object from
the appropriate registry.
The keyword arguments or configuration object you pass
are forwarded directly to the constructor:

```python
def attach_projection(self, name: str, lat: float = 0.0, lon: float = 0.0,
                      fov: float = 90.0, **kwargs):
    from panorai.factory.panorai_factory import PanoraiFactory
    self.projection = PanoraiFactory.get_projection(
        name, lat=lat, lon=lon, fov=fov, **kwargs
    )
```

`PanoraiFactory` performs minimal processing before delegating to the registry:

```python
@classmethod
def get_projection(cls, name: str, lat: float, lon: float, fov: float, **kwargs):
    available = ProjectionRegistry.available_projections()
    kwargs["phi1_deg"] = lat
    kwargs["lam0_deg"] = lon
    kwargs["fov_deg"] = fov
    if name not in available:
        raise ProjectionNotFoundError(name, available)
    return ProjectionRegistry.create(name, **kwargs)
```

Every sampler, blender or projection can be built from a **config object** or
direct keyword parameters.
When both are supplied the config takes precedence,
as seen in the sampler base class:

```python
class Sampler(ABC):
    def __init__(self, config: Optional[SamplerConfig] = None, **kwargs: Any):
        if config is not None:
            self.config = config
        else:
            self.config = SamplerConfig(**kwargs)
```

This design lets you quickly attach components with simple parameters or manage
shared settings via `ConfigManager`.
All attachments ultimately flow through the
factory, ensuring a consistent creation mechanism.

---
### Factory Helpers (Advanced)
Use `PanoraiFactory` to load files or arrays and directly access registered components.
```python
from panorai.factory.panorai_factory import PanoraiFactory
import numpy as np

# Load an equirectangular image
eq_img = PanoraiFactory.load_image("pano.jpg")

# Create a gnomonic face from a NumPy array
arr = np.zeros((256, 256, 3), dtype=np.uint8)
face = PanoraiFactory.create_data_from_array(arr, data_type="gnomonic_face",
                                             lat=0, lon=0, fov=90)

# Obtain a sampler or blender directly
sampler = PanoraiFactory.get_sampler("fibonacci", n_points=6)
blender = PanoraiFactory.get_blender("feathering")
```


## **📌 Summary**
| Feature                 | Function |
|-------------------------|----------|
| Load Image              | `DataFactory.from_file()` |
| Convert to Gnomonic     | `to_gnomonic(lat, lon, fov)` |
| Convert to Face Set     | `to_gnomonic_face_set(fov, sampling_method)` |
| Convert Back to EQ      | `to_equirectangular(eq_shape, blend_method)` |
| Use Samplers & Blenders | `ConfigManager`, `BlenderRegistry` |
---

## Samplers

Samplers define how tangent points are chosen when generating face sets.
The strategy affects coverage and the number of faces:

- **`cube`** – six orthogonal faces.
- **`icosahedron`** – vertices of an icosahedron; can be subdivided for density.
- **`fibonacci`** – nearly uniform distribution using the Fibonacci spiral.
- **`spiral`** – a simple spiral path around the sphere.
- **`blue_noise`** – random placement while keeping points apart.

```python
eq_image.attach_sampler("cube")             # basic 6 faces
eq_image.attach_sampler("fibonacci", n_points=20)
faces = eq_image.to_gnomonic_face_set(fov=60)
```

## Blenders

Blenders merge multiple faces back into a panorama.
They control how overlaps are resolved:

- **`average`** – uniform averaging of pixels.
- **`feathering`** – smooth, distance-based weighting.
- **`gaussian`** – Gaussian weights projected onto the sphere.
- **`closest`** – choose the closest face for every pixel.
- **`huber`** – robust averaging that reduces outlier impact.

```python
face_set.attach_blender("gaussian", sig=1.0)
result = face_set.to_equirectangular(eq_shape=(512, 1024))
```

## Point Cloud Export

`GnomonicFace` and `GnomonicFaceSet` objects can be transformed into a
`PCD` point cloud via their respective `to_pcd()` methods.
The conversion is
implemented in `PCDHandler`, which also provides convenience helpers such as
`create_axis_arrows()` for quick Open3D visualisation or gradient masking
functions used during conversion.

The PCD modules and all container `to_pcd()` names remain part of the 3.x
compatibility surface. Install Open3D separately with
`pip install "panorai[pcd]"` before using them.

## **📚 Next Steps**
- Experiment with **different samplers (`"cube"`, `"fibonacci"`)**.
- Try **blenders (`"closest"`, `"average"`)** for optimal reconstructions.
- Use **Torch tensors** for deep learning integration.

🔗 **[PanorAi Documentation](docs/_build/html/index.html)** (Link to full API reference)

---
## Running Tests
To run the tests execute:
```bash
pytest
```

The library uses a `paths.yaml` file to store paths to datasets and checkpoints.
By default this file is expected in the project root, but you can override the
location by setting the `PANORAI_PATHS` environment variable.

```python
from panorai.path_config import get_path

ckpt_path = get_path("metric3d", "ckpt_file")
```

## Building Documentation
To generate the HTML documentation run:
```bash
cd docs
make html
```
The output will be written to `docs/_build/html/index.html`.

## Pre-commit Hook for Documentation
To automatically check for documentation issues before each commit, install
[pre-commit](https://pre-commit.com/):
```bash
pip install pre-commit
pre-commit install
```
The hook runs `sphinx-build -n -W` to fail the commit if any warnings or broken
references are found in the RST files.
