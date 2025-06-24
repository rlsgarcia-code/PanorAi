# **PanorAi: Spherical Image Processing & Projection**

**PanorAi** is a framework for working with **spherical (equirectangular) images**, enabling efficient transformation into **Gnomonic projections** and back to equirectangular format. It provides flexible **samplers** and **blenders** to optimize projection and reconstruction processes.

---

## Data Types

The library revolves around three main data containers:

- **`EquirectangularImage`** – holds a full panorama and exposes methods such as `to_gnomonic` and `to_gnomonic_face_set`.
- **`GnomonicFace`** – represents a single rectilinear face with methods like `to_equirectangular`.
- **`GnomonicFaceSet`** – a collection of gnomonic faces that can be blended back into an equirectangular image.

Each container includes a convenient `show()` method that leverages **PIL** to quickly preview the underlying image data.

`DataFactory` can create these objects from arrays, dictionaries or files, allowing the data type to drive the processing pipeline.

### MultiChannelHandler

`MultiChannelHandler` helps when your data is stored in multiple channels
(for example an RGB image plus a depth or mask channel). It can **stack** a
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

### Transformation Flow

The main data containers can transform into each other using the built‑in
projection helpers. The diagram below illustrates the typical direction of
each conversion:

```
EquirectangularImage
    |     \-- to_gnomonic_face_set --> GnomonicFaceSet -- to_equirectangular -->
    |                                                ^
    |                                                |
    \-- to_gnomonic ----------> GnomonicFace -- to_equirectangular --/
```

Both **`GnomonicFace`** and **`GnomonicFaceSet`** can be retro‑projected back to
an equirectangular panorama. This step often happens *after image processing* on
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
```

### **1️⃣ Load an Equirectangular Image**
Convert an image to an **EquirectangularImage** object.
```python
from panorai.data import DataFactory

eq_image = DataFactory.from_file("path/to/image.png", data_type="equirectangular")
```

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

### Attach Methods
Each data type can **attach** processing components at runtime:

```python
# Attach a sampler to control how multiple faces are sampled
eq_image.attach_sampler("fibonacci", n_points=8)

# Override the projection used by a gnomonic face
face.attach_projection("gnomonic", lat=30, lon=45, fov=75)

# Attach a blender to merge a set of faces
face_set.attach_blender("feathering")
```

---

## **🛠️ Advanced Usage**

### **4️⃣ Convert to Multiple Gnomonic Faces**
Use **sampling strategies** (e.g., `"cube"`, `"fibonacci"`) to extract multiple faces.
```python
face_set = eq_image.to_gnomonic_face_set(fov=60, sampling_method="cube")
face_set[0].show()  # View first face
```

### **5️⃣ Reconstruct Using a Blender**
Back-project multiple faces using different blending methods (`"closest"`, `"average"`).
```python
eq_reconstructed = face_set.to_equirectangular(eq_shape=(512, 1024), blend_method="closest")
eq_reconstructed.show()
```

---

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
`attach_sampler`, `attach_projection`, and `attach_blender` helpers. These
simply call **`PanoraiFactory`** which in turn pulls the requested object from
the appropriate registry. The keyword arguments or configuration object you pass
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
direct keyword parameters. When both are supplied the config takes precedence,
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
shared settings via `ConfigManager`. All attachments ultimately flow through the
factory, ensuring a consistent creation mechanism.

---


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

Samplers define how tangent points are chosen when generating face sets. The strategy affects coverage and the number of faces:

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

Blenders merge multiple faces back into a panorama. They control how overlaps are resolved:

- **`average`** – uniform averaging of pixels.
- **`feathering`** – smooth, distance-based weighting.
- **`gaussian`** – Gaussian weights projected onto the sphere.
- **`closest`** – choose the closest face for every pixel.
- **`huber`** – robust averaging that reduces outlier impact.

```python
face_set.attach_blender("gaussian", sig=1.0)
result = face_set.to_equirectangular(eq_shape=(512, 1024))
```

## **📚 Next Steps**
- Experiment with **different samplers (`"cube"`, `"fibonacci"`)**.
- Try **blenders (`"closest"`, `"average"`)** for optimal reconstructions.
- Use **Torch tensors** for deep learning integration.

🔗 **[PanorAi Documentation](docs/_build/html/index.html)** (Link to full API reference)

---
## Extra Model Dependencies
Training certain models requires installing extra packages from their
respective **`requirements.txt`** files. Run the following commands for any
models you wish to train:

- **DepthAnythingV2**
  ```bash
  pip install -r panorai_models/DepthAnythingV2/requirements.txt
  ```
- **Metric3D**
  ```bash
  pip install -r panorai_models/Metric3D/requirements_v2.txt
  ```
- **Dust3r**
  ```bash
  pip install -r panorai_models/Dust3r/requirements.txt
  ```
- **ZoeDepth**
  ```bash
  pip install transformers
  ```

These models will be skipped if their dependencies are not installed.

---
## Running Tests
To run the tests execute:
```bash
pytest
```

## Building Documentation
To generate the HTML documentation run:
```bash
cd docs
make html
```
The output will be written to `docs/_build/html/index.html`.
