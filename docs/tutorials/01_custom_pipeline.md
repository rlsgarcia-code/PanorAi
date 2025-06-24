~~~md
# Custom Pipeline Example

This tutorial builds a small end‑to‑end pipeline similar to the one
described in the project README. We load a panorama, attach a sampler,
run a dummy processing step and blend the result back.

```python
from panorai.data import DataFactory
from panorai.preprocessing.preprocessor import Preprocessor
from panorai.config.config_manager import ConfigManager

# Load panorama and sample views
pano = DataFactory.from_file("my_pano.jpg", data_type="equirectangular")

# Optional preprocessing using NumPy helper
pano.data = Preprocessor.preprocess_eq(pano.data, delta_lat=5, resize_factor=0.5)

# Attach a fibonacci sampler and generate faces
pano.attach_sampler("fibonacci", n_points=8)
faces = pano.to_gnomonic_face_set(fov=60)

# Pretend to run a neural network on each face
for face in faces:
    face.data = 255 - face.data

# Recombine using a gaussian blender with config
cfg = ConfigManager.create("gnomonic_config", fov_deg=60)
faces.attach_blender("gaussian", sig=1.2)
result = faces.to_equirectangular(eq_shape=(512, 1024), blend_method="gaussian")
result.show()
```
~~~
