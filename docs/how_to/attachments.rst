.. _howto-attachments:

Customising attachments
=======================

Data containers can attach samplers, projections and blenders at
runtime. This lets you try different projection strategies without
reloading your image.

Attach a sampler
----------------

```python
from panorai.data import DataFactory

pano = DataFactory.from_file("pano.jpg", data_type="equirectangular")
# Fibonacci distribution of 8 faces
pano.attach_sampler("fibonacci", n_points=8)
faces = pano.to_gnomonic_face_set(fov=60)
```

Attach a projection to a face
-----------------------------

```python
face = pano.to_gnomonic(lat=0, lon=45, fov=90)
face.attach_projection("gnomonic", lat=30, lon=60, fov=75)
```

Attach a blender with custom config
-----------------------------------

```python
from panorai.blenders.registry import BlenderRegistry

faces.attach_blender("gaussian", sig=1.2)
result = faces.to_equirectangular(eq_shape=(512, 1024))
```

Using configuration objects
---------------------------

```python
from panorai.config.config_manager import ConfigManager

cfg = ConfigManager.create("gnomonic_config", fov_deg=120,
                           x_points=512, y_points=512)
# Values from cfg are forwarded to the attachment helper
pano.attach_projection("gnomonic", fov=cfg.fov_deg)
```

The same pattern is used throughout the project and mirrors the
examples shown in the README and :doc:`../tutorials/01_custom_pipeline`.
