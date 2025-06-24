.. ───────────────────────────────────────────────────────────────
.. 🗂  docs/how_to/image_processing.rst  ── Samplers · Blenders · Preprocessing
.. ───────────────────────────────────────────────────────────────

Image-processing recipes
========================

Convert many faces
------------------

```python
faces = eq.to_gnomonic_face_set(fov=75, sampling_method="fibonacci", n_points=20)
```

Attach a custom blender
-----------------------

```python
faces.attach_blender("closest")           # pixel from nearest centre
pano = faces.to_equirectangular(eq_shape=(512,1024))
```

Multi-channel trick
-------------------

```python
from panorai.data.multi_handler import MultiChannelHandler
handler = MultiChannelHandler({"rgb": rgb, "mask": mask})
handler.apply_projection(projector.project)
```

Preprocess without containers
-----------------------------

```python
from panorai.preprocessing.preprocessor import Preprocessor
arr = Preprocessor.preprocess_eq(eq.data, delta_lat=5, resize_factor=0.5)
```

---


Samplers & blenders at a glance
-------------------------------

.. include:: ../reference/samplers_blenders.rst
   :start-after: <!-- cut:start -->
   :end-before:  <!-- cut:end -->