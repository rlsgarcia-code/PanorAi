~~~md
# Custom Pipeline Example

This tutorial walks through a slightly more advanced workflow where a
set of gnomonic faces is processed and blended back into a panorama.

```python
from panorai.data import DataFactory

# Load panorama and sample views
pano = DataFactory.from_file("my_pano.jpg", data_type="equirectangular")
faces = pano.to_gnomonic_face_set(fov=60, sampling_method="fibonacci", n_points=8)

# Pretend to run a neural network on each face
for face in faces:
    face.data = 255 - face.data

# Recombine using a gaussian blender
result = faces.to_equirectangular(eq_shape=(512, 1024), blend_method="gaussian")
result.show()
```
~~~
