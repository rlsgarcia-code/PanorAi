"""Quick start demo used in the documentation."""

from panorai.data import DataFactory

# Load a panorama
pano = DataFactory.from_file("pano.jpg", data_type="equirectangular")

# Extract a cube of views
faces = pano.to_gnomonic_face_set(fov=90, sampling_method="cube")

# Dummy processing step
for f in faces:
    f.data = 255 - f.data

# Blend back to equirectangular
result = faces.to_equirectangular(eq_shape=(512, 1024))
result.show()
