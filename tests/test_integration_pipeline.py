import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("skimage")

# Import packages to ensure registries are populated
import panorai.samplers  # registers default samplers
import panorai.blenders  # registers default blenders
import panorai.projections  # registers gnomonic projection

from panorai.factory.panorai_factory import PanoraiFactory
from panorai.preprocessing.preprocessor import Preprocessor


def test_end_to_end_pipeline():
    # Create a simple synthetic equirectangular image
    data = np.zeros((8, 16, 3), dtype=np.uint8)
    data[..., 0] = 255  # red channel to visualize rotation

    # Create EquirectangularImage object through factory
    eq = PanoraiFactory.create_data_from_array(data, "equirectangular")

    # Apply preprocessing steps
    processed = Preprocessor.preprocess_eq(
        eq.data,
        shadow_angle=10.0,
        delta_lat=5.0,
        delta_lon=10.0,
        resize_factor=0.5,
    )
    eq.data = processed

    # Sample gnomonic faces using the default sampler
    faceset = eq.to_gnomonic_face_set(fov=90)
    assert len(faceset) == 6

    # Blend the faces back into an equirectangular image
    eq_back = faceset.to_equirectangular(eq.shape)
    assert eq_back.shape == eq.shape
