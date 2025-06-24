import pytest
np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("skimage")

import panorai.samplers
import panorai.blenders
import panorai.projections

from panorai.factory.panorai_factory import PanoraiFactory


def test_to_gnomonic_preserves_source():
    data = np.random.randint(0, 256, (4, 8, 3), dtype=np.uint8)
    eq = PanoraiFactory.create_data_from_array(data, "equirectangular")
    before = eq.data.copy()

    eq.to_gnomonic(0.0, 0.0, 90.0)

    assert np.array_equal(eq.data, before)


def test_face_to_equirectangular_preserves_source():
    data = np.random.randint(0, 256, (4, 8, 3), dtype=np.uint8)
    eq = PanoraiFactory.create_data_from_array(data, "equirectangular")
    face = eq.to_gnomonic(0.0, 0.0, 90.0)
    before = face.data.copy()

    face.to_equirectangular(eq.shape[:2])

    assert np.array_equal(face.data, before)
