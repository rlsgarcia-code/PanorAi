import importlib

import pytest

np = pytest.importorskip("numpy")
try:
    # Access an attribute that only exists on a fully installed NumPy
    _ = np.exceptions
except Exception:
    pytest.skip("NumPy is not fully installed", allow_module_level=True)
cv2 = pytest.importorskip("cv2")
pytest.importorskip("skimage")
PIL_Image = pytest.importorskip("PIL.Image")

# Import packages only after the optional runtime dependencies were verified.
importlib.import_module("panorai.samplers")
importlib.import_module("panorai.blenders")
importlib.import_module("panorai.projections")
PanoraiFactory = importlib.import_module(
    "panorai.factory.panorai_factory"
).PanoraiFactory
Preprocessor = importlib.import_module(
    "panorai.preprocessing.preprocessor"
).Preprocessor


def test_visual_pipeline_output(tmp_path):
    height, width = 32, 64
    x_grad = np.linspace(0, 255, width, dtype=np.uint8)
    y_grad = np.linspace(0, 255, height, dtype=np.uint8)
    data = np.zeros((height, width, 3), dtype=np.uint8)
    data[..., 0] = x_grad
    data[..., 1] = y_grad[:, None]
    for i in range(min(height, width)):
        data[i, i] = [255, 255, 255]
        data[i, width - i - 1] = [255, 255, 255]

    eq = PanoraiFactory.create_data_from_array(data, "equirectangular")

    processed = Preprocessor.preprocess_eq(
        eq.data,
        shadow_angle=5.0,
        delta_lat=10.0,
        delta_lon=20.0,
        resize_factor=1.0,
    )
    eq.data = processed

    faceset = eq.to_gnomonic_face_set(fov=90)
    assert len(faceset) == 6

    eq_back = faceset.to_equirectangular(eq.shape)
    assert eq_back.shape == eq.shape

    orig_path = tmp_path / "original.png"
    recon_path = tmp_path / "reconstructed.png"
    PIL_Image.fromarray(np.array(eq)).save(orig_path)
    PIL_Image.fromarray(np.array(eq_back)).save(recon_path)

    for idx, face in enumerate(faceset):
        face_path = tmp_path / f"face_{idx}.png"
        PIL_Image.fromarray(np.array(face)).save(face_path)
        assert face_path.exists()

    assert orig_path.exists()
    assert recon_path.exists()
