import importlib

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("skimage")

# Import packages only after the optional runtime dependencies were verified.
importlib.import_module("panorai.samplers")
importlib.import_module("panorai.blenders")
importlib.import_module("panorai.projections")
DataFactory = importlib.import_module("panorai.data.factory").DataFactory
PanoraiFactory = importlib.import_module(
    "panorai.factory.panorai_factory"
).PanoraiFactory
SamplerRegistry = importlib.import_module("panorai.samplers.registry").SamplerRegistry


def test_multi_channel_round_trip():
    rgb = np.zeros((4, 8, 3), dtype=np.uint8)
    depth = np.ones((4, 8), dtype=np.float32)
    eq = DataFactory.from_dict({"rgb": rgb, "depth": depth}, "equirectangular")

    faceset = eq.to_gnomonic_face_set(fov=90)
    assert len(faceset) == 6

    eq_back = faceset.to_equirectangular(eq.shape)
    assert eq_back.is_multi_channel()
    assert set(eq_back.get_channels()) == {"rgb", "depth"}
    assert eq_back.data["rgb"].shape == rgb.shape
    assert eq_back.data["depth"].shape == depth.shape


def test_sampler_registry_and_faces():
    assert "icosahedron" in SamplerRegistry.available_samplers()

    eq = PanoraiFactory.create_data_from_array(
        np.zeros((4, 8, 3), dtype=np.uint8), "equirectangular"
    )
    eq.attach_sampler("icosahedron", subdivisions=0)
    tangent_points = eq.sampler.get_tangent_points()

    faceset = eq.to_gnomonic_face_set(fov=90)
    assert len(faceset) == len(tangent_points)
