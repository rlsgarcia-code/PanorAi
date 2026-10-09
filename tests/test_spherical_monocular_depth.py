from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "spherical_monocular_depth"
    / "protocol.py"
)
SPEC = importlib.util.spec_from_file_location("spherical_depth_protocol", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
protocol = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = protocol
SPEC.loader.exec_module(protocol)


def test_axial_to_radial_matches_hand_derived_pinhole_rays() -> None:
    axial = np.ones((2, 2), dtype=np.float32)
    radial = protocol.axial_to_radial(axial)
    expected = np.sqrt(1.0 + 0.5**2 + 0.5**2)
    np.testing.assert_allclose(radial, expected, rtol=1e-6)


def test_depth_metrics_use_explicit_mask_and_solid_angle() -> None:
    truth = np.full((4, 8), 2.0, dtype=np.float32)
    prediction = truth.copy()
    prediction[0, 0] = 200.0
    validity = np.ones_like(truth, dtype=bool)
    validity[0, 0] = False
    metrics = protocol.depth_metrics(prediction, truth, validity)
    assert metrics["valid_pixels"] == 31
    assert metrics["abs_rel"] == pytest.approx(0.0)
    assert metrics["delta_1"] == pytest.approx(1.0)


def test_scale_invariant_structure_ignores_global_prediction_scale() -> None:
    longitude = np.linspace(-np.pi, np.pi, 64, endpoint=False)
    truth = np.broadcast_to(3.0 + 0.4 * np.cos(longitude), (32, 64)).astype(np.float32)
    validity = np.ones_like(truth, dtype=bool)
    metrics = protocol.scale_invariant_structure_metrics(truth * 7.0, truth, validity)
    assert metrics["optimal_prediction_scale"] == pytest.approx(1.0 / 7.0)
    assert metrics["scale_aligned_relative_3d_rmse"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["scale_invariant_log_rmse"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["log_depth_correlation"] == pytest.approx(1.0)


def test_surface_normals_are_invariant_to_global_range_scale() -> None:
    truth = np.ones((64, 128), dtype=np.float32)
    validity = np.ones_like(truth, dtype=bool)
    metrics = protocol.normal_structure_metrics(
        truth * 9.0, truth, validity, stride_px=2, row_chunk=5
    )
    assert metrics["normal_valid_samples"] > 0
    assert metrics["normal_mean_deg"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["normal_median_deg"] == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize(
    ("source_shape", "expected"),
    [((3414, 8248), (4128, 8256)), ((4267, 10332), (5184, 10368))],
)
def test_native_angular_shape_matches_p74_sampling(
    source_shape: tuple[int, int], expected: tuple[int, int]
) -> None:
    assert protocol.native_angular_erp_shape(source_shape) == expected


@pytest.mark.parametrize(
    ("erp_shape", "expected_face_size"),
    [((4128, 8256), 2656), ((5184, 10368), 3328)],
)
def test_native_cube_faces_do_not_minify_at_face_center(
    erp_shape: tuple[int, int], expected_face_size: int
) -> None:
    assert protocol.native_angular_cube_face_size(erp_shape) == expected_face_size


def test_seam_score_measures_adjacent_erp_boundary_columns() -> None:
    prediction = np.zeros((3, 5), dtype=np.float32)
    prediction[:, -1] = [1.0, 2.0, 3.0]
    validity = np.ones_like(prediction, dtype=bool)
    assert protocol.seam_score(prediction, validity) == {"rows": 3, "mae_m": 2.0}


def test_spherical_transpose_port_preserves_shape_and_parameter_identity() -> None:
    torch = pytest.importorskip("torch")
    source = torch.nn.Sequential(
        torch.nn.Conv2d(2, 3, 3, padding=1),
        torch.nn.ConvTranspose2d(3, 2, 3, stride=2, padding=1, output_padding=1),
    )
    parameters = tuple(source.parameters())
    report = protocol.port_metric3d_spatial_layers(source)
    result = source(torch.randn(1, 2, 8, 16))
    assert result.shape == (1, 2, 16, 32)
    assert report["ported_layer_count"] == 2
    assert report["all_parameter_identities_preserved"] is True
    assert all(
        after is before for after, before in zip(source.parameters(), parameters)
    )


def test_chunked_port_matches_full_lattice_port() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    reference = torch.nn.Sequential(
        torch.nn.Conv2d(2, 4, 3, padding=1, groups=2),
        torch.nn.Conv2d(4, 3, 1),
        torch.nn.ConvTranspose2d(3, 2, 3, stride=2, padding=1, output_padding=1),
    )
    chunked = torch.nn.Sequential(
        torch.nn.Conv2d(2, 4, 3, padding=1, groups=2),
        torch.nn.Conv2d(4, 3, 1),
        torch.nn.ConvTranspose2d(3, 2, 3, stride=2, padding=1, output_padding=1),
    )
    chunked.load_state_dict(reference.state_dict())
    protocol.port_metric3d_spatial_layers(reference)
    report = protocol.port_metric3d_spatial_layers(chunked, max_sampled_elements=10)
    values = torch.randn(1, 2, 8, 16)
    torch.testing.assert_close(chunked(values), reference(values), rtol=1e-5, atol=1e-6)
    assert report["max_sampled_elements_per_chunk"] == 10


def test_binary_ply_contains_decimated_canonical_points(tmp_path: Path) -> None:
    radial = np.full((4, 8), 2.0, dtype=np.float32)
    rgb = np.zeros((4, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    validity = np.ones((4, 8), dtype=bool)
    path = tmp_path / "cloud.ply"
    report = protocol.write_binary_ply(path, radial, rgb, validity, max_points=8)
    payload = path.read_bytes()
    header, body = payload.split(b"end_header\n", 1)
    assert b"format binary_little_endian 1.0" in header
    assert f"element vertex {report['written_points']}".encode() in header
    assert report["written_points"] <= 8
    assert len(body) == report["written_points"] * 15
