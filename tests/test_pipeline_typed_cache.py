from __future__ import annotations

import numpy as np
import pytest

import panorai as pa
from panorai.data import RegionAnalysisResult
from panorai.pipeline import PipelineCacheError, RasterGrid
from panorai.ops import spherical as S
from panorai.pipeline._cache_typed import decode_typed, encode_typed


def _round_trip(value):
    arrays, state = encode_typed(value)
    assert all(item.dtype != object for item in arrays.values())
    return decode_typed({name: item.copy() for name, item in arrays.items()}, state)


def test_typed_cache_round_trips_mixed_bundle_grid_and_observations() -> None:
    shape = (6, 12)
    grid = RasterGrid(
        domain="erp",
        shape_hw=shape,
        erp_shape_hw=shape,
        alignment="erp-pixel-centres",
    )
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    support = np.ones(shape, dtype=bool)
    validity = np.ones(shape, dtype=bool)
    validity[1, 2] = False
    values[1, 2] = np.nan
    field = pa.RasterField("depth", values, "depth", support, validity, grid, units="m")
    bearings = np.asarray(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)))
    observations = pa.SphericalObservations(
        bearings,
        footprints_xyz=(bearings[:1], bearings[1:]),
        scores=np.asarray((0.9, 0.8)),
        support=np.asarray((True, True)),
        data_validity=np.asarray((True, True)),
        valid=np.asarray((True, True)),
        source_face_ids=np.asarray(("front", "right"), dtype=object),
        source_indices=np.asarray((0, 1)),
        source_pixels_xy=(np.asarray(((2.0, 3.0),)), np.asarray(((1.0, 1.0),))),
        geometry="points",
        labels=np.asarray((2, 4)),
        metadata=({"origin": "detector"}, {"origin": "detector"}),
    )
    bundle = pa.ResultBundle(
        {
            "depth": field,
            "classification": pa.ClassificationResult(
                np.asarray((0.1, 0.9), dtype=np.float32)
            ),
            "detections": observations,
        },
        descriptions={"depth": {"stage": "test"}},
    )

    restored = _round_trip(bundle)

    assert isinstance(restored, pa.ResultBundle)
    np.testing.assert_array_equal(restored["depth"].values, field.values)
    np.testing.assert_array_equal(restored["depth"].support, field.support)
    np.testing.assert_array_equal(restored["depth"].validity, field.validity)
    assert restored["depth"].grid == field.grid
    np.testing.assert_array_equal(
        restored["classification"].value, bundle["classification"].value
    )
    np.testing.assert_array_equal(
        restored["detections"].bearings_xyz, observations.bearings_xyz
    )
    assert list(restored["detections"].source_face_ids) == ["front", "right"]


def test_typed_cache_round_trips_view_rasters_and_materialized_views() -> None:
    panorama = pa.EquirectangularImage(
        np.arange(8 * 16 * 3, dtype=np.float32).reshape(8, 16, 3)
    )
    faces = panorama.views("cube", size=5)
    field = pa.RasterField(
        "image",
        tuple(face.image for face in faces),
        "image",
        tuple(face._workflow_support["image"] for face in faces),
        tuple(face.validity("image") for face in faces),
        RasterGrid(
            domain="views",
            shape_hw=(5, 5),
            erp_shape_hw=(8, 16),
            alignment="gnomonic-pixel-centres",
            view_ids=tuple(face.face_id for face in faces),
            view_specs=tuple(
                (
                    face.spec.center_lat_deg,
                    face.spec.center_lon_deg,
                    face.spec.hfov_deg,
                    face.spec.vfov_deg,
                    face.spec.roll_deg,
                    face.spec.output_shape_hw,
                )
                for face in faces
            ),
            longitude_periodic=False,
        ),
    )

    restored_field = _round_trip(field)
    restored_faces = _round_trip(faces)

    assert restored_field.grid == field.grid
    for actual, expected in zip(restored_field.values, field.values, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert [item.face_id for item in restored_faces] == [item.face_id for item in faces]
    assert restored_faces._workflow["layout"] == "cube"
    np.testing.assert_allclose(
        restored_faces.reconstruct().image,
        faces.reconstruct().image,
        equal_nan=True,
    )


def test_typed_cache_round_trips_complete_region_analysis() -> None:
    labels = np.zeros((12, 24), dtype=np.int32)
    labels[3:9, :4] = 1
    labels[3:9, -4:] = 1
    labels[4:8, 10:14] = 2
    result = RegionAnalysisResult(
        measurements=S.measure_regions(labels),
        adjacency=S.region_adjacency_graph(labels),
        boundaries=S.extract_boundaries(labels),
        contours=S.trace_contours(labels),
    )

    restored = _round_trip(result)

    assert isinstance(restored, RegionAnalysisResult)
    assert restored.interface == result.interface
    assert restored.measurements.shape_hw == labels.shape
    assert restored.adjacency == result.adjacency
    assert restored.boundaries is not None
    assert restored.contours is not None
    assert len(restored.contours.contours) == len(result.contours.contours)
    for actual, expected in zip(
        restored.contours.contours,
        result.contours.contours,
        strict=True,
    ):
        assert actual.label == expected.label
        assert actual.closed == expected.closed
        assert actual.angular_length_rad == expected.angular_length_rad
        np.testing.assert_array_equal(actual.bearings_xyz, expected.bearings_xyz)


def test_content_cache_preserves_fortran_layout(tmp_path) -> None:
    value = np.asfortranarray(np.arange(24, dtype=np.float32).reshape(4, 6))
    pipeline = pa.Pipeline(
        pa.Parallel(value=None),
        pa.Select("value"),
        memory=tmp_path,
    )

    pipeline(value)
    restored = pipeline(value)

    assert restored.flags.f_contiguous
    assert not restored.flags.c_contiguous
    np.testing.assert_array_equal(restored, value)


def test_torch_cache_allows_detached_tensor_and_rejects_autograd(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    pipeline = pa.Pipeline(
        pa.Parallel(value=None),
        pa.Select("value"),
        memory=tmp_path,
    )
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    pipeline(value)
    restored = pipeline(value)

    assert torch.equal(restored, value)
    assert restored.dtype == value.dtype
    assert restored.device == value.device

    differentiable = value.clone().requires_grad_()
    with pytest.raises(PipelineCacheError, match="disconnect autograd"):
        pipeline.with_params()(differentiable)


def test_typed_cache_rejects_unsafe_metadata_and_unregistered_results() -> None:
    bundle = pa.ResultBundle(
        {"value": np.ones((2, 2), dtype=np.float32)},
        descriptions={"value": {"callable": lambda: None}},
    )
    with pytest.raises(ValueError, match="unsupported metadata type"):
        encode_typed(bundle)

    with pytest.raises(ValueError, match="no registered typed cache codec"):
        encode_typed(object())


@pytest.mark.parametrize(
    "operation",
    (
        S.Resample(np.asarray(((0.0, 0.0, 1.0),))),
        S.AreaAverage(),
        S.AreaHistogram(bins=np.asarray((0.0, 1.0))),
        S.Filter2D(np.ones((3, 3))),
        S.BoxBlur(),
        S.MedianBlur(),
        S.BilateralFilter(),
        S.Sobel(),
        S.Gradient(),
        S.Laplacian(),
        S.Canny(0.1, 0.2),
        S.GaussianBlur(),
        S.Resize((4, 8)),
        S.Rotate(np.eye(3)),
        S.SLIC(n_segments=8),
        S.GaussianPyramid(2),
        S.LaplacianPyramid(2),
    ),
)
def test_every_configured_numpy_operation_has_a_safe_pipeline_codec(operation) -> None:
    pipeline = pa.Pipeline(pa.OnSphere(operation))

    restored = pa.Pipeline.from_config(pipeline.to_config())

    assert restored.to_config() == pipeline.to_config()
