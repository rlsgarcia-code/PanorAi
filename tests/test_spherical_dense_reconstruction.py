from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pytest

from panorai.applications import SphericalDenseReconstruction
from panorai.data import EquirectangularImage
from panorai.evaluation import (
    aggregate_surface_reconstruction,
    evaluate_surface_reconstruction,
)
from panorai.geometry import (
    camera_rays_to_world_points,
    erp_pixels_to_rays,
    relative_camera_pose,
    world_points_to_camera_rays,
)
from panorai.pipeline import Pipeline
from panorai.pipeline._cache_typed import decode_typed, encode_typed
from panorai.pipeline._serialization import pipeline_from_config, pipeline_to_config
from panorai.reconstruction import (
    BuildRegisteredReferenceSets,
    ColorizeSurface,
    EstimateMultiViewRanges,
    ExtractSurface,
    FuseTSDF,
    MultiViewRangeMap,
    MultiViewRangeSet,
    RegisteredSphericalScene,
    RegisteredSphericalView,
    SphericalCameraPose,
    SurfaceMesh,
    TSDFSpec,
    TSDFVolume,
    ValidateDenseReconstruction,
    colorize_surface,
    extract_surface,
    fuse_tsdf,
)
from panorai.reconstruction._dense import _bilinear_sample
from panorai.stereo import SphericalStereoOptions
from panorai.stereo.panoramic import (
    FullSphereStereoOptions,
    FullSphereStereoResult,
    OverlappingCubemapPlanner,
)


def _scene(*, color: float = 0.5) -> RegisteredSphericalScene:
    views = []
    for index, x in enumerate((-0.2, 0.0, 0.2)):
        pose = SphericalCameraPose(
            panorama_id=f"camera-{index}",
            rotation_world_to_panorama=np.eye(3),
            center_world=np.asarray((x, 0.0, 0.0)),
        )
        image = EquirectangularImage(np.full((32, 64, 3), color, dtype=np.float32))
        views.append(RegisteredSphericalView(image=image, pose=pose))
    return RegisteredSphericalScene(views=tuple(views), units="m")


def _plane_range_map(view: RegisteredSphericalView) -> MultiViewRangeMap:
    height, width = view.shape_hw
    yy, xx = np.indices((height, width), dtype=np.float64)
    rays = erp_pixels_to_rays(np.stack((xx, yy), axis=-1), (height, width))
    valid = rays[..., 2] > 0.2
    radial = np.full((height, width), np.nan, dtype=np.float32)
    radial[valid] = (2.0 / rays[..., 2][valid]).astype(np.float32)
    count = np.where(valid, 2, 0).astype(np.uint16)
    return MultiViewRangeMap(
        reference_id=view.panorama_id,
        pose=view.pose,
        radial_range=radial,
        validity=valid,
        sampling_support=np.ones((height, width), dtype=bool),
        observable=valid,
        confidence=np.where(valid, 1.0, 0.0).astype(np.float32),
        observable_source_count=count,
        source_count=count,
        source_consensus_count=count,
        contribution_count=count,
        source_ids=("source-a", "source-b"),
    )


def _plane_ranges(scene: RegisteredSphericalScene) -> MultiViewRangeSet:
    return MultiViewRangeSet(
        scene=scene,
        maps=tuple(_plane_range_map(view) for view in scene.views),
    )


def _plane_spec() -> TSDFSpec:
    return TSDFSpec(
        voxel_size=0.05,
        bounds=((-1.0, -1.0, 1.5), (1.0, 1.0, 2.5)),
        min_surface_views=2,
    )


def _analytic_range_set(shape: str) -> MultiViewRangeSet:
    height, width = 96, 192
    yy, xx = np.indices((height, width), dtype=np.float64)
    rays = erp_pixels_to_rays(np.stack((xx, yy), axis=-1), (height, width))
    centers = (
        (-3.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (0.0, -3.0, 0.0),
        (0.0, 3.0, 0.0),
        (0.0, 0.0, -3.0),
        (0.0, 0.0, 3.0),
    )
    views = []
    maps = []
    for index, raw_center in enumerate(centers):
        center = np.asarray(raw_center)
        pose = SphericalCameraPose(f"camera-{index}", np.eye(3), center)
        view = RegisteredSphericalView(
            image=EquirectangularImage(
                np.full((height, width, 3), 0.5, dtype=np.float32)
            ),
            pose=pose,
        )
        if shape == "sphere":
            centre_dot = np.einsum("...i,i->...", rays, center)
            discriminant = centre_dot**2 - (np.dot(center, center) - 1.0)
            valid = discriminant >= 0.0
            radial = -centre_dot - np.sqrt(np.maximum(discriminant, 0.0))
            valid &= radial > 0.0
        elif shape == "box":
            inverse = np.divide(
                1.0,
                rays,
                out=np.full_like(rays, np.inf),
                where=np.abs(rays) > 1e-12,
            )
            first = (-0.75 - center) * inverse
            second = (0.75 - center) * inverse
            near = np.max(np.minimum(first, second), axis=-1)
            far = np.min(np.maximum(first, second), axis=-1)
            valid = (far >= near) & (far > 0.0)
            radial = np.where(near > 0.0, near, far)
            valid &= radial > 0.0
        else:
            raise AssertionError(shape)
        radial = np.where(valid, radial, np.nan).astype(np.float32)
        count = np.where(valid, 5, 0).astype(np.uint16)
        views.append(view)
        maps.append(
            MultiViewRangeMap(
                reference_id=view.panorama_id,
                pose=pose,
                radial_range=radial,
                validity=valid,
                sampling_support=np.ones((height, width), dtype=bool),
                observable=valid,
                confidence=np.where(valid, 1.0, 0.0),
                observable_source_count=count,
                source_count=count,
                source_consensus_count=count,
                contribution_count=count,
                source_ids=tuple(
                    f"camera-{other}" for other in range(6) if other != index
                ),
            )
        )
    scene = RegisteredSphericalScene(views=tuple(views), units="m")
    return MultiViewRangeSet(scene=scene, maps=tuple(maps))


def _full_sphere_result(item: MultiViewRangeMap) -> FullSphereStereoResult:
    shape = item.radial_range.shape
    disagreement = np.full(shape, np.nan, dtype=np.float32)
    disagreement[item.validity] = 0.0
    return FullSphereStereoResult(
        radial_range=item.radial_range,
        validity=item.validity,
        confidence=item.confidence,
        disagreement=disagreement,
        sampling_support=item.sampling_support,
        observable=item.observable,
        max_parallax_deg=np.where(item.observable, 5.0, 0.0),
        observable_source_count=item.observable_source_count,
        source_count=item.source_count,
        source_consensus_count=item.source_consensus_count,
        contribution_count=item.contribution_count,
        source_ids=item.source_ids,
        diagnostics={"units": "m"},
    )


def test_camera_world_transform_and_relative_pose_oracles() -> None:
    angle = np.deg2rad(32.0)
    reference_rotation = np.asarray(
        (
            (np.cos(angle), 0.0, np.sin(angle)),
            (0.0, 1.0, 0.0),
            (-np.sin(angle), 0.0, np.cos(angle)),
        )
    )
    source_rotation = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, np.cos(angle), np.sin(angle)),
            (0.0, -np.sin(angle), np.cos(angle)),
        )
    )
    reference_center = np.asarray((0.2, -0.4, 1.3))
    source_center = np.asarray((-0.7, 0.3, 0.5))
    points = np.asarray(((1.2, 0.5, 3.0), (-0.4, 1.1, 2.3)))

    bearings, ranges = world_points_to_camera_rays(
        points, reference_rotation, reference_center
    )
    restored = camera_rays_to_world_points(
        bearings, ranges, reference_rotation, reference_center
    )
    np.testing.assert_allclose(restored, points, atol=1e-12)

    rotation, translation = relative_camera_pose(
        reference_rotation,
        reference_center,
        source_rotation,
        source_center,
    )
    reference_points = (points - reference_center) @ reference_rotation.T
    source_points = (points - source_center) @ source_rotation.T
    np.testing.assert_allclose(
        reference_points @ rotation.T + translation, source_points, atol=1e-12
    )


def test_camera_world_transform_preserves_torch_autograd() -> None:
    torch = pytest.importorskip("torch")
    points = torch.tensor(((1.0, 0.5, 2.0),), dtype=torch.float64, requires_grad=True)
    bearings, ranges = world_points_to_camera_rays(
        points, torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    )
    restored = camera_rays_to_world_points(
        bearings,
        ranges,
        torch.eye(3, dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
    )
    restored.sum().backward()
    torch.testing.assert_close(restored, points)
    torch.testing.assert_close(points.grad, torch.ones_like(points))


def test_registered_scene_builds_sorted_metric_reference_sets() -> None:
    scene = _scene()
    stage = BuildRegisteredReferenceSets()
    workspace = stage.execute(scene).value
    assert tuple(item.reference_id for item in workspace.reference_sets.sets) == (
        "camera-0",
        "camera-1",
        "camera-2",
    )
    registered = scene.registered_set("camera-1")
    assert tuple(item.source_id for item in registered.sources) == (
        "camera-0",
        "camera-2",
    )
    assert all(item.pose.scale_status == "metric" for item in registered.sources)
    assert all(item.pose.units == "m" for item in registered.sources)


def test_range_stage_runs_the_real_full_sphere_mvs_engine() -> None:
    height, width = 16, 32
    yy, xx = np.indices((height, width), dtype=np.float32)
    texture = np.stack(
        (
            (np.sin(xx * 0.5) + 1.0) / 2.0,
            (np.cos(yy * 0.5) + 1.0) / 2.0,
            ((xx // 3 + yy // 3) % 2).astype(np.float32),
        ),
        axis=-1,
    )
    views = tuple(
        RegisteredSphericalView(
            image=EquirectangularImage(np.roll(texture, index, axis=1)),
            pose=SphericalCameraPose(
                panorama_id=f"camera-{index}",
                rotation_world_to_panorama=np.eye(3),
                center_world=np.asarray((center_x, 0.0, 0.0)),
            ),
        )
        for index, center_x in enumerate((-0.1, 0.0, 0.1))
    )
    scene = RegisteredSphericalScene(views=views, units="m")
    options = FullSphereStereoOptions(
        planner=OverlappingCubemapPlanner(face_size=8, face_fov_deg=100.0),
        stereo=SphericalStereoOptions(
            min_range=0.5,
            max_range=2.0,
            num_hypotheses=3,
            window_size=3,
            pole_margin_fraction=0.0,
            min_texture_std=0.0,
            min_confidence=0.0,
            max_matching_cost=1.0,
            bidirectional_consistency=False,
            filter_backend="numpy",
            pyramid_levels=1,
        ),
        min_parallax_deg=0.0,
        min_source_evidence=1,
        matching_cost="intensity-gradient",
        local_aggregation="none",
        regularizer="none",
        erp_feature_tap_deg=None,
        face_cost_tap_deg=None,
    )

    workspace = BuildRegisteredReferenceSets().execute(scene).value
    estimated = EstimateMultiViewRanges(options).execute(workspace).value

    assert estimated.ranges is not None
    assert tuple(item.reference_id for item in estimated.ranges.maps) == (
        "camera-0",
        "camera-1",
        "camera-2",
    )
    assert all(
        item.radial_range.shape == (height, width) for item in estimated.ranges.maps
    )
    assert all(np.any(item.validity) for item in estimated.ranges.maps)


def test_erp_sampling_wraps_seam_handles_poles_and_does_not_promote_nan() -> None:
    values = np.asarray(((1.0, np.nan, 3.0, 5.0), (1.0, np.nan, 3.0, 5.0)))
    validity = np.isfinite(values)
    sampled, accepted = _bilinear_sample(
        values,
        np.asarray(((0.0, 0.0), (-0.5, -0.5), (3.5, 1.5))),
        validity=validity,
    )
    assert accepted.tolist() == [True, True, True]
    assert sampled[0] == 1.0
    assert sampled[1] == pytest.approx(3.0)
    assert sampled[2] == pytest.approx(3.0)


def test_tsdf_accepts_registered_erps_with_different_resolutions() -> None:
    views = []
    for index, (height, x) in enumerate(((16, -0.2), (24, 0.0), (32, 0.2))):
        pose = SphericalCameraPose(
            f"camera-{index}", np.eye(3), np.asarray((x, 0.0, 0.0))
        )
        views.append(
            RegisteredSphericalView(
                image=EquirectangularImage(
                    np.full((height, 2 * height, 3), 0.5, dtype=np.float32)
                ),
                pose=pose,
            )
        )
    scene = RegisteredSphericalScene(views=tuple(views), units="m")
    ranges = MultiViewRangeSet(
        scene=scene,
        maps=tuple(_plane_range_map(view) for view in views),
    )
    surface = extract_surface(fuse_tsdf(ranges, _plane_spec()))
    assert np.median(np.abs(surface.vertices_xyz[:, 2] - 2.0)) <= 0.05


def test_tsdf_plane_surface_meets_analytic_voxel_oracle() -> None:
    ranges = _plane_ranges(_scene())
    volume = fuse_tsdf(ranges, _plane_spec())
    surface = extract_surface(volume)

    error = np.abs(surface.vertices_xyz[:, 2] - 2.0)
    assert np.median(error) <= volume.voxel_size
    assert np.percentile(error, 95) <= 2.0 * volume.voxel_size
    # For this camera-facing plane, positive TSDF/free space is toward -Z.
    expected = np.asarray((0.0, 0.0, -1.0))
    angular = np.degrees(np.arccos(np.clip(surface.normals_xyz @ expected, -1.0, 1.0)))
    assert np.median(angular) <= 10.0
    assert np.all(surface.vertex_weight > 0.0)
    assert np.all(surface.vertex_view_count >= 2)

    repeated = fuse_tsdf(ranges, _plane_spec())
    np.testing.assert_array_equal(repeated.tsdf_zyx, volume.tsdf_zyx)
    np.testing.assert_array_equal(repeated.weight_zyx, volume.weight_zyx)
    np.testing.assert_array_equal(repeated.view_count_zyx, volume.view_count_zyx)


def test_extract_surface_contains_only_known_scikit_image_numpy_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import skimage.measure

    real_marching_cubes = skimage.measure.marching_cubes

    def marching_cubes_with_known_warning(*args, **kwargs):
        warnings.warn(
            "Setting the shape on a NumPy array has been deprecated in NumPy 2.5.",
            DeprecationWarning,
            stacklevel=1,
        )
        return real_marching_cubes(*args, **kwargs)

    monkeypatch.setattr(
        skimage.measure,
        "marching_cubes",
        marching_cubes_with_known_warning,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        surface = extract_surface(fuse_tsdf(_plane_ranges(_scene()), _plane_spec()))
    assert len(surface.faces) > 0

    def marching_cubes_with_unexpected_warning(*args, **kwargs):
        warnings.warn(
            "unexpected marching-cubes warning",
            DeprecationWarning,
            stacklevel=1,
        )
        return real_marching_cubes(*args, **kwargs)

    monkeypatch.setattr(
        skimage.measure,
        "marching_cubes",
        marching_cubes_with_unexpected_warning,
    )
    with (
        warnings.catch_warnings(),
        pytest.raises(DeprecationWarning, match="unexpected marching-cubes warning"),
    ):
        warnings.simplefilter("error")
        extract_surface(fuse_tsdf(_plane_ranges(_scene()), _plane_spec()))


@pytest.mark.parametrize("shape", ("sphere", "box"))
def test_tsdf_surface_matches_independent_analytic_shape_oracle(shape: str) -> None:
    voxel_size = 0.1
    volume = fuse_tsdf(
        _analytic_range_set(shape),
        TSDFSpec(
            voxel_size=voxel_size,
            bounds=((-1.3, -1.3, -1.3), (1.3, 1.3, 1.3)),
            min_surface_views=2,
        ),
    )
    surface = extract_surface(volume)
    if shape == "sphere":
        error = np.abs(np.linalg.norm(surface.vertices_xyz, axis=1) - 1.0)
    else:
        q = np.abs(surface.vertices_xyz) - 0.75
        signed_distance = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(
            np.max(q, axis=1), 0.0
        )
        error = np.abs(signed_distance)
    assert np.median(error) <= voxel_size
    assert np.percentile(error, 95) <= 2.0 * voxel_size


def test_tsdf_fails_explicitly_when_bound_exceeds_limit() -> None:
    with pytest.raises(ValueError, match="exceeding max_voxels"):
        fuse_tsdf(
            _plane_ranges(_scene()),
            TSDFSpec(
                voxel_size=0.01,
                bounds=((-1.0, -1.0, 1.0), (1.0, 1.0, 3.0)),
                max_voxels=1_000,
            ),
        )


def test_unknown_tsdf_does_not_generate_a_surface() -> None:
    shape = (8, 8, 8)
    volume = TSDFVolume(
        tsdf_zyx=np.ones(shape, dtype=np.float32),
        weight_zyx=np.zeros(shape, dtype=np.float32),
        view_count_zyx=np.zeros(shape, dtype=np.uint16),
        origin_xyz=np.zeros(3),
        voxel_size=0.1,
        truncation_distance=0.4,
        units="m",
        frame="world",
        min_surface_views=2,
    )
    with pytest.raises(ValueError, match="minimum view count"):
        extract_surface(volume)


def test_black_is_a_valid_surface_color() -> None:
    ranges = _plane_ranges(_scene(color=0.0))
    volume = fuse_tsdf(ranges, _plane_spec())
    surface = colorize_surface(
        extract_surface(volume),
        ranges,
        truncation_distance=volume.truncation_distance,
    )
    assert surface.color_validity is not None
    assert np.any(surface.color_validity)
    np.testing.assert_array_equal(
        surface.colors_rgb[surface.color_validity],
        np.zeros((surface.color_validity.sum(), 3), dtype=np.float32),
    )


def test_application_matches_public_stage_composition(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    scene = _scene()
    source_ranges = _plane_ranges(scene)

    class AnalyticMVS:
        interface = "test-analytic-mvs"
        stability = "test"

        def __init__(self, options) -> None:
            self.options = options

        def describe(self):
            return {"interface": self.interface}

        def estimate(self, registered):
            return _full_sphere_result(
                source_ranges.by_reference(registered.reference_id)
            )

    monkeypatch.setattr(
        "panorai.reconstruction._dense.FullSphereMultiViewStereo", AnalyticMVS
    )
    method = SphericalDenseReconstruction(
        voxel_size=0.05,
        bounds=((-1.0, -1.0, 1.5), (1.0, 1.0, 2.5)),
        colorize=False,
        retain_intermediates="all",
    )
    application_result = method(scene)

    value = BuildRegisteredReferenceSets().execute(scene).value
    value = EstimateMultiViewRanges(method.mvs_options).execute(value).value
    value = FuseTSDF(method._tsdf_spec()).execute(value).value
    value = ExtractSurface().execute(value).value
    value = ColorizeSurface(False).execute(value).value
    stage_result = ValidateDenseReconstruction("all").execute(value).value

    assert application_result.accepted
    assert stage_result.accepted
    np.testing.assert_array_equal(
        application_result.surface.vertices_xyz, stage_result.surface.vertices_xyz
    )
    np.testing.assert_array_equal(
        application_result.volume.tsdf_zyx, stage_result.volume.tsdf_zyx
    )

    base_pipeline = method.as_pipeline()
    cached_pipeline = Pipeline(*base_pipeline.named_steps.items(), memory=tmp_path)
    first = cached_pipeline.run(scene)
    second = cached_pipeline.run(scene)
    assert first.output.accepted
    assert second.records[0].name == "cache"
    np.testing.assert_array_equal(
        first.output.surface.vertices_xyz, second.output.surface.vertices_xyz
    )


def test_reviewed_profile_requires_three_views_and_round_trips_config(tmp_path) -> None:
    method = SphericalDenseReconstruction.reviewed_profile(
        voxel_size=0.1,
        bounds=((-1.0, -1.0, 1.0), (1.0, 1.0, 3.0)),
    )
    path = method.save(tmp_path / "dense.json")
    restored = SphericalDenseReconstruction.load(path)
    assert restored == method
    assert method.with_params(bounds=((-2, -2, 0), (2, 2, 4))).profile_id
    assert method.with_params(weighting="confidence").profile_id is None
    two_view_scene = replace(_scene(), views=_scene().views[:2])
    with pytest.raises(ValueError, match="at least three"):
        method(two_view_scene)

    restored_pipeline = pipeline_from_config(pipeline_to_config(method.as_pipeline()))
    assert restored_pipeline.describe() == method.as_pipeline().describe()


def test_dense_result_has_safe_pickle_free_typed_cache_codec() -> None:
    ranges = _plane_ranges(_scene())
    volume = fuse_tsdf(ranges, _plane_spec())
    surface = extract_surface(volume)
    from panorai.reconstruction import SphericalDenseReconstructionResult

    result = SphericalDenseReconstructionResult(
        status="accepted",
        reasons=(),
        diagnostics={"oracle": "analytic-plane"},
        surface=surface,
        ranges=ranges,
        volume=volume,
    )
    arrays, state = encode_typed(result)
    restored = decode_typed(arrays, state)
    assert restored.status == "accepted"
    np.testing.assert_array_equal(restored.volume.tsdf_zyx, volume.tsdf_zyx)
    np.testing.assert_array_equal(restored.surface.faces, surface.faces)
    assert all(array.dtype != object for array in arrays.values())


def test_surface_evaluation_is_exact_for_identical_mesh() -> None:
    mesh = extract_surface(fuse_tsdf(_plane_ranges(_scene()), _plane_spec()))
    evaluation = evaluate_surface_reconstruction(mesh, mesh, distance_threshold=0.1)
    assert evaluation.accuracy_median == pytest.approx(0.0, abs=1e-12)
    assert evaluation.completeness_median_distance == pytest.approx(0.0, abs=1e-12)
    assert evaluation.precision == 1.0
    assert evaluation.recall == 1.0
    assert evaluation.fscore == 1.0
    assert evaluation.median_normal_error_deg == pytest.approx(0.0, abs=1e-5)


def test_surface_evaluation_aggregate_is_scene_balanced_and_fail_closed() -> None:
    mesh = extract_surface(fuse_tsdf(_plane_ranges(_scene()), _plane_spec()))
    first = evaluate_surface_reconstruction(mesh, mesh, distance_threshold=0.1)
    displaced = replace(
        mesh, vertices_xyz=mesh.vertices_xyz + np.asarray((0.0, 0.0, 0.05))
    )
    second = evaluate_surface_reconstruction(displaced, mesh, distance_threshold=0.1)

    report = aggregate_surface_reconstruction({"z-scene": second, "a-scene": first})

    assert tuple(report.per_scene) == ("a-scene", "z-scene")
    assert report.scene_count == 2
    assert report.aggregate["accuracy_median"] == pytest.approx(0.025, abs=1e-4)
    with pytest.raises(ValueError, match="same distance_threshold"):
        aggregate_surface_reconstruction(
            {
                "first": first,
                "changed": evaluate_surface_reconstruction(
                    mesh, mesh, distance_threshold=0.2
                ),
            }
        )


def test_surface_mesh_rejects_faces_outside_vertex_array() -> None:
    with pytest.raises(ValueError, match="out-of-range"):
        SurfaceMesh(
            vertices_xyz=np.zeros((2, 3)),
            faces=np.asarray(((0, 1, 2),)),
            normals_xyz=np.zeros((2, 3)),
            vertex_weight=np.ones(2),
            vertex_view_count=np.ones(2),
        )
