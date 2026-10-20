from __future__ import annotations

from panorai_streetview.models import ImagePairRecord, ImageRecord, ProjectRecord
from panorai_streetview.planner import build_plan


def image(*, pose: bool = False) -> ImageRecord:
    metadata = {}
    if pose:
        metadata["pose"] = {
            "rotation_world_to_panorama": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "center_world": [0, 0, 0],
        }
    return ImageRecord(
        id="img_1",
        filename="one.png",
        object_name="a.png",
        sha256="a" * 64,
        media_type="image/png",
        width=8,
        height=4,
        size_bytes=12,
        metadata=metadata,
    )


def project(workflow: str, *, pose: bool = False, pair: bool = False) -> ProjectRecord:
    images = [image(pose=pose)]
    pairs = []
    if pair:
        second = image(pose=pose).model_copy(
            update={"id": "img_2", "filename": "two.png", "sha256": "b" * 64}
        )
        images.append(second)
        pairs.append(
            ImagePairRecord(
                id="pair-1",
                reference_image_id="img_1",
                source_image_id="img_2",
                relation="registered_overlap",
                baseline_m=1.0,
                minimum_directional_overlap=0.8,
            )
        )
    return ProjectRecord(
        id="sv_test",
        name="test",
        workflow=workflow,
        created_at="2026-10-19T00:00:00+00:00",
        updated_at="2026-10-19T00:00:00+00:00",
        panorai_runtime={"version": "test"},
        images=images,
        pairs=pairs,
    )


def test_registered_erp_is_buildable_only_with_explicit_pose() -> None:
    blocked = build_plan(project("spherical_registered"))
    poses_only = build_plan(project("spherical_registered", pose=True))
    ready = build_plan(project("spherical_registered", pose=True, pair=True))

    assert blocked.can_build_now is False
    assert blocked.stages[1].status == "requires_input"
    assert poses_only.can_build_now is False
    assert poses_only.stages[2].status == "requires_input"
    assert ready.can_build_now is True
    assert ready.stages[1].status == "ready"


def test_registered_erp_rejects_non_rotation_matrix() -> None:
    malformed = project("spherical_registered", pose=True)
    malformed.images[0].metadata["pose"]["rotation_world_to_panorama"][0][0] = 2

    plan = build_plan(malformed)

    assert plan.can_build_now is False
    assert plan.stages[1].status == "requires_input"


def test_registered_erp_rejects_numeric_pose_overflow() -> None:
    malformed = project("spherical_registered", pose=True, pair=True)
    malformed.images[0].metadata["pose"]["center_world"][0] = int("9" * 400)

    plan = build_plan(malformed)

    assert plan.can_build_now is False
    assert plan.stages[1].status == "requires_input"


def test_unregistered_erp_names_public_panorai_engines() -> None:
    plan = build_plan(project("spherical_unregistered"))
    engines = {stage.id: stage.engine for stage in plan.stages}

    assert engines["spherical-features"] == "panorai.features.SphericalFeaturePipeline"
    assert (
        engines["relative-poses"] == "panorai.estimators.SphericalRelativePoseEstimator"
    )
    assert engines["global-map"] == "panorai.reconstruction.SphericalGlobalMapper"
    assert plan.can_build_now is False


def test_uncalibrated_perspective_names_public_observable_fov_route() -> None:
    plan = build_plan(project("perspective_uncalibrated"))
    calibration = next(
        stage for stage in plan.stages if stage.id == "robust-calibration"
    )

    assert calibration.engine == "panorai.calibration.calibrate_rotation_pair"
    assert calibration.status == "planned"
    assert plan.can_build_now is False
    assert "pelo menos duas" in plan.next_action


def test_perspective_groups_each_require_two_sources() -> None:
    capture = project("perspective_uncalibrated", pair=True)
    capture.images[0].metadata["capture_group"] = "a"
    capture.images[1].metadata["capture_group"] = "b"

    incomplete = build_plan(capture)

    assert incomplete.can_build_now is False
    assert "cada capture_group" in incomplete.next_action

    capture.images[1].metadata["capture_group"] = "a"
    ready = build_plan(capture)
    assert ready.can_build_now is True

    capture.images[0].metadata["capture_group"] = "group one"
    capture.images[1].metadata["capture_group"] = "group one"
    unsafe = build_plan(capture)
    assert unsafe.can_build_now is False
    assert "cada capture_group" in unsafe.next_action


def test_calibrated_perspective_uses_stitching_camera_validation() -> None:
    invalid_metadata = [
        {"horizontal_fov_deg": "wide"},
        {"horizontal_fov_deg": 180.0},
        {"intrinsics": {"fx": 0.0}},
        {"horizontal_fov_deg": 92.0, "distortion": "radial"},
        {"intrinsics": []},
    ]
    for metadata in invalid_metadata:
        capture = project("perspective_calibrated", pair=True)
        capture.images[0].metadata = metadata
        capture.images[1].metadata = {"horizontal_fov_deg": 92.0}

        plan = build_plan(capture)
        camera = next(stage for stage in plan.stages if stage.id == "camera-metadata")

        assert plan.can_build_now is False
        assert camera.status == "requires_input"


def test_uncalibrated_perspective_rejects_invalid_optional_camera_metadata() -> None:
    invalid_metadata = [
        {"horizontal_fov_deg": "wide"},
        {"horizontal_fov_deg": 180.0},
        {"intrinsics": {"fx": 0.0}},
        {"distortion": "radial"},
        {"intrinsics": []},
    ]
    for metadata in invalid_metadata:
        capture = project("perspective_uncalibrated", pair=True)
        capture.images[0].metadata = metadata
        capture.images[1].metadata = {}

        plan = build_plan(capture)
        calibration = next(
            stage for stage in plan.stages if stage.id == "robust-calibration"
        )

        assert plan.can_build_now is False
        assert calibration.status == "requires_input"
        assert "metadados da câmera" in plan.next_action

    valid_without_fov = project("perspective_uncalibrated", pair=True)
    valid_without_fov.images[0].metadata = {}
    valid_without_fov.images[1].metadata = {"distortion": "none"}
    assert build_plan(valid_without_fov).can_build_now is True
