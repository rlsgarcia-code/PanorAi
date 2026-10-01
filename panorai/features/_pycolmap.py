"""Optional PyCOLMAP database export for PanorAi virtual-camera rigs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ._models import (
    GnomonicRig,
    PyCOLMAPExportResult,
    SphericalFeatureMatches,
    SphericalFeatureSet,
)


def export_pycolmap(
    *,
    rig: GnomonicRig | Sequence[GnomonicRig],
    features: SphericalFeatureSet | Sequence[SphericalFeatureSet],
    output_database: str | Path,
    matches: SphericalFeatureMatches | Sequence[SphericalFeatureMatches] | None = None,
) -> PyCOLMAPExportResult:
    """Write cameras, rigs, keypoints, descriptors, and optional matches.

    PyCOLMAP remains the owner of geometric verification, tracks,
    triangulation, registration, and bundle adjustment. This function only
    materializes PanorAi evidence in its database schema.
    """

    try:
        import pycolmap
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "PyCOLMAP export is optional; install PanorAi with "
            "`pip install panorai[pycolmap]`."
        ) from exc
    rigs = [rig] if isinstance(rig, GnomonicRig) else list(rig)
    feature_sets = (
        [features] if isinstance(features, SphericalFeatureSet) else list(features)
    )
    match_sets: list[SphericalFeatureMatches] = []
    if matches is not None:
        match_sets = (
            [matches] if isinstance(matches, SphericalFeatureMatches) else list(matches)
        )
    if not rigs or len(rigs) != len(feature_sets):
        raise ValueError(
            "rig and features must contain the same non-zero number of panoramas"
        )
    by_panorama = {item.panorama_id: item for item in rigs}
    if len(by_panorama) != len(rigs):
        raise ValueError("rig panorama_id values must be unique")
    for feature_set in feature_sets:
        if feature_set.panorama_id not in by_panorama:
            raise ValueError("each feature set must have a matching rig panorama_id")

    database_path = Path(output_database)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        raise FileExistsError(
            f"refusing to modify existing COLMAP database: {database_path}"
        )
    camera_ids: dict[str, int] = {}
    image_ids: dict[str, int] = {}
    feature_rows: dict[str, np.ndarray] = {}
    panorama_rig_rotations: dict[str, np.ndarray] = {}
    local_rows: dict[tuple[str, int], tuple[int, int]] = {}
    next_camera_id = 1
    next_image_id = 1
    next_rig_id = 1
    next_frame_id = 1

    with pycolmap.Database.open(database_path) as database:
        for feature_set in feature_sets:
            panorama_rig = by_panorama[feature_set.panorama_id]
            cameras_by_face = {
                camera.face_id: camera for camera in panorama_rig.cameras
            }
            unknown_faces = sorted(set(feature_set.face_ids) - set(cameras_by_face))
            if unknown_faces:
                raise ValueError(
                    f"features reference faces absent from rig: {unknown_faces}"
                )
            reference = panorama_rig.cameras[0]
            panorama_rig_rotations[panorama_rig.panorama_id] = np.asarray(
                reference.R_panorama_from_face, dtype=np.float64
            )
            colmap_rig = pycolmap.Rig(rig_id=next_rig_id)
            sensor_ids: dict[str, Any] = {}
            face_image_ids: dict[str, int] = {}

            for camera in panorama_rig.cameras:
                camera_id = next_camera_id
                next_camera_id += 1
                camera_key = f"{panorama_rig.panorama_id}/{camera.face_id}"
                camera_ids[camera_key] = camera_id
                K = np.asarray(camera.K, dtype=np.float64)
                colmap_camera = pycolmap.Camera(
                    camera_id=camera_id,
                    model="PINHOLE",
                    width=int(camera.width),
                    height=int(camera.height),
                    params=np.asarray((K[0, 0], K[1, 1], K[0, 2], K[1, 2])),
                    has_prior_focal_length=True,
                )
                database.write_camera(colmap_camera, use_camera_id=True)
                sensor = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera_id)
                sensor_ids[camera.face_id] = sensor
                if camera.face_id == reference.face_id:
                    colmap_rig.add_ref_sensor(sensor)
                else:
                    face_from_reference = np.asarray(
                        camera.R_panorama_from_face, dtype=np.float64
                    ).T @ np.asarray(reference.R_panorama_from_face, dtype=np.float64)
                    transform = np.concatenate(
                        (face_from_reference, np.zeros((3, 1), dtype=np.float64)),
                        axis=1,
                    )
                    colmap_rig.add_sensor(sensor, pycolmap.Rigid3d(transform))
            database.write_rig(colmap_rig, use_rig_id=True)

            frame = pycolmap.Frame(frame_id=next_frame_id, rig_id=next_rig_id)
            for camera in panorama_rig.cameras:
                image_id = next_image_id
                next_image_id += 1
                face_image_ids[camera.face_id] = image_id
                image_key = f"{panorama_rig.panorama_id}/{camera.face_id}"
                image_ids[image_key] = image_id
                frame.add_data_id(pycolmap.data_t(sensor_ids[camera.face_id], image_id))
            database.write_frame(frame, use_frame_id=True)

            grouped: dict[str, list[int]] = defaultdict(list)
            for feature_index, feature in enumerate(feature_set.features):
                grouped[feature.face_id].append(feature_index)
            for camera in panorama_rig.cameras:
                rows = np.asarray(grouped.get(camera.face_id, []), dtype=np.int64)
                image_key = f"{panorama_rig.panorama_id}/{camera.face_id}"
                image_id = face_image_ids[camera.face_id]
                image = pycolmap.Image(
                    name=f"{image_key}.png",
                    camera_id=camera_ids[image_key],
                    image_id=image_id,
                    frame_id=next_frame_id,
                )
                database.write_image(image, use_image_id=True)
                keypoints = (
                    np.stack(
                        [feature_set.features[index].pixel_xy for index in rows]
                    ).astype(np.float32)
                    if len(rows)
                    else np.empty((0, 2), dtype=np.float32)
                )
                descriptors = feature_set.descriptors[rows]
                database.write_keypoints(image_id, keypoints)
                database.write_descriptors(
                    image_id,
                    _pycolmap_descriptors(pycolmap, feature_set, descriptors),
                )
                feature_rows[image_key] = rows
                for local_index, feature_index in enumerate(rows):
                    local_rows[(feature_set.panorama_id, int(feature_index))] = (
                        image_id,
                        local_index,
                    )
            next_rig_id += 1
            next_frame_id += 1

        pairs: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for match_set in match_sets:
            for left, right, valid in zip(
                match_set.feature_indices_a,
                match_set.feature_indices_b,
                match_set.valid,
            ):
                if not valid:
                    continue
                left_row = local_rows.get((match_set.panorama_id_a, int(left)))
                right_row = local_rows.get((match_set.panorama_id_b, int(right)))
                if left_row is None or right_row is None:
                    raise ValueError(
                        "matches reference feature sets absent from export"
                    )
                image_a, row_a = left_row
                image_b, row_b = right_row
                if image_a < image_b:
                    pairs[(image_a, image_b)].append((row_a, row_b))
                else:
                    pairs[(image_b, image_a)].append((row_b, row_a))
        for (image_a, image_b), rows in pairs.items():
            database.write_matches(
                image_a,
                image_b,
                np.asarray(sorted(set(rows)), dtype=np.uint32),
            )

    return PyCOLMAPExportResult(
        database_path=str(database_path),
        camera_ids=camera_ids,
        image_ids=image_ids,
        feature_rows=feature_rows,
        panorama_rig_rotations=panorama_rig_rotations,
    )


def _pycolmap_descriptors(
    pycolmap: Any, feature_set: SphericalFeatureSet, descriptors: np.ndarray
):
    extractor_type = (
        pycolmap.FeatureExtractorType.SIFT
        if feature_set.extractor_name == "sift"
        else pycolmap.FeatureExtractorType.UNDEFINED
    )
    if np.issubdtype(descriptors.dtype, np.floating):
        floating = pycolmap.FeatureDescriptorsFloat(
            type=extractor_type,
            data=np.asarray(descriptors, dtype=np.float32),
        )
        return floating.to_bytes()
    return pycolmap.FeatureDescriptors(
        type=extractor_type,
        data=np.asarray(descriptors, dtype=np.uint8),
    )
