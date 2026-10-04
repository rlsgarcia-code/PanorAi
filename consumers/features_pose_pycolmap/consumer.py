#!/usr/bin/env python3
"""Installed-wheel consumer for spherical features, pose, and PyCOLMAP."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np


HEIGHT = 128
WIDTH = 256
YAW_SHIFT_PX = 4


def _assert_installed_origin(package_file: str, source_root: Path) -> Path:
    origin = Path(package_file).resolve()
    try:
        origin.relative_to(source_root.resolve())
    except ValueError:
        return origin
    raise AssertionError(
        f"panorai resolved inside the source checkout: {origin} (root={source_root})"
    )


def _textured_panorama() -> np.ndarray:
    y, x = np.indices((HEIGHT, WIDTH))
    rng = np.random.default_rng(20261004)
    noise = rng.integers(0, 55, size=(HEIGHT, WIDTH), dtype=np.uint8)
    checker = (((x // 13) + (y // 11)) % 2) * 115
    rings = (
        np.sin(np.hypot(x - WIDTH * 0.37, y - HEIGHT * 0.61) / 3.7) + 1
    ) * 35
    base = np.clip(noise.astype(np.int16) + checker + rings, 0, 255).astype(
        np.uint8
    )
    return np.stack(
        (
            base,
            np.roll(base, 7, axis=1),
            np.bitwise_xor(base, ((x * 17 + y * 29) % 256).astype(np.uint8)),
        ),
        axis=-1,
    )


def _verify_database(pycolmap, database_path, result, rigs, feature_sets, matches):
    with pycolmap.Database.open(database_path) as database:
        assert database.num_cameras() == 12
        assert database.num_rigs() == 2
        assert database.num_frames() == 2
        assert database.num_images() == 12
        assert database.num_keypoints() == sum(len(item) for item in feature_sets)
        assert database.num_descriptors() == sum(len(item) for item in feature_sets)
        assert database.num_matches() == int(matches.valid.sum())

        stored_rigs = database.read_all_rigs()
        assert len(stored_rigs) == 2
        checked_rows = 0
        for image_key, image_id in result.image_ids.items():
            panorama_id, face_id = image_key.split("/", 1)
            index = 0 if panorama_id == "left" else 1
            features = feature_sets[index]
            rig = rigs[index]
            camera = next(item for item in rig.cameras if item.face_id == face_id)
            rows = result.feature_rows[image_key]
            keypoints = database.read_keypoints(image_id)
            descriptors = database.read_descriptors(image_id).data
            assert keypoints.shape == (len(rows), 2)
            assert descriptors.shape == (len(rows), 128)
            assert np.array_equal(
                descriptors,
                np.rint(features.descriptors[list(rows)]).astype(np.uint8),
            )
            if len(rows) == 0:
                continue
            expected_keypoints = features.pixels_xy[list(rows)] + 0.5
            assert np.allclose(keypoints, expected_keypoints, atol=1e-6)
            stored_K = database.read_camera(
                result.camera_ids[image_key]
            ).calibration_matrix()
            expected_K = camera.K.copy()
            expected_K[0, 2] += 0.5
            expected_K[1, 2] += 0.5
            assert np.allclose(stored_K, expected_K, atol=1e-12)
            local = np.concatenate(
                (
                    (keypoints[:, :1] - stored_K[0, 2]) / stored_K[0, 0],
                    (keypoints[:, 1:2] - stored_K[1, 2]) / stored_K[1, 1],
                    np.ones((len(rows), 1)),
                ),
                axis=1,
            )
            reconstructed = local @ camera.R_panorama_from_face.T
            reconstructed /= np.linalg.norm(reconstructed, axis=1, keepdims=True)
            assert np.allclose(
                reconstructed,
                features.bearings[list(rows)],
                atol=1e-6,
            )
            checked_rows += len(rows)
        assert checked_rows > 0
    return checked_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()

    import cv2
    import panorai
    import pycolmap
    from panorai.estimators import RelativePoseOptions, SphericalRelativePoseEstimator
    from panorai.features import SphericalFeaturePipeline

    origin = _assert_installed_origin(panorai.__file__, args.source_root)
    left = _textured_panorama()
    right = np.roll(left, YAW_SHIFT_PX, axis=1)
    pipeline = SphericalFeaturePipeline.from_preset(
        "sift-bf",
        face_sampler="cube",
        face_shape_hw=96,
        face_fov_deg=100.0,
        edge_margin_px=4,
        max_features=220,
    )
    pipeline_description = pipeline.describe()
    assert pipeline_description["interface"] == "panorai-spherical-features/v1"
    assert pipeline_description["stability"] == "stable"
    assert pipeline_description["experimental_extensions"] == [
        "build_virtual_camera_rig",
        "export_pycolmap",
    ]
    features_left = pipeline.extract(left, panorama_id="left")
    features_right = pipeline.extract(right, panorama_id="right")
    matches = pipeline.match(features_left, features_right)
    correspondences = matches.to_bearing_correspondences()
    assert len(features_left) > 0 and len(features_right) > 0
    assert len(matches) >= 30
    assert features_left.describe()["stability"] == "stable"
    assert matches.describe()["stability"] == "stable"
    assert correspondences.bearings_a.shape == (len(matches), 3)
    assert np.allclose(
        np.linalg.norm(correspondences.bearings_a, axis=1), 1.0, atol=1e-6
    )
    assert np.allclose(
        np.linalg.norm(correspondences.bearings_b, axis=1), 1.0, atol=1e-6
    )
    assert np.array_equal(
        correspondences.weights,
        correspondences.valid.astype(np.float32),
    )

    estimator = SphericalRelativePoseEstimator(
        RelativePoseOptions(
            max_angular_error_deg=0.5,
            min_num_trials=8,
            max_num_trials=80,
            min_inliers=10,
            local_optimization_steps=2,
            minimal_solver_starts=12,
            random_seed=7,
        )
    )
    pose = estimator.estimate(correspondences)
    assert pose is not None
    assert pose.describe()["interface"] == "panorai-spherical-relative-pose/v1"
    assert pose.num_inliers >= 20
    assert not pose.quality_report.accepted
    assert pose.quality_report.model_competition.preferred_model == "rotation-only"
    assert "competing-model:rotation-only" in pose.quality_report.rejection_reasons

    rigs = (
        pipeline.build_virtual_camera_rig(left, panorama_id="left"),
        pipeline.build_virtual_camera_rig(right, panorama_id="right"),
    )
    database_path = Path.cwd() / "features.db"
    export = pipeline.export_pycolmap(
        rig=rigs,
        features=(features_left, features_right),
        matches=matches,
        output_database=database_path,
    )
    assert export.interface == "panorai-pycolmap-export/v1"
    checked_rows = _verify_database(
        pycolmap,
        database_path,
        export,
        rigs,
        (features_left, features_right),
        matches,
    )

    report = {
        "status": "ok",
        "consumer": "features-pose-pycolmap",
        "origin": str(origin),
        "panorai_version": version("panorai"),
        "opencv_version": cv2.__version__,
        "pycolmap_version": pycolmap.__version__,
        "features": [len(features_left), len(features_right)],
        "matches": len(matches),
        "pose_inliers": pose.num_inliers,
        "pose_accepted": pose.quality_report.accepted,
        "pose_rejections": list(pose.quality_report.rejection_reasons),
        "database_cameras": len(export.camera_ids),
        "database_images": len(export.image_ids),
        "verified_feature_rows": checked_rows,
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
