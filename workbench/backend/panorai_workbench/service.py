"""PanorAi-backed pair analysis with JSON-safe diagnostics."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import math
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def capabilities() -> dict[str, Any]:
    import panorai
    from .catalog import experiment_catalog
    from panorai.features import available_presets

    optional: dict[str, bool] = {}
    for name, module in {
        "features": "panorai.features",
        "relative_pose": "panorai.estimators",
        "object_hypotheses": "panorai.object_localization",
        "segmentation": "panorai.experimental.deep_learning.segmentation",
    }.items():
        try:
            __import__(module)
            optional[name] = True
        except (ImportError, ModuleNotFoundError):
            optional[name] = False
    catalog = experiment_catalog()
    return {
        "backend": "panorai-workbench/v0",
        "panorai_version": panorai.__version__,
        "runtime": catalog["runtime"],
        "catalog": catalog["catalog"],
        "api_families": catalog["families"],
        "catalog_counts": catalog["counts"],
        "feature_presets": list(available_presets()),
        "capabilities": optional,
        "stability": {
            "features": "stable PanorAi surface",
            "relative_pose": "experimental PanorAi estimator",
            "object_hypotheses": "experimental PanorAi workflow",
            "segmentation": "experimental optional workflow",
        },
    }


def single_view_academy_status() -> dict[str, Any]:
    """Describe the exact PanorAi surfaces available to the academy workspace.

    Missing estimators stay explicit.  The Workbench does not replace a public
    PanorAi capability with OpenCV or a private implementation.
    """

    import panorai
    from panorai.geometry import erp_pixels_to_rays, rays_to_erp_pixels

    return {
        "schema": "panorai-workbench-academy-capabilities/v1",
        "case": "single-view-geometry",
        "panorai_version": panorai.__version__,
        "available": [
            {
                "id": "erp-pixel-to-bearing",
                "public_api": f"{erp_pixels_to_rays.__module__.replace('._engine', '')}.erp_pixels_to_rays",
                "stability": "stable",
            },
            {
                "id": "bearing-to-erp-pixel",
                "public_api": f"{rays_to_erp_pixels.__module__.replace('._engine', '')}.rays_to_erp_pixels",
                "stability": "stable",
            },
        ],
        "requires_panorai_update": [
            {
                "id": "great-circle-line-fit",
                "reason": "no public robust spherical line/great-circle fitting API",
                "owner": "PanorAI library agent",
            },
            {
                "id": "vanishing-direction-estimator",
                "reason": "no public single-panorama vanishing-direction estimator",
                "owner": "PanorAI library agent",
            },
            {
                "id": "single-view-calibration",
                "reason": "no public calibration-from-orthogonal-directions application",
                "owner": "PanorAI library agent",
            },
        ],
        "policy": (
            "Backend computer-vision primitives come only from documented PanorAi Main APIs. "
            "The browser may visualize manual constructions for learning, but they are not "
            "reported as PanorAi estimators."
        ),
    }


def lift_erp_points(
    points_xy: Any,
    image_shape_hw: tuple[int, int],
) -> dict[str, Any]:
    """Lift selected ERP pixels through PanorAi's canonical public geometry API."""

    from panorai.geometry import erp_pixels_to_rays

    height, width = image_shape_hw
    if height <= 0 or width <= 0:
        raise ValueError("image_shape_hw must contain positive height and width")
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape (N, 2)")
    if len(points) > 128:
        raise ValueError("at most 128 points may be lifted per request")
    if not np.isfinite(points).all():
        raise ValueError("points_xy must contain only finite coordinates")
    rays = erp_pixels_to_rays(points, (height, width))
    return {
        "schema": "panorai-workbench-erp-rays/v1",
        "execution": {
            "api_level": "geometry",
            "public_api": "panorai.geometry.erp_pixels_to_rays",
            "coordinate_contract": "geometry-v1 pixel centres, top-left origin",
        },
        "image_shape_hw": [height, width],
        "points_xy": points.tolist(),
        "rays_xyz": np.asarray(rays, dtype=np.float64).tolist(),
    }


def _feature_payload(feature: Any) -> dict[str, Any]:
    source = None
    if feature.source_erp_xy is not None:
        value = np.asarray(feature.source_erp_xy, dtype=np.float64)
        if value.shape == (2,) and np.isfinite(value).all():
            source = value.tolist()
    return {
        "id": feature.feature_id,
        "face_id": feature.face_id,
        "erp_xy": source,
        "bearing_xyz": np.asarray(feature.bearing_xyz, dtype=np.float64).tolist(),
        "response": float(feature.response),
        "scale": float(feature.scale),
        "angle_deg": float(feature.angle_deg),
        "octave": int(feature.octave),
        "valid": bool(feature.valid),
    }


def _epipolar_normal(
    essential: np.ndarray, bearing_a: np.ndarray
) -> list[float] | None:
    normal = essential @ bearing_a
    norm = float(np.linalg.norm(normal))
    if not math.isfinite(norm) or norm <= 1e-12:
        return None
    return (normal / norm).tolist()


def analyze_pair(
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    panorama_id_a: str = "view-a",
    panorama_id_b: str = "view-b",
    profile: str = "relative-pose-reference",
    max_angular_error_deg: float = 0.5,
    random_seed: int = 7,
) -> dict[str, Any]:
    """Run feature extraction, matching and relative pose on two ERP arrays.

    The response deliberately exposes intermediate evidence. A returned pose is
    not presented as accepted unless PanorAi's quality report accepts it.
    """

    from panorai.applications import SphericalRelativePose
    from panorai.estimators import RelativePoseOptions
    from panorai.features import SphericalFeaturePipeline

    if image_a.ndim != 3 or image_a.shape[2] not in {3, 4}:
        raise ValueError("image_a must be an HWC RGB/RGBA array")
    if image_b.ndim != 3 or image_b.shape[2] not in {3, 4}:
        raise ValueError("image_b must be an HWC RGB/RGBA array")
    if profile == "relative-pose-reference":
        pipeline = SphericalFeaturePipeline.for_relative_pose()
    elif profile in {"sift-flann", "sift-bf", "orb-hamming", "akaze-hamming"}:
        pipeline = SphericalFeaturePipeline.from_preset(profile)
    else:
        raise ValueError(f"unsupported feature profile: {profile}")

    features_a = pipeline.extract(image_a, panorama_id=panorama_id_a)
    features_b = pipeline.extract(image_b, panorama_id=panorama_id_b)
    matches = pipeline.match(features_a, features_b)
    application = SphericalRelativePose(
        frontend_config=pipeline.config,
        options=RelativePoseOptions(
            max_angular_error_deg=max_angular_error_deg,
            random_seed=random_seed,
        ),
    )
    pose = application.estimate_matches(matches)

    essential = None if pose is None else np.asarray(pose.essential_matrix)
    rows: list[dict[str, Any]] = []
    for index in range(len(matches)):
        residual_deg = None
        inlier = False
        epipolar_normal_b = None
        if pose is not None:
            residual_deg = math.degrees(float(pose.residuals_rad[index]))
            inlier = bool(pose.inlier_mask[index])
            epipolar_normal_b = _epipolar_normal(essential, matches.bearings_a[index])
        rows.append(
            {
                "index": index,
                "feature_index_a": int(matches.feature_indices_a[index]),
                "feature_index_b": int(matches.feature_indices_b[index]),
                "bearing_a": matches.bearings_a[index].tolist(),
                "bearing_b": matches.bearings_b[index].tolist(),
                "descriptor_distance": float(matches.descriptor_distances[index]),
                "ratio_score": (
                    None
                    if matches.ratio_scores is None
                    else float(matches.ratio_scores[index])
                ),
                "mutual": None
                if matches.mutual is None
                else bool(matches.mutual[index]),
                "valid": bool(matches.valid[index]),
                "inlier": inlier,
                "residual_deg": residual_deg,
                "epipolar_plane_normal_b": epipolar_normal_b,
            }
        )

    pose_payload = None
    if pose is not None:
        pose_payload = {
            "rotation_b_from_a": pose.R.tolist(),
            "translation_direction_b": pose.t.tolist(),
            "essential_matrix": pose.essential_matrix.tolist(),
            "accepted": bool(pose.quality_report.accepted),
            "rejection_reasons": list(pose.quality_report.rejection_reasons),
            "num_inliers": int(pose.num_inliers),
            "num_trials": int(pose.num_trials),
            "median_parallax_deg": float(pose.median_parallax_deg),
            "cheirality_ratio": float(pose.cheirality_ratio),
            "degenerate": bool(pose.degenerate),
            "degeneracy_reasons": list(pose.degeneracy_reasons),
            "diagnostics": _jsonable(pose.describe()),
        }

    return {
        "schema": "panorai-workbench-pair-analysis/v0",
        "profile": profile,
        "execution": {
            "api_level": "applications",
            "public_api": application.application_id,
            "application": _jsonable(application.describe()),
        },
        "images": {
            "a": {"id": panorama_id_a, "shape_hwc": list(image_a.shape)},
            "b": {"id": panorama_id_b, "shape_hwc": list(image_b.shape)},
        },
        "features": {
            "a": [_feature_payload(feature) for feature in features_a.features],
            "b": [_feature_payload(feature) for feature in features_b.features],
            "diagnostics_a": _jsonable(features_a.describe()),
            "diagnostics_b": _jsonable(features_b.describe()),
        },
        "matches": rows,
        "match_diagnostics": _jsonable(matches.describe()),
        "pose": pose_payload,
        "interpretation": {
            "translation": "unit direction only; metric scale is unobservable from this pair",
            "without_triangulation": "show epipolar great circle and angular residual, not a unique reprojection point",
            "acceptance": "pose exists and pose.accepted are distinct states",
        },
    }
