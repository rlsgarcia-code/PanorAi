#!/usr/bin/env python3
"""Validate CAM-guided hypotheses with fixed-geometry semantic holdouts.

The benchmark keeps registration fixed and removes one view's semantic field
at a time. The held-out CAM is opened only for scoring projected hypotheses.
Stanford2D3D remains external to the repository.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "4")

from panorai.geometry import (  # noqa: E402
    GnomonicSpec,
    erp_pixels_to_rays,
    gnomonic_rotation,
    gnomonic_pixels_to_rays,
    rays_to_erp_pixels,
    rays_to_gnomonic_pixels,
)
from panorai.object_localization import (  # noqa: E402
    HypothesisGnomonicRegion,
    MultiviewObjectHypothesis,
    RegisteredBearingObservation,
    RegisteredBearingTrack,
    RegisteredPoseInconsistencyError,
    RegisteredRadialDepth,
    RegisteredSemanticEvidence,
    SemanticConcept,
    SphericalObjectHypothesisConfig,
    SphericalObjectHypothesisGenerator,
    SphericalObjectHypothesisMethod,
    SphericalObjectHypothesisMethodConfig,
    hypothesis_view_consistency,
)
from panorai.reconstruction import (  # noqa: E402
    SphericalCameraPose,
    SphericalGlobalMapper,
    SphericalGlobalMapperOptions,
    SphericalReconstructionDiagnostics,
    SphericalReconstructionResult,
    SphericalTrack,
    SphericalTrackObservation,
)


SCHEMA = "panorai-multiview-cam-hypothesis-loo/v7"
WINDOW_METRIC_SHAPE_HW = (128, 256)
STANFORD_ROOT = Path("data/stanford2d3d")
STANFORD_VIEW_NAMES = (
    "camera_a81c748601ef4e5db4b74c4494ee61bc_office_3_frame_equirectangular_domain",
    "camera_aa113cffd2b046bb8a224f9b1bdc2e38_office_3_frame_equirectangular_domain",
    "camera_72d012b85380419b90251a8ee076daca_office_3_frame_equirectangular_domain",
    "camera_fc0fdf04f4ca49c680797066b95e8f7b_office_3_frame_equirectangular_domain",
)
STANFORD_PANORAI_FROM_DATASET = np.diag((1.0, -1.0, 1.0))
SHAPE_HW = (512, 1024)
STANFORD_MANUAL_REGIONS = {
    f"stanford2d3d::area_2::{STANFORD_VIEW_NAMES[0]}": {
        "desk": ((0, 540, 760, 875), (850, 600, 1380, 930)),
        "bookcase": ((1740, 300, 2005, 860),),
    },
    f"stanford2d3d::area_2::{STANFORD_VIEW_NAMES[1]}": {
        "desk": ((1240, 535, 1775, 850), (1800, 570, 2048, 960), (0, 570, 190, 960)),
        "bookcase": ((1090, 250, 1285, 880),),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _angular_error(first: np.ndarray, second: np.ndarray) -> float:
    return math.acos(
        float(
            np.clip(
                np.dot(first, second)
                / (np.linalg.norm(first) * np.linalg.norm(second)),
                -1.0,
                1.0,
            )
        )
    )


def _nearest_rotation(value: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(np.asarray(value, dtype=np.float64))
    sign = np.linalg.det(left @ right)
    return left @ np.diag((1.0, 1.0, sign)) @ right


def _feature_pipeline():
    from panorai.features import SphericalFeaturePipeline

    return SphericalFeaturePipeline.from_preset(
        "sift-flann",
        face_sampler="cube",
        face_shape_hw=512,
        face_fov_deg=95.0,
        max_features=3000,
        ratio_test=0.72,
        angular_dedup_threshold_deg=0.15,
        edge_margin_px=12,
        validity_scale_margin=1.5,
    )


def build_reference_bearing_tracks(
    *,
    images: dict[str, np.ndarray],
    supports: dict[str, np.ndarray],
    reference_id: str,
) -> tuple[tuple[RegisteredBearingTrack, ...], dict[str, Any]]:
    """Build point-free reference tracks from descriptor-valid matches.

    No depth, scanner XYZ, camera pose, or latent point participates in track
    construction. The downstream multiray gate alone decides whether the
    supplied registered poses are consistent with these image observations.
    """

    pipeline = _feature_pipeline()
    features = {
        view_id: pipeline.extract(
            image,
            panorama_id=view_id,
            validity_mask=supports[view_id],
        )
        for view_id, image in images.items()
    }
    reference = features[reference_id]
    accepted: dict[int, dict[str, tuple[int, np.ndarray, float]]] = defaultdict(dict)
    valid_match_counts: dict[str, int] = {}
    retained_match_counts: dict[str, int] = {}
    for target_id in sorted(set(images) - {reference_id}):
        matches = pipeline.match(reference, features[target_id])
        valid_rows = np.flatnonzero(matches.valid)
        valid_match_counts[target_id] = int(len(valid_rows))
        used_reference: set[int] = set()
        used_target: set[int] = set()
        retained = 0
        for row in sorted(
            valid_rows,
            key=lambda index: (
                float(matches.descriptor_distances[index]),
                int(matches.feature_indices_a[index]),
                int(matches.feature_indices_b[index]),
            ),
        ):
            reference_index = int(matches.feature_indices_a[row])
            target_index = int(matches.feature_indices_b[row])
            if reference_index in used_reference or target_index in used_target:
                continue
            accepted[reference_index][target_id] = (
                target_index,
                np.asarray(matches.bearings_b[row], dtype=np.float64),
                float(matches.descriptor_distances[row]),
            )
            used_reference.add(reference_index)
            used_target.add(target_index)
            retained += 1
        retained_match_counts[target_id] = retained

    tracks = []
    for reference_index, targets in sorted(accepted.items()):
        observations = [
            RegisteredBearingObservation(
                reference_id,
                reference_index,
                np.asarray(reference.bearings[reference_index], dtype=np.float64),
            )
        ]
        observations.extend(
            RegisteredBearingObservation(
                target_id,
                targets[target_id][0],
                targets[target_id][1],
            )
            for target_id in sorted(targets)
        )
        tracks.append(
            RegisteredBearingTrack(
                f"descriptor-track-{reference_index:06d}", tuple(observations)
            )
        )
    if not tracks:
        raise RuntimeError("descriptor matching produced no point-free tracks")
    return tuple(tracks), {
        "method": "reference-descriptor-bearing-tracks-no-input-points",
        "reference_id": reference_id,
        "feature_counts": {key: len(value) for key, value in features.items()},
        "descriptor_valid_match_counts": valid_match_counts,
        "retained_one_to_one_match_counts": retained_match_counts,
        "bearing_track_count": len(tracks),
        "track_length_histogram": {
            str(length): sum(len(track.observations) == length for track in tracks)
            for length in sorted({len(track.observations) for track in tracks})
        },
        "uses_input_point_xyz": False,
        "uses_depth_or_scanner_xyz": False,
        "uses_registered_pose_during_matching": False,
    }


def build_mapper_refined_reconstruction(
    *,
    images: dict[str, np.ndarray],
    supports: dict[str, np.ndarray],
    reference_id: str,
) -> tuple[SphericalReconstructionResult, dict[str, Any]]:
    """Run PanorAi reconstruction and expose only its post-refinement result."""

    pipeline = _feature_pipeline()
    ordered_ids = tuple(sorted(images))
    features = {
        view_id: pipeline.extract(
            images[view_id],
            panorama_id=view_id,
            validity_mask=supports[view_id],
        )
        for view_id in ordered_ids
    }
    pairwise_matches = tuple(
        pipeline.match(features[first], features[second])
        for first, second in combinations(ordered_ids, 2)
    )
    mapper = SphericalGlobalMapper()
    reconstruction = mapper.reconstruct(
        matches=pairwise_matches,
        reference_id=reference_id,
    )
    stage_messages = tuple(reconstruction.diagnostics.stage_messages)
    refined = (
        reconstruction.success
        and "filter-retriangulate-final-refinement" in stage_messages
    )
    if reconstruction.success and not refined:
        raise RuntimeError(
            "successful mapper result omitted final sparse refinement provenance"
        )
    active_observations = sum(
        observation.active
        for track in reconstruction.tracks
        if track.valid
        for observation in track.observations
    )
    inactive_observations = sum(
        not observation.active
        for track in reconstruction.tracks
        for observation in track.observations
    )
    return reconstruction, {
        "method": "panorai-spherical-global-mapper-post-refinement",
        "reference_id": reference_id,
        "feature_counts": {key: len(value) for key, value in features.items()},
        "pair_count": len(pairwise_matches),
        "pairwise_match_counts": {
            f"{item.panorama_id_a}::{item.panorama_id_b}": len(item)
            for item in pairwise_matches
        },
        "pairwise_valid_match_counts": {
            f"{item.panorama_id_a}::{item.panorama_id_b}": int(item.valid.sum())
            for item in pairwise_matches
        },
        "success": reconstruction.success,
        "failure_reasons": list(reconstruction.failure_reasons),
        "stage_messages": list(stage_messages),
        "final_refinement_present": refined,
        "refined_track_count": len(reconstruction.tracks),
        "valid_track_count": sum(track.valid for track in reconstruction.tracks),
        "active_observation_count": active_observations,
        "inactive_observation_count": inactive_observations,
        "reprojection_median_deg": (reconstruction.diagnostics.reprojection_median_deg),
        "reprojection_p90_deg": reconstruction.diagnostics.reprojection_p90_deg,
        "reprojection_max_deg": reconstruction.diagnostics.reprojection_max_deg,
        "multiview_corroboration_passed": (
            reconstruction.diagnostics.multiview_corroboration_passed
        ),
        "scale": reconstruction.scale,
        "uses_depth_or_scanner_xyz": False,
        "uses_external_registered_pose": False,
        "point_provenance": "mapper-filter-retriangulate-final-refinement",
    }


def build_reference_registered_reconstruction(
    *,
    images: dict[str, np.ndarray],
    supports: dict[str, np.ndarray],
    poses: dict[str, SphericalCameraPose],
    reference_id: str,
    point_from_reference: Callable[[int, Any], np.ndarray | None],
    maximum_residual_deg: float = 1.5,
) -> tuple[SphericalReconstructionResult, dict[str, Any]]:
    """Create reference-registered tracks from real descriptor matches.

    The point oracle is independent of descriptor and CAM values. Matches only
    propose which target feature observes the reference point; registered-pose
    reprojection performs the final admission.
    """

    pipeline = _feature_pipeline()
    features = {
        view_id: pipeline.extract(
            image,
            panorama_id=view_id,
            validity_mask=supports[view_id],
        )
        for view_id, image in images.items()
    }
    reference = features[reference_id]
    accepted: dict[int, dict[str, tuple[int, np.ndarray, float]]] = defaultdict(dict)
    match_counts: dict[str, int] = {}
    for target_id in sorted(set(images) - {reference_id}):
        matches = pipeline.match(reference, features[target_id])
        match_counts[target_id] = int(np.asarray(matches.valid).sum())
        order = np.argsort(matches.descriptor_distances)
        for row in order:
            if not matches.valid[row]:
                continue
            reference_index = int(matches.feature_indices_a[row])
            if target_id in accepted[reference_index]:
                continue
            point = point_from_reference(reference_index, reference)
            if point is None:
                continue
            pose = poses[target_id]
            predicted = pose.R @ (point - pose.center)
            if np.linalg.norm(predicted) <= 1e-9:
                continue
            predicted /= np.linalg.norm(predicted)
            measured = np.asarray(matches.bearings_b[row], dtype=np.float64)
            residual = _angular_error(predicted, measured)
            if math.degrees(residual) <= maximum_residual_deg:
                accepted[reference_index][target_id] = (
                    int(matches.feature_indices_b[row]),
                    measured,
                    residual,
                )

    tracks: list[SphericalTrack] = []
    for reference_index, targets in sorted(accepted.items()):
        if not targets:
            continue
        point = point_from_reference(reference_index, reference)
        if point is None:
            continue
        reference_bearing = np.asarray(reference.bearings[reference_index])
        reference_pose = poses[reference_id]
        predicted_reference = reference_pose.R @ (point - reference_pose.center)
        reference_residual = _angular_error(reference_bearing, predicted_reference)
        observations = [
            SphericalTrackObservation(
                reference_id,
                reference_index,
                reference_bearing,
                reference_residual,
            )
        ]
        observations.extend(
            SphericalTrackObservation(target_id, *targets[target_id])
            for target_id in sorted(targets)
        )
        tracks.append(
            SphericalTrack(
                f"reference-track-{reference_index:06d}",
                point,
                tuple(observations),
            )
        )
    if not tracks:
        raise RuntimeError("reference registration produced no tracks")
    reconstruction = SphericalReconstructionResult(
        success=True,
        failure_reasons=(),
        poses=tuple(poses[key] for key in sorted(poses)),
        tracks=tuple(tracks),
        pairwise_edges=(),
        admitted_edge_indices=(),
        reference_panorama_id=reference_id,
        diagnostics=SphericalReconstructionDiagnostics(
            candidate_match_count=sum(match_counts.values()),
            track_count=len(tracks),
            reprojection_median_deg=float(
                np.median(
                    [
                        math.degrees(item.residual_rad)
                        for track in tracks
                        for item in track.observations
                    ]
                )
            ),
        ),
        options=SphericalGlobalMapperOptions(),
        scale="metric-reference-registration",
        _points_xyz=np.stack([item.point_xyz for item in tracks]),
    )
    diagnostics = {
        "feature_counts": {key: len(value) for key, value in features.items()},
        "valid_match_counts_from_reference": match_counts,
        "track_count": len(tracks),
        "track_length_histogram": {
            str(length): sum(len(track.observations) == length for track in tracks)
            for length in sorted({len(track.observations) for track in tracks})
        },
    }
    return reconstruction, diagnostics


def _normalize_channels(values: np.ndarray, support: np.ndarray) -> np.ndarray:
    source = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    output = np.zeros_like(source, dtype=np.float32)
    for index, channel in enumerate(source):
        selected = channel[support]
        if selected.size == 0:
            continue
        low = float(selected.min())
        span = float(selected.max() - low)
        if span > np.finfo(np.float64).eps:
            output[index, support] = ((selected - low) / span).astype(np.float32)
    return output


def _imagenet_evidence(
    images: dict[str, np.ndarray],
    supports: dict[str, np.ndarray],
    class_channels: tuple[tuple[str, int], ...] = (("desk", 526), ("bookcase", 453)),
) -> tuple[RegisteredSemanticEvidence, ...]:
    import torch

    from panorai.experimental.deep_learning import (
        ImageNetFCN,
        load_pretrained_imagenet_model,
        port_module_with_report,
        regrid_dense_lattice_to_erp,
    )

    loaded = load_pretrained_imagenet_model("resnet18", progress=False)
    model = ImageNetFCN(loaded.model, "resnet18").eval()
    report = port_module_with_report(model)
    if report.remaining_planar_spatial_layers:
        raise RuntimeError("planar layers remain in the spherical ImageNet model")
    transforms = loaded.weights.transforms()
    mean = torch.tensor(transforms.mean)[None, :, None, None]
    std = torch.tensor(transforms.std)[None, :, None, None]
    ordered = sorted(images)
    batch = (
        torch.stack(
            [torch.from_numpy(images[key].copy()).permute(2, 0, 1) for key in ordered]
        ).float()
        / 255.0
    )
    with torch.inference_mode():
        dense = model.forward_dense((batch - mean) / std)
        selected = dense.logits[:, tuple(index for _, index in class_channels)]
        aligned = regrid_dense_lattice_to_erp(
            selected, SHAPE_HW, model.lattice_geometry
        )
    values = aligned.values.cpu().numpy()
    model_support = aligned.support.cpu().numpy().astype(bool)
    result = []
    for index, view_id in enumerate(ordered):
        support = supports[view_id] & model_support
        result.append(
            RegisteredSemanticEvidence(
                view_id=view_id,
                channel_ids=tuple(name for name, _ in class_channels),
                scores=_normalize_channels(values[index], support),
                support=support,
                vocabulary="imagenet-1k",
                source_id=f"spherical-resnet18-imagenet:{view_id}",
                provenance={
                    "architecture": "resnet18",
                    "weights": str(loaded.weights),
                    "checkpoint_sha256": loaded.checkpoint.sha256,
                    "direct_spherical_erp": True,
                    "multiface": False,
                    "input_shape_hw": list(SHAPE_HW),
                    "class_indices": [index for _, index in class_channels],
                },
            )
        )
    return tuple(result)


def _sample_concept(
    evidence: RegisteredSemanticEvidence,
    concept: SemanticConcept,
    bearing: np.ndarray,
) -> tuple[bool, float]:
    pixel = np.asarray(rays_to_erp_pixels(bearing, evidence.shape_hw).pixels_xy)
    column = int(math.floor(float(pixel[0]) + 0.5)) % evidence.shape_hw[1]
    row = min(
        max(int(math.floor(float(pixel[1]) + 0.5)), 0),
        evidence.shape_hw[0] - 1,
    )
    if not evidence.support[row, column]:
        return False, 0.0
    channels = {name: index for index, name in enumerate(evidence.channel_ids)}
    score = sum(
        weight
        * (
            float(evidence.scores[channels[proxy], row, column])
            if proxy in channels
            else 0.0
        )
        for proxy, weight in concept.proxy_weights
    )
    return True, float(score)


def _concept_values(
    evidence: RegisteredSemanticEvidence, concept: SemanticConcept
) -> np.ndarray:
    channels = {name: index for index, name in enumerate(evidence.channel_ids)}
    values = np.zeros(evidence.shape_hw, dtype=np.float64)
    for proxy, weight in concept.proxy_weights:
        if proxy in channels:
            values += weight * evidence.scores[channels[proxy]]
    values[~evidence.support] = 0.0
    return values


def spherical_score_percentile(
    evidence: RegisteredSemanticEvidence,
    concept: SemanticConcept,
    score: float,
) -> float:
    """Return solid-angle mass at or below one within-class CAM score."""

    values = _concept_values(evidence, concept)
    height, width = evidence.shape_hw
    latitudes = math.pi / 2.0 - (np.arange(height) + 0.5) * math.pi / height
    weights = np.broadcast_to(np.cos(latitudes)[:, None], (height, width))
    valid_weights = weights[evidence.support]
    if valid_weights.size == 0:
        raise ValueError("semantic evidence support is empty")
    return float(
        valid_weights[values[evidence.support] <= score].sum() / valid_weights.sum()
    )


def _window_mask(spec: GnomonicSpec, shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    rays = np.asarray(
        erp_pixels_to_rays(np.column_stack((xx.ravel(), yy.ravel())), shape_hw)
    )
    return np.asarray(rays_to_gnomonic_pixels(rays, spec).valid).reshape(shape_hw)


def _sample_metric_grid(array: np.ndarray) -> np.ndarray:
    target_height, target_width = WINDOW_METRIC_SHAPE_HW
    source_height, source_width = array.shape
    rows = np.clip(
        np.rint((np.arange(target_height) + 0.5) * source_height / target_height - 0.5),
        0,
        source_height - 1,
    ).astype(np.int64)
    columns = np.mod(
        np.rint(
            (np.arange(target_width) + 0.5) * source_width / target_width - 0.5
        ).astype(np.int64),
        source_width,
    )
    return array[np.ix_(rows, columns)]


@dataclass(frozen=True, slots=True)
class _IndependentCamComponent:
    region: HypothesisGnomonicRegion
    mask: np.ndarray
    activation_mass: float


def _periodic_mask_components(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return deterministic 8-connected components with ERP seam wrapping."""

    source = np.asarray(mask, dtype=bool)
    height, width = source.shape
    visited = np.zeros_like(source)
    components: list[np.ndarray] = []
    offsets = tuple(
        (dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy != 0 or dx != 0
    )
    for start_y, start_x in np.argwhere(source):
        if visited[start_y, start_x]:
            continue
        visited[start_y, start_x] = True
        stack = [(int(start_y), int(start_x))]
        cells: list[tuple[int, int]] = []
        while stack:
            row, column = stack.pop()
            cells.append((row, column))
            for dy, dx in offsets:
                neighbour_row = row + dy
                if neighbour_row < 0 or neighbour_row >= height:
                    continue
                neighbour_column = (column + dx) % width
                if (
                    source[neighbour_row, neighbour_column]
                    and not visited[neighbour_row, neighbour_column]
                ):
                    visited[neighbour_row, neighbour_column] = True
                    stack.append((neighbour_row, neighbour_column))
        components.append(np.asarray(sorted(cells), dtype=np.int64))
    return tuple(components)


def _fit_component_region(
    *,
    view_id: str,
    cells_yx: np.ndarray,
    values: np.ndarray,
    config: SphericalObjectHypothesisConfig,
    minimum_fov_deg: float = 8.0,
    maximum_fov_deg: float = 120.0,
    margin_deg: float = 3.0,
) -> HypothesisGnomonicRegion:
    shape_hw = values.shape
    pixels = np.column_stack(
        (cells_yx[:, 1].astype(np.float64), cells_yx[:, 0].astype(np.float64))
    )
    bearings = np.asarray(erp_pixels_to_rays(pixels, shape_hw), dtype=np.float64)
    latitude_weights = np.cos(
        math.pi / 2.0
        - (cells_yx[:, 0].astype(np.float64) + 0.5) * math.pi / shape_hw[0]
    )
    weights = values[cells_yx[:, 0], cells_yx[:, 1]] * latitude_weights
    if float(weights.sum()) <= 1e-12:
        weights = np.ones(len(bearings), dtype=np.float64)
    weights /= weights.sum()
    vector = np.sum(bearings * weights[:, None], axis=0)
    if float(np.linalg.norm(vector)) <= 1e-12:
        vector = bearings[int(np.argmax(weights))]
    center = vector / np.linalg.norm(vector)
    longitude = math.degrees(math.atan2(float(center[0]), float(center[2])))
    latitude = math.degrees(math.asin(float(np.clip(center[1], -1.0, 1.0))))
    probe = GnomonicSpec(
        center_lat_deg=latitude,
        center_lon_deg=longitude,
        hfov_deg=maximum_fov_deg,
        vfov_deg=maximum_fov_deg,
        output_shape_hw=config.gnomonic_output_shape_hw,
    )
    rotation = np.asarray(gnomonic_rotation(probe), dtype=np.float64)
    local = bearings @ rotation
    positive = local[:, 2] > 1e-12
    horizontal = np.full(len(bearings), 90.0, dtype=np.float64)
    vertical = np.full(len(bearings), 90.0, dtype=np.float64)
    horizontal[positive] = np.degrees(
        np.abs(np.arctan2(local[positive, 0], local[positive, 2]))
    )
    vertical[positive] = np.degrees(
        np.abs(np.arctan2(local[positive, 1], local[positive, 2]))
    )
    hfov = min(
        max(
            minimum_fov_deg,
            2.0 * (float(horizontal.max()) + margin_deg),
        ),
        maximum_fov_deg,
    )
    vfov = min(
        max(
            minimum_fov_deg,
            2.0 * (float(vertical.max()) + margin_deg),
        ),
        maximum_fov_deg,
    )
    spec = GnomonicSpec(
        center_lat_deg=latitude,
        center_lon_deg=longitude,
        hfov_deg=hfov,
        vfov_deg=vfov,
        output_shape_hw=config.gnomonic_output_shape_hw,
    )
    angular_radius = float(
        np.degrees(np.max(np.arccos(np.clip(bearings @ center, -1.0, 1.0))))
    )
    coverage = float(
        np.mean(np.asarray(rays_to_gnomonic_pixels(bearings, spec).valid, dtype=bool))
    )
    if len(bearings) > 512:
        selected = np.linspace(0, len(bearings) - 1, 512).astype(np.int64)
        retained_bearings = bearings[selected]
    else:
        retained_bearings = bearings
    return HypothesisGnomonicRegion(
        view_id=view_id,
        spec=spec,
        center_bearing_xyz=center,
        angular_radius_deg=angular_radius,
        support_bearings_xyz=retained_bearings,
        track_feature_indices=(),
        sparse_support_coverage=coverage,
        semantic_evidence_available=True,
        semantic_supporting=False,
        extent_source="independent-local-cam-component-threshold-0.60",
    )


def build_independent_cam_region_provider(
    evidence: tuple[RegisteredSemanticEvidence, ...],
    concepts: tuple[SemanticConcept, ...],
    config: SphericalObjectHypothesisConfig,
    *,
    threshold: float = 0.60,
    minimum_component_cells: int = 4,
    maximum_components: int = 12,
    maximum_fallback_angle_deg: float = 35.0,
    minimum_anchor_fov_deg: float = 8.0,
    maximum_anchor_fov_deg: float = 120.0,
    anchor_margin_deg: float = 3.0,
) -> tuple[
    Callable[[MultiviewObjectHypothesis], tuple[HypothesisGnomonicRegion, ...]],
    dict[str, Any],
]:
    """Freeze per-view CAM components before associating them to hypotheses."""

    components: dict[tuple[str, str], tuple[_IndependentCamComponent, ...]] = {}
    raw_counts: dict[str, int] = {}
    retained_counts: dict[str, int] = {}
    for item in sorted(evidence, key=lambda value: value.view_id):
        for concept in sorted(concepts, key=lambda value: value.concept_id):
            values = _sample_metric_grid(_concept_values(item, concept))
            support = _sample_metric_grid(item.support).astype(bool)
            found = _periodic_mask_components(support & (values >= threshold))
            key = f"{item.view_id}::{concept.concept_id}"
            raw_counts[key] = len(found)
            proposals = []
            for cells in found:
                if len(cells) < minimum_component_cells:
                    continue
                mask = np.zeros(WINDOW_METRIC_SHAPE_HW, dtype=bool)
                mask[cells[:, 0], cells[:, 1]] = True
                region = _fit_component_region(
                    view_id=item.view_id,
                    cells_yx=cells,
                    values=values,
                    config=config,
                    minimum_fov_deg=minimum_anchor_fov_deg,
                    maximum_fov_deg=maximum_anchor_fov_deg,
                    margin_deg=anchor_margin_deg,
                )
                proposals.append(
                    _IndependentCamComponent(
                        region=region,
                        mask=mask,
                        activation_mass=float(values[mask].sum()),
                    )
                )
            proposals.sort(
                key=lambda value: (
                    -value.activation_mass,
                    value.region.spec.center_lat_deg,
                    value.region.spec.center_lon_deg,
                )
            )
            retained = tuple(proposals[:maximum_components])
            components[(item.view_id, concept.concept_id)] = retained
            retained_counts[key] = len(retained)

    def provider(
        hypothesis: MultiviewObjectHypothesis,
    ) -> tuple[HypothesisGnomonicRegion, ...]:
        regions = []
        projected_by_view = {item.view_id: item for item in hypothesis.view_regions}
        evidence_view_ids = tuple(sorted(item.view_id for item in evidence))
        for view_id in evidence_view_ids:
            local = components.get((view_id, hypothesis.concept_id), ())
            observations = tuple(
                item
                for item in hypothesis.observations
                if item.view_id == view_id and item.supported
            )
            projected = projected_by_view.get(view_id)
            if not local or projected is None:
                continue
            bearings = (
                np.stack([item.bearing_xyz for item in observations])
                if observations
                else np.asarray(projected.center_bearing_xyz)[None, :]
            )
            pixels = np.asarray(
                rays_to_erp_pixels(bearings, WINDOW_METRIC_SHAPE_HW).pixels_xy
            )
            columns = (
                np.floor(pixels[:, 0] + 0.5).astype(np.int64)
                % WINDOW_METRIC_SHAPE_HW[1]
            )
            rows = np.clip(
                np.floor(pixels[:, 1] + 0.5).astype(np.int64),
                0,
                WINDOW_METRIC_SHAPE_HW[0] - 1,
            )
            ranked = []
            for component in local:
                inside = component.mask[rows, columns]
                angles = np.degrees(
                    np.arccos(
                        np.clip(
                            bearings @ component.region.center_bearing_xyz,
                            -1.0,
                            1.0,
                        )
                    )
                )
                ranked.append(
                    (
                        -int(inside.sum()),
                        float(angles.min()),
                        -component.activation_mass,
                        component,
                        inside,
                    )
                )
            _, minimum_angle, _, selected, inside = min(
                ranked, key=lambda value: value[:3]
            )
            if not np.any(inside) and minimum_angle > maximum_fallback_angle_deg:
                continue
            track_features = (
                tuple(
                    sorted(
                        (observation.track_id, observation.feature_index)
                        for observation, keep in zip(observations, inside, strict=True)
                        if keep
                    )
                )
                if observations
                else ()
            )
            regions.append(
                replace(
                    selected.region,
                    track_feature_indices=track_features,
                    sparse_support_coverage=(
                        float(inside.mean()) if observations else 0.0
                    ),
                    semantic_supporting=bool(track_features),
                )
            )
        return tuple(regions)

    return provider, {
        "threshold": threshold,
        "minimum_component_cells": minimum_component_cells,
        "maximum_components_per_view_concept": maximum_components,
        "maximum_fallback_angle_deg": maximum_fallback_angle_deg,
        "minimum_anchor_fov_deg": minimum_anchor_fov_deg,
        "maximum_anchor_fov_deg": maximum_anchor_fov_deg,
        "anchor_margin_deg": anchor_margin_deg,
        "component_shape_hw": list(WINDOW_METRIC_SHAPE_HW),
        "raw_component_counts": raw_counts,
        "retained_component_counts": retained_counts,
        "extent_source": "independent-local-cam-component-fov-anchor",
        "common_projected_3d_regions_used_for_transfer": False,
    }


def gnomonic_window_metrics(
    evidence: RegisteredSemanticEvidence,
    concept: SemanticConcept,
    spec: GnomonicSpec,
) -> dict[str, float | None]:
    """Measure semantic coverage and angular cost of one proposed face."""

    values = _sample_metric_grid(_concept_values(evidence, concept))
    support = _sample_metric_grid(evidence.support).astype(bool)
    window = _window_mask(spec, WINDOW_METRIC_SHAPE_HW) & support
    height, width = WINDOW_METRIC_SHAPE_HW
    latitudes = math.pi / 2.0 - (np.arange(height) + 0.5) * math.pi / height
    weights = np.broadcast_to(np.cos(latitudes)[:, None], (height, width))
    support_area = float(weights[support].sum())
    window_area = float(weights[window].sum())
    semantic_mass = float((values[support] * weights[support]).sum())
    window_mass = float((values[window] * weights[window]).sum())
    high_response = support & (values >= 0.75)
    high_area = float(weights[high_response].sum())
    global_mean = semantic_mass / support_area if support_area > 0.0 else 0.0
    window_mean = window_mass / window_area if window_area > 0.0 else 0.0
    return {
        "solid_angle_fraction": window_area / support_area if support_area else None,
        "semantic_mass_recall": window_mass / semantic_mass if semantic_mass else None,
        "high_response_recall": (
            float(weights[window & high_response].sum()) / high_area
            if high_area
            else None
        ),
        "mean_score_enrichment": (
            window_mean / global_mean if global_mean > 0.0 else None
        ),
    }


def semantic_leave_one_view_out(
    reconstruction: SphericalReconstructionResult | None,
    evidence: tuple[RegisteredSemanticEvidence, ...],
    concepts: tuple[SemanticConcept, ...],
    config: SphericalObjectHypothesisConfig,
    *,
    poses: tuple[SphericalCameraPose, ...] | None = None,
    bearing_tracks: tuple[RegisteredBearingTrack, ...] | None = None,
    reconstruction_scale: str = "arbitrary",
    reference_view_id: str | None = None,
) -> dict[str, Any]:
    """Generate from N-1 semantic views and score only on the held-out CAM."""

    if len(evidence) < 3:
        raise ValueError("leave-one-view-out requires at least three semantic views")
    has_reconstruction = reconstruction is not None
    has_bearing_geometry = poses is not None or bearing_tracks is not None
    if has_reconstruction == has_bearing_geometry:
        raise ValueError(
            "provide exactly one geometry source: a refined reconstruction or "
            "registered poses plus point-free bearing tracks"
        )
    if has_bearing_geometry and (poses is None or bearing_tracks is None):
        raise ValueError("the point-free path requires poses and bearing_tracks")
    generator = SphericalObjectHypothesisGenerator(config)

    def generate(selected_evidence: tuple[RegisteredSemanticEvidence, ...]):
        if reconstruction is not None:
            return generator.generate(reconstruction, selected_evidence, concepts)
        assert poses is not None and bearing_tracks is not None
        return generator.generate_from_bearings(
            poses=poses,
            tracks=bearing_tracks,
            semantic_evidence=selected_evidence,
            concepts=concepts,
            scale=reconstruction_scale,
            reference_view_id=reference_view_id,
        )

    all_view_result = generate(evidence)
    folds = []
    for held_out in sorted(evidence, key=lambda item: item.view_id):
        training = tuple(item for item in evidence if item.view_id != held_out.view_id)
        result = generate(training)
        hypotheses = []
        for hypothesis in result.hypotheses:
            if hypothesis.spatial.center_xyz is None:
                continue
            held_out_region = next(
                (
                    item
                    for item in hypothesis.view_regions
                    if item.view_id == held_out.view_id
                ),
                None,
            )
            if held_out_region is None:
                continue
            ray = np.asarray(held_out_region.center_bearing_xyz, dtype=np.float64)
            local_scores = {}
            local_support = {}
            local_percentiles = {}
            for concept in concepts:
                supported, score = _sample_concept(held_out, concept, ray)
                local_scores[concept.concept_id] = score
                local_support[concept.concept_id] = supported
                local_percentiles[concept.concept_id] = (
                    spherical_score_percentile(held_out, concept, score)
                    if supported
                    else None
                )
            ordered_scores = sorted(
                local_scores.items(), key=lambda item: (-item[1], item[0])
            )
            rank = next(
                index + 1
                for index, (concept_id, _) in enumerate(ordered_scores)
                if concept_id == hypothesis.concept_id
            )
            nearest_all_view = min(
                (
                    float(
                        np.linalg.norm(
                            hypothesis.spatial.center_xyz - candidate.spatial.center_xyz
                        )
                    )
                    for candidate in all_view_result.hypotheses
                    if candidate.concept_id == hypothesis.concept_id
                    and candidate.spatial.center_xyz is not None
                ),
                default=None,
            )
            concept = next(
                item for item in concepts if item.concept_id == hypothesis.concept_id
            )
            window_metrics = gnomonic_window_metrics(
                held_out, concept, held_out_region.spec
            )
            pair_consistency = [
                item
                for item in hypothesis_view_consistency(hypothesis)
                if held_out.view_id in (item.view_id_a, item.view_id_b)
            ]
            hypotheses.append(
                {
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "concept_id": hypothesis.concept_id,
                    "state": hypothesis.state,
                    "generation_semantic_score": hypothesis.semantic_score,
                    "training_view_ids": list(hypothesis.view_ids),
                    "track_count": len(hypothesis.track_ids),
                    "held_out_supported": local_support[hypothesis.concept_id],
                    "held_out_score": local_scores[hypothesis.concept_id],
                    "held_out_within_class_spherical_percentile": local_percentiles[
                        hypothesis.concept_id
                    ],
                    "held_out_concept_rank": rank,
                    "projected_bearing_xyz": ray.tolist(),
                    "gnomonic_region": {
                        "center_lat_deg": held_out_region.spec.center_lat_deg,
                        "center_lon_deg": held_out_region.spec.center_lon_deg,
                        "hfov_deg": held_out_region.spec.hfov_deg,
                        "vfov_deg": held_out_region.spec.vfov_deg,
                        "output_shape_hw": list(held_out_region.spec.output_shape_hw),
                        "angular_radius_deg": held_out_region.angular_radius_deg,
                        "sparse_support_coverage": (
                            held_out_region.sparse_support_coverage
                        ),
                        **window_metrics,
                    },
                    "held_out_pair_consistency": {
                        "pair_count": len(pair_consistency),
                        "shared_track_count_median": (
                            float(
                                np.median(
                                    [
                                        item.shared_track_count
                                        for item in pair_consistency
                                    ]
                                )
                            )
                            if pair_consistency
                            else None
                        ),
                        "track_jaccard_mean": (
                            float(
                                np.mean(
                                    [item.track_jaccard for item in pair_consistency]
                                )
                            )
                            if pair_consistency
                            else None
                        ),
                        "smaller_view_track_coverage_mean": (
                            float(
                                np.mean(
                                    [
                                        item.smaller_view_track_coverage
                                        for item in pair_consistency
                                    ]
                                )
                            )
                            if pair_consistency
                            else None
                        ),
                    },
                    "nearest_all_view_same_concept_distance": nearest_all_view,
                }
            )
        folds.append(
            {
                "held_out_view_id": held_out.view_id,
                "generation_semantic_view_ids": sorted(
                    item.view_id for item in training
                ),
                "hypotheses": hypotheses,
            }
        )
    records = [item for fold in folds for item in fold["hypotheses"]]
    supported = [item for item in records if item["held_out_supported"]]
    confirmed_supported = [item for item in supported if item["state"] == "confirmed"]

    def percentile_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        percentiles = [
            float(item["held_out_within_class_spherical_percentile"]) for item in items
        ]
        return {
            "count": len(items),
            "median": float(np.median(percentiles)) if percentiles else None,
            "top_quartile_rate": (
                sum(value >= 0.75 for value in percentiles) / len(percentiles)
                if percentiles
                else None
            ),
            "top_decile_rate": (
                sum(value >= 0.90 for value in percentiles) / len(percentiles)
                if percentiles
                else None
            ),
        }

    def median_field(items: list[dict[str, Any]], *keys: str) -> float | None:
        values = []
        for item in items:
            value: Any = item
            for key in keys:
                value = value[key]
            if value is not None:
                values.append(float(value))
        return float(np.median(values)) if values else None

    return {
        "protocol": "fixed-registered-geometry-leave-one-semantic-view-out",
        "geometry_source": (
            "refined-sparse-reconstruction"
            if reconstruction is not None
            else "registered-poses-plus-point-free-bearing-tracks"
        ),
        "uses_input_point_xyz": reconstruction is not None,
        "semantic_view_count": len(evidence),
        "all_view_hypothesis_count": len(all_view_result.hypotheses),
        "all_view_confirmed_count": len(all_view_result.confirmed),
        "folds": folds,
        "aggregate": {
            "projected_hypothesis_count": len(records),
            "held_out_supported_count": len(supported),
            "held_out_top1_count": sum(
                item["held_out_concept_rank"] == 1 for item in supported
            ),
            "held_out_top1_rate": (
                sum(item["held_out_concept_rank"] == 1 for item in supported)
                / len(supported)
                if supported
                else None
            ),
            "held_out_score_median": (
                float(np.median([item["held_out_score"] for item in supported]))
                if supported
                else None
            ),
            "within_class_spherical_percentile": percentile_summary(supported),
            "confirmed_within_class_spherical_percentile": percentile_summary(
                confirmed_supported
            ),
            "gnomonic_window": {
                "semantic_mass_recall_median": median_field(
                    supported, "gnomonic_region", "semantic_mass_recall"
                ),
                "high_response_recall_median": median_field(
                    supported, "gnomonic_region", "high_response_recall"
                ),
                "solid_angle_fraction_median": median_field(
                    supported, "gnomonic_region", "solid_angle_fraction"
                ),
                "mean_score_enrichment_median": median_field(
                    supported, "gnomonic_region", "mean_score_enrichment"
                ),
                "confirmed_semantic_mass_recall_median": median_field(
                    confirmed_supported, "gnomonic_region", "semantic_mass_recall"
                ),
            },
            "held_out_track_consistency": {
                "jaccard_median": median_field(
                    records, "held_out_pair_consistency", "track_jaccard_mean"
                ),
                "smaller_view_coverage_median": median_field(
                    records,
                    "held_out_pair_consistency",
                    "smaller_view_track_coverage_mean",
                ),
                "shared_track_count_median": median_field(
                    records,
                    "held_out_pair_consistency",
                    "shared_track_count_median",
                ),
            },
            "repeat_with_all_view_within_cluster_radius_rate": (
                sum(
                    item["nearest_all_view_same_concept_distance"] is not None
                    and item["nearest_all_view_same_concept_distance"]
                    <= (config.spatial_cluster_radius or math.inf)
                    for item in records
                )
                / len(records)
                if records
                else None
            ),
        },
    }


def evaluate_complete_hypothesis_method(
    *,
    reconstruction: SphericalReconstructionResult,
    evidence: tuple[RegisteredSemanticEvidence, ...],
    concepts: tuple[SemanticConcept, ...],
    radial_depth: tuple[RegisteredRadialDepth, ...],
    generator_config: SphericalObjectHypothesisConfig,
) -> dict[str, Any]:
    """Run independent CAM regions, transfer, affinity, fusion, and Soft-NMS."""

    provider, provider_diagnostics = build_independent_cam_region_provider(
        evidence,
        concepts,
        generator_config,
    )
    method = SphericalObjectHypothesisMethod(
        SphericalObjectHypothesisMethodConfig(generator=generator_config)
    )
    result = method.run(
        reconstruction=reconstruction,
        semantic_evidence=evidence,
        concepts=concepts,
        radial_depth=radial_depth,
        independent_region_provider=provider,
    )
    within = [
        item for item in result.transfer_evaluations if item.role == "within-hypothesis"
    ]
    reduction_transfer = [
        item for item in result.transfer_evaluations if item.role == "reduction-cue"
    ]
    reduction = result.reduction
    records = () if reduction is None else reduction.records
    affinities = () if reduction is None else reduction.pair_affinities
    anchor_fusion = result.anchor_fusion

    def median(values: list[float]) -> float | None:
        return float(np.median(values)) if values else None

    def region_record(region: HypothesisGnomonicRegion) -> dict[str, Any]:
        return {
            "view_id": region.view_id,
            "center_lat_deg": region.spec.center_lat_deg,
            "center_lon_deg": region.spec.center_lon_deg,
            "hfov_deg": region.spec.hfov_deg,
            "vfov_deg": region.spec.vfov_deg,
            "output_shape_hw": list(region.spec.output_shape_hw),
            "sparse_support_coverage": region.sparse_support_coverage,
            "extent_source": region.extent_source,
        }

    def anchor_record(anchor: Any) -> dict[str, Any]:
        return {
            "anchor_id": anchor.anchor_id,
            "view_id": anchor.view_id,
            "concept_id": anchor.concept_id,
            "vocabulary": anchor.vocabulary,
            "score": anchor.score,
            "source_hypothesis_ids": list(anchor.source_hypothesis_ids),
            "proposal_family_ids": list(anchor.proposal_family_ids),
            "provenance": dict(anchor.provenance),
            "region": region_record(anchor.region),
        }

    def cluster_record(cluster: Any) -> dict[str, Any]:
        return {
            "cluster_id": cluster.cluster_id,
            "concept_id": cluster.concept_id,
            "vocabulary": cluster.vocabulary,
            "anchor_ids": list(cluster.anchor_ids),
            "view_ids": list(cluster.view_ids),
            "source_hypothesis_ids": list(cluster.source_hypothesis_ids),
            "proposal_family_ids": list(cluster.proposal_family_ids),
            "score": cluster.score,
            "state": cluster.state,
            "minimum_pair_affinity": cluster.minimum_pair_affinity,
            "consensus_regions": [
                region_record(region) for region in cluster.consensus_regions
            ],
        }

    return {
        "protocol": (
            "independent-local-cam-components-plus-spherical-anchor-fusion/v2"
        ),
        "independent_region_provider": provider_diagnostics,
        "method_diagnostics": dict(result.diagnostics),
        "generated_hypothesis_count": len(result.generation.hypotheses),
        "confirmed_hypothesis_count": len(result.generation.confirmed),
        "regional_utility": {
            "evaluation_count": len(result.region_evaluations),
            "semantic_mass_recall_median": median(
                [
                    item.metrics.semantic_mass_recall
                    for item in result.region_evaluations
                    if item.metrics.semantic_mass_recall is not None
                ]
            ),
            "observed_sphere_fraction_median": median(
                [
                    item.metrics.observed_sphere_fraction
                    for item in result.region_evaluations
                    if item.metrics.observed_sphere_fraction is not None
                ]
            ),
            "mean_score_enrichment_median": median(
                [
                    item.metrics.mean_score_enrichment
                    for item in result.region_evaluations
                    if item.metrics.mean_score_enrichment is not None
                ]
            ),
        },
        "within_hypothesis_transfer": {
            "count": len(within),
            "symmetric_iou_median": median([item.symmetric_iou for item in within]),
            "symmetric_iou_p90": (
                float(np.quantile([item.symmetric_iou for item in within], 0.9))
                if within
                else None
            ),
            "positive_iou_rate": (
                sum(item.symmetric_iou > 0.0 for item in within) / len(within)
                if within
                else None
            ),
        },
        "reduction_transfer": {
            "count": len(reduction_transfer),
            "symmetric_iou_median": median(
                [item.symmetric_iou for item in reduction_transfer]
            ),
        },
        "reduction": {
            "pair_cue_count": len(result.reduction_pair_cues),
            "pair_affinity_count": len(affinities),
            "positive_affinity_count": sum(item.affinity > 0.0 for item in affinities),
            "record_count": len(records),
            "fused_record_count": sum(
                item.action in {"fused", "fused_soft_nms"} for item in records
            ),
            "soft_nms_record_count": sum(
                item.action in {"soft_nms", "fused_soft_nms"} for item in records
            ),
            "actions": {
                action: sum(item.action == action for item in records)
                for action in ("selected", "fused", "soft_nms", "fused_soft_nms")
            },
            "diagnostics": None if reduction is None else dict(reduction.diagnostics),
        },
        "anchor_fusion": (
            None
            if anchor_fusion is None
            else {
                "diagnostics": dict(anchor_fusion.diagnostics),
                "retained_anchor_count": len(anchor_fusion.retained_anchors),
                "suppressed_anchor_count": len(anchor_fusion.suppressed_anchor_ids),
                "suppressed_cluster_count": len(anchor_fusion.suppressed_cluster_ids),
                "suppressed_cluster_ids": list(anchor_fusion.suppressed_cluster_ids),
                "pair_affinity_count": len(anchor_fusion.pair_affinities),
                "positive_pair_count": sum(
                    item.hard_compatible
                    and item.affinity
                    >= method.config.anchor_fusion.pair_affinity_threshold
                    for item in anchor_fusion.pair_affinities
                ),
                "anchor_decisions": [
                    {
                        "anchor": anchor_record(item.anchor),
                        "action": item.action,
                        "representative_anchor_id": item.representative_anchor_id,
                        "solid_angle_iou": item.solid_angle_iou,
                    }
                    for item in anchor_fusion.anchor_decisions
                ],
                "clusters": [cluster_record(item) for item in anchor_fusion.clusters],
                "cluster_decisions": [
                    {
                        "cluster": cluster_record(item.cluster),
                        "action": item.action,
                        "representative_cluster_id": (item.representative_cluster_id),
                        "minimum_view_iou": item.minimum_view_iou,
                    }
                    for item in anchor_fusion.cluster_decisions
                ],
                "pair_affinities": [
                    {
                        "anchor_id_a": item.anchor_id_a,
                        "anchor_id_b": item.anchor_id_b,
                        "hard_compatible": item.hard_compatible,
                        "hard_reasons": list(item.hard_reasons),
                        "factors": dict(item.factors),
                        "available_association_cues": list(
                            item.available_association_cues
                        ),
                        "affinity": item.affinity,
                    }
                    for item in anchor_fusion.pair_affinities
                ],
            }
        ),
        "transfer_evaluations": [
            {
                "hypothesis_id_a": item.hypothesis_id_a,
                "hypothesis_id_b": item.hypothesis_id_b,
                "view_id_a": item.view_id_a,
                "view_id_b": item.view_id_b,
                "role": item.role,
                "forward_iou": item.forward_iou,
                "reverse_iou": item.reverse_iou,
                "symmetric_iou": item.symmetric_iou,
            }
            for item in result.transfer_evaluations
        ],
        "reduction_records": [
            {
                "representative_id": item.representative.hypothesis_id,
                "concept_id": item.representative.concept_id,
                "source_hypothesis_ids": list(item.source_hypothesis_ids),
                "original_scores": [list(value) for value in item.original_scores],
                "reduced_score": item.reduced_score,
                "action": item.action,
                "suppressed_by": list(item.suppressed_by),
            }
            for item in records
        ],
    }


def add_stanford_manual_region_metrics(evaluation: dict[str, Any]) -> None:
    """Evaluate centres and proposed windows against frozen RGB rectangles."""

    eligible: list[dict[str, Any]] = []
    for fold in evaluation["folds"]:
        regions = STANFORD_MANUAL_REGIONS.get(fold["held_out_view_id"])
        for hypothesis in fold["hypotheses"]:
            boxes = None if regions is None else regions.get(hypothesis["concept_id"])
            hypothesis["manual_region_evaluated"] = boxes is not None
            if boxes is None:
                hypothesis["manual_region_hit"] = None
                continue
            pixel = np.asarray(
                rays_to_erp_pixels(
                    np.asarray(hypothesis["projected_bearing_xyz"], dtype=np.float64),
                    (1024, 2048),
                ).pixels_xy
            )
            x, y = float(pixel[0]), float(pixel[1])
            hit = any(x0 <= x < x1 and y0 <= y < y1 for x0, y0, x1, y1 in boxes)
            hypothesis["manual_region_hit"] = hit
            metric_shape = (256, 512)
            region = hypothesis["gnomonic_region"]
            spec = GnomonicSpec(
                center_lat_deg=region["center_lat_deg"],
                center_lon_deg=region["center_lon_deg"],
                hfov_deg=region["hfov_deg"],
                vfov_deg=region["vfov_deg"],
                output_shape_hw=tuple(region["output_shape_hw"]),
            )
            window = _window_mask(spec, metric_shape)
            latitudes = (
                math.pi / 2.0
                - (np.arange(metric_shape[0]) + 0.5) * math.pi / metric_shape[0]
            )
            weights = np.broadcast_to(np.cos(latitudes)[:, None], metric_shape)
            window_area = float(weights[window].sum())
            instance_metrics = []
            for x0, y0, x1, y1 in boxes:
                annotation = np.zeros(metric_shape, dtype=bool)
                mx0 = int(math.floor(x0 * metric_shape[1] / 2048))
                mx1 = int(math.ceil(x1 * metric_shape[1] / 2048))
                my0 = int(math.floor(y0 * metric_shape[0] / 1024))
                my1 = int(math.ceil(y1 * metric_shape[0] / 1024))
                annotation[my0:my1, mx0:mx1] = True
                intersection = float(weights[window & annotation].sum())
                annotation_area = float(weights[annotation].sum())
                union = float(weights[window | annotation].sum())
                instance_metrics.append(
                    {
                        "annotation_recall": (
                            intersection / annotation_area if annotation_area else None
                        ),
                        "window_precision": (
                            intersection / window_area if window_area else None
                        ),
                        "spherical_iou": intersection / union if union else None,
                    }
                )
            hypothesis["manual_region_window"] = max(
                instance_metrics,
                key=lambda item: (
                    item["spherical_iou"] if item["spherical_iou"] is not None else -1.0
                ),
            )
            eligible.append(hypothesis)
    confirmed = [item for item in eligible if item["state"] == "confirmed"]

    def median_window(items: list[dict[str, Any]], key: str) -> float | None:
        values = [
            item["manual_region_window"][key]
            for item in items
            if item["manual_region_window"][key] is not None
        ]
        return float(np.median(values)) if values else None

    evaluation["manual_region_evaluation"] = {
        "annotation": (
            "best single-instance match among frozen RGB-only rectangles from "
            "stanford_office3_regions.json"
        ),
        "eligible_view_count": len(STANFORD_MANUAL_REGIONS),
        "projected_hypothesis_count": len(eligible),
        "hit_count": sum(item["manual_region_hit"] for item in eligible),
        "hit_rate": (
            sum(item["manual_region_hit"] for item in eligible) / len(eligible)
            if eligible
            else None
        ),
        "confirmed_projected_hypothesis_count": len(confirmed),
        "confirmed_hit_count": sum(item["manual_region_hit"] for item in confirmed),
        "confirmed_hit_rate": (
            sum(item["manual_region_hit"] for item in confirmed) / len(confirmed)
            if confirmed
            else None
        ),
        "window_annotation_recall_median": median_window(eligible, "annotation_recall"),
        "window_precision_median": median_window(eligible, "window_precision"),
        "window_spherical_iou_median": median_window(eligible, "spherical_iou"),
        "confirmed_window_annotation_recall_median": median_window(
            confirmed, "annotation_recall"
        ),
        "confirmed_window_precision_median": median_window(
            confirmed, "window_precision"
        ),
        "confirmed_window_spherical_iou_median": median_window(
            confirmed, "spherical_iou"
        ),
    }


def _stanford_pose(path: Path, view_id: str) -> SphericalCameraPose:
    matrix = np.loadtxt(path).reshape(3, 4)
    rotation_dataset = matrix[:, :3]
    translation = matrix[:, 3]
    center_dataset = -rotation_dataset.T @ translation
    basis = STANFORD_PANORAI_FROM_DATASET
    # Change both the camera-coordinate and shared world-coordinate bases. A
    # one-sided reflection would have determinant -1 and is not a valid pose.
    center = basis @ center_dataset
    rotation = _nearest_rotation(basis @ rotation_dataset @ basis)
    return SphericalCameraPose(view_id, rotation, center)


def _load_stanford(
    root: Path,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, SphericalCameraPose],
    Callable[[int, Any], np.ndarray | None],
    dict[str, Any],
]:
    from panorai.image_processing import spherical_resize

    images: dict[str, np.ndarray] = {}
    supports: dict[str, np.ndarray] = {}
    poses: dict[str, SphericalCameraPose] = {}
    source_paths = {}
    for name in STANFORD_VIEW_NAMES:
        view_id = f"stanford2d3d::area_2::{name}"
        rgb_path = root / "rgb" / "area_2" / f"{name}.png"
        pose_path = root / "poses" / "area_2" / f"{name}.txt"
        source = np.asarray(Image.open(rgb_path).convert("RGB"))
        images[view_id] = np.clip(
            np.rint(spherical_resize(source, SHAPE_HW)), 0, 255
        ).astype(np.uint8)
        supports[view_id] = np.ones(SHAPE_HW, dtype=bool)
        poses[view_id] = _stanford_pose(pose_path, view_id)
        source_paths[view_id] = {
            "rgb": str(rgb_path),
            "rgb_sha256": _sha256(rgb_path),
            "pose": str(pose_path),
            "pose_sha256": _sha256(pose_path),
        }
    reference_id = f"stanford2d3d::area_2::{STANFORD_VIEW_NAMES[0]}"
    depth_path = root / "depth" / "area_2" / f"{STANFORD_VIEW_NAMES[0]}.npy"
    depth = np.load(depth_path, mmap_mode="r")

    def point_from_reference(index: int, features: Any) -> np.ndarray | None:
        pixel = np.asarray(features.source_erp_xy[index], dtype=np.float64)
        column = (
            int(math.floor((pixel[0] + 0.5) * depth.shape[1] / SHAPE_HW[1]))
            % depth.shape[1]
        )
        row = min(
            max(int(math.floor((pixel[1] + 0.5) * depth.shape[0] / SHAPE_HW[0])), 0),
            depth.shape[0] - 1,
        )
        radial_range = float(depth[row, column])
        if not math.isfinite(radial_range) or radial_range <= 0.0:
            return None
        pose = poses[reference_id]
        return pose.center + pose.R.T @ (features.bearings[index] * radial_range)

    provenance = {
        "dataset": "Stanford2D3D",
        "license": "Stanford 2D-3D-S license agreement; local research copy",
        "views": source_paths,
        "reference_depth": str(depth_path),
        "reference_depth_sha256": _sha256(depth_path),
        "reference_id": reference_id,
    }
    return images, supports, poses, point_from_reference, provenance


def align_reconstruction_to_reference_poses(
    reconstruction: SphericalReconstructionResult,
    reference_poses: dict[str, SphericalCameraPose],
) -> tuple[SphericalReconstructionResult, dict[str, Any]]:
    """Apply one evaluation-only similarity from mapper to metric pose frame."""

    reconstruction.require_success()
    view_ids = tuple(sorted(item.panorama_id for item in reconstruction.poses))
    if len(view_ids) < 3 or any(view_id not in reference_poses for view_id in view_ids):
        raise ValueError("similarity alignment requires at least three matched poses")
    mapper_poses = {item.panorama_id: item for item in reconstruction.poses}
    source = np.stack([mapper_poses[key].center for key in view_ids])
    target = np.stack([reference_poses[key].center for key in view_ids])
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_zero = source - source_mean
    target_zero = target - target_mean
    left, singular, right = np.linalg.svd(source_zero.T @ target_zero)
    row_rotation = left @ right
    if np.linalg.det(row_rotation) < 0.0:
        left[:, -1] *= -1.0
        row_rotation = left @ right
    denominator = float(np.sum(source_zero**2))
    if denominator <= 1e-12:
        raise ValueError("mapper camera centres cannot determine similarity scale")
    scale = float(singular.sum() / denominator)
    translation = target_mean - scale * source_mean @ row_rotation

    def transform_points(points: np.ndarray) -> np.ndarray:
        return scale * np.asarray(points, dtype=np.float64) @ row_rotation + translation

    aligned_poses = tuple(
        SphericalCameraPose(
            pose.panorama_id,
            _nearest_rotation(np.asarray(pose.R) @ row_rotation),
            transform_points(np.asarray(pose.center)[None, :])[0],
        )
        for pose in reconstruction.poses
    )
    aligned_tracks = tuple(
        replace(
            track,
            point_xyz=transform_points(np.asarray(track.point_xyz)[None, :])[0],
        )
        for track in reconstruction.tracks
    )
    aligned_points = transform_points(reconstruction.points_xyz)
    aligned = replace(
        reconstruction,
        poses=aligned_poses,
        tracks=aligned_tracks,
        scale="metric-evaluation-similarity-aligned-stanford",
        _points_xyz=aligned_points,
    )
    center_errors = [
        float(
            np.linalg.norm(
                aligned.pose(view_id).center - reference_poses[view_id].center
            )
        )
        for view_id in view_ids
    ]
    rotation_errors = []
    for view_id in view_ids:
        delta = aligned.pose(view_id).R @ reference_poses[view_id].R.T
        cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
        rotation_errors.append(math.degrees(math.acos(cosine)))
    return aligned, {
        "role": "evaluation-only-mapper-to-dataset-similarity",
        "scale": scale,
        "row_rotation": row_rotation.tolist(),
        "translation": translation.tolist(),
        "camera_center_rmse_m": float(np.sqrt(np.mean(np.square(center_errors)))),
        "camera_center_max_error_m": max(center_errors),
        "camera_rotation_median_error_deg": float(np.median(rotation_errors)),
        "camera_rotation_max_error_deg": max(rotation_errors),
        "used_for_feature_matching_or_mapper_refinement": False,
    }


def load_stanford_transfer_depth(
    root: Path,
    *,
    output_shape_hw: tuple[int, int] = WINDOW_METRIC_SHAPE_HW,
) -> tuple[RegisteredRadialDepth, ...]:
    """Load radial range with invalid-aware spherical resizing for evaluation."""

    from panorai.image_processing import spherical_resize

    result = []
    for name in STANFORD_VIEW_NAMES:
        view_id = f"stanford2d3d::area_2::{name}"
        path = root / "depth" / "area_2" / f"{name}.npy"
        source = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float64)
        valid = np.isfinite(source) & (source > 0.0)
        numerator = spherical_resize(np.where(valid, source, 0.0), output_shape_hw)
        denominator = spherical_resize(valid.astype(np.float64), output_shape_hw)
        support = denominator >= 1.0 - 1e-6
        values = np.full(output_shape_hw, np.nan, dtype=np.float64)
        values[support] = numerator[support] / denominator[support]
        result.append(
            RegisteredRadialDepth(
                view_id=view_id,
                values=values,
                support=support,
                units="m",
                source_id=f"stanford2d3d-radial-depth:{view_id}",
                provenance={
                    "path": str(path),
                    "sha256": _sha256(path),
                    "resampling": "invalid-aware-bilinear-spherical-resize",
                    "output_shape_hw": list(output_shape_hw),
                },
            )
        )
    return tuple(result)


def _save_projection_panel(
    output: Path,
    images: dict[str, np.ndarray],
    evaluation: dict[str, Any],
) -> None:
    tile_width, tile_height = 768, 384
    folds = evaluation["folds"]
    panel = Image.new("RGB", (tile_width, len(folds) * tile_height), "black")
    palette = ((239, 83, 80), (66, 165, 245), (102, 187, 106), (255, 202, 40))
    concept_ids = sorted(
        {item["concept_id"] for fold in folds for item in fold["hypotheses"]}
    )
    colors = {
        concept_id: palette[index % len(palette)]
        for index, concept_id in enumerate(concept_ids)
    }

    def draw_region_outline(
        draw: ImageDraw.ImageDraw,
        record: dict[str, Any],
        color: tuple[int, int, int],
    ) -> None:
        region = record["gnomonic_region"]
        spec = GnomonicSpec(
            center_lat_deg=region["center_lat_deg"],
            center_lon_deg=region["center_lon_deg"],
            hfov_deg=region["hfov_deg"],
            vfov_deg=region["vfov_deg"],
            output_shape_hw=tuple(region["output_shape_hw"]),
        )
        height, width = spec.output_shape_hw
        samples = 96
        horizontal = np.linspace(0.0, width - 1.0, samples)
        vertical = np.linspace(0.0, height - 1.0, samples)
        edges = (
            np.column_stack((horizontal, np.zeros(samples))),
            np.column_stack((np.full(samples, width - 1.0), vertical)),
            np.column_stack((horizontal[::-1], np.full(samples, height - 1.0))),
            np.column_stack((np.zeros(samples), vertical[::-1])),
        )
        for edge in edges:
            rays = np.asarray(gnomonic_pixels_to_rays(edge, spec).rays_xyz)
            pixels = np.asarray(
                rays_to_erp_pixels(rays, (tile_height, tile_width)).pixels_xy
            )
            for first, second in zip(pixels[:-1], pixels[1:], strict=True):
                if abs(float(first[0] - second[0])) < tile_width / 2:
                    draw.line(
                        (
                            float(first[0]),
                            float(first[1]),
                            float(second[0]),
                            float(second[1]),
                        ),
                        fill=color,
                        width=2,
                    )

    for fold_index, fold in enumerate(folds):
        source = Image.fromarray(images[fold["held_out_view_id"]]).resize(
            (tile_width, tile_height), Image.Resampling.LANCZOS
        )
        draw = ImageDraw.Draw(source)
        confirmed = [
            item for item in fold["hypotheses"] if item["state"] == "confirmed"
        ]
        candidates = confirmed or fold["hypotheses"]
        candidates = sorted(
            candidates,
            key=lambda item: (
                -item["track_count"],
                -item["generation_semantic_score"],
                item["hypothesis_id"],
            ),
        )[:16]
        draw.rectangle((0, 0, tile_width, 20), fill=(0, 0, 0))
        draw.text(
            (6, 5),
            f"held out: {fold['held_out_view_id']} | shown {len(candidates)} "
            f"of {len(fold['hypotheses'])}",
            fill="white",
        )
        for hypothesis in candidates:
            color = colors[hypothesis["concept_id"]]
            draw_region_outline(draw, hypothesis, color)
            pixel = np.asarray(
                rays_to_erp_pixels(
                    np.asarray(hypothesis["projected_bearing_xyz"], dtype=np.float64),
                    SHAPE_HW,
                ).pixels_xy
            )
            x = float(pixel[0]) / SHAPE_HW[1] * tile_width
            y = float(pixel[1]) / SHAPE_HW[0] * tile_height
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=3)
            draw.text((x + 7, y - 6), hypothesis["concept_id"], fill=color)
        panel.paste(source, (0, fold_index * tile_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output, quality=92)


def _save_anchor_fusion_panel(
    output: Path,
    images: dict[str, np.ndarray],
    complete_method_evaluation: dict[str, Any],
) -> None:
    """Render post-association consensus FOVs, not discovery/search windows."""

    fusion = complete_method_evaluation.get("anchor_fusion")
    if fusion is None:
        return
    clusters = fusion["clusters"]
    confirmed = [item for item in clusters if item["state"] == "confirmed"]
    selected = sorted(
        confirmed or clusters,
        key=lambda item: (-item["score"], item["cluster_id"]),
    )[:12]
    tile_width, tile_height = 1024, 512
    ordered_views = sorted(images)
    panel = Image.new("RGB", (tile_width, len(ordered_views) * tile_height), "black")
    colors = {"bookcase": (239, 83, 80), "desk": (66, 165, 245)}

    def draw_outline(
        draw: ImageDraw.ImageDraw,
        region: dict[str, Any],
        color: tuple[int, int, int],
    ) -> None:
        spec = GnomonicSpec(
            center_lat_deg=region["center_lat_deg"],
            center_lon_deg=region["center_lon_deg"],
            hfov_deg=region["hfov_deg"],
            vfov_deg=region["vfov_deg"],
            output_shape_hw=tuple(region["output_shape_hw"]),
        )
        height, width = spec.output_shape_hw
        samples = 128
        horizontal = np.linspace(0.0, width - 1.0, samples)
        vertical = np.linspace(0.0, height - 1.0, samples)
        edges = (
            np.column_stack((horizontal, np.zeros(samples))),
            np.column_stack((np.full(samples, width - 1.0), vertical)),
            np.column_stack((horizontal[::-1], np.full(samples, height - 1.0))),
            np.column_stack((np.zeros(samples), vertical[::-1])),
        )
        for edge in edges:
            rays = np.asarray(gnomonic_pixels_to_rays(edge, spec).rays_xyz)
            pixels = np.asarray(
                rays_to_erp_pixels(rays, (tile_height, tile_width)).pixels_xy
            )
            for first, second in zip(pixels[:-1], pixels[1:], strict=True):
                if abs(float(first[0] - second[0])) < tile_width / 2:
                    draw.line(
                        (
                            float(first[0]),
                            float(first[1]),
                            float(second[0]),
                            float(second[1]),
                        ),
                        fill=color,
                        width=3,
                    )

    for row, view_id in enumerate(ordered_views):
        tile = Image.fromarray(images[view_id]).resize(
            (tile_width, tile_height), Image.Resampling.LANCZOS
        )
        draw = ImageDraw.Draw(tile)
        records = []
        for cluster in selected:
            region = next(
                (
                    item
                    for item in cluster["consensus_regions"]
                    if item["view_id"] == view_id
                ),
                None,
            )
            if region is not None:
                records.append((cluster, region))
        draw.rectangle((0, 0, tile_width, 24), fill=(0, 0, 0))
        draw.text(
            (6, 6),
            f"anchor consensus | {view_id} | shown {len(records)} clusters",
            fill="white",
        )
        for cluster, region in records:
            color = colors.get(cluster["concept_id"], (102, 187, 106))
            draw_outline(draw, region, color)
            longitude = math.radians(region["center_lon_deg"])
            latitude = math.radians(region["center_lat_deg"])
            center = np.asarray(
                (
                    math.cos(latitude) * math.sin(longitude),
                    math.sin(latitude),
                    math.cos(latitude) * math.cos(longitude),
                ),
                dtype=np.float64,
            )
            pixel = np.asarray(
                rays_to_erp_pixels(center, (tile_height, tile_width)).pixels_xy
            )
            x, y = float(pixel[0]), float(pixel[1])
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)
            draw.text(
                (x + 6, y - 8),
                f"{cluster['concept_id']} {cluster['score']:.2f}",
                fill=color,
            )
        panel.paste(tile, (0, row * tile_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output, quality=94)


def _save_anchor_fusion_audit_panel(
    output: Path,
    images: dict[str, np.ndarray],
    evidence: tuple[RegisteredSemanticEvidence, ...],
    concepts: tuple[SemanticConcept, ...],
    complete_method_evaluation: dict[str, Any],
) -> None:
    """Show proposals, NMS decisions, cross-view merge, and final consensus."""

    fusion = complete_method_evaluation.get("anchor_fusion")
    if fusion is None or not fusion["clusters"]:
        return
    confirmed = [item for item in fusion["clusters"] if item["state"] == "confirmed"]
    winner = sorted(
        confirmed or fusion["clusters"],
        key=lambda item: (-item["score"], item["cluster_id"]),
    )[0]
    winner_families = set(winner["proposal_family_ids"])
    winner_views = tuple(winner["view_ids"])
    decisions = [
        item
        for item in fusion["anchor_decisions"]
        if item["anchor"]["view_id"] in winner_views
        and item["anchor"]["concept_id"] == winner["concept_id"]
        and winner_families.intersection(item["anchor"]["proposal_family_ids"])
    ]
    decision_by_id = {
        item["anchor"]["anchor_id"]: item for item in fusion["anchor_decisions"]
    }
    pair_id = tuple(sorted(winner["anchor_ids"]))
    pair = next(
        (
            item
            for item in fusion["pair_affinities"]
            if (item["anchor_id_a"], item["anchor_id_b"]) == pair_id
        ),
        None,
    )
    family_cluster_decisions = [
        item
        for item in fusion["cluster_decisions"]
        if item["cluster"]["concept_id"] == winner["concept_id"]
        and set(item["cluster"]["proposal_family_ids"]) == winner_families
        and tuple(item["cluster"]["view_ids"]) == winner_views
    ]
    evidence_by_view = {item.view_id: item for item in evidence}
    concept_by_id = {item.concept_id: item for item in concepts}

    tile_width, tile_height = 600, 300
    header_height = 108
    panel = Image.new(
        "RGB",
        (4 * tile_width, header_height + len(winner_views) * tile_height),
        (18, 20, 24),
    )
    panel_draw = ImageDraw.Draw(panel)
    retained_clusters = sum(
        item["action"] == "retained" for item in family_cluster_decisions
    )
    suppressed_clusters = len(family_cluster_decisions) - retained_clusters
    factors = {} if pair is None else pair["factors"]
    factor_text = "  ".join(f"{key}={value:.3f}" for key, value in factors.items())
    affinity_text = "n/a" if pair is None else f"{pair['affinity']:.3f}"
    panel_draw.text(
        (16, 12),
        f"SPHERICAL FOV ANCHOR AUDIT  |  concept={winner['concept_id']}  "
        f"|  cluster score={winner['score']:.3f}",
        fill=(255, 255, 255),
    )
    panel_draw.text(
        (16, 38),
        f"complete-link merge: affinity={affinity_text}  {factor_text}",
        fill=(120, 220, 255),
    )
    panel_draw.text(
        (16, 64),
        f"cluster family: {len(family_cluster_decisions)} raw -> "
        f"{retained_clusters} retained + {suppressed_clusters} suppressed  |  "
        "no ERP masks retained",
        fill=(210, 210, 210),
    )
    panel_draw.text(
        (16, 88),
        "colors: proposals=cyan  kept=green  IoU-suppressed=red  "
        "capacity-suppressed=orange  selected=gold  consensus=magenta",
        fill=(210, 210, 210),
    )

    def heatmap_tile(view_id: str) -> Image.Image:
        tile = Image.fromarray(images[view_id]).resize(
            (tile_width, tile_height), Image.Resampling.LANCZOS
        )
        evidence_item = evidence_by_view[view_id]
        concept = concept_by_id[winner["concept_id"]]
        values = np.asarray(_concept_values(evidence_item, concept), dtype=np.float32)
        support = np.asarray(evidence_item.support, dtype=np.uint8) * 255
        field = np.asarray(
            Image.fromarray(values, mode="F").resize(
                (tile_width, tile_height), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        valid = (
            np.asarray(
                Image.fromarray(support).resize(
                    (tile_width, tile_height), Image.Resampling.NEAREST
                ),
                dtype=np.uint8,
            )
            > 0
        )
        field = np.clip(field, 0.0, 1.0)
        low = np.asarray((30.0, 70.0, 210.0), dtype=np.float32)
        middle = np.asarray((255.0, 225.0, 60.0), dtype=np.float32)
        high = np.asarray((230.0, 35.0, 55.0), dtype=np.float32)
        first_half = field[..., None] * 2.0
        color = np.where(
            (field < 0.5)[..., None],
            low + first_half * (middle - low),
            middle + (field[..., None] - 0.5) * 2.0 * (high - middle),
        )
        base = np.asarray(tile, dtype=np.float32)
        alpha = (0.62 * np.sqrt(field) * valid)[..., None]
        blended = base * (1.0 - alpha) + color * alpha
        return Image.fromarray(np.clip(blended, 0.0, 255.0).astype(np.uint8))

    def draw_outline(
        draw: ImageDraw.ImageDraw,
        region: dict[str, Any],
        color: tuple[int, int, int],
        *,
        width: int = 3,
        dashed: bool = False,
    ) -> None:
        spec = GnomonicSpec(
            center_lat_deg=region["center_lat_deg"],
            center_lon_deg=region["center_lon_deg"],
            hfov_deg=region["hfov_deg"],
            vfov_deg=region["vfov_deg"],
            output_shape_hw=tuple(region["output_shape_hw"]),
        )
        height, width_pixels = spec.output_shape_hw
        samples = 128
        horizontal = np.linspace(0.0, width_pixels - 1.0, samples)
        vertical = np.linspace(0.0, height - 1.0, samples)
        edges = (
            np.column_stack((horizontal, np.zeros(samples))),
            np.column_stack((np.full(samples, width_pixels - 1.0), vertical)),
            np.column_stack((horizontal[::-1], np.full(samples, height - 1.0))),
            np.column_stack((np.zeros(samples), vertical[::-1])),
        )
        for edge in edges:
            rays = np.asarray(gnomonic_pixels_to_rays(edge, spec).rays_xyz)
            pixels = np.asarray(
                rays_to_erp_pixels(rays, (tile_height, tile_width)).pixels_xy
            )
            for index, (first, second) in enumerate(
                zip(pixels[:-1], pixels[1:], strict=True)
            ):
                if dashed and (index // 5) % 2:
                    continue
                if abs(float(first[0] - second[0])) < tile_width / 2:
                    draw.line(
                        (
                            float(first[0]),
                            float(first[1]),
                            float(second[0]),
                            float(second[1]),
                        ),
                        fill=color,
                        width=width,
                    )

    stage_titles = (
        "1  CAM + local component (threshold 0.60)",
        "2  generated scale/aspect FOV proposals",
        "3  spherical intra-view NMS decisions",
        "4  selected pair -> multiview consensus",
    )
    for row, view_id in enumerate(winner_views):
        local_decisions = [
            item for item in decisions if item["anchor"]["view_id"] == view_id
        ]
        base_components = [
            item
            for item in local_decisions
            if item["anchor"]["provenance"].get("scale_factor") == 1.0
            and item["anchor"]["provenance"].get("aspect_ratio") == 1.0
        ] or [local_decisions[0]]
        selected_anchor_id = next(
            anchor_id
            for anchor_id in winner["anchor_ids"]
            if decision_by_id[anchor_id]["anchor"]["view_id"] == view_id
        )
        selected_anchor = decision_by_id[selected_anchor_id]["anchor"]
        consensus = next(
            item for item in winner["consensus_regions"] if item["view_id"] == view_id
        )
        short_view = view_id.split("::")[-1]
        if len(short_view) > 34:
            short_view = short_view[:31] + "..."
        for stage, title in enumerate(stage_titles):
            tile = (
                heatmap_tile(view_id)
                if stage == 0
                else Image.fromarray(images[view_id]).resize(
                    (tile_width, tile_height), Image.Resampling.LANCZOS
                )
            )
            draw = ImageDraw.Draw(tile)
            draw.rectangle((0, 0, tile_width, 34), fill=(0, 0, 0))
            draw.text((8, 5), title, fill=(255, 255, 255))
            draw.text((8, 19), short_view, fill=(190, 190, 190))
            if stage == 0:
                for item in base_components:
                    draw_outline(
                        draw, item["anchor"]["region"], (255, 255, 255), width=4
                    )
            elif stage == 1:
                for item in local_decisions:
                    draw_outline(
                        draw, item["anchor"]["region"], (40, 210, 235), width=2
                    )
            elif stage == 2:
                for item in local_decisions:
                    action = item["action"]
                    color = (
                        (70, 220, 100)
                        if action == "retained"
                        else (240, 75, 75)
                        if action == "suppressed-overlap"
                        else (255, 165, 50)
                    )
                    draw_outline(
                        draw,
                        item["anchor"]["region"],
                        color,
                        width=3 if action == "retained" else 2,
                        dashed=action != "retained",
                    )
                counts = {
                    action: sum(item["action"] == action for item in local_decisions)
                    for action in (
                        "retained",
                        "suppressed-overlap",
                        "suppressed-capacity",
                    )
                }
                draw.rectangle(
                    (0, tile_height - 22, tile_width, tile_height), fill=(0, 0, 0)
                )
                draw.text(
                    (8, tile_height - 18),
                    f"kept {counts['retained']} | IoU-out {counts['suppressed-overlap']} "
                    f"| cap-out {counts['suppressed-capacity']}",
                    fill=(255, 255, 255),
                )
            else:
                draw_outline(draw, selected_anchor["region"], (255, 194, 45), width=4)
                draw_outline(draw, consensus, (235, 70, 210), width=4)
                draw.rectangle(
                    (0, tile_height - 22, tile_width, tile_height), fill=(0, 0, 0)
                )
                draw.text(
                    (8, tile_height - 18),
                    f"gold=input {selected_anchor['region']['hfov_deg']:.1f}x"
                    f"{selected_anchor['region']['vfov_deg']:.1f} deg  ->  "
                    f"magenta=final {consensus['hfov_deg']:.1f}x"
                    f"{consensus['vfov_deg']:.1f} deg",
                    fill=(255, 255, 255),
                )
            panel.paste(tile, (stage * tile_width, header_height + row * tile_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output, quality=95)


def _run_cohort(
    *,
    name: str,
    loaded: tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, SphericalCameraPose],
        Callable[[int, Any], np.ndarray | None],
        dict[str, Any],
    ],
    evidence_factory: Callable[
        [dict[str, np.ndarray], dict[str, np.ndarray]],
        tuple[RegisteredSemanticEvidence, ...],
    ],
    concepts: tuple[SemanticConcept, ...],
    cluster_radius: float | None,
    output_dir: Path,
    geometry_source: str,
    bearing_geometry: tuple[tuple[RegisteredBearingTrack, ...], dict[str, Any]]
    | None = None,
    refined_geometry: tuple[SphericalReconstructionResult, dict[str, Any]]
    | None = None,
    complete_method_geometry: tuple[
        SphericalReconstructionResult,
        tuple[RegisteredRadialDepth, ...],
        dict[str, Any],
    ]
    | None = None,
) -> dict[str, Any]:
    images, supports, poses, point_oracle, provenance = loaded
    del point_oracle
    started = time.perf_counter()
    if geometry_source == "refined-mapper":
        if refined_geometry is None:
            reconstruction, registration = build_mapper_refined_reconstruction(
                images=images,
                supports=supports,
                reference_id=provenance["reference_id"],
            )
        else:
            reconstruction, registration = refined_geometry
        bearing_tracks = None
        if not reconstruction.success:
            return {
                "cohort": name,
                "status": "mapper_failed",
                "geometry_source": geometry_source,
                "provenance": provenance,
                "registration": registration,
                "configuration": {"shape_hw": list(SHAPE_HW)},
                "evaluation": None,
                "complete_method_evaluation": None,
                "projection_panel": None,
                "anchor_fusion_panel": None,
                "anchor_fusion_audit_panel": None,
                "elapsed_seconds": time.perf_counter() - started,
            }
    elif geometry_source == "point-free-given-poses":
        reconstruction = None
        if bearing_geometry is None:
            bearing_tracks, registration = build_reference_bearing_tracks(
                images=images,
                supports=supports,
                reference_id=provenance["reference_id"],
            )
        else:
            bearing_tracks, registration = bearing_geometry
    else:
        raise ValueError(f"unsupported geometry_source: {geometry_source}")

    config = SphericalObjectHypothesisConfig(
        minimum_observation_score=0.35,
        minimum_track_score=0.35,
        minimum_independent_views=2,
        minimum_tracks_per_hypothesis=2,
        ambiguity_margin=0.05,
        maximum_ambiguous_track_fraction=0.50,
        maximum_track_residual_deg=1.5,
        spatial_cluster_radius=cluster_radius,
    )
    evidence = evidence_factory(images, supports)
    try:
        if reconstruction is not None:
            evaluation = semantic_leave_one_view_out(
                reconstruction,
                evidence,
                concepts,
                config,
            )
        else:
            assert bearing_tracks is not None
            evaluation = semantic_leave_one_view_out(
                None,
                evidence,
                concepts,
                config,
                poses=tuple(poses[key] for key in sorted(poses)),
                bearing_tracks=bearing_tracks,
                reconstruction_scale="metric-registered-poses-unverified",
                reference_view_id=provenance["reference_id"],
            )
    except RegisteredPoseInconsistencyError as error:
        return {
            "cohort": name,
            "status": "pose_inconsistent",
            "geometry_source": geometry_source,
            "provenance": provenance,
            "registration": registration,
            "pose_consistency": {
                "input_track_count": error.input_track_count,
                "nondegenerate_track_count": error.nondegenerate_track_count,
                "rejected_track_count": error.rejected_track_count,
                "rejected_fraction": error.rejected_fraction,
                "reason_counts": dict(error.reason_counts),
            },
            "configuration": {
                "shape_hw": list(SHAPE_HW),
                "maximum_pose_inconsistent_track_fraction": (
                    config.maximum_pose_inconsistent_track_fraction
                ),
            },
            "evaluation": None,
            "complete_method_evaluation": None,
            "projection_panel": None,
            "anchor_fusion_panel": None,
            "anchor_fusion_audit_panel": None,
            "elapsed_seconds": time.perf_counter() - started,
        }
    panel_path = output_dir / f"{name}-held-out-projections.jpg"
    _save_projection_panel(panel_path, images, evaluation)
    complete_method_evaluation = None
    complete_method_alignment = None
    anchor_fusion_panel_path = None
    anchor_fusion_audit_panel_path = None
    if complete_method_geometry is not None:
        aligned_reconstruction, radial_depth, complete_method_alignment = (
            complete_method_geometry
        )
        complete_method_evaluation = evaluate_complete_hypothesis_method(
            reconstruction=aligned_reconstruction,
            evidence=evidence,
            concepts=concepts,
            radial_depth=radial_depth,
            generator_config=replace(config, spatial_cluster_radius=0.80),
        )
        anchor_fusion_panel_path = output_dir / f"{name}-anchor-fusion.jpg"
        _save_anchor_fusion_panel(
            anchor_fusion_panel_path,
            images,
            complete_method_evaluation,
        )
        anchor_fusion_audit_panel_path = output_dir / f"{name}-anchor-fusion-audit.jpg"
        _save_anchor_fusion_audit_panel(
            anchor_fusion_audit_panel_path,
            images,
            evidence,
            concepts,
            complete_method_evaluation,
        )
    return {
        "cohort": name,
        "status": "evaluated",
        "geometry_source": geometry_source,
        "provenance": provenance,
        "registration": registration,
        "configuration": {
            "shape_hw": list(SHAPE_HW),
            "spatial_cluster_radius": cluster_radius,
            "spatial_cluster_radius_units": (
                "auto-baseline-fraction" if cluster_radius is None else "m"
            ),
            "minimum_observation_score": 0.35,
            "minimum_track_score": 0.35,
            "minimum_independent_views": 2,
            "minimum_tracks_per_hypothesis": 2,
        },
        "concepts": [
            {
                "concept_id": item.concept_id,
                "display_name": item.display_name,
                "proxy_weights": list(item.proxy_weights),
                "vocabulary": item.vocabulary,
            }
            for item in concepts
        ],
        "evaluation": evaluation,
        "complete_method_alignment": complete_method_alignment,
        "complete_method_evaluation": complete_method_evaluation,
        "projection_panel": str(panel_path),
        "anchor_fusion_panel": (
            None if anchor_fusion_panel_path is None else str(anchor_fusion_panel_path)
        ),
        "anchor_fusion_audit_panel": (
            None
            if anchor_fusion_audit_panel_path is None
            else str(anchor_fusion_audit_panel_path)
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    stanford_files = []
    for name in STANFORD_VIEW_NAMES:
        stanford_files.extend(
            (
                args.stanford_root / "rgb" / "area_2" / f"{name}.png",
                args.stanford_root / "poses" / "area_2" / f"{name}.txt",
            )
        )
    stanford_files.append(
        args.stanford_root / "depth" / "area_2" / f"{STANFORD_VIEW_NAMES[0]}.npy"
    )
    paths = tuple(stanford_files)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing benchmark inputs: {missing}")
    imagenet_checkpoint = (
        Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"
    )
    if not imagenet_checkpoint.is_file():
        raise FileNotFoundError(imagenet_checkpoint)
    return {
        "ready": True,
        "geometry_source": args.geometry_source,
        "stanford_view_count": len(STANFORD_VIEW_NAMES),
        "imagenet_checkpoint": str(imagenet_checkpoint),
        "imagenet_checkpoint_sha256": _sha256(imagenet_checkpoint),
        "dataset_or_checkpoint_bytes_will_be_copied": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    readiness = preflight(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stanford_loaded = _load_stanford(args.stanford_root)
    if args.geometry_source == "refined-mapper":
        stanford_refined = build_mapper_refined_reconstruction(
            images=stanford_loaded[0],
            supports=stanford_loaded[1],
            reference_id=stanford_loaded[4]["reference_id"],
        )
        stanford_bearings = None
        if stanford_refined[0].success:
            aligned, alignment = align_reconstruction_to_reference_poses(
                stanford_refined[0], stanford_loaded[2]
            )
            stanford_complete_method = (
                aligned,
                load_stanford_transfer_depth(args.stanford_root),
                alignment,
            )
        else:
            stanford_complete_method = None
    else:
        stanford_refined = None
        stanford_complete_method = None
        stanford_bearings = build_reference_bearing_tracks(
            images=stanford_loaded[0],
            supports=stanford_loaded[1],
            reference_id=stanford_loaded[4]["reference_id"],
        )
    stanford = _run_cohort(
        name="stanford2d3d-office3",
        loaded=stanford_loaded,
        evidence_factory=_imagenet_evidence,
        concepts=(
            SemanticConcept("desk", "Desk", (("desk", 1.0),), "imagenet-1k"),
            SemanticConcept(
                "bookcase", "Bookcase", (("bookcase", 1.0),), "imagenet-1k"
            ),
        ),
        cluster_radius=None if stanford_refined is not None else 0.80,
        output_dir=args.output_dir,
        geometry_source=args.geometry_source,
        bearing_geometry=stanford_bearings,
        refined_geometry=stanford_refined,
        complete_method_geometry=stanford_complete_method,
    )
    if stanford["evaluation"] is not None:
        add_stanford_manual_region_metrics(stanford["evaluation"])
    result = {
        "schema": SCHEMA,
        "scope": (
            f"geometry_source={args.geometry_source}; held-out semantic view is "
            "used only for scoring; association/localization evidence, not "
            "segmentation accuracy"
        ),
        "preflight": readiness,
        "cohorts": [stanford],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "license_boundary": (
            "Stanford2D3D, CAM tensors, panels, and all dataset bytes remain "
            "external to the PanorAi distribution."
        ),
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                item["cohort"]: (
                    item["evaluation"]["aggregate"]
                    if item["evaluation"] is not None
                    else {
                        "status": item["status"],
                        "geometry_source": item["geometry_source"],
                        "pose_consistency": item.get("pose_consistency"),
                        "failure_reasons": item["registration"].get("failure_reasons"),
                    }
                )
                for item in result["cohorts"]
            },
            indent=2,
        )
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stanford-root", type=Path, default=STANFORD_ROOT)
    parser.add_argument(
        "--geometry-source",
        choices=("refined-mapper", "point-free-given-poses"),
        default="refined-mapper",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.preflight:
        print(json.dumps(preflight(args), indent=2, sort_keys=True))
        return
    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --preflight is used")
    run(args)


if __name__ == "__main__":
    main()
