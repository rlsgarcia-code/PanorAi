#!/usr/bin/env python3
"""Generate outcome-blind external audit evidence for spherical applications.

The command deliberately separates outcome-blind pair selection from metric
execution.  ``select`` may inspect only frozen split/covariate/overlap records;
``evaluate-core`` consumes the immutable selection and evaluates SLIC, DoG and
stereo. Model checkpoints and datasets remain external to the repository.
Reused source records are labelled development evidence and cannot satisfy
held-out promotion gates.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panorai.applications import (  # noqa: E402
    SphericalDoGFeatures,
    SphericalSLIC,
    SphericalStereoDepth,
)
from panorai.features import (  # noqa: E402
    FeatureMatcher,
    OpenCVTangentDescriptorV2Config,
    SphericalDoGDetector,
    SphericalDoGDetectorConfig,
    SphericalDoGSIFTConfig,
    SphericalDoGSIFTExtractor,
)
from panorai.geometry import erp_pixels_to_rays, rays_to_erp_pixels  # noqa: E402
from panorai.image_processing._slic import (  # noqa: E402
    _assign_labels,
    _erp_rays,
    _solid_angle_weights,
)
from panorai.stereo import SphericalStereoOptions  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "contracts/spherical-application-promotion-v1.json"
SELECTION_CONTRACT = "panorai-prospective-application-selection/v1"
EVIDENCE_CONTRACT = "panorai-spherical-application-promotion/v1"
CORPUS_NAMES = {
    "matterport360": "Matterport3D",
    "stanford2d3d": "Stanford2D3D",
}
FRAME_FROM_PANORAI = {
    "matterport360": np.diag((1.0, 1.0, -1.0)),
    "stanford2d3d": np.diag((1.0, -1.0, 1.0)),
}
SELECTION_POLICY = {
    "partition": "evaluation",
    "datasets": sorted(CORPUS_NAMES),
    "minimum_directional_cloud_overlap": 0.50,
    "minimum_baseline_m": 0.15,
    "maximum_baseline_m": 2.50,
    "pairs_per_corpus": 3,
    "ordering": (
        "descending overlap, then ascending baseline, then pair_id; "
        "prefer unused spatial groups and views before deterministic fill"
    ),
    "method_outputs_used": False,
    "promotion_split": "development",
    "heldout_eligible": False,
    "heldout_ineligibility_reason": (
        "source records were used by earlier repository evaluations"
    ),
}


def sha256_file(path: Path) -> str:
    """Return a content hash without interpreting the file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def write_immutable_json(path: Path, value: Any) -> None:
    """Create an evidence file, refusing to replace different bytes."""

    payload = canonical_json(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace different evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _view_parts(view_id: str) -> tuple[str, str, str]:
    parts = view_id.split("::")
    if len(parts) != 3 or parts[0] not in CORPUS_NAMES:
        raise ValueError(f"unsupported canonical view id: {view_id!r}")
    return parts[0], parts[1], parts[2]


def _asset_paths(dataset_root: Path, view_id: str) -> dict[str, Path]:
    _, group, stem = _view_parts(view_id)
    return {
        "rgb": dataset_root / "rgb" / group / f"{stem}.png",
        "depth": dataset_root / "depth" / group / f"{stem}.npy",
    }


def _asset_record(dataset_root: Path, view_id: str) -> dict[str, Any]:
    result: dict[str, Any] = {"view_id": view_id}
    for kind, path in _asset_paths(dataset_root, view_id).items():
        if not path.is_file():
            raise FileNotFoundError(path)
        result[kind] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return result


def select_pairs(
    split_records: Iterable[dict[str, Any]],
    overlap_records: Iterable[dict[str, Any]],
    dataset_roots: dict[str, Path],
) -> list[dict[str, Any]]:
    """Select outcome-blind external development pairs using the policy."""

    overlaps = {item["pair_id"]: item for item in overlap_records}
    selected: list[dict[str, Any]] = []
    for dataset_id in sorted(CORPUS_NAMES):
        candidates = []
        for item in split_records:
            if item.get("dataset_id") != dataset_id:
                continue
            if item.get("partition") != SELECTION_POLICY["partition"]:
                continue
            overlap = overlaps.get(item.get("pair_id"))
            if overlap is None:
                continue
            minimum_overlap = float(overlap["minimum_directional_cloud_overlap"])
            baseline = float(item["covariates"]["baseline_m"])
            if minimum_overlap < SELECTION_POLICY["minimum_directional_cloud_overlap"]:
                continue
            if not (
                SELECTION_POLICY["minimum_baseline_m"]
                <= baseline
                <= SELECTION_POLICY["maximum_baseline_m"]
            ):
                continue
            candidates.append((item, overlap, minimum_overlap, baseline))
        candidates.sort(key=lambda row: (-row[2], row[3], row[0]["pair_id"]))

        chosen: list[tuple[dict[str, Any], dict[str, Any], float, float]] = []
        used_groups: set[str] = set()
        used_views: set[str] = set()
        for prefer_new_group in (True, False):
            for candidate in candidates:
                item = candidate[0]
                if candidate in chosen:
                    continue
                views = {item["from_view_id"], item["to_view_id"]}
                if views & used_views:
                    continue
                if prefer_new_group and item["spatial_group_id"] in used_groups:
                    continue
                chosen.append(candidate)
                used_groups.add(item["spatial_group_id"])
                used_views.update(views)
                if len(chosen) == SELECTION_POLICY["pairs_per_corpus"]:
                    break
            if len(chosen) == SELECTION_POLICY["pairs_per_corpus"]:
                break
        if len(chosen) != SELECTION_POLICY["pairs_per_corpus"]:
            raise RuntimeError(
                f"{dataset_id}: selected {len(chosen)} independent pairs; "
                f"requires {SELECTION_POLICY['pairs_per_corpus']}"
            )

        for item, overlap, minimum_overlap, baseline in chosen:
            selected.append(
                {
                    "pair_id": item["pair_id"],
                    "dataset_id": dataset_id,
                    "corpus": CORPUS_NAMES[dataset_id],
                    "split": SELECTION_POLICY["promotion_split"],
                    "source_partition": item["partition"],
                    "source_dataset_split": item["dataset_split"],
                    "spatial_group_id": item["spatial_group_id"],
                    "independence_component_id": item["independence_component_id"],
                    "minimum_directional_cloud_overlap": minimum_overlap,
                    "baseline_m": baseline,
                    "from": _asset_record(
                        dataset_roots[dataset_id], item["from_view_id"]
                    ),
                    "to": _asset_record(dataset_roots[dataset_id], item["to_view_id"]),
                    "reference": item["reference"],
                    "selection_provenance": {
                        "split_record_pair_id": item["pair_id"],
                        "overlap_record_pair_id": overlap["pair_id"],
                        "outcome_blind": True,
                        "heldout_eligible": SELECTION_POLICY["heldout_eligible"],
                        "heldout_ineligibility_reason": SELECTION_POLICY[
                            "heldout_ineligibility_reason"
                        ],
                    },
                }
            )
    return selected


def build_selection_manifest(
    split_path: Path,
    overlap_path: Path,
    dataset_roots: dict[str, Path],
) -> dict[str, Any]:
    return {
        "contract": SELECTION_CONTRACT,
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "selection_policy": SELECTION_POLICY,
        "sources": {
            "split_jsonl": {
                "path": str(split_path.resolve()),
                "sha256": sha256_file(split_path),
            },
            "overlap_jsonl": {
                "path": str(overlap_path.resolve()),
                "sha256": sha256_file(overlap_path),
            },
            "dataset_roots": {
                name: str(path.resolve()) for name, path in dataset_roots.items()
            },
        },
        "licenses": {
            "Matterport3D": "external local copy; governed by Matterport3D terms",
            "Stanford2D3D": "external local copy; governed by Stanford2D3D terms",
        },
        "records": select_pairs(
            read_jsonl(split_path), read_jsonl(overlap_path), dataset_roots
        ),
    }


def _periodically_connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if not len(points):
        return False
    height, width = mask.shape
    start = tuple(int(value) for value in points[0])
    seen = {start}
    queue = deque((start,))
    while queue:
        y, x = queue.popleft()
        neighbours = ((y, (x - 1) % width), (y, (x + 1) % width))
        if y:
            neighbours += ((y - 1, x),)
        if y + 1 < height:
            neighbours += ((y + 1, x),)
        for neighbour in neighbours:
            if mask[neighbour] and neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return len(seen) == int(mask.sum())


def _slic_oracle_protocol() -> dict[str, Any]:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    requirement = next(
        item
        for item in protocol["requirements"]
        if item["id"] == "slic-analytic-contract"
    )
    oracle = requirement.get("oracle")
    if not isinstance(oracle, dict):
        raise ValueError("SLIC analytic requirement has no oracle configuration")
    return oracle


def _dog_oracle_protocol() -> dict[str, Any]:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    requirement = next(
        item for item in protocol["requirements"] if item["id"] == "dog-analytic-blobs"
    )
    oracle = requirement.get("oracle")
    if not isinstance(oracle, dict):
        raise ValueError("DoG analytic requirement has no oracle configuration")
    if oracle.get("contract") != "panorai-dog-analytic-oracles/v2":
        raise ValueError("DoG analytic requirement has the wrong oracle contract")
    return oracle


def _axis_angle_rotation(axis_xyz: Iterable[float], angle_deg: float) -> np.ndarray:
    axis = np.asarray(tuple(axis_xyz), dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    outer = np.outer(axis, axis)
    angle = math.radians(angle_deg)
    return (
        np.cos(angle) * np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * outer
    )


def _rotation_oracle_scene(rays: np.ndarray, oracle: dict[str, Any]) -> np.ndarray:
    sites = np.asarray(oracle["sites_xyz"], dtype=np.float64)
    sites /= np.linalg.norm(sites, axis=1, keepdims=True)
    palette = np.asarray(oracle["palette_rgb"], dtype=np.float64)
    regions = np.argmax(rays @ sites.T, axis=-1)
    texture = float(oracle["texture_amplitude"]) * np.einsum(
        "...j,cj->...c",
        rays,
        np.asarray(oracle["texture_vectors_xyz"], dtype=np.float64),
    )
    return np.clip(palette[regions] + texture, 0.0, 1.0).astype(np.float32)


def _planar_oracle_scene(side: int, oracle: dict[str, Any]) -> np.ndarray:
    coordinate = (np.arange(side, dtype=np.float64) + 0.5) / side * 2.0 - 1.0
    v, u = np.meshgrid(coordinate, coordinate, indexing="ij")
    centres = np.asarray(oracle["centres_uv"], dtype=np.float64)
    squared_distance = (u[..., None] - centres[:, 0]) ** 2 + (
        v[..., None] - centres[:, 1]
    ) ** 2
    logits = -squared_distance / float(oracle["blend_temperature"])
    logits -= np.max(logits, axis=-1, keepdims=True)
    mixture = np.exp(logits)
    mixture /= np.sum(mixture, axis=-1, keepdims=True)
    color = mixture @ np.asarray(oracle["palette_rgb"], dtype=np.float64)
    texture = float(oracle["texture_amplitude"]) * np.einsum(
        "...j,cj->...c",
        np.stack((u, v), axis=-1),
        np.asarray(oracle["texture_vectors_uv"], dtype=np.float64),
    )
    return np.clip(color + texture, 0.0, 1.0).astype(np.float32)


def _optimal_label_agreement(
    reference: np.ndarray,
    candidate: np.ndarray,
    weights: np.ndarray,
) -> float:
    from scipy.optimize import linear_sum_assignment

    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    weights = np.asarray(weights, dtype=np.float64)
    reference_valid = (reference >= 0) & np.isfinite(weights) & (weights > 0)
    overlap_valid = reference_valid & (candidate >= 0)
    reference_labels, reference_index = np.unique(
        reference[overlap_valid], return_inverse=True
    )
    candidate_labels, candidate_index = np.unique(
        candidate[overlap_valid], return_inverse=True
    )
    if not len(reference_labels) or not len(candidate_labels):
        return 0.0
    overlap = np.zeros((len(reference_labels), len(candidate_labels)), dtype=np.float64)
    np.add.at(overlap, (reference_index, candidate_index), weights[overlap_valid])
    rows, columns = linear_sum_assignment(-overlap)
    agreement = float(overlap[rows, columns].sum() / weights[reference_valid].sum())
    return float(np.clip(agreement, 0.0, 1.0))


def _label_boundaries(labels: np.ndarray, *, periodic_x: bool) -> np.ndarray:
    labels = np.asarray(labels)
    valid = labels >= 0
    boundary = np.zeros(labels.shape, dtype=bool)
    horizontal = valid[:, :-1] & valid[:, 1:] & (labels[:, :-1] != labels[:, 1:])
    boundary[:, :-1] |= horizontal
    boundary[:, 1:] |= horizontal
    if periodic_x:
        seam = valid[:, -1] & valid[:, 0] & (labels[:, -1] != labels[:, 0])
        boundary[:, -1] |= seam
        boundary[:, 0] |= seam
    vertical = valid[:-1] & valid[1:] & (labels[:-1] != labels[1:])
    boundary[:-1] |= vertical
    boundary[1:] |= vertical
    return boundary


def _boundary_f1(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    tolerance_pixels: int,
    periodic_x: bool,
) -> float:
    from scipy.ndimage import binary_dilation

    reference_boundary = _label_boundaries(reference, periodic_x=periodic_x)
    candidate_boundary = _label_boundaries(candidate, periodic_x=periodic_x)
    if not reference_boundary.any() and not candidate_boundary.any():
        return 1.0
    structure = np.ones((3, 3), dtype=bool)
    dilated_reference = reference_boundary
    dilated_candidate = candidate_boundary

    def dilate_once(value: np.ndarray) -> np.ndarray:
        if not periodic_x:
            return binary_dilation(value, structure=structure)
        longitude_padded = np.pad(value, ((0, 0), (1, 1)), mode="wrap")
        padded = np.pad(
            longitude_padded,
            ((1, 1), (0, 0)),
            mode="constant",
            constant_values=False,
        )
        return binary_dilation(padded, structure=structure)[1:-1, 1:-1]

    for _ in range(tolerance_pixels):
        dilated_reference = dilate_once(dilated_reference)
        dilated_candidate = dilate_once(dilated_candidate)
    precision = float(
        np.count_nonzero(candidate_boundary & dilated_reference)
        / max(1, np.count_nonzero(candidate_boundary))
    )
    recall = float(
        np.count_nonzero(reference_boundary & dilated_candidate)
        / max(1, np.count_nonzero(reference_boundary))
    )
    return (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )


def _sample_erp_labels_nearest(labels: np.ndarray, rays: np.ndarray) -> np.ndarray:
    height, width = labels.shape
    projected = rays_to_erp_pixels(rays, labels.shape)
    pixels = np.asarray(projected.pixels_xy)
    x = np.floor(pixels[..., 0] + 0.5).astype(np.int64) % width
    y = np.clip(np.floor(pixels[..., 1] + 0.5).astype(np.int64), 0, height - 1)
    return labels[y, x]


def _evaluate_slic_rotation(oracle: dict[str, Any]) -> dict[str, Any]:
    shape = tuple(oracle["shape_hw"])
    rays = _erp_rays(shape)
    original_image = _rotation_oracle_scene(rays, oracle)
    application = SphericalSLIC(
        n_segments=oracle["n_segments"],
        compactness=oracle["compactness"],
        max_num_iter=oracle["max_num_iter"],
        min_size_factor=oracle["min_size_factor"],
        start_label=0,
    )
    original = application(original_image).values
    weights = _solid_angle_weights(shape)
    cases = []
    for specification in oracle["rotations"]:
        rotation = _axis_angle_rotation(
            specification["axis_xyz"], specification["angle_deg"]
        )
        rotated_image = _rotation_oracle_scene(rays @ rotation, oracle)
        rotated = application(rotated_image).values
        rotated_on_original = _sample_erp_labels_nearest(rotated, rays @ rotation.T)
        cases.append(
            {
                "axis_xyz": specification["axis_xyz"],
                "angle_deg": specification["angle_deg"],
                "weighted_label_agreement": _optimal_label_agreement(
                    original, rotated_on_original, weights
                ),
                "boundary_f1": _boundary_f1(
                    original,
                    rotated_on_original,
                    tolerance_pixels=oracle["boundary_tolerance_pixels"],
                    periodic_x=True,
                ),
                "rotated_label_count": int(len(np.unique(rotated))),
            }
        )
    return {
        "minimum_weighted_label_agreement": min(
            item["weighted_label_agreement"] for item in cases
        ),
        "minimum_boundary_f1": min(item["boundary_f1"] for item in cases),
        "original_label_count": int(len(np.unique(original))),
        "cases": cases,
    }


def _evaluate_slic_planar_limit(oracle: dict[str, Any]) -> list[dict[str, Any]]:
    from skimage.segmentation import slic

    height, width = oracle["erp_shape_hw"]
    solid_angles = _solid_angle_weights((height, width))
    results = []
    for side in oracle["patch_side_pixels_largest_to_smallest"]:
        y0 = (height - side) // 2
        x0 = (width - side) // 2
        patch = _planar_oracle_scene(side, oracle)
        panorama = np.full((height, width, 3), 0.5, dtype=np.float32)
        panorama[y0 : y0 + side, x0 : x0 + side] = patch
        valid = np.zeros((height, width), dtype=bool)
        valid[y0 : y0 + side, x0 : x0 + side] = True
        application = SphericalSLIC(
            n_segments=oracle["n_segments"],
            compactness=oracle["compactness"],
            max_num_iter=oracle["max_num_iter"],
            min_size_factor=oracle["min_size_factor"],
            start_label=0,
        )
        spherical = application(panorama, validity_mask=valid).values[
            y0 : y0 + side, x0 : x0 + side
        ]
        planar = slic(
            patch,
            n_segments=oracle["n_segments"],
            compactness=oracle["compactness"],
            max_num_iter=oracle["max_num_iter"],
            enforce_connectivity=True,
            min_size_factor=oracle["min_size_factor"],
            start_label=0,
            convert2lab=True,
            channel_axis=-1,
        )
        weights = solid_angles[y0 : y0 + side, x0 : x0 + side]
        results.append(
            {
                "side_pixels": int(side),
                "angular_extent_deg": float(side * 180.0 / height),
                "label_agreement": _optimal_label_agreement(planar, spherical, weights),
                "boundary_f1": _boundary_f1(
                    planar,
                    spherical,
                    tolerance_pixels=oracle["boundary_tolerance_pixels"],
                    periodic_x=False,
                ),
                "spherical_label_count": int(len(np.unique(spherical))),
                "planar_label_count": int(len(np.unique(planar))),
            }
        )
    return results


def evaluate_slic_analytic() -> dict[str, Any]:
    shape = (48, 96)
    y, x = np.indices(shape)
    textured = np.stack(
        (
            0.5 + 0.25 * np.sin(2.0 * np.pi * x / shape[1]),
            0.5 + 0.25 * np.cos(2.0 * np.pi * y / shape[0]),
            0.5 + 0.2 * np.sin(4.0 * np.pi * x / shape[1]),
        ),
        axis=-1,
    ).astype(np.float32)
    image = np.full((*shape, 3), 0.5, dtype=np.float32)
    application = SphericalSLIC(n_segments=12, max_num_iter=15, min_size_factor=0.25)
    first = application(image).values
    second = application(image).values
    labels = np.unique(first)
    weights = _solid_angle_weights(shape)

    seam_image = np.zeros((*shape, 3), dtype=np.uint8)
    seam_image[..., 2] = 96
    seam_image[:, :8] = (255, 0, 0)
    seam_image[:, -8:] = (255, 0, 0)
    seam = SphericalSLIC(
        n_segments=8,
        compactness=2.0,
        max_num_iter=15,
        min_size_factor=0.25,
        start_label=0,
    )(seam_image).values

    mask = np.ones(shape, dtype=bool)
    mask[:4] = False
    mask[-3:] = False
    mask[20:24, 40:48] = False
    masked = SphericalSLIC(
        n_segments=9, compactness=6.0, max_num_iter=12, start_label=7
    )(textured, validity_mask=mask).values

    rng = np.random.default_rng(8)
    rays = rng.normal(size=(80, 3))
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    centres = rays[[2, 17, 31, 48, 67]]
    colours = rng.normal(size=(80, 3))
    centre_colours = colours[[2, 17, 31, 48, 67]]
    angle = np.deg2rad(73.0)
    rotation = np.asarray(
        (
            (np.cos(angle), 0.0, np.sin(angle)),
            (0.0, 1.0, 0.0),
            (-np.sin(angle), 0.0, np.cos(angle)),
        )
    )
    assigned = _assign_labels(
        colours, rays, centre_colours, centres, compactness=7.0, angular_scale=0.8
    )
    rotated = _assign_labels(
        colours,
        rays @ rotation.T,
        centre_colours,
        centres @ rotation.T,
        compactness=7.0,
        angular_scale=0.8,
    )

    erp_rays = _erp_rays((360, 720))
    row, column = 179, 360
    intrinsic = math.acos(
        float(np.clip(erp_rays[row, column] @ erp_rays[row + 1, column + 1], -1, 1))
    )
    planar = math.hypot(math.pi / 360.0, math.pi / 360.0)
    planar_relative_error = abs(intrinsic - planar) / planar
    oracle = _slic_oracle_protocol()
    rotation_result = _evaluate_slic_rotation(oracle["rotation"])
    planar_results = _evaluate_slic_planar_limit(oracle["planar_limit"])
    smallest_planar = planar_results[-1]
    rotation_passed = bool(
        rotation_result["minimum_weighted_label_agreement"]
        >= oracle["rotation"]["minimum_weighted_label_agreement"]
        and rotation_result["minimum_boundary_f1"]
        >= oracle["rotation"]["minimum_boundary_f1"]
    )
    planar_passed = bool(
        smallest_planar["label_agreement"]
        >= oracle["planar_limit"]["minimum_smallest_patch_label_agreement"]
        and smallest_planar["boundary_f1"]
        >= oracle["planar_limit"]["minimum_smallest_patch_boundary_f1"]
    )

    return {
        "application": "slic",
        "evaluation": "analytic",
        "metrics": {
            "solid_angle_4pi": math.isclose(
                float(weights.sum()), 4.0 * math.pi, abs_tol=1e-12
            ),
            "connected_labels": all(
                _periodically_connected(first == label) for label in labels
            ),
            "deterministic": bool(np.array_equal(first, second)),
            "periodic_seam": bool(
                np.array_equal(seam[:, 0], seam[:, -1])
                and _periodically_connected(seam == seam[shape[0] // 2, 0])
            ),
            "analytic_cardinalities_and_masks": bool(
                len(labels) == 12
                and np.all(masked[~mask] == -1)
                and masked[mask].min() == 7
            ),
            "controlled_rotation_stability": rotation_passed,
            "planar_low_distortion_limit": planar_passed,
        },
        "details": {
            "shape_hw": list(shape),
            "observed_label_count": int(len(labels)),
            "solid_angle_sum": float(weights.sum()),
            "equatorial_one_pixel_diagonal_relative_error": planar_relative_error,
            "assignment_kernel_rotation_invariant": bool(
                np.array_equal(assigned, rotated)
            ),
            "oracle": oracle,
            "rotation": rotation_result,
            "planar_limit": planar_results,
            "promotion_limitations": [
                name
                for name, passed in (
                    (
                        "public-wrapper controlled rotation oracle failed",
                        rotation_passed,
                    ),
                    (
                        "public-wrapper planar low-distortion oracle failed",
                        planar_passed,
                    ),
                )
                if not passed
            ],
        },
    }


def _angular_distance_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return np.rad2deg(np.arccos(np.clip(np.sum(left * right, axis=-1), -1.0, 1.0)))


def _blob_image(
    shape_hw: tuple[int, int],
    centres: np.ndarray,
    sigma_deg: float,
    *,
    background: float = 25.0,
    amplitude: float = 220.0,
) -> np.ndarray:
    y, x = np.indices(shape_hw, dtype=np.float64)
    rays = erp_pixels_to_rays(np.stack((x, y), axis=-1), shape_hw)
    image = np.full(shape_hw, background, dtype=np.float64)
    for centre in centres:
        distance = np.arccos(np.clip(rays @ centre, -1.0, 1.0))
        image += amplitude * np.exp(-0.5 * (distance / np.deg2rad(sigma_deg)) ** 2)
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def _bearing(longitude_deg: float, latitude_deg: float) -> np.ndarray:
    longitude = math.radians(longitude_deg)
    latitude = math.radians(latitude_deg)
    return np.asarray(
        (
            math.sin(longitude) * math.cos(latitude),
            math.sin(latitude),
            math.cos(longitude) * math.cos(latitude),
        )
    )


def _dog_configs(
    oracle: dict[str, Any],
) -> tuple[SphericalDoGDetectorConfig, SphericalDoGSIFTConfig]:
    detector_values = dict(oracle["detector_config"])
    detector_values["selection_grid_shape"] = tuple(
        detector_values["selection_grid_shape"]
    )
    detector_config = SphericalDoGDetectorConfig(**detector_values)
    wrapper_values = oracle["wrapper_config"]
    descriptor_values = dict(wrapper_values["descriptor"])
    descriptor_values["scale_multipliers"] = tuple(
        descriptor_values["scale_multipliers"]
    )
    descriptor_values["algorithm_parameters"] = tuple(
        tuple(item) for item in descriptor_values["algorithm_parameters"]
    )
    descriptor_config = OpenCVTangentDescriptorV2Config(**descriptor_values)
    wrapper_config = SphericalDoGSIFTConfig(
        octaves=detector_config.octaves,
        levels_per_octave=detector_config.levels_per_octave,
        base_sigma_px=detector_config.base_sigma_px,
        contrast_threshold=detector_config.contrast_threshold,
        edge_threshold=detector_config.edge_threshold,
        max_features=detector_config.max_keypoints,
        patch_size=int(wrapper_values["patch_size"]),
        descriptor_radius_sigmas=float(wrapper_values["descriptor_radius_sigmas"]),
        minimum_valid_fraction=float(wrapper_values["minimum_valid_fraction"]),
        minimum_valid_support_fraction=(detector_config.minimum_valid_support_fraction),
        angular_dedup_threshold_deg=detector_config.angular_dedup_threshold_deg,
        scale_dedup_log2=detector_config.scale_dedup_log2,
        selection_policy=detector_config.selection_policy,
        selection_grid_shape=detector_config.selection_grid_shape,
        descriptor_config=descriptor_config,
        convolution_backend=detector_config.convolution_backend,
    )
    return detector_config, wrapper_config


def _dog_wrapper_is_consistent(
    image: np.ndarray,
    direct: Any,
    wrapper_config: SphericalDoGSIFTConfig,
    panorama_id: str,
) -> dict[str, Any]:
    delegated = SphericalDoGSIFTExtractor(wrapper_config).detect(
        image, panorama_id=panorama_id
    )
    detection_identity = all(
        np.array_equal(getattr(direct, field), getattr(delegated, field))
        for field in (
            "source_erp_xy",
            "bearings",
            "responses",
            "scales_deg",
            "octaves",
            "levels",
            "refined_levels",
        )
    )
    features = SphericalDoGFeatures(wrapper_config)(image, panorama_id=panorama_id)
    subset = True
    for source_xy, bearing, scale in zip(
        features.source_erp_xy,
        features.bearings,
        (item.scale for item in features.features),
        strict=True,
    ):
        exact = (
            np.all(
                np.isclose(direct.source_erp_xy, source_xy, rtol=0.0, atol=1e-12),
                axis=1,
            )
            & np.all(np.isclose(direct.bearings, bearing, rtol=0.0, atol=1e-12), axis=1)
            & np.isclose(direct.scales_deg, scale, rtol=0.0, atol=1e-12)
        )
        if not np.any(exact):
            subset = False
            break
    return {
        "detector_delegation_exact": bool(detection_identity),
        "described_features_are_detector_subset": bool(subset),
        "descriptor_feature_count": len(features),
    }


def _evaluate_dog_case(case: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    from scipy.optimize import linear_sum_assignment

    shape = tuple(int(item) for item in case["shape_hw"])
    sigma_deg = float(case["blob_sigma_deg"])
    lon_lat = np.asarray(case["centres_lon_lat_deg"], dtype=np.float64)
    centres = np.stack([_bearing(*item) for item in lon_lat])
    renderer = oracle["renderer"]
    image = _blob_image(
        shape,
        centres,
        sigma_deg,
        background=float(renderer["background"]),
        amplitude=float(renderer["amplitude"]),
    )
    detector_config, wrapper_config = _dog_configs(oracle)
    panorama_id = str(case["id"])
    detected = SphericalDoGDetector(detector_config).detect(
        image, panorama_id=panorama_id
    )
    cost = np.empty((len(centres), len(detected)), dtype=np.float64)
    if len(detected):
        cost = np.rad2deg(np.arccos(np.clip(centres @ detected.bearings.T, -1.0, 1.0)))
        expected_indices, detected_indices = linear_sum_assignment(cost)
        assignments = dict(zip(expected_indices.tolist(), detected_indices.tolist()))
    else:
        assignments = {}
    half_step = 0.5 * 180.0 / shape[0]
    maximum_assignment_radius = 3.0 * sigma_deg
    unmatched_error = float(oracle["assignment"]["unmatched_error_deg"])
    unmatched_scale_error = float(oracle["assignment"]["unmatched_scale_level_error"])
    maximum_scale_error = float(oracle["maximum_scale_level_error"])
    blob_results = []
    accepted_detection_indices: set[int] = set()
    for blob_index, (longitude, latitude) in enumerate(lon_lat):
        detection_index = assignments.get(blob_index)
        angular_error = (
            float(cost[blob_index, detection_index])
            if detection_index is not None
            else unmatched_error
        )
        if angular_error > maximum_assignment_radius:
            detection_index = None
            angular_error = unmatched_error
        if detection_index is None:
            observed_scale = None
            scale_error = unmatched_scale_error
        else:
            accepted_detection_indices.add(detection_index)
            observed_scale = float(detected.scales_deg[detection_index])
            scale_error = abs(
                math.log2(observed_scale / sigma_deg)
                * detector_config.levels_per_octave
            )
        location_pass = angular_error <= half_step
        scale_pass = scale_error <= maximum_scale_error
        blob_results.append(
            {
                "blob_index": blob_index,
                "longitude_deg": float(longitude),
                "latitude_deg": float(latitude),
                "seam": bool(abs(longitude) >= 170.0),
                "high_latitude": bool(abs(latitude) >= 60.0),
                "detection_index": detection_index,
                "angular_error_deg": angular_error,
                "observed_scale_deg": observed_scale,
                "scale_level_error": float(scale_error),
                "location_pass": bool(location_pass),
                "scale_pass": bool(scale_pass),
            }
        )
    wrapper = _dog_wrapper_is_consistent(image, detected, wrapper_config, panorama_id)
    return {
        "id": panorama_id,
        "shape_hw": list(shape),
        "blob_sigma_deg": sigma_deg,
        "lattice_half_step_deg": half_step,
        "detector_keypoint_count": len(detected),
        "unmatched_blob_count": sum(
            item["detection_index"] is None for item in blob_results
        ),
        "unmatched_detection_count": len(detected) - len(accepted_detection_indices),
        "blobs": blob_results,
        "wrapper": wrapper,
    }


def evaluate_dog_analytic() -> dict[str, Any]:
    oracle = _dog_oracle_protocol()
    cases = [_evaluate_dog_case(case, oracle) for case in oracle["cases"]]
    blobs = [blob for case in cases for blob in case["blobs"]]
    seam = [blob for blob in blobs if blob["seam"]]
    high_latitude = [blob for blob in blobs if blob["high_latitude"]]
    location_passes = [bool(blob["location_pass"]) for blob in blobs]
    subgroup_pass = bool(
        seam
        and high_latitude
        and all(blob["location_pass"] and blob["scale_pass"] for blob in seam)
        and all(blob["location_pass"] and blob["scale_pass"] for blob in high_latitude)
    )
    wrapper_consistent = all(
        case["wrapper"]["detector_delegation_exact"]
        and case["wrapper"]["described_features_are_detector_subset"]
        for case in cases
    )
    return {
        "application": "dog",
        "evaluation": "analytic",
        "metrics": {
            "isolated_blob_detection_rate": float(np.mean(location_passes)),
            "maximum_angular_error_deg": max(
                float(blob["angular_error_deg"]) for blob in blobs
            ),
            "lattice_half_step_deg": max(
                float(case["lattice_half_step_deg"]) for case in cases
            ),
            "maximum_scale_level_error": max(
                float(blob["scale_level_error"]) for blob in blobs
            ),
            "seam_and_high_latitude_corpus": subgroup_pass,
            "public_wrapper_consistent": wrapper_consistent,
        },
        "details": {
            "oracle": oracle,
            "cases": cases,
            "public_wrapper_consistent": wrapper_consistent,
            "promotion_limitations": [
                message
                for message, passed in (
                    (
                        "not every isolated blob localized within half a lattice step",
                        all(location_passes),
                    ),
                    (
                        "one or more blob scales exceeded one DoG level of error",
                        all(blob["scale_pass"] for blob in blobs),
                    ),
                    ("seam or high-latitude subgroup failed", subgroup_pass),
                    (
                        "high-level wrapper diverged from detector output",
                        wrapper_consistent,
                    ),
                )
                if not passed
            ],
        },
    }


def _resize_image(image: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, shape_hw[::-1], interpolation=cv2.INTER_AREA)


def _resize_depth(depth: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    return cv2.resize(depth, shape_hw[::-1], interpolation=cv2.INTER_NEAREST)


def _load_rgb(asset: dict[str, Any], shape_hw: tuple[int, int]) -> np.ndarray:
    path = Path(asset["path"])
    if sha256_file(path) != asset["sha256"]:
        raise ValueError(f"RGB checksum changed: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return cv2.cvtColor(_resize_image(image, shape_hw), cv2.COLOR_BGR2RGB)


def _load_depth(asset: dict[str, Any], shape_hw: tuple[int, int]) -> np.ndarray:
    path = Path(asset["path"])
    if sha256_file(path) != asset["sha256"]:
        raise ValueError(f"depth checksum changed: {path}")
    depth = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth must be HW: {path}")
    return _resize_depth(depth, shape_hw)


def _panorai_pose(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    dataset_id = record["dataset_id"]
    adapter = FRAME_FROM_PANORAI[dataset_id]
    reference = record["reference"]
    raw_rotation = np.asarray(reference["R_to_from"], dtype=np.float64)
    raw_translation = np.asarray(reference["t_to_from_m"], dtype=np.float64)
    return adapter.T @ raw_rotation @ adapter, adapter.T @ raw_translation


def _sample_nearest(values: np.ndarray, xy: np.ndarray) -> np.ndarray:
    height, width = values.shape[:2]
    finite = np.isfinite(xy).all(axis=1)
    safe = np.where(finite[:, None], xy, 0.0)
    x = np.mod(np.rint(safe[:, 0]).astype(np.int64), width)
    y = np.clip(np.rint(safe[:, 1]).astype(np.int64), 0, height - 1)
    sampled = np.asarray(values[y, x], dtype=np.float64)
    sampled[~finite] = np.nan
    return sampled


def _visible_feature_oracle(
    bearings: np.ndarray,
    source_xy: np.ndarray,
    depth_a: np.ndarray,
    depth_b: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    depth = _sample_nearest(depth_a, source_xy)
    points_b = (rotation @ (bearings * depth[:, None]).T).T + translation
    range_b = np.linalg.norm(points_b, axis=1)
    expected_bearings = np.divide(
        points_b,
        range_b[:, None],
        out=np.zeros_like(points_b),
        where=range_b[:, None] > 0,
    )
    projection = rays_to_erp_pixels(expected_bearings, depth_b.shape)
    expected_xy = np.asarray(projection.pixels_xy)
    target_depth = _sample_nearest(depth_b, expected_xy)
    tolerance = np.maximum(0.15, 0.05 * range_b)
    visible = (
        np.isfinite(depth)
        & (depth > 0)
        & np.isfinite(target_depth)
        & (target_depth > 0)
        & np.isfinite(range_b)
        & (range_b > 0)
        & np.asarray(projection.valid, dtype=bool)
        & (np.abs(target_depth - range_b) <= tolerance)
    )
    return expected_bearings, visible


def _minimum_angular_errors(expected: np.ndarray, observed: np.ndarray) -> np.ndarray:
    if len(expected) == 0:
        return np.empty(0, dtype=np.float64)
    if len(observed) == 0:
        return np.full(len(expected), 180.0)
    cosine = np.clip(expected @ observed.T, -1.0, 1.0)
    return np.rad2deg(np.arccos(np.max(cosine, axis=1)))


def evaluate_dog_external_pair(
    record: dict[str, Any], shape_hw: tuple[int, int] = (192, 384)
) -> dict[str, Any]:
    image_a = _load_rgb(record["from"]["rgb"], shape_hw)
    image_b = _load_rgb(record["to"]["rgb"], shape_hw)
    depth_a = _load_depth(record["from"]["depth"], shape_hw)
    depth_b = _load_depth(record["to"]["depth"], shape_hw)
    rotation, translation = _panorai_pose(record)
    config = SphericalDoGSIFTConfig(
        octaves=3,
        levels_per_octave=3,
        max_features=600,
        patch_size=32,
        contrast_threshold=0.003,
        convolution_backend="numpy",
    )
    application = SphericalDoGFeatures(config)
    features_a = application(image_a, panorama_id=record["from"]["view_id"])
    features_b = application(image_b, panorama_id=record["to"]["view_id"])
    expected, visible = _visible_feature_oracle(
        features_a.bearings,
        features_a.source_erp_xy,
        depth_a,
        depth_b,
        rotation,
        translation,
    )
    errors = _minimum_angular_errors(expected[visible], features_b.bearings)
    repeatable = errors <= 1.0
    matches = FeatureMatcher().match(features_a, features_b)
    match_expected, match_visible = _visible_feature_oracle(
        features_a.bearings[matches.feature_indices_a],
        features_a.source_erp_xy[matches.feature_indices_a],
        depth_a,
        depth_b,
        rotation,
        translation,
    )
    match_errors = _angular_distance_deg(match_expected, matches.bearings_b)
    verified = match_visible & (match_errors <= 1.0) & matches.valid
    return {
        "pair_id": record["pair_id"],
        "visible_feature_count": int(visible.sum()),
        "repeatable_feature_count": int(repeatable.sum()),
        "repeatability": float(repeatable.mean()) if len(repeatable) else 0.0,
        "visible_location_errors_deg": errors.tolist(),
        "match_count": int(len(matches)),
        "verified_match_count": int(verified.sum()),
        "verified_inlier_precision": (float(verified.mean()) if len(verified) else 0.0),
    }


def aggregate_dog_external(
    records: list[dict[str, Any]], corpus: str
) -> dict[str, Any]:
    pairs = [evaluate_dog_external_pair(item) for item in records]
    visible = sum(item["visible_feature_count"] for item in pairs)
    repeatable = sum(item["repeatable_feature_count"] for item in pairs)
    matches = sum(item["match_count"] for item in pairs)
    verified = sum(item["verified_match_count"] for item in pairs)
    location_errors = [
        error for item in pairs for error in item["visible_location_errors_deg"]
    ]
    splits = {item["split"] for item in records}
    if len(splits) != 1:
        raise ValueError("external DoG records must have one evidence split")
    return {
        "application": "dog",
        "evaluation": "external",
        "split": splits.pop(),
        "corpus": corpus,
        "metrics": {
            "repeatability": repeatable / visible if visible else 0.0,
            "median_location_error_deg": (
                float(np.median(location_errors)) if location_errors else 180.0
            ),
            "verified_inlier_precision": verified / matches if matches else 0.0,
        },
        "details": {"pairs": pairs},
    }


def _render_textured_sphere(
    camera_center: np.ndarray,
    shape_hw: tuple[int, int],
    radius: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.indices(shape_hw, dtype=np.float32)
    rays = erp_pixels_to_rays(np.stack((x, y), axis=-1), shape_hw)
    centre_dot_ray = np.sum(rays * camera_center, axis=-1)
    distance = -centre_dot_ray + np.sqrt(
        centre_dot_ray**2 + radius**2 - float(camera_center @ camera_center)
    )
    points = camera_center + distance[..., None] * rays
    image = np.stack(
        (
            0.5
            + 0.2 * np.sin(5.0 * points[..., 0] + 2.0 * points[..., 2])
            + 0.2 * np.cos(7.0 * points[..., 1]),
            0.5
            + 0.25 * np.sin(4.0 * points[..., 1] - 3.0 * points[..., 2])
            + 0.15 * np.cos(8.0 * points[..., 0]),
            0.5
            + 0.25 * np.cos(6.0 * points[..., 2] + points[..., 0])
            + 0.15 * np.sin(9.0 * points[..., 1]),
        ),
        axis=-1,
    )
    return np.clip(image, 0, 1).astype(np.float32), distance.astype(np.float32)


def _solid_angle_metric_weights(shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    rows = np.arange(height, dtype=np.float64)
    latitude = math.pi / 2.0 - (rows + 0.5) * math.pi / height
    weights = np.cos(latitude)[:, None] * np.ones((1, width), dtype=np.float64)
    return weights / weights.sum()


def stereo_metrics(
    estimate: np.ndarray,
    valid: np.ndarray,
    confidence: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    oracle = np.isfinite(reference) & (reference > 0)
    accepted = oracle & valid & np.isfinite(estimate) & (estimate > 0)
    weights = _solid_angle_metric_weights(reference.shape)
    coverage = float(weights[accepted].sum() / weights[oracle].sum())
    if not np.any(accepted):
        return {
            "abs_rel": float("inf"),
            "delta1": 0.0,
            "coverage": coverage,
            "errors": [],
            "confidences": [],
        }
    relative = np.abs(estimate[accepted] - reference[accepted]) / reference[accepted]
    ratio = np.maximum(
        estimate[accepted] / reference[accepted],
        reference[accepted] / estimate[accepted],
    )
    accepted_weights = weights[accepted]
    accepted_weights /= accepted_weights.sum()
    return {
        "abs_rel": float(np.sum(accepted_weights * relative)),
        "delta1": float(np.sum(accepted_weights * (ratio < 1.25))),
        "coverage": coverage,
        "errors": relative.tolist(),
        "confidences": confidence[accepted].tolist(),
    }


def _stereo_options(*, analytic: bool) -> SphericalStereoOptions:
    return SphericalStereoOptions(
        min_range=2.5 if analytic else 0.30,
        max_range=5.5 if analytic else 12.0,
        num_hypotheses=128 if analytic else 96,
        window_size=5,
        pole_margin_fraction=0.0,
        min_texture_std=0.002,
        min_confidence=0.0,
        max_matching_cost=0.9,
        bidirectional_consistency=False,
        filter_backend="numpy",
        pyramid_levels=1,
    )


def evaluate_stereo_analytic() -> dict[str, Any]:
    shape = (64, 128)
    centre_b_in_a = np.asarray((0.35, 0.03, 0.12), dtype=np.float32)
    first, reference = _render_textured_sphere(np.zeros(3), shape)
    second, _ = _render_textured_sphere(centre_b_in_a, shape)
    result = SphericalStereoDepth(_stereo_options(analytic=True))(
        first, second, np.eye(3), -centre_b_in_a
    )
    metrics = stereo_metrics(
        result.range, result.validity_mask, result.confidence, reference
    )
    return {
        "application": "stereo",
        "evaluation": "analytic",
        "corpus": "textured-sphere",
        "metrics": {key: metrics[key] for key in ("abs_rel", "delta1", "coverage")},
        "details": {
            "shape_hw": list(shape),
            "options": asdict(_stereo_options(analytic=True)),
        },
    }


def evaluate_stereo_external_pair(
    record: dict[str, Any], shape_hw: tuple[int, int] = (96, 192)
) -> dict[str, Any]:
    image_a = _load_rgb(record["from"]["rgb"], shape_hw)
    image_b = _load_rgb(record["to"]["rgb"], shape_hw)
    depth_a = _load_depth(record["from"]["depth"], shape_hw)
    rotation, translation = _panorai_pose(record)
    application = SphericalStereoDepth(_stereo_options(analytic=False))
    result = application(image_a, image_b, rotation, translation)
    metrics = stereo_metrics(
        result.range, result.validity_mask, result.confidence, depth_a
    )
    return {
        "pair_id": record["pair_id"],
        "abs_rel": metrics["abs_rel"],
        "delta1": metrics["delta1"],
        "coverage": metrics["coverage"],
        "errors": metrics["errors"],
        "confidences": metrics["confidences"],
    }


def _confidence_metrics(pairs: list[dict[str, Any]]) -> tuple[bool, float, list[float]]:
    errors = np.asarray(
        [error for item in pairs for error in item["errors"]], dtype=np.float64
    )
    confidence = np.asarray(
        [value for item in pairs for value in item["confidences"]], dtype=np.float64
    )
    if len(errors) < 4 or np.all(confidence == confidence[0]):
        return False, 0.0, []
    edges = np.quantile(confidence, (0.0, 0.25, 0.5, 0.75, 1.0))
    bins = []
    for index in range(4):
        include = (confidence >= edges[index]) & (
            confidence <= edges[index + 1]
            if index == 3
            else confidence < edges[index + 1]
        )
        bins.append(float(np.mean(errors[include])) if np.any(include) else math.inf)
    monotonic = all(left >= right for left, right in zip(bins, bins[1:]))
    overall = float(np.mean(errors))
    reduction = (overall - bins[-1]) / overall if overall > 0 else 0.0
    return monotonic, reduction, bins


def _stereo_degeneration_checks(
    rotation: np.ndarray, translation: np.ndarray
) -> dict[str, dict[str, Any]]:
    shape = (48, 96)
    blank = np.zeros((*shape, 3), dtype=np.uint8)
    rng = np.random.default_rng(90210)
    texture = rng.integers(0, 256, size=(*shape, 3), dtype=np.uint8)
    cases = {
        "textureless": (blank, blank, rotation, translation),
        "illumination": (
            texture,
            np.clip(texture.astype(np.float32) * 0.05 + 240.0, 0, 255).astype(np.uint8),
            rotation,
            translation,
        ),
        "occlusion": (texture, blank, rotation, translation),
        "motion": (
            texture,
            np.roll(texture, shape[1] // 3, axis=1),
            rotation,
            translation,
        ),
        "zero_baseline": (texture, texture, rotation, np.zeros(3, dtype=np.float64)),
    }
    application = SphericalStereoDepth(_stereo_options(analytic=False))
    results: dict[str, dict[str, Any]] = {}
    for name, (reference, source, case_rotation, case_translation) in cases.items():
        try:
            result = application(reference, source, case_rotation, case_translation)
        except (RuntimeError, ValueError) as exc:
            results[name] = {
                "diagnosed": True,
                "abstained": False,
                "exception": type(exc).__name__,
            }
            continue
        valid_fraction = float(result.validity_mask.mean())
        results[name] = {
            "diagnosed": False,
            "abstained": valid_fraction <= 0.05,
            "valid_fraction": valid_fraction,
        }
    return results


def aggregate_stereo_external(
    records: list[dict[str, Any]], corpus: str
) -> dict[str, Any]:
    pairs = [evaluate_stereo_external_pair(item) for item in records]
    monotonic, reduction, confidence_bins = _confidence_metrics(pairs)
    first = records[0]
    rotation, translation = _panorai_pose(first)
    degeneration_checks = _stereo_degeneration_checks(rotation, translation)
    degeneration_gate = all(
        item["diagnosed"] or item["abstained"] for item in degeneration_checks.values()
    )
    splits = {item["split"] for item in records}
    if len(splits) != 1:
        raise ValueError("external stereo records must have one evidence split")
    return {
        "application": "stereo",
        "evaluation": "external",
        "split": splits.pop(),
        "corpus": corpus,
        "metrics": {
            "median_abs_rel": float(np.median([item["abs_rel"] for item in pairs])),
            "median_delta1": float(np.median([item["delta1"] for item in pairs])),
            "accepted_coverage": float(np.median([item["coverage"] for item in pairs])),
            "confidence_error_monotonic": monotonic,
            "top_confidence_quartile_error_reduction": reduction,
            "degenerations_diagnosed_or_abstained": degeneration_gate,
        },
        "details": {
            "pairs": pairs,
            "confidence_quartile_mean_relative_errors": confidence_bins,
            "degeneration_checks": degeneration_checks,
            "options": asdict(_stereo_options(analytic=False)),
        },
    }


def evaluate_core(selection: dict[str, Any]) -> dict[str, Any]:
    if selection.get("contract") != SELECTION_CONTRACT:
        raise ValueError("selection manifest has the wrong contract")
    if selection.get("protocol_sha256") != sha256_file(DEFAULT_PROTOCOL):
        raise ValueError("selection manifest targets a different protocol")
    records = selection.get("records")
    if not isinstance(records, list):
        raise ValueError("selection records must be a list")
    evidence = [
        evaluate_slic_analytic(),
        evaluate_dog_analytic(),
        evaluate_stereo_analytic(),
    ]
    for corpus in sorted(CORPUS_NAMES.values()):
        subset = [item for item in records if item.get("corpus") == corpus]
        if not subset:
            raise ValueError(f"selection has no records for {corpus}")
        evidence.append(aggregate_dog_external(subset, corpus))
        evidence.append(aggregate_stereo_external(subset, corpus))
    return {
        "contract": EVIDENCE_CONTRACT,
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "selection_sha256": hashlib.sha256(canonical_json(selection)).hexdigest(),
        "records": evidence,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--split-jsonl", type=Path, required=True)
    select.add_argument("--overlap-jsonl", type=Path, required=True)
    select.add_argument("--matterport-root", type=Path, required=True)
    select.add_argument("--stanford-root", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    core = subparsers.add_parser("evaluate-core")
    core.add_argument("--selection", type=Path, required=True)
    core.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "select":
        manifest = build_selection_manifest(
            args.split_jsonl.resolve(),
            args.overlap_jsonl.resolve(),
            {
                "matterport360": args.matterport_root.resolve(),
                "stanford2d3d": args.stanford_root.resolve(),
            },
        )
        write_immutable_json(args.output.resolve(), manifest)
    else:
        evidence = evaluate_core(_load_json_object(args.selection.resolve()))
        write_immutable_json(args.output.resolve(), evidence)


if __name__ == "__main__":
    main()
