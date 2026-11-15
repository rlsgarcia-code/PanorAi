"""Portable revision documents and checksummed numerical exports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from panorai.geometry import erp_pixels_to_rays
from panorai.reconstruction import SphericalCameraPose
from ._models import (
    INTERFACE,
    PlaneCorrespondence,
    PlanarPointObservation,
    PlanarRegion,
    RegisteredPlanarScene,
    RegisteredPlanarView,
)

SCENE_SCHEMA = "panorai-planar-scene/v1"


def encode_planar_mask(mask):
    """Row-major RLE: alternating [start, length] pairs for True runs."""
    array = np.asarray(mask)
    if array.ndim != 2 or array.dtype != np.bool_:
        raise ValueError("mask must be boolean HW")
    flat = array.ravel()
    transitions = np.diff(np.r_[False, flat, False].astype(np.int8))
    starts, ends = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
    pairs = np.stack((starts, ends - starts), axis=1).ravel().tolist()
    return {"shape_hw": list(array.shape), "runs": pairs}


def decode_planar_mask(value, shape_hw=None):
    if not isinstance(value, dict) or set(value) != {"shape_hw", "runs"}:
        raise ValueError("expected RLE mask with shape_hw and runs")
    shape = value["shape_hw"]
    if len(shape) != 2 or any(type(x) is not int or x < 1 for x in shape):
        raise ValueError("invalid mask shape")
    if shape_hw is not None and tuple(shape) != tuple(shape_hw):
        raise ValueError("mask does not align with native ERP")
    size = shape[0] * shape[1]
    if size > 160_000_000:
        raise ValueError("mask exceeds 160 million pixels")
    runs = value["runs"]
    if len(runs) % 2 or any(type(x) is not int for x in runs):
        raise ValueError("RLE runs must be integer pairs")
    result = np.zeros(size, dtype=bool)
    previous = 0
    for start, length in zip(runs[::2], runs[1::2]):
        if start < previous or length < 1 or start + length > size:
            raise ValueError("RLE runs must be ordered, disjoint, and in bounds")
        result[start : start + length] = True
        previous = start + length
    return result.reshape(shape)


def planar_scene_from_document(
    document: Mapping[str, Any], images: Mapping[str, np.ndarray]
):
    """Load an explicit geometry-v1 scene; image keys resolve only via images.

    pose_convention is mandatory. camera_to_world uses R and center_world;
    world_to_camera uses R and either center_world or translation. No guessed
    convention, camera model, scale, support or missing-range validity.
    """
    if (
        document.get("schema") != SCENE_SCHEMA
        or document.get("geometry_convention") != "panorai-geometry-v1"
    ):
        raise ValueError("unsupported planar scene or geometry convention")
    views = []
    for item in document["views"]:
        image = images[item["image_key"]]
        shape = image.shape[:2]
        rotation = np.asarray(item["rotation"], dtype=np.float64)
        convention = item["pose_convention"]
        if convention not in {"world_to_camera", "camera_to_world"}:
            raise ValueError(
                "pose_convention must be world_to_camera or camera_to_world"
            )
        if convention == "camera_to_world":
            rotation = rotation.T
        if "center_world" in item:
            center = np.asarray(item["center_world"], dtype=np.float64)
        elif convention == "world_to_camera" and "translation" in item:
            center = -rotation.T @ np.asarray(item["translation"], dtype=np.float64)
        else:
            raise ValueError("explicit camera center is required")
        support = (
            np.ones(shape, bool)
            if item.get("support") == "full"
            else decode_planar_mask(item["support"], shape)
        )
        baseline = item.get("baseline_range")
        if baseline is not None:
            baseline = np.asarray(baseline, dtype=np.float64)
        validity = (
            None
            if baseline is None
            else decode_planar_mask(item["baseline_validity"], shape)
        )
        views.append(
            RegisteredPlanarView(
                item["view_id"],
                image,
                SphericalCameraPose(item["view_id"], rotation, center),
                support,
                baseline,
                validity,
            )
        )
    shapes = {v.view_id: v.support.shape for v in views}
    regions = []
    for item in document.get("regions", []):
        shape = shapes[item["view_id"]]
        kwargs = {
            name: decode_planar_mask(item[name], shape)
            if item.get(name) is not None
            else None
            for name in ("positive", "negative", "roi")
        }
        regions.append(
            PlanarRegion(
                item["plane_id"],
                item["view_id"],
                decode_planar_mask(item["mask"], shape),
                item.get("state", "draft"),
                item.get("label", "plane"),
                **kwargs,
                interpretation=item.get("interpretation", "final"),
                history=tuple(item.get("history", [])),
                source=item.get("source", "human"),
            )
        )
    tracks = []
    for item in document.get("correspondences", []):
        observations = tuple(
            PlanarPointObservation(
                o["view_id"], tuple(o["pixel_xy"]), o.get("weight", 1)
            )
            for o in item["observations"]
        )
        tracks.append(
            PlaneCorrespondence(
                item["plane_id"],
                item["track_id"],
                observations,
                item.get("source", "human"),
            )
        )
    return RegisteredPlanarScene(
        tuple(views),
        tuple(regions),
        tuple(tracks),
        document.get("scene_id", "scene"),
        document.get("revision", "unversioned"),
        document.get("units", ""),
    )


def region_document(region):
    result = {
        "plane_id": region.plane_id,
        "view_id": region.view_id,
        "label": region.label,
        "state": region.state,
        "mask": encode_planar_mask(region.mask),
        "interpretation": region.interpretation,
        "source": region.source,
        "history": [dict(h) for h in region.history],
    }
    for name in ("positive", "negative", "roi"):
        if getattr(region, name) is not None:
            result[name] = encode_planar_mask(getattr(region, name))
    return result


def planar_result_summary(workspace):
    return {
        "interface": INTERFACE,
        "scene_id": workspace.scene.scene_id,
        "revision": workspace.scene.revision,
        "planes": [
            {
                "plane_id": p.plane_id,
                "state": p.state,
                "normal_world": None
                if p.normal_world is None
                else p.normal_world.tolist(),
                "offset_m": p.offset_m,
                "covariance": None if p.covariance is None else p.covariance.tolist(),
                "tangent_basis": None
                if p.tangent_basis is None
                else p.tangent_basis.tolist(),
                "diagnostics": dict(p.diagnostics),
                "inlier_track_ids": list(p.inlier_track_ids),
                "heldout_track_ids": list(p.heldout_track_ids),
                "covariance_origin_world": None
                if p.covariance_origin_world is None
                else p.covariance_origin_world.tolist(),
            }
            for p in workspace.planes
        ],
        "tracks": [
            {
                "plane_id": t.correspondence.plane_id,
                "track_id": t.correspondence.track_id,
                "source": t.correspondence.source,
                "observations": [
                    {"view_id": o.view_id, "pixel_xy": list(o.pixel_xy)}
                    for o in t.correspondence.observations
                ],
                "point_world": None
                if t.point_world is None
                else t.point_world.tolist(),
                "sigma_m": t.sigma_m,
                "parallax_deg": t.parallax_deg,
                "residual_deg": t.residual_deg,
                "reason": t.reason,
            }
            for t in workspace.tracks
        ],
        "proposals": [region_document(r) for r in workspace.proposals],
        "regions": [region_document(r) for r in workspace.scene.regions],
        "segmentation_alternatives": [
            {
                "plane_id": p,
                "view_id": v,
                "masks": [encode_planar_mask(m) for m in masks],
            }
            for p, v, masks in workspace.segmentation_alternatives
        ],
        "ranges": [
            {
                "view_id": r.view_id,
                "shape_hw": list(r.range.shape),
                "valid_pixels": int(r.validity.sum()),
                "conflict_pixels": int(r.conflict.sum()),
                "plane_ids": list(r.plane_ids),
            }
            for r in workspace.ranges
        ],
        "uncertainty": "fixed-pose linearized standard deviation; excludes pose/mask/systematic errors",
    }


def export_planar_result(workspace, directory, *, pipeline_manifest=None):
    """Export native NPZs, binary PLYs, masks/planes/tracks and SHA-256 manifest.

    A new directory is required. Each PLY contains fitted surface points;
    measured triangulated support remains separately identified in result.json.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    summary = planar_result_summary(workspace)
    (directory / "result.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    views = {v.view_id: v for v in workspace.scene.views}
    for index, result in enumerate(workspace.ranges):
        view = views[result.view_id]
        prefix = f"view-{index:04d}"
        np.savez_compressed(
            directory / f"{prefix}.npz",
            range=result.range,
            validity=result.validity,
            support=view.support,
            uncertainty_m=result.uncertainty_m,
            plane_index=result.plane_index,
            conflict=result.conflict,
            composed_range=result.composed_range,
            composed_validity=result.composed_validity,
            source=result.source,
        )
        count = int(result.validity.sum())
        with (directory / f"{prefix}.ply").open("wb") as stream:
            header = f"ply\nformat binary_little_endian 1.0\ncomment world coordinates; metres; plane inferred\nelement vertex {count}\nproperty double x\nproperty double y\nproperty double z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
            stream.write(header.encode("ascii"))
            dtype = np.dtype([("xyz", "<f8", (3,)), ("rgb", "u1", (3,))])
            h, w = result.range.shape
            for row in range(0, h, 128):
                ys, xs = np.where(result.validity[row : row + 128])
                ys += row
                if not len(xs):
                    continue
                rays = (
                    erp_pixels_to_rays(
                        np.stack((xs, ys), axis=-1).astype(float), (h, w)
                    )
                    @ view.pose.R
                )
                points = view.pose.center + rays * result.range[ys, xs, None]
                records = np.empty(len(xs), dtype=dtype)
                records["xyz"], records["rgb"] = points, view.image[ys, xs]
                stream.write(records.tobytes())
    files = []
    for path in sorted(directory.iterdir()):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        files.append(
            {
                "file": path.name,
                "sha256": digest.hexdigest(),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema": "panorai-planar-export/v1",
        "interface": INTERFACE,
        "scene_id": workspace.scene.scene_id,
        "revision": workspace.scene.revision,
        "quantity": "radial_range",
        "units": "m",
        "geometry_convention": "panorai-geometry-v1",
        "files": files,
        "pipeline": pipeline_manifest,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    return manifest
