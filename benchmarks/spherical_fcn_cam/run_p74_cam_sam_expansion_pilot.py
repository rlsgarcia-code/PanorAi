#!/usr/bin/env python3
"""Expand CAM-guided SAM masks across spherical charts and residual peaks.

The workflow has two explicit phases:

1. same-instance expansion follows only frontiers touched by a two-of-three
   majority consensus between the SAM alternatives;
2. new-instance discovery suppresses the accumulated possible envelope from a
   predeclared proxy ensemble, prompts the strongest remaining spherical peaks,
   and accepts a hypothesis only when at least two alternatives agree.

Every SAM alternative is retained in decoder order.  Consensus is a majority
vote and envelope is a union, never a hidden best-mask choice.
Only one RGB gnomonic face and image embedding exist at a time.
"""

from __future__ import annotations

import argparse
from collections import deque
import gc
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.spherical_fcn_cam.run_p74_cam_sam_prompt_pilot import (  # noqa: E402
    EXPECTED_MULTIMASK_COUNT,
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    _download_model,
    _draw_cross,
    _mask_overlay,
    _panel,
    _resolve_device,
    _sha256,
    erp_pixel_from_lon_lat,
    pairwise_mask_iou,
)
from panorai.geometry import (  # noqa: E402
    GnomonicSpec,
    equirectangular_to_gnomonic,
    gnomonic_pixels_to_rays,
    gnomonic_to_equirectangular,
    rays_to_gnomonic_pixels,
)

PIPE_PROXY_WEIGHTS = {
    653: 0.35,  # milk can: vertical cylinders
    821: 0.30,  # steel arch bridge: curved tubular structures
    822: 0.15,  # steel drum: metal cylinders
    427: 0.10,  # barrel: broad cylinders
    758: 0.10,  # reel: circular/curved structures
}


def load_dense_proxy_evidence(path: Path) -> dict[str, Any]:
    """Load and validate the compact native proxy lattice."""

    with np.load(path, allow_pickle=False) as stored:
        schema = stored["schema"].item()
        indices = stored["class_indices"].astype(np.int64)
        names = stored["class_names"].astype(str)
        logits = stored["logits"].astype(np.float32)
        support = stored["support"].astype(bool)
    if schema != "panorai-internimage-proxy-evidence/v1":
        raise ValueError(f"unsupported evidence schema: {schema!r}")
    if logits.ndim != 3 or logits.shape[0] != len(indices):
        raise ValueError("proxy logits must have shape (C,H,W)")
    if support.shape != logits.shape[-2:]:
        raise ValueError("proxy support does not match the logit lattice")
    if len(set(indices.tolist())) != len(indices):
        raise ValueError("proxy class indices must be unique")
    return {
        "class_indices": indices,
        "class_names": names,
        "logits": logits,
        "support": support,
    }


def normalized_proxy_ensemble(
    evidence: dict[str, Any], weights: dict[int, float]
) -> np.ndarray:
    """Combine independently normalized positive proxy evidence."""

    if not weights or any(value <= 0 for value in weights.values()):
        raise ValueError("proxy weights must be nonempty and positive")
    indices = evidence["class_indices"].tolist()
    logits = evidence["logits"]
    support = evidence["support"]
    combined = np.zeros(support.shape, dtype=np.float64)
    total = 0.0
    for class_index, weight in weights.items():
        if class_index not in indices:
            raise ValueError(f"proxy class {class_index} is absent from the evidence")
        channel = np.maximum(logits[indices.index(class_index)].astype(np.float64), 0.0)
        values = channel[support]
        scale = float(values.max() - values.min())
        normalized = np.zeros_like(channel)
        if scale > np.finfo(np.float64).eps:
            normalized[support] = (values - values.min()) / scale
        combined += weight * normalized
        total += weight
    combined /= total
    return np.where(support, np.clip(combined, 0.0, 1.0), 0.0)


def _direction_from_cell(
    row: int, column: int, shape_hw: tuple[int, int]
) -> tuple[float, float]:
    height, width = shape_hw
    longitude = ((column + 0.5) / width * 360.0) - 180.0
    latitude = 90.0 - ((row + 0.5) / height * 180.0)
    return longitude, latitude


def _unit_ray(longitude_deg: float, latitude_deg: float) -> np.ndarray:
    longitude = math.radians(longitude_deg)
    latitude = math.radians(latitude_deg)
    cosine = math.cos(latitude)
    return np.asarray(
        [
            cosine * math.sin(longitude),
            math.sin(latitude),
            cosine * math.cos(longitude),
        ],
        dtype=np.float64,
    )


def _lon_lat(ray: np.ndarray) -> tuple[float, float]:
    ray = ray / np.linalg.norm(ray)
    return math.degrees(math.atan2(ray[0], ray[2])), math.degrees(math.asin(ray[1]))


def angular_distance_degrees(
    first: tuple[float, float], second: tuple[float, float]
) -> float:
    one = _unit_ray(*first)
    two = _unit_ray(*second)
    return math.degrees(math.acos(float(np.clip(np.dot(one, two), -1.0, 1.0))))


def greedy_spherical_peaks(
    evidence: np.ndarray,
    support: np.ndarray,
    exclusion: np.ndarray,
    *,
    maximum_count: int,
    minimum_separation_degrees: float,
    relative_threshold: float,
) -> list[dict[str, float | int]]:
    """Select strong residual cells with geodesic non-maximum suppression."""

    if evidence.shape != support.shape or evidence.shape != exclusion.shape:
        raise ValueError("evidence, support, and exclusion must have matching shapes")
    valid = support & ~exclusion & np.isfinite(evidence)
    if not valid.any() or maximum_count <= 0:
        return []
    residual = np.where(valid, evidence, -np.inf)
    maximum = float(np.max(residual))
    if not math.isfinite(maximum) or maximum <= 0:
        return []
    order = np.argsort(residual.ravel())[::-1]
    selected: list[dict[str, float | int]] = []
    for flat_index in order:
        score = float(residual.ravel()[flat_index])
        if score < relative_threshold * maximum:
            break
        row, column = np.unravel_index(flat_index, residual.shape)
        longitude, latitude = _direction_from_cell(row, column, residual.shape)
        if any(
            angular_distance_degrees(
                (longitude, latitude),
                (float(item["longitude_degrees"]), float(item["latitude_degrees"])),
            )
            < minimum_separation_degrees
            for item in selected
        ):
            continue
        selected.append(
            {
                "row": int(row),
                "column": int(column),
                "score": score,
                "longitude_degrees": longitude,
                "latitude_degrees": latitude,
            }
        )
        if len(selected) >= maximum_count:
            break
    return selected


def spherical_dilate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Dilate a coarse ERP mask with longitude wrap and no latitude wrap."""

    if mask.ndim != 2 or radius_cells < 0:
        raise ValueError("mask must be HW and radius_cells nonnegative")
    result = mask.astype(bool).copy()
    source = result.copy()
    for delta_y in range(-radius_cells, radius_cells + 1):
        for delta_x in range(-radius_cells, radius_cells + 1):
            if delta_x * delta_x + delta_y * delta_y > radius_cells * radius_cells:
                continue
            shifted = np.roll(source, delta_x, axis=1)
            if delta_y > 0:
                shifted = np.pad(shifted[:-delta_y], ((delta_y, 0), (0, 0)))
            elif delta_y < 0:
                shifted = np.pad(shifted[-delta_y:], ((0, -delta_y), (0, 0)))
            result |= shifted
    return result


def erp_mask_to_lattice(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    resized = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
        (shape_hw[1], shape_hw[0]), Image.Resampling.NEAREST
    )
    return np.asarray(resized) > 0


def consensus_frontiers(
    masks: np.ndarray, *, band_pixels: int, minimum_pixels: int
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Find chart edges touched by a two-of-three SAM majority consensus."""

    if masks.ndim != 3 or len(masks) != EXPECTED_MULTIMASK_COUNT:
        raise ValueError("expected exactly three NHW masks")
    height, width = masks.shape[-2:]
    if not 0 < band_pixels < min(height, width) // 2:
        raise ValueError("invalid frontier band")
    consensus = masks.sum(axis=0) >= 2
    envelope = masks.any(axis=0)
    regions = {
        "top": (slice(0, band_pixels), slice(None)),
        "bottom": (slice(height - band_pixels, height), slice(None)),
        "left": (slice(None), slice(0, band_pixels)),
        "right": (slice(None), slice(width - band_pixels, width)),
    }
    frontiers = []
    for side, region in regions.items():
        local_y, local_x = np.where(consensus[region])
        if len(local_y) < minimum_pixels:
            continue
        if side == "bottom":
            local_y += height - band_pixels
        if side == "right":
            local_x += width - band_pixels
        if side == "top":
            point = [float(np.median(local_x)), 0.0]
        elif side == "bottom":
            point = [float(np.median(local_x)), float(height - 1)]
        elif side == "left":
            point = [0.0, float(np.median(local_y))]
        else:
            point = [float(width - 1), float(np.median(local_y))]
        frontiers.append(
            {
                "side": side,
                "consensus_pixels_in_band": int(len(local_y)),
                "face_point_xy": point,
            }
        )
    return consensus, envelope, frontiers


def maximum_pairwise_iou(masks: np.ndarray) -> float:
    """Return the strongest agreement between two distinct SAM alternatives."""

    matrix = np.asarray(pairwise_mask_iou(masks), dtype=np.float64)
    if matrix.shape != (EXPECTED_MULTIMASK_COUNT, EXPECTED_MULTIMASK_COUNT):
        raise ValueError("expected exactly three masks")
    return float(np.max(matrix[np.triu_indices(EXPECTED_MULTIMASK_COUNT, k=1)]))


def advance_chart(
    spec: GnomonicSpec,
    frontier_xy: tuple[float, float],
    *,
    step_fraction: float,
) -> tuple[GnomonicSpec, np.ndarray, tuple[float, float]]:
    """Advance a chart centre geodesically beyond one touched frontier."""

    frontier_ray = gnomonic_pixels_to_rays(
        np.asarray(frontier_xy, dtype=np.float64), spec
    ).rays_xyz
    center_ray = _unit_ray(spec.center_lon_deg, spec.center_lat_deg)
    cosine = float(np.clip(np.dot(center_ray, frontier_ray), -1.0, 1.0))
    angle = math.acos(cosine)
    if angle <= 1e-8:
        raise ValueError("frontier direction is indistinguishable from chart centre")
    tangent = (frontier_ray - cosine * center_ray) / math.sin(angle)
    step = math.radians(step_fraction * min(spec.hfov_deg, spec.vfov_deg))
    new_ray = math.cos(step) * center_ray + math.sin(step) * tangent
    longitude, latitude = _lon_lat(new_ray)
    next_spec = GnomonicSpec(
        center_lat_deg=latitude,
        center_lon_deg=longitude,
        hfov_deg=spec.hfov_deg,
        vfov_deg=spec.vfov_deg,
        output_shape_hw=spec.output_shape_hw,
    )
    projected = rays_to_gnomonic_pixels(frontier_ray, next_spec)
    if not bool(projected.valid):
        raise RuntimeError("frontier prompt does not lie in the advanced chart")
    prompt_xy = tuple(float(value) for value in projected.pixels_xy)
    return next_spec, frontier_ray, prompt_xy


class SamSession:
    """One loaded SAM model with transient per-face embeddings."""

    def __init__(self, snapshot: Path, device_name: str) -> None:
        import torch
        from transformers import SamModel, SamProcessor

        self.torch = torch
        self.device = _resolve_device(device_name, torch)
        self.processor = SamProcessor.from_pretrained(snapshot, local_files_only=True)
        self.model = (
            SamModel.from_pretrained(
                snapshot, local_files_only=True, use_safetensors=True
            )
            .eval()
            .to(self.device)
        )

    def predict(
        self, face: Image.Image, positive_points_xy: list[tuple[float, float]]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        inputs = self.processor(
            images=face,
            input_points=[[list(point) for point in positive_points_xy]],
            input_labels=[[1] * len(positive_points_xy)],
            return_tensors="pt",
        )
        original_sizes = inputs["original_sizes"].cpu()
        reshaped_sizes = inputs["reshaped_input_sizes"].cpu()
        model_keys = {"pixel_values", "input_points", "input_labels"}
        model_inputs = {}
        for key, value in inputs.items():
            if key not in model_keys:
                continue
            if self.torch.is_floating_point(value):
                value = value.to(dtype=self.torch.float32)
            model_inputs[key] = value.to(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model(**model_inputs, multimask_output=True)
        if self.device == "mps":
            self.torch.mps.synchronize()
        seconds = time.perf_counter() - started
        processed = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            original_sizes,
            reshaped_sizes,
            binarize=True,
        )[0]
        while processed.ndim > 3 and processed.shape[0] == 1:
            processed = processed.squeeze(0)
        masks = processed.numpy().astype(bool)
        scores = (
            outputs.iou_scores.detach().cpu().reshape(-1).numpy().astype(np.float64)
        )
        if (
            masks.shape[0] != EXPECTED_MULTIMASK_COUNT
            or len(scores) != EXPECTED_MULTIMASK_COUNT
        ):
            raise RuntimeError(
                f"SAM returned masks={masks.shape}, scores={scores.shape}"
            )
        del outputs, model_inputs, inputs
        return masks, scores, seconds

    def close(self) -> None:
        del self.model, self.processor
        gc.collect()
        if self.device == "mps":
            self.torch.mps.empty_cache()
        elif self.device == "cuda":
            self.torch.cuda.empty_cache()


def _face_from_erp(source: np.ndarray, spec: GnomonicSpec) -> Image.Image:
    projected = equirectangular_to_gnomonic(source, spec, interpolation="bilinear").data
    return Image.fromarray(
        np.clip(np.rint(projected), 0, 255).astype(np.uint8), mode="RGB"
    )


def _back_project_masks(
    masks: np.ndarray, spec: GnomonicSpec, output_shape_hw: tuple[int, int]
) -> list[np.ndarray]:
    return [
        gnomonic_to_equirectangular(
            mask.astype(np.uint8),
            spec,
            output_shape_hw,
            interpolation="nearest",
            fill_value=0,
        ).data.astype(bool)
        for mask in masks
    ]


def _save_all_mask_sheet(
    face: Image.Image,
    masks: np.ndarray,
    scores: np.ndarray,
    prompt_xy: tuple[float, float],
    title: str,
    output_path: Path,
) -> None:
    prompted = face.copy()
    _draw_cross(prompted, prompt_xy)
    panels = [_panel(prompted, title, (560, 560))]
    for index, (mask, score) in enumerate(zip(masks, scores, strict=True)):
        panels.append(
            _panel(
                _mask_overlay(face, mask, prompt_xy=prompt_xy),
                f"mask {index + 1}/3 - decoder order - IoU {score:.3f}",
                (560, 560),
            )
        )
    sheet = Image.new("RGB", (1120, 1120), (235, 235, 235))
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 2) * 560, (index // 2) * 560))
    sheet.save(output_path, quality=95)


def _save_masks(
    output_dir: Path,
    prefix: str,
    masks: np.ndarray,
    erp_masks: list[np.ndarray],
) -> list[dict[str, str]]:
    files = []
    for index, (mask, erp_mask) in enumerate(
        zip(masks, erp_masks, strict=True), start=1
    ):
        face_name = f"{prefix}-mask-{index:02d}-face.png"
        erp_name = f"{prefix}-mask-{index:02d}-erp.png"
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
            output_dir / face_name
        )
        Image.fromarray(erp_mask.astype(np.uint8) * 255, mode="L").save(
            output_dir / erp_name
        )
        files.append({"face_mask": face_name, "erp_mask": erp_name})
    return files


def _overlay_regions(
    base: Image.Image,
    regions: list[tuple[np.ndarray, tuple[int, int, int]]],
) -> Image.Image:
    result = np.asarray(base.convert("RGB"), dtype=np.float32).copy()
    for mask, color in regions:
        if not mask.any():
            continue
        target = np.asarray(color, dtype=np.float32)
        result[mask] = 0.52 * result[mask] + 0.48 * target
    return Image.fromarray(
        np.clip(np.rint(result), 0, 255).astype(np.uint8), mode="RGB"
    )


def _heat_overlay(base: Image.Image, evidence: np.ndarray) -> Image.Image:
    resized = Image.fromarray(
        np.rint(evidence * 255).astype(np.uint8), mode="L"
    ).resize(base.size, Image.Resampling.BILINEAR)
    values = np.asarray(resized, dtype=np.float32) / 255.0
    heat = np.stack(
        (
            np.clip(2.0 * values, 0, 1),
            np.clip(2.0 - 2.0 * np.abs(values - 0.5), 0, 1),
            np.clip(2.0 * (1.0 - values), 0, 1),
        ),
        axis=-1,
    )
    rgb = np.asarray(base, dtype=np.float32) / 255.0
    return Image.fromarray(
        np.rint(255 * np.clip(0.58 * rgb + 0.42 * heat, 0, 1)).astype(np.uint8)
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--face-size", type=int, default=1024)
    parser.add_argument("--same-instance-fov", type=float, default=30.0)
    parser.add_argument("--new-instance-fov", type=float, default=70.0)
    parser.add_argument("--max-same-instance-faces", type=int, default=4)
    parser.add_argument("--max-new-instances", type=int, default=3)
    parser.add_argument("--minimum-mask-agreement", type=float, default=0.50)
    parser.add_argument("--frontier-band", type=int, default=32)
    parser.add_argument("--frontier-min-pixels", type=int, default=128)
    parser.add_argument("--accept-model-license", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 0.0 <= args.minimum_mask_agreement <= 1.0:
        raise ValueError("--minimum-mask-agreement must be in [0, 1]")
    record = json.loads(args.record.read_text(encoding="utf-8"))
    evidence = load_dense_proxy_evidence(args.evidence)
    prior = normalized_proxy_ensemble(evidence, PIPE_PROXY_WEIGHTS)
    source_path = Path(record["source_path"])
    preview_path = args.record.parent / "input-erp.jpg"
    source = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.float32)
    preview = Image.open(preview_path).convert("RGB")
    preview_shape = (preview.height, preview.width)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidate = next(
        item
        for item in record["candidate_predictions"]
        if item["class_name"] == "milk can"
    )
    seed_stats = candidate["cam_statistics"]
    seed_spec = GnomonicSpec(
        center_lat_deg=seed_stats["peak_latitude_degrees"],
        center_lon_deg=seed_stats["peak_longitude_degrees"],
        hfov_deg=args.same_instance_fov,
        vfov_deg=args.same_instance_fov,
        output_shape_hw=(args.face_size, args.face_size),
    )
    seed_prompt = ((args.face_size - 1) / 2.0, (args.face_size - 1) / 2.0)
    snapshot, checkpoint_files = _download_model(
        args.cache_dir, accept_model_license=args.accept_model_license
    )
    session = SamSession(snapshot, args.device)

    global_consensus = np.zeros(preview_shape, dtype=bool)
    global_envelope = np.zeros(preview_shape, dtype=bool)
    same_steps: list[dict[str, Any]] = []
    queue: deque[tuple[GnomonicSpec, tuple[float, float], int | None, str]] = deque(
        [(seed_spec, seed_prompt, None, "seed")]
    )
    visited_centres: list[tuple[float, float]] = []
    while queue and len(same_steps) < args.max_same_instance_faces:
        spec, prompt_xy, parent, source_frontier = queue.popleft()
        centre = (spec.center_lon_deg, spec.center_lat_deg)
        if any(
            angular_distance_degrees(centre, previous) < 0.30 * spec.hfov_deg
            for previous in visited_centres
        ):
            continue
        visited_centres.append(centre)
        face = _face_from_erp(source, spec)
        masks, scores, seconds = session.predict(face, [prompt_xy])
        erp_masks = _back_project_masks(masks, spec, preview_shape)
        consensus, envelope, frontiers = consensus_frontiers(
            masks,
            band_pixels=args.frontier_band,
            minimum_pixels=args.frontier_min_pixels,
        )
        consensus_erp = np.sum(erp_masks, axis=0) >= 2
        envelope_erp = np.logical_or.reduce(erp_masks)
        global_consensus |= consensus_erp
        global_envelope |= envelope_erp
        step_index = len(same_steps)
        prefix = f"same-step-{step_index:02d}"
        mask_files = _save_masks(args.output_dir, prefix, masks, erp_masks)
        sheet_name = f"{prefix}-all-masks.jpg"
        _save_all_mask_sheet(
            face,
            masks,
            scores,
            prompt_xy,
            f"same instance step {step_index} - {source_frontier}",
            args.output_dir / sheet_name,
        )
        step_record = {
            "step_index": step_index,
            "parent_step": parent,
            "source_frontier": source_frontier,
            "center_longitude_degrees": spec.center_lon_deg,
            "center_latitude_degrees": spec.center_lat_deg,
            "fov_degrees": spec.hfov_deg,
            "prompt_xy": list(prompt_xy),
            "predicted_iou_decoder_order": scores.tolist(),
            "pairwise_iou": pairwise_mask_iou(masks),
            "consensus_face_fraction": float(consensus.mean()),
            "envelope_face_fraction": float(envelope.mean()),
            "frontiers": frontiers,
            "inference_seconds": seconds,
            "mask_files": mask_files,
            "all_masks_sheet": sheet_name,
        }
        same_steps.append(step_record)
        for frontier in frontiers:
            if len(same_steps) + len(queue) >= args.max_same_instance_faces:
                break
            next_spec, _, next_prompt = advance_chart(
                spec,
                tuple(frontier["face_point_xy"]),
                step_fraction=0.65,
            )
            next_centre = (next_spec.center_lon_deg, next_spec.center_lat_deg)
            if any(
                angular_distance_degrees(next_centre, previous) < 0.30 * spec.hfov_deg
                for previous in visited_centres
            ):
                frontier["queued"] = False
                frontier["skip_reason"] = "visited-centre"
                continue
            frontier["queued"] = True
            frontier["next_center_longitude_degrees"] = next_spec.center_lon_deg
            frontier["next_center_latitude_degrees"] = next_spec.center_lat_deg
            queue.append((next_spec, next_prompt, step_index, frontier["side"]))
        del face, masks, erp_masks, consensus, envelope

    Image.fromarray(global_consensus.astype(np.uint8) * 255, mode="L").save(
        args.output_dir / "same-instance-consensus-erp.png"
    )
    Image.fromarray(global_envelope.astype(np.uint8) * 255, mode="L").save(
        args.output_dir / "same-instance-envelope-erp.png"
    )

    lattice_exclusion = spherical_dilate(
        erp_mask_to_lattice(global_envelope, prior.shape), radius_cells=2
    )
    peak_candidates = greedy_spherical_peaks(
        prior,
        evidence["support"],
        lattice_exclusion,
        maximum_count=max(12, 4 * args.max_new_instances),
        minimum_separation_degrees=12.0,
        relative_threshold=0.30,
    )
    new_instances: list[dict[str, Any]] = []
    instance_envelopes: list[np.ndarray] = []
    evaluated_candidates: list[dict[str, Any]] = []
    for peak in peak_candidates:
        if len(new_instances) >= args.max_new_instances:
            break
        longitude = float(peak["longitude_degrees"])
        latitude = float(peak["latitude_degrees"])
        erp_x, erp_y = erp_pixel_from_lon_lat(longitude, latitude, preview_shape)
        if global_envelope[
            int(round(erp_y)) % preview.height, int(round(erp_x)) % preview.width
        ]:
            continue
        if any(
            mask[int(round(erp_y)) % preview.height, int(round(erp_x)) % preview.width]
            for mask in instance_envelopes
        ):
            continue
        spec = GnomonicSpec(
            center_lat_deg=latitude,
            center_lon_deg=longitude,
            hfov_deg=args.new_instance_fov,
            vfov_deg=args.new_instance_fov,
            output_shape_hw=(args.face_size, args.face_size),
        )
        prompt_xy = ((args.face_size - 1) / 2.0, (args.face_size - 1) / 2.0)
        face = _face_from_erp(source, spec)
        masks, scores, seconds = session.predict(face, [prompt_xy])
        erp_masks = _back_project_masks(masks, spec, preview_shape)
        consensus_erp = np.sum(erp_masks, axis=0) >= 2
        envelope_erp = np.logical_or.reduce(erp_masks)
        pairwise_iou = pairwise_mask_iou(masks)
        maximum_agreement = maximum_pairwise_iou(masks)
        accepted = maximum_agreement >= args.minimum_mask_agreement
        candidate_number = len(evaluated_candidates) + 1
        prefix = f"new-candidate-{candidate_number:02d}"
        mask_files = _save_masks(args.output_dir, prefix, masks, erp_masks)
        sheet_name = f"{prefix}-all-masks.jpg"
        _save_all_mask_sheet(
            face,
            masks,
            scores,
            prompt_xy,
            (
                f"new candidate {candidate_number} - evidence {peak['score']:.3f} - "
                f"{'accepted' if accepted else 'rejected'}"
            ),
            args.output_dir / sheet_name,
        )
        candidate_record = {
            "candidate_number": candidate_number,
            "accepted": accepted,
            "acceptance_reason": (
                "two-mask-agreement"
                if accepted
                else "maximum-pairwise-iou-below-threshold"
            ),
            "peak": peak,
            "fov_degrees": spec.hfov_deg,
            "predicted_iou_decoder_order": scores.tolist(),
            "pairwise_iou": pairwise_iou,
            "maximum_pairwise_iou": maximum_agreement,
            "majority_consensus_erp_fraction": float(consensus_erp.mean()),
            "envelope_erp_fraction": float(envelope_erp.mean()),
            "inference_seconds": seconds,
            "mask_files": mask_files,
            "all_masks_sheet": sheet_name,
        }
        evaluated_candidates.append(candidate_record)
        if accepted:
            instance_number = len(new_instances) + 1
            candidate_record["instance_number"] = instance_number
            new_instances.append(candidate_record)
            instance_envelopes.append(envelope_erp)
        del face, masks, erp_masks

    colors = [(255, 30, 120), (0, 210, 255), (255, 190, 0), (80, 230, 80)]
    same_overlay = _overlay_regions(
        preview,
        [
            (global_envelope, (0, 210, 255)),
            (global_consensus, (255, 30, 120)),
        ],
    )
    for centre in visited_centres:
        _draw_cross(
            same_overlay,
            erp_pixel_from_lon_lat(centre[0], centre[1], preview_shape),
            radius=9,
        )
    new_overlay = _overlay_regions(
        preview,
        [
            (mask, colors[index % len(colors)])
            for index, mask in enumerate(instance_envelopes)
        ],
    )
    draw = ImageDraw.Draw(new_overlay)
    for instance in new_instances:
        peak = instance["peak"]
        xy = erp_pixel_from_lon_lat(
            float(peak["longitude_degrees"]),
            float(peak["latitude_degrees"]),
            preview_shape,
        )
        _draw_cross(new_overlay, xy, radius=9)
        draw.text(
            (xy[0] + 10, xy[1] + 6), str(instance["instance_number"]), fill="white"
        )
    summary = Image.new("RGB", (3 * 640, 430), (235, 235, 235))
    summary.paste(
        _panel(_heat_overlay(preview, prior), "Piping proxy ensemble", (640, 430)),
        (0, 0),
    )
    summary.paste(
        _panel(
            same_overlay, "Same instance: cyan envelope, magenta consensus", (640, 430)
        ),
        (640, 0),
    )
    summary.paste(
        _panel(
            new_overlay,
            "Geometry-stable residual hypotheses: every color is one prompt",
            (640, 430),
        ),
        (1280, 0),
    )
    summary_path = args.output_dir / "spherical-expansion-summary.jpg"
    summary.save(summary_path, quality=95)

    result = {
        "schema": "panorai-p74-cam-sam-spherical-expansion/v1",
        "sample_id": record["sample_id"],
        "predeclared_proxy_weights": {
            str(key): value for key, value in PIPE_PROXY_WEIGHTS.items()
        },
        "same_instance": {
            "rule": "expand only the two-of-three majority consensus when it touches a chart frontier",
            "requested_max_faces": args.max_same_instance_faces,
            "processed_face_count": len(same_steps),
            "steps": same_steps,
            "global_consensus_erp_fraction": float(global_consensus.mean()),
            "global_envelope_erp_fraction": float(global_envelope.mean()),
        },
        "new_instances": {
            "rule": "suppress the same-instance envelope, geodesically NMS residual peaks, and retain geometry-stable hypotheses with two-mask agreement",
            "requested_max_instances": args.max_new_instances,
            "minimum_mask_agreement": args.minimum_mask_agreement,
            "candidate_peaks": peak_candidates,
            "evaluated_candidate_count": len(evaluated_candidates),
            "evaluated_candidates": evaluated_candidates,
            "accepted_instance_count": len(new_instances),
            "acceptance_scope": "geometric mask stability only; semantic identity remains unverified",
            "instances": new_instances,
        },
        "sam": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": MODEL_LICENSE,
            "checkpoint_files": checkpoint_files,
            "device": session.device,
            "multimask_output": True,
            "all_alternatives_retained": True,
        },
        "projection": {
            "one_transient_face_at_a_time": True,
            "standalone_rgb_faces_retained": False,
            "all_mask_sheets_embed_each_transient_face_for_audit": True,
        },
        "artifacts": {
            "summary": summary_path.name,
            "same_instance_consensus": "same-instance-consensus-erp.png",
            "same_instance_envelope": "same-instance-envelope-erp.png",
        },
        "source": {
            "record": str(args.record.resolve()),
            "evidence": str(args.evidence.resolve()),
            "erp": str(source_path.resolve()),
            "erp_sha256": _sha256(source_path),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "limitations": [
            "P74 has no instance-mask ground truth.",
            "Same-instance propagation uses a two-of-three mask majority and can stop early.",
            "Residual proxy evidence is only 32x64 and may merge adjacent pipes or emphasize glare.",
            "Accepted new-instance colors pass mask-agreement gating but are not verified industrial identities.",
        ],
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    session.close()
    del source
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
