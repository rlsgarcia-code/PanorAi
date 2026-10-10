#!/usr/bin/env python3
"""Compare point-only and enriched SAM 2.1 prompts on frozen P74 charts.

The chart centres come from a prior spherical CAM/SAM expansion result so the
comparison changes only the segmenter and prompt.  Same-instance continuation
uses the previous two-of-three majority mask, an enclosing box, interior
positive points, and exterior negative points.  Residual candidates use the
local CAM component to construct the box and points, but not as a mask input.

All three decoder alternatives are retained.  One RGB face and one SAM image
embedding are live at a time; standalone RGB faces are never saved.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.spherical_fcn_cam.run_p74_cam_sam_expansion_pilot import (  # noqa: E402
    PIPE_PROXY_WEIGHTS,
    _back_project_masks,
    _face_from_erp,
    _heat_overlay,
    _overlay_regions,
    load_dense_proxy_evidence,
    maximum_pairwise_iou,
    normalized_proxy_ensemble,
)
from benchmarks.spherical_fcn_cam.run_p74_cam_sam_prompt_pilot import (  # noqa: E402
    EXPECTED_MULTIMASK_COUNT,
    _draw_cross,
    _mask_overlay,
    _panel,
    _resolve_device,
    erp_pixel_from_lon_lat,
    pairwise_mask_iou,
)
from panorai.geometry import (  # noqa: E402
    GnomonicSpec,
    equirectangular_to_gnomonic,
)

MODEL_ID = "facebook/sam2.1-hiera-large"
MODEL_REVISION = "665f8e2ad61cf5f53d65644ff27c8ee525124610"
MODEL_LICENSE = "Apache-2.0"
MODEL_FILES = {
    "config.json": (5705, None),
    "preprocessor_config.json": (683, None),
    "processor_config.json": (95, None),
    "model.safetensors": (
        897_897_416,
        "dc407dce21301fd94abb395c5099b4f2c455fdc8a8f261ac3d0ea6d4cd197230",
    ),
}
CAM_COMPONENT_RELATIVE_THRESHOLD = 0.70
CAM_COMPONENT_ABSOLUTE_THRESHOLD = 0.35


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(
    cache_dir: Path, *, accept_model_license: bool
) -> tuple[Path, list[dict[str, Any]]]:
    """Acquire the pinned official checkpoint and verify immutable metadata."""

    if not accept_model_license:
        raise PermissionError(
            f"pass --accept-model-license after reviewing {MODEL_ID} ({MODEL_LICENSE})"
        )
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=cache_dir,
            allow_patterns=list(MODEL_FILES),
        )
    )
    records = []
    for filename, (expected_size, expected_hash) in MODEL_FILES.items():
        path = snapshot / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size != expected_size:
            raise RuntimeError(
                f"unexpected size for {filename}: {size} != {expected_size}"
            )
        actual_hash = _sha256(path)
        if expected_hash is not None and actual_hash != expected_hash:
            raise RuntimeError(
                f"unexpected SHA-256 for {filename}: {actual_hash} != {expected_hash}"
            )
        records.append(
            {"filename": filename, "size_bytes": size, "sha256": actual_hash}
        )
    return snapshot, records


def majority_mask(masks: np.ndarray) -> np.ndarray:
    if masks.ndim != 3 or len(masks) != EXPECTED_MULTIMASK_COUNT:
        raise ValueError("expected exactly three NHW masks")
    return masks.sum(axis=0) >= 2


def connected_component_at_anchor(
    mask: np.ndarray, anchor_xy: tuple[float, float]
) -> np.ndarray:
    """Return the 8-connected component at, or nearest to, one face point."""

    if mask.ndim != 2:
        raise ValueError("mask must be HW")
    if not mask.any():
        raise ValueError("mask has no foreground")
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        raise ValueError("mask has no connected components")
    height, width = mask.shape
    x = int(np.clip(round(anchor_xy[0]), 0, width - 1))
    y = int(np.clip(round(anchor_xy[1]), 0, height - 1))
    label = int(labels[y, x])
    if label == 0:
        ys, xs = np.where(mask)
        nearest = int(np.argmin((xs - anchor_xy[0]) ** 2 + (ys - anchor_xy[1]) ** 2))
        label = int(labels[ys[nearest], xs[nearest]])
    return labels == label


def cam_component_mask(
    face_evidence: np.ndarray,
    anchor_xy: tuple[float, float],
    *,
    relative_threshold: float = CAM_COMPONENT_RELATIVE_THRESHOLD,
    absolute_threshold: float = CAM_COMPONENT_ABSOLUTE_THRESHOLD,
    fallback_radius: int = 48,
) -> tuple[np.ndarray, float]:
    """Extract the thresholded CAM component containing the frozen peak."""

    if face_evidence.ndim != 2 or not np.isfinite(face_evidence).all():
        raise ValueError("face evidence must be a finite HW array")
    height, width = face_evidence.shape
    x = int(np.clip(round(anchor_xy[0]), 0, width - 1))
    y = int(np.clip(round(anchor_xy[1]), 0, height - 1))
    anchor_score = float(face_evidence[y, x])
    threshold = max(absolute_threshold, relative_threshold * anchor_score)
    foreground = face_evidence >= threshold
    if foreground.any():
        component = connected_component_at_anchor(foreground, anchor_xy)
    else:
        component = np.zeros_like(foreground)
    if np.count_nonzero(component) < 64:
        yy, xx = np.ogrid[:height, :width]
        component = (xx - anchor_xy[0]) ** 2 + (
            yy - anchor_xy[1]
        ) ** 2 <= fallback_radius**2
    return component, threshold


def _spaced_interior_points(
    component: np.ndarray,
    anchor_xy: tuple[float, float],
    count: int,
) -> list[tuple[float, float]]:
    """Choose deterministic, well-inside positives without sampling boundaries."""

    distance = ndimage.distance_transform_edt(component)
    selected: list[tuple[float, float]] = []
    height, width = component.shape
    ax = int(np.clip(round(anchor_xy[0]), 0, width - 1))
    ay = int(np.clip(round(anchor_xy[1]), 0, height - 1))
    if component[ay, ax]:
        selected.append((float(anchor_xy[0]), float(anchor_xy[1])))
    working = distance.copy()
    suppression = max(8, int(round(min(height, width) * 0.04)))
    yy, xx = np.ogrid[:height, :width]
    while len(selected) < count and float(working.max()) > 0:
        y, x = np.unravel_index(int(np.argmax(working)), working.shape)
        point = (float(x), float(y))
        if not any(
            (x - px) ** 2 + (y - py) ** 2 < suppression**2 for px, py in selected
        ):
            selected.append(point)
        working[(xx - x) ** 2 + (yy - y) ** 2 <= suppression**2] = 0
    if not selected:
        raise ValueError("could not choose an interior point")
    return selected


def prompt_from_component(
    component: np.ndarray,
    anchor_xy: tuple[float, float],
    *,
    positive_count: int = 3,
    box_padding: int = 12,
    negative_offset: int = 10,
) -> dict[str, Any]:
    """Build one object prompt from a connected face component."""

    component = connected_component_at_anchor(component.astype(bool), anchor_xy)
    ys, xs = np.where(component)
    height, width = component.shape
    x1 = max(0, int(xs.min()) - box_padding)
    y1 = max(0, int(ys.min()) - box_padding)
    x2 = min(width - 1, int(xs.max()) + box_padding)
    y2 = min(height - 1, int(ys.max()) + box_padding)
    positives = _spaced_interior_points(component, anchor_xy, positive_count)
    mid_x = 0.5 * (x1 + x2)
    mid_y = 0.5 * (y1 + y2)
    candidates = [
        (max(0.0, x1 - negative_offset), mid_y),
        (min(width - 1.0, x2 + negative_offset), mid_y),
        (mid_x, max(0.0, y1 - negative_offset)),
        (mid_x, min(height - 1.0, y2 + negative_offset)),
    ]
    negatives = []
    for x, y in candidates:
        iy = int(np.clip(round(y), 0, height - 1))
        ix = int(np.clip(round(x), 0, width - 1))
        if not component[iy, ix] and (x, y) not in negatives:
            negatives.append((float(x), float(y)))
    return {
        "component": component,
        "points_xy": positives + negatives,
        "labels": [1] * len(positives) + [0] * len(negatives),
        "box_xyxy": [float(x1), float(y1), float(x2), float(y2)],
        "positive_count": len(positives),
        "negative_count": len(negatives),
    }


class Sam21Session:
    """Pinned SAM 2.1 model with reusable per-face image embeddings."""

    def __init__(self, snapshot: Path, device_name: str) -> None:
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.torch = torch
        self.device = _resolve_device(device_name, torch)
        self.processor = Sam2Processor.from_pretrained(snapshot, local_files_only=True)
        self.model = (
            Sam2Model.from_pretrained(
                snapshot, local_files_only=True, use_safetensors=True
            )
            .eval()
            .to(self.device)
        )

    def encode(self, face: Image.Image) -> tuple[list[Any], float]:
        inputs = self.processor(images=face, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.torch.float32)
        started = time.perf_counter()
        with self.torch.inference_mode():
            embeddings = self.model.get_image_embeddings(pixel_values)
        self._synchronize()
        seconds = time.perf_counter() - started
        del inputs, pixel_values
        return embeddings, seconds

    def predict(
        self,
        embeddings: list[Any],
        face_shape_hw: tuple[int, int],
        *,
        points_xy: list[tuple[float, float]],
        labels: list[int],
        box_xyxy: list[float] | None = None,
        input_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        height, width = face_shape_hw
        processor_args: dict[str, Any] = {
            "original_sizes": [[height, width]],
            "input_points": [[[list(point) for point in points_xy]]],
            "input_labels": [[labels]],
            "return_tensors": "pt",
        }
        if box_xyxy is not None:
            processor_args["input_boxes"] = [[box_xyxy]]
        inputs = self.processor(**processor_args)
        model_inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if key in {"input_points", "input_labels", "input_boxes"}
        }
        if input_mask is not None:
            model_inputs["input_masks"] = self.torch.from_numpy(
                input_mask.astype(np.float32)[None, None]
            ).to(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model(
                image_embeddings=embeddings,
                **model_inputs,
                multimask_output=True,
            )
        self._synchronize()
        seconds = time.perf_counter() - started
        processed = self.processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            [[height, width]],
            binarize=True,
        )[0]
        masks = processed.numpy().astype(bool)
        while masks.ndim > 3 and masks.shape[0] == 1:
            masks = masks.squeeze(0)
        scores = (
            outputs.iou_scores.detach().cpu().reshape(-1).numpy().astype(np.float64)
        )
        presence = float(outputs.object_score_logits.detach().cpu().reshape(-1)[0])
        if masks.shape[0] != EXPECTED_MULTIMASK_COUNT:
            raise RuntimeError(f"SAM 2.1 returned unexpected masks {masks.shape}")
        del outputs, model_inputs, inputs
        return masks, scores, presence, seconds

    def memory_bytes(self) -> dict[str, int | None]:
        if self.device == "mps":
            return {
                "current_allocated": int(self.torch.mps.current_allocated_memory()),
                "driver_allocated": int(self.torch.mps.driver_allocated_memory()),
            }
        if self.device == "cuda":
            return {
                "current_allocated": int(self.torch.cuda.memory_allocated()),
                "driver_allocated": int(self.torch.cuda.memory_reserved()),
            }
        return {"current_allocated": None, "driver_allocated": None}

    def _synchronize(self) -> None:
        if self.device == "mps":
            self.torch.mps.synchronize()
        elif self.device == "cuda":
            self.torch.cuda.synchronize()

    def release_cache(self) -> None:
        gc.collect()
        if self.device == "mps":
            self.torch.mps.empty_cache()
        elif self.device == "cuda":
            self.torch.cuda.empty_cache()

    def close(self) -> None:
        del self.model, self.processor
        gc.collect()


def _draw_prompt(
    face: Image.Image,
    points_xy: list[tuple[float, float]],
    labels: list[int],
    box_xyxy: list[float] | None,
) -> Image.Image:
    result = face.copy()
    draw = ImageDraw.Draw(result)
    if box_xyxy is not None:
        draw.rectangle(tuple(box_xyxy), outline=(0, 230, 255), width=5)
    for point, label in zip(points_xy, labels, strict=True):
        if label == 1:
            _draw_cross(result, point, radius=12)
        else:
            x, y = point
            draw.line((x - 10, y - 10, x + 10, y + 10), fill=(255, 30, 30), width=5)
            draw.line((x - 10, y + 10, x + 10, y - 10), fill=(255, 30, 30), width=5)
    return result


def _save_prompt_sheet(
    face: Image.Image,
    masks: np.ndarray,
    scores: np.ndarray,
    *,
    points_xy: list[tuple[float, float]],
    labels: list[int],
    box_xyxy: list[float] | None,
    title: str,
    path: Path,
) -> None:
    prompted = _draw_prompt(face, points_xy, labels, box_xyxy)
    panels = [_panel(prompted, title, (560, 560))]
    positive = next(
        point for point, label in zip(points_xy, labels, strict=True) if label == 1
    )
    for index, (mask, score) in enumerate(zip(masks, scores, strict=True)):
        panels.append(
            _panel(
                _mask_overlay(face, mask, prompt_xy=positive),
                f"mask {index + 1}/3 - decoder order - IoU {score:.3f}",
                (560, 560),
            )
        )
    sheet = Image.new("RGB", (1120, 1120), (235, 235, 235))
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 2) * 560, (index // 2) * 560))
    sheet.save(path, quality=95)


def _save_masks(
    output_dir: Path,
    prefix: str,
    masks: np.ndarray,
    erp_masks: list[np.ndarray],
) -> list[dict[str, str]]:
    records = []
    for number, (mask, erp_mask) in enumerate(
        zip(masks, erp_masks, strict=True), start=1
    ):
        face_name = f"{prefix}-mask-{number:02d}-face.png"
        erp_name = f"{prefix}-mask-{number:02d}-erp.png"
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
            output_dir / face_name
        )
        Image.fromarray(erp_mask.astype(np.uint8) * 255, mode="L").save(
            output_dir / erp_name
        )
        records.append({"face_mask": face_name, "erp_mask": erp_name})
    return records


def _mask_metrics(masks: np.ndarray, erp_masks: list[np.ndarray]) -> dict[str, Any]:
    consensus = majority_mask(masks)
    consensus_erp = np.sum(erp_masks, axis=0) >= 2
    envelope = masks.any(axis=0)
    envelope_erp = np.logical_or.reduce(erp_masks)
    return {
        "pairwise_iou": pairwise_mask_iou(masks),
        "maximum_pairwise_iou": maximum_pairwise_iou(masks),
        "majority_face_fraction": float(consensus.mean()),
        "envelope_face_fraction": float(envelope.mean()),
        "majority_erp_fraction": float(consensus_erp.mean()),
        "envelope_erp_fraction": float(envelope_erp.mean()),
    }


def _jaccard(first: np.ndarray, second: np.ndarray) -> float:
    union = np.count_nonzero(first | second)
    return 1.0 if union == 0 else float(np.count_nonzero(first & second) / union)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--charts-from", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--face-size", type=int, default=1024)
    parser.add_argument("--maximum-residual-candidates", type=int, default=5)
    parser.add_argument("--accept-model-license", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    record = json.loads(args.record.read_text(encoding="utf-8"))
    chart_record = json.loads(args.charts_from.read_text(encoding="utf-8"))
    if record["sample_id"] != chart_record["sample_id"]:
        raise ValueError("record and frozen charts refer to different samples")
    evidence = load_dense_proxy_evidence(args.evidence)
    prior = normalized_proxy_ensemble(evidence, PIPE_PROXY_WEIGHTS)
    source_path = Path(record["source_path"])
    source = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.float32)
    preview = Image.open(args.record.parent / "input-erp.jpg").convert("RGB")
    preview_shape = (preview.height, preview.width)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    snapshot, checkpoint_files = download_model(
        args.cache_dir, accept_model_license=args.accept_model_license
    )
    session = Sam21Session(snapshot, args.device)
    same_records: list[dict[str, Any]] = []
    enriched_consensus_by_step: dict[int, np.ndarray] = {}
    point_global = np.zeros(preview_shape, dtype=bool)
    enriched_global = np.zeros(preview_shape, dtype=bool)

    for frozen in chart_record["same_instance"]["steps"]:
        step_index = int(frozen["step_index"])
        parent_step = frozen["parent_step"]
        spec = GnomonicSpec(
            center_lat_deg=float(frozen["center_latitude_degrees"]),
            center_lon_deg=float(frozen["center_longitude_degrees"]),
            hfov_deg=float(frozen["fov_degrees"]),
            vfov_deg=float(frozen["fov_degrees"]),
            output_shape_hw=(args.face_size, args.face_size),
        )
        base_point = tuple(float(value) for value in frozen["prompt_xy"])
        face = _face_from_erp(source, spec)
        embeddings, encode_seconds = session.encode(face)
        point_masks, point_scores, point_presence, point_seconds = session.predict(
            embeddings,
            (args.face_size, args.face_size),
            points_xy=[base_point],
            labels=[1],
        )
        point_erp = _back_project_masks(point_masks, spec, preview_shape)
        point_consensus_erp = np.sum(point_erp, axis=0) >= 2
        point_global |= point_consensus_erp
        point_prefix = f"same-{step_index:02d}-point"
        point_sheet = f"{point_prefix}-all-masks.jpg"
        _save_prompt_sheet(
            face,
            point_masks,
            point_scores,
            points_xy=[base_point],
            labels=[1],
            box_xyxy=None,
            title=f"SAM2.1 point-only - same step {step_index}",
            path=args.output_dir / point_sheet,
        )
        point_files = _save_masks(args.output_dir, point_prefix, point_masks, point_erp)

        if parent_step is None:
            enriched_masks = point_masks.copy()
            enriched_scores = point_scores.copy()
            enriched_presence = point_presence
            enriched_seconds = 0.0
            enriched_points = [base_point]
            enriched_labels = [1]
            enriched_box = None
            projected_prior = majority_mask(point_masks)
            shared_with_point = True
        else:
            parent_erp = enriched_consensus_by_step[int(parent_step)]
            projected_prior = equirectangular_to_gnomonic(
                parent_erp.astype(np.uint8),
                spec,
                interpolation="nearest",
            ).data.astype(bool)
            prompt = prompt_from_component(projected_prior, base_point)
            enriched_points = prompt["points_xy"]
            enriched_labels = prompt["labels"]
            enriched_box = prompt["box_xyxy"]
            enriched_masks, enriched_scores, enriched_presence, enriched_seconds = (
                session.predict(
                    embeddings,
                    (args.face_size, args.face_size),
                    points_xy=enriched_points,
                    labels=enriched_labels,
                    box_xyxy=enriched_box,
                    input_mask=projected_prior,
                )
            )
            shared_with_point = False
        enriched_erp = _back_project_masks(enriched_masks, spec, preview_shape)
        enriched_consensus = majority_mask(enriched_masks)
        enriched_consensus_erp = np.sum(enriched_erp, axis=0) >= 2
        enriched_consensus_by_step[step_index] = enriched_consensus_erp
        enriched_global |= enriched_consensus_erp
        enriched_prefix = f"same-{step_index:02d}-enriched"
        enriched_sheet = f"{enriched_prefix}-all-masks.jpg"
        _save_prompt_sheet(
            face,
            enriched_masks,
            enriched_scores,
            points_xy=enriched_points,
            labels=enriched_labels,
            box_xyxy=enriched_box,
            title=f"SAM2.1 enriched - same step {step_index}",
            path=args.output_dir / enriched_sheet,
        )
        enriched_files = _save_masks(
            args.output_dir, enriched_prefix, enriched_masks, enriched_erp
        )
        prior_overlap = np.count_nonzero(projected_prior)
        same_records.append(
            {
                "step_index": step_index,
                "parent_step": parent_step,
                "center_longitude_degrees": spec.center_lon_deg,
                "center_latitude_degrees": spec.center_lat_deg,
                "fov_degrees": spec.hfov_deg,
                "encode_seconds": encode_seconds,
                "point_only": {
                    "points_xy": [list(base_point)],
                    "labels": [1],
                    "predicted_iou_decoder_order": point_scores.tolist(),
                    "object_presence_logit": point_presence,
                    "decode_seconds": point_seconds,
                    "metrics": _mask_metrics(point_masks, point_erp),
                    "all_masks_sheet": point_sheet,
                    "mask_files": point_files,
                },
                "enriched": {
                    "shared_with_point_control": shared_with_point,
                    "points_xy": [list(point) for point in enriched_points],
                    "labels": enriched_labels,
                    "box_xyxy": enriched_box,
                    "input_mask_fraction": float(projected_prior.mean()),
                    "predicted_iou_decoder_order": enriched_scores.tolist(),
                    "object_presence_logit": enriched_presence,
                    "decode_seconds": enriched_seconds,
                    "metrics": _mask_metrics(enriched_masks, enriched_erp),
                    "input_mask_recall_by_majority": (
                        None
                        if prior_overlap == 0
                        else float(
                            np.count_nonzero(projected_prior & enriched_consensus)
                            / prior_overlap
                        )
                    ),
                    "all_masks_sheet": enriched_sheet,
                    "mask_files": enriched_files,
                },
                "point_vs_enriched_majority_iou": _jaccard(
                    majority_mask(point_masks), enriched_consensus
                ),
                "memory_bytes_after_face": session.memory_bytes(),
            }
        )
        del embeddings
        session.release_cache()
        del face, point_masks, enriched_masks, point_erp, enriched_erp

    residual_records: list[dict[str, Any]] = []
    point_residual_masks: list[np.ndarray] = []
    component_residual_masks: list[np.ndarray] = []
    frozen_candidates = chart_record["new_instances"]["evaluated_candidates"]
    for frozen in frozen_candidates[: args.maximum_residual_candidates]:
        candidate_number = int(frozen["candidate_number"])
        peak = frozen["peak"]
        spec = GnomonicSpec(
            center_lat_deg=float(peak["latitude_degrees"]),
            center_lon_deg=float(peak["longitude_degrees"]),
            hfov_deg=float(frozen["fov_degrees"]),
            vfov_deg=float(frozen["fov_degrees"]),
            output_shape_hw=(args.face_size, args.face_size),
        )
        center = ((args.face_size - 1) / 2.0, (args.face_size - 1) / 2.0)
        face = _face_from_erp(source, spec)
        embeddings, encode_seconds = session.encode(face)
        point_masks, point_scores, point_presence, point_seconds = session.predict(
            embeddings,
            (args.face_size, args.face_size),
            points_xy=[center],
            labels=[1],
        )
        point_erp = _back_project_masks(point_masks, spec, preview_shape)
        point_consensus_erp = np.sum(point_erp, axis=0) >= 2
        point_residual_masks.append(point_consensus_erp)

        face_prior = equirectangular_to_gnomonic(
            prior.astype(np.float32), spec, interpolation="bilinear"
        ).data
        component, threshold = cam_component_mask(face_prior, center)
        prompt = prompt_from_component(component, center)
        component_masks, component_scores, component_presence, component_seconds = (
            session.predict(
                embeddings,
                (args.face_size, args.face_size),
                points_xy=prompt["points_xy"],
                labels=prompt["labels"],
                box_xyxy=prompt["box_xyxy"],
            )
        )
        component_erp = _back_project_masks(component_masks, spec, preview_shape)
        component_consensus_erp = np.sum(component_erp, axis=0) >= 2
        component_residual_masks.append(component_consensus_erp)

        modes = []
        for name, masks, scores, presence, seconds, points, labels, box, erp_masks in (
            (
                "point",
                point_masks,
                point_scores,
                point_presence,
                point_seconds,
                [center],
                [1],
                None,
                point_erp,
            ),
            (
                "component",
                component_masks,
                component_scores,
                component_presence,
                component_seconds,
                prompt["points_xy"],
                prompt["labels"],
                prompt["box_xyxy"],
                component_erp,
            ),
        ):
            prefix = f"residual-{candidate_number:02d}-{name}"
            sheet = f"{prefix}-all-masks.jpg"
            _save_prompt_sheet(
                face,
                masks,
                scores,
                points_xy=points,
                labels=labels,
                box_xyxy=box,
                title=f"SAM2.1 {name} - residual {candidate_number}",
                path=args.output_dir / sheet,
            )
            modes.append(
                {
                    "mode": name,
                    "points_xy": [list(point) for point in points],
                    "labels": labels,
                    "box_xyxy": box,
                    "predicted_iou_decoder_order": scores.tolist(),
                    "object_presence_logit": presence,
                    "decode_seconds": seconds,
                    "metrics": _mask_metrics(masks, erp_masks),
                    "all_masks_sheet": sheet,
                    "mask_files": _save_masks(
                        args.output_dir, prefix, masks, erp_masks
                    ),
                }
            )
        residual_records.append(
            {
                "candidate_number": candidate_number,
                "peak": peak,
                "fov_degrees": spec.hfov_deg,
                "encode_seconds": encode_seconds,
                "cam_component_threshold": threshold,
                "cam_component_face_fraction": float(component.mean()),
                "point_vs_component_majority_iou": _jaccard(
                    majority_mask(point_masks), majority_mask(component_masks)
                ),
                "modes": modes,
                "memory_bytes_after_face": session.memory_bytes(),
            }
        )
        del embeddings
        session.release_cache()
        del face, point_masks, component_masks, point_erp, component_erp

    colors = [
        (255, 30, 120),
        (0, 210, 255),
        (255, 190, 0),
        (80, 230, 80),
        (160, 80, 255),
    ]
    same_point_overlay = _overlay_regions(preview, [(point_global, (255, 30, 120))])
    same_enriched_overlay = _overlay_regions(
        preview, [(enriched_global, (0, 210, 255))]
    )
    point_new_overlay = _overlay_regions(
        preview,
        [
            (mask, colors[index % len(colors)])
            for index, mask in enumerate(point_residual_masks)
        ],
    )
    component_new_overlay = _overlay_regions(
        preview,
        [
            (mask, colors[index % len(colors)])
            for index, mask in enumerate(component_residual_masks)
        ],
    )
    for index, frozen in enumerate(
        frozen_candidates[: args.maximum_residual_candidates]
    ):
        peak = frozen["peak"]
        xy = erp_pixel_from_lon_lat(
            float(peak["longitude_degrees"]),
            float(peak["latitude_degrees"]),
            preview_shape,
        )
        _draw_cross(point_new_overlay, xy, radius=7)
        _draw_cross(component_new_overlay, xy, radius=7)
        ImageDraw.Draw(point_new_overlay).text(
            (xy[0] + 8, xy[1] + 4), str(index + 1), fill="white"
        )
        ImageDraw.Draw(component_new_overlay).text(
            (xy[0] + 8, xy[1] + 4), str(index + 1), fill="white"
        )
    panels = [
        _panel(_heat_overlay(preview, prior), "Frozen proxy ensemble", (640, 430)),
        _panel(same_point_overlay, "Same instance: point-only majority", (640, 430)),
        _panel(
            same_enriched_overlay, "Same instance: propagated-mask majority", (640, 430)
        ),
        _panel(point_new_overlay, "Residuals: point-only majority", (640, 430)),
        _panel(
            component_new_overlay,
            "Residuals: CAM box + positive/negative points",
            (640, 430),
        ),
    ]
    summary = Image.new("RGB", (3 * 640, 2 * 430), (235, 235, 235))
    for index, panel in enumerate(panels):
        summary.paste(panel, ((index % 3) * 640, (index // 3) * 430))
    summary_name = "sam21-prompt-comparison-summary.jpg"
    summary.save(args.output_dir / summary_name, quality=95)

    result = {
        "schema": "panorai-p74-sam21-prompt-study/v1",
        "sample_id": record["sample_id"],
        "comparison_contract": (
            "same frozen chart centres and residual directions; point-only versus enriched prompts"
        ),
        "same_instance": {
            "frozen_chart_count": len(same_records),
            "point_global_majority_erp_fraction": float(point_global.mean()),
            "enriched_global_majority_erp_fraction": float(enriched_global.mean()),
            "global_majority_iou": _jaccard(point_global, enriched_global),
            "steps": same_records,
        },
        "residual_candidates": {
            "count": len(residual_records),
            "cam_component_relative_threshold": CAM_COMPONENT_RELATIVE_THRESHOLD,
            "cam_component_absolute_threshold": CAM_COMPONENT_ABSOLUTE_THRESHOLD,
            "candidates": residual_records,
        },
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": MODEL_LICENSE,
            "checkpoint_files": checkpoint_files,
            "device": session.device,
            "multimask_output": True,
            "all_alternatives_retained": True,
            "parameter_count": int(
                sum(parameter.numel() for parameter in session.model.parameters())
            ),
        },
        "projection": {
            "one_transient_face_and_embedding_at_a_time": True,
            "standalone_rgb_faces_retained": False,
            "face_size": args.face_size,
        },
        "source": {
            "record": str(args.record.resolve()),
            "evidence": str(args.evidence.resolve()),
            "frozen_charts": str(args.charts_from.resolve()),
            "erp": str(source_path.resolve()),
            "erp_sha256": _sha256(source_path),
        },
        "artifacts": {"summary": summary_name},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "limitations": [
            "P74 has no instance-mask ground truth.",
            "SAM self-scores and mask agreement are not semantic verification.",
            "The propagated mask is a binary reprojection through a 512x1024 ERP preview.",
            "The residual CAM prior is only 32x64 and component thresholds are predeclared.",
            "The experiment compares frozen charts and does not yet rerun dynamic frontier selection.",
        ],
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    session.close()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
