#!/usr/bin/env python3
"""Prompt official SAM from one frozen spherical CAM direction.

This benchmark intentionally keeps every multimask alternative in decoder
order.  It does not choose a best mask and never stores a standalone RGB
gnomonic face.  The face exists only in memory while SAM and the contact sheet
are produced; binary masks and their ERP reprojections are retained.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panorai.geometry import (  # noqa: E402
    GnomonicSpec,
    equirectangular_to_gnomonic,
    gnomonic_to_equirectangular,
)

MODEL_ID = "facebook/sam-vit-base"
MODEL_REVISION = "70c1a07f894ebb5b307fd9eaaee97b9dfc16068f"
MODEL_LICENSE = "Apache-2.0"
MODEL_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")
EXPECTED_MULTIMASK_COUNT = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def select_candidate(record: dict[str, Any], class_name: str) -> dict[str, Any]:
    """Return one declared CAM candidate without using any SAM result."""

    candidates = record.get("candidate_predictions")
    if not isinstance(candidates, list):
        raise ValueError("record has no candidate_predictions list")
    matches = [item for item in candidates if item.get("class_name") == class_name]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {class_name!r} candidate, got {len(matches)}"
        )
    candidate = matches[0]
    statistics = candidate.get("cam_statistics")
    if not isinstance(statistics, dict):
        raise ValueError(f"candidate {class_name!r} has no rendered CAM statistics")
    for key in ("peak_latitude_degrees", "peak_longitude_degrees"):
        if key not in statistics or not math.isfinite(float(statistics[key])):
            raise ValueError(f"candidate {class_name!r} has no finite {key}")
    return candidate


def make_prompt_spec(
    candidate: dict[str, Any], *, fov_deg: float, face_size: int
) -> GnomonicSpec:
    """Center a square tangent view exactly on the frozen CAM peak."""

    statistics = candidate["cam_statistics"]
    return GnomonicSpec(
        center_lat_deg=float(statistics["peak_latitude_degrees"]),
        center_lon_deg=float(statistics["peak_longitude_degrees"]),
        hfov_deg=fov_deg,
        vfov_deg=fov_deg,
        output_shape_hw=(face_size, face_size),
    )


def erp_pixel_from_lon_lat(
    longitude_deg: float, latitude_deg: float, shape_hw: tuple[int, int]
) -> tuple[float, float]:
    """Map a canonical spherical direction to an ERP pixel-center coordinate."""

    height, width = shape_hw
    x = ((longitude_deg + 180.0) / 360.0) * width - 0.5
    y = ((90.0 - latitude_deg) / 180.0) * height - 0.5
    return x % width, min(max(y, -0.5), height - 0.5)


def ordered_mask_records(
    masks: np.ndarray, scores: np.ndarray, erp_masks: list[np.ndarray]
) -> list[dict[str, Any]]:
    """Describe all alternatives in decoder order without sorting or selection."""

    if masks.ndim != 3:
        raise ValueError(f"masks must have shape (N,H,W), got {masks.shape}")
    if len(masks) != len(scores) or len(masks) != len(erp_masks):
        raise ValueError("mask, score, and ERP-mask counts differ")
    records: list[dict[str, Any]] = []
    for index, (mask, score, erp_mask) in enumerate(
        zip(masks, scores, erp_masks, strict=True)
    ):
        records.append(
            {
                "decoder_index": index,
                "display_number": index + 1,
                "predicted_iou": float(score),
                "face_area_fraction": float(np.mean(mask)),
                "erp_raster_fraction": float(np.mean(erp_mask)),
                "contains_positive_prompt": bool(
                    mask[mask.shape[0] // 2, mask.shape[1] // 2]
                ),
            }
        )
    return records


def pairwise_mask_iou(masks: np.ndarray) -> list[list[float]]:
    """Return a symmetric IoU matrix while preserving decoder order."""

    if masks.ndim != 3:
        raise ValueError(f"masks must have shape (N,H,W), got {masks.shape}")
    count = len(masks)
    matrix = np.eye(count, dtype=np.float64)
    for first in range(count):
        for second in range(first + 1, count):
            intersection = np.count_nonzero(masks[first] & masks[second])
            union = np.count_nonzero(masks[first] | masks[second])
            value = 1.0 if union == 0 else intersection / union
            matrix[first, second] = matrix[second, first] = value
    return matrix.tolist()


def _solid_angle_fraction(mask: np.ndarray) -> float:
    height, _ = mask.shape
    latitude = (
        math.pi / 2.0 - (np.arange(height, dtype=np.float64) + 0.5) * math.pi / height
    )
    weights = np.cos(latitude)[:, None]
    return float(np.sum(mask * weights) / (mask.shape[1] * np.sum(weights)))


def _draw_cross(
    image: Image.Image, xy: tuple[float, float], *, radius: int = 14
) -> None:
    draw = ImageDraw.Draw(image)
    x, y = xy
    draw.line((x - radius, y, x + radius, y), fill=(255, 230, 0), width=5)
    draw.line((x, y - radius, x, y + radius), fill=(255, 230, 0), width=5)
    draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(0, 0, 0), width=2)


def _mask_overlay(
    image: Image.Image,
    mask: np.ndarray,
    *,
    prompt_xy: tuple[float, float] | None = None,
) -> Image.Image:
    base = image.convert("RGBA")
    color = Image.new("RGBA", base.size, (255, 30, 120, 0))
    alpha = Image.fromarray((mask.astype(np.uint8) * 126), mode="L")
    color.putalpha(alpha)
    result = Image.alpha_composite(base, color).convert("RGB")
    if prompt_xy is None:
        prompt_xy = ((result.width - 1) / 2.0, (result.height - 1) / 2.0)
    _draw_cross(result, prompt_xy)
    return result


def _panel(image: Image.Image, title: str, size: tuple[int, int]) -> Image.Image:
    width, height = size
    canvas = Image.new("RGB", size, "white")
    fitted = ImageOps.contain(image.convert("RGB"), (width - 16, height - 56))
    x = (width - fitted.width) // 2
    y = 48 + (height - 48 - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 14), title, fill="black", font=ImageFont.load_default())
    return canvas


def render_contact_sheet(
    *,
    erp_preview: Image.Image,
    cam_overlay: Image.Image,
    face: Image.Image,
    masks: np.ndarray,
    scores: np.ndarray,
    spec: GnomonicSpec,
    output_path: Path,
) -> None:
    """Render input, prior, prompt, and every decoder mask in a fixed 2x3 sheet."""

    if len(masks) != EXPECTED_MULTIMASK_COUNT:
        raise ValueError(
            f"expected {EXPECTED_MULTIMASK_COUNT} SAM masks, got {len(masks)}"
        )
    located = erp_preview.copy()
    direction_xy = erp_pixel_from_lon_lat(
        spec.center_lon_deg, spec.center_lat_deg, (located.height, located.width)
    )
    _draw_cross(located, direction_xy)
    prompted = face.copy()
    _draw_cross(prompted, ((face.width - 1) / 2.0, (face.height - 1) / 2.0))

    panels = [
        _panel(located, "P74 G100: CAM direction", (640, 430)),
        _panel(
            cam_overlay, "InternImage-G prior: milk can / process vessel", (640, 430)
        ),
        _panel(prompted, "One temporary gnomonic view + positive point", (640, 430)),
    ]
    for index, (mask, score) in enumerate(zip(masks, scores, strict=True)):
        title = (
            f"SAM mask {index + 1}/{len(masks)} - decoder order - "
            f"predicted IoU {float(score):.3f} - area {100.0 * float(mask.mean()):.1f}%"
        )
        panels.append(_panel(_mask_overlay(face, mask), title, (640, 430)))

    sheet = Image.new("RGB", (3 * 640, 2 * 430), (235, 235, 235))
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 3) * 640, (index // 3) * 430))
    sheet.save(output_path, quality=95)


def _resolve_device(requested: str, torch: Any) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _download_model(
    cache_dir: Path, *, accept_model_license: bool
) -> tuple[Path, list[dict[str, Any]]]:
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
    files = []
    for name in MODEL_FILES:
        path = snapshot / name
        if not path.is_file():
            raise FileNotFoundError(f"downloaded snapshot is missing {name}")
        files.append(
            {
                "filename": name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return snapshot, files


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--class-name", default="milk can")
    parser.add_argument("--fov-deg", type=float, default=70.0)
    parser.add_argument("--face-size", type=int, default=1024)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--accept-model-license", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 0.0 < args.fov_deg < 180.0:
        raise ValueError("--fov-deg must be strictly between 0 and 180")
    if args.face_size <= 0:
        raise ValueError("--face-size must be positive")

    record = _load_json(args.record.resolve())
    candidate = select_candidate(record, args.class_name)
    spec = make_prompt_spec(candidate, fov_deg=args.fov_deg, face_size=args.face_size)
    source_path = Path(record["source_path"]).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    preview_path = args.record.parent / "input-erp.jpg"
    cam_path = args.record.parent / candidate["overlay"]
    if not preview_path.is_file() or not cam_path.is_file():
        raise FileNotFoundError("the frozen ERP preview or CAM overlay is missing")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.float32)
    face_array = equirectangular_to_gnomonic(
        source, spec, interpolation="bilinear"
    ).data
    face_uint8 = np.clip(np.rint(face_array), 0.0, 255.0).astype(np.uint8)
    face = Image.fromarray(face_uint8, mode="RGB")

    snapshot, checkpoint_files = _download_model(
        args.cache_dir.resolve(), accept_model_license=args.accept_model_license
    )
    import torch
    from transformers import SamModel, SamProcessor

    device = _resolve_device(args.device, torch)
    processor = SamProcessor.from_pretrained(snapshot, local_files_only=True)
    model = (
        SamModel.from_pretrained(snapshot, local_files_only=True, use_safetensors=True)
        .eval()
        .to(device)
    )
    prompt_xy = [(args.face_size - 1) / 2.0, (args.face_size - 1) / 2.0]
    inputs = processor(
        images=face,
        input_points=[[[prompt_xy[0], prompt_xy[1]]]],
        input_labels=[[1]],
        return_tensors="pt",
    )
    original_sizes = inputs["original_sizes"].cpu()
    reshaped_sizes = inputs["reshaped_input_sizes"].cpu()
    model_keys = {
        "pixel_values",
        "input_points",
        "input_labels",
        "input_boxes",
        "input_masks",
    }
    model_inputs: dict[str, Any] = {}
    for key, value in inputs.items():
        if key not in model_keys or not hasattr(value, "to"):
            continue
        if torch.is_floating_point(value):
            value = value.to(dtype=torch.float32)
        model_inputs[key] = value.to(device)

    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**model_inputs, multimask_output=True)
    inference_seconds = time.perf_counter() - started
    processed = processor.image_processor.post_process_masks(
        outputs.pred_masks.detach().cpu(),
        original_sizes,
        reshaped_sizes,
        binarize=True,
    )[0]
    while processed.ndim > 3 and processed.shape[0] == 1:
        processed = processed.squeeze(0)
    masks = processed.numpy().astype(bool)
    scores = outputs.iou_scores.detach().cpu().reshape(-1).numpy().astype(np.float64)
    if (
        masks.shape[0] != EXPECTED_MULTIMASK_COUNT
        or len(scores) != EXPECTED_MULTIMASK_COUNT
    ):
        raise RuntimeError(
            f"official SAM multimask contract expected 3 alternatives; got "
            f"masks={masks.shape}, scores={scores.shape}"
        )

    preview = Image.open(preview_path).convert("RGB")
    preview_array = np.asarray(preview)
    erp_masks: list[np.ndarray] = []
    for index, mask in enumerate(masks):
        erp_mask = gnomonic_to_equirectangular(
            mask.astype(np.uint8),
            spec,
            preview_array.shape[:2],
            interpolation="nearest",
            fill_value=0,
        ).data.astype(bool)
        erp_masks.append(erp_mask)
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
            args.output_dir / f"mask-{index + 1:02d}-face.png"
        )
        Image.fromarray(erp_mask.astype(np.uint8) * 255, mode="L").save(
            args.output_dir / f"mask-{index + 1:02d}-erp.png"
        )
        erp_prompt_xy = erp_pixel_from_lon_lat(
            spec.center_lon_deg,
            spec.center_lat_deg,
            (preview.height, preview.width),
        )
        erp_overlay = _mask_overlay(preview, erp_mask, prompt_xy=erp_prompt_xy)
        erp_overlay.save(
            args.output_dir / f"mask-{index + 1:02d}-erp-overlay.jpg", quality=95
        )

    contact_sheet_path = args.output_dir / "all-sam-masks-contact-sheet.jpg"
    render_contact_sheet(
        erp_preview=preview,
        cam_overlay=Image.open(cam_path).convert("RGB"),
        face=face,
        masks=masks,
        scores=scores,
        spec=spec,
        output_path=contact_sheet_path,
    )

    alternatives = ordered_mask_records(masks, scores, erp_masks)
    for item, erp_mask in zip(alternatives, erp_masks, strict=True):
        item["erp_solid_angle_fraction"] = _solid_angle_fraction(erp_mask)
        number = item["display_number"]
        item["face_mask"] = f"mask-{number:02d}-face.png"
        item["erp_mask"] = f"mask-{number:02d}-erp.png"
        item["erp_overlay"] = f"mask-{number:02d}-erp-overlay.jpg"

    result = {
        "schema": "panorai-p74-cam-sam-multimask-pilot/v1",
        "selection_protocol": {
            "sample": record["sample_id"],
            "selected_before_sam_review": True,
            "reason": "milk can was the frozen top-1 InternImage-G process-vessel proxy on G100",
            "mask_selection_performed": False,
            "mask_order": "unchanged official decoder order",
        },
        "semantic_prior": {
            "industrial_class": "process_vessel",
            "imagenet_class_index": candidate["class_index"],
            "imagenet_class_name": candidate["class_name"],
            "rank": candidate["rank"],
            "probability": candidate["probability"],
            "peak_latitude_degrees": spec.center_lat_deg,
            "peak_longitude_degrees": spec.center_lon_deg,
        },
        "projection": {
            "type": "single temporary gnomonic view",
            "hfov_degrees": spec.hfov_deg,
            "vfov_degrees": spec.vfov_deg,
            "shape_hw": list(spec.output_shape_hw),
            "standalone_rgb_face_retained": False,
            "face_visualization_embedded_in_contact_sheet": True,
            "no_multiface_inference": True,
        },
        "prompt": {
            "positive_points_xy": [prompt_xy],
            "negative_points_xy": [],
            "box_xyxy": None,
            "mask_prior": None,
            "description": "one positive point placed at the frozen CAM peak",
        },
        "sam": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": MODEL_LICENSE,
            "checkpoint_files": checkpoint_files,
            "device": device,
            "dtype": str(next(model.parameters()).dtype),
            "multimask_output": True,
            "expected_mask_count": EXPECTED_MULTIMASK_COUNT,
            "returned_mask_count": len(alternatives),
            "inference_seconds": inference_seconds,
        },
        "alternatives": alternatives,
        "alternative_pairwise_iou": pairwise_mask_iou(masks),
        "contact_sheet": contact_sheet_path.name,
        "source": {
            "record": str(args.record.resolve()),
            "erp": str(source_path),
            "erp_sha256": _sha256(source_path),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
        "limitations": [
            "There is no P74 segmentation ground truth.",
            "SAM is class-agnostic; the semantic label comes from the ImageNet proxy.",
            "A single positive point is intentionally ambiguous and the three masks may be nested.",
            "The retained contact sheet embeds the temporary face only to audit every mask.",
        ],
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    del outputs, model_inputs, inputs, model, processor, source, face_array, face_uint8
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
