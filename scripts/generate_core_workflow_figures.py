#!/usr/bin/env python3
"""Generate core-workflow figures from a checksum-pinned CC0 ERP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from panorai import EquirectangularImage  # noqa: E402
from panorai.core.numpy import (  # noqa: E402
    SphericalGaussianBlur2d,
    SphericalGradient,
    SphericalResize,
)
from panorai.data import FacePointBatch, FaceRegionBatch, FaceViewOutput  # noqa: E402


EXPECTED_INPUT_SHA256 = (
    "5333c2cecc55468fcbc64a252f4bc097fab064cae70bead4f76d6c2e6101c524"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=REPOSITORY_ROOT
        / "docs/_static/tutorials/nature-reserve-forest-erp.jpg",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "docs/_static/tutorials",
    )
    return parser.parse_args()


def _label(panel: Image.Image, label: str, *, width: int = 190) -> Image.Image:
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((12, 12, 12 + width, 44), radius=8, fill=(15, 23, 42))
    draw.text((24, 22), label, fill=(255, 255, 255))
    return panel


def _response_panel(
    response: np.ndarray,
    label: str,
    *,
    signed: bool = False,
    size: tuple[int, int] = (600, 300),
) -> Image.Image:
    scale = max(float(np.percentile(np.abs(response), 99.5)), 1e-8)
    if signed:
        normalized = np.clip(response / scale, -1.0, 1.0)
        neutral = np.asarray((15, 23, 42), dtype=np.float32)
        negative = np.asarray((37, 99, 235), dtype=np.float32)
        positive = np.asarray((239, 68, 68), dtype=np.float32)
        rgb = np.broadcast_to(neutral, (*normalized.shape, 3)).copy()
        rgb += np.maximum(-normalized, 0.0)[..., None] * (negative - neutral)
        rgb += np.maximum(normalized, 0.0)[..., None] * (positive - neutral)
    else:
        display = np.rint(np.clip(np.abs(response) / scale, 0.0, 1.0) * 255)
        rgb = cv2.cvtColor(
            cv2.applyColorMap(display.astype(np.uint8), cv2.COLORMAP_TURBO),
            cv2.COLOR_BGR2RGB,
        )
    return _label(
        Image.fromarray(np.rint(rgb).astype(np.uint8)).resize(
            size, Image.Resampling.BILINEAR
        ),
        label,
    )


def _save_projected_canny(rgb: np.ndarray, output: Path) -> dict[str, object]:
    panorama = EquirectangularImage(rgb)

    def canny(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        smooth = cv2.GaussianBlur(gray, (5, 5), 1.0)
        return cv2.Canny(smooth, 40, 120)

    edges = panorama.process_views(
        canny, layout="cube", size=256, fov=100, blend="closest"
    ).image
    source = _label(Image.fromarray(rgb).resize((600, 300)), "Input ERP")
    shown = np.repeat(np.asarray(edges)[..., None], 3, axis=-1)
    result = _label(Image.fromarray(shown).resize((600, 300)), "Fused Canny ERP")
    figure = Image.new("RGB", (1200, 300), (15, 23, 42))
    figure.paste(source, (0, 0))
    figure.paste(result, (600, 0))
    figure.save(output, quality=91, optimize=True, progressive=True)
    return {
        "route": "cube/256/100deg -> OpenCV Gaussian+Canny -> closest fusion",
        "edge_pixels": int(np.count_nonzero(edges)),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def _save_gradient(gray: np.ndarray, output: Path) -> dict[str, object]:
    components = SphericalGradient(operator="scharr")(gray[None, None])
    east = components[0, 0]
    north = components[0, 1]
    magnitude = np.hypot(east, north)
    panels = (
        _response_panel(east, "East component", signed=True, size=(400, 200)),
        _response_panel(north, "North component", signed=True, size=(400, 200)),
        _response_panel(magnitude, "Magnitude", size=(400, 200)),
    )
    figure = Image.new("RGB", (1200, 200), (15, 23, 42))
    for index, panel in enumerate(panels):
        figure.paste(panel, (index * 400, 0))
    figure.save(output, quality=91, optimize=True, progressive=True)
    return {
        "operator": "sphere-native Scharr",
        "components": ["east", "north", "magnitude"],
        "display": "signed components and absolute magnitude; 99.5 percentile",
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def _save_dog(rgb: np.ndarray, gray: np.ndarray, output: Path) -> dict[str, object]:
    current = gray[None, None]
    small = SphericalGaussianBlur2d(sigma=1.0)
    large = SphericalGaussianBlur2d(sigma=2.0)
    resize = SphericalResize(scale_factor=0.5)

    responses: list[np.ndarray] = []
    shapes: list[list[int]] = []
    for _ in range(3):
        response = large(current) - small(current)
        responses.append(response[0, 0])
        shapes.append(list(response.shape[-2:]))
        current = resize(small(current))

    source = _label(
        Image.fromarray(rgb).resize((600, 300), Image.Resampling.LANCZOS),
        "Input ERP",
    )
    panels = [source]
    panels.extend(
        _response_panel(response, f"DoG octave {index}")
        for index, response in enumerate(responses)
    )
    figure = Image.new("RGB", (1200, 600), (15, 23, 42))
    for index, panel in enumerate(panels):
        figure.paste(panel, ((index % 2) * 600, (index // 2) * 300))
    figure.save(output, quality=91, optimize=True, progressive=True)
    return {
        "input_sha256": EXPECTED_INPUT_SHA256,
        "working_shape_hw": [256, 512],
        "octave_shape_hw": shapes,
        "sigma_pair": [1.0, 2.0],
        "display": "absolute DoG normalized at per-octave 99.5 percentile",
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def _save_point_return(rgb: np.ndarray, output: Path) -> dict[str, object]:
    panorama = EquirectangularImage(rgb)
    views = panorama.views("cube", size=192, fov=100)

    def detect(face):
        gray = cv2.cvtColor(face.image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        found = cv2.goodFeaturesToTrack(gray, 45, 0.01, 5)
        pixels = (
            np.empty((0, 2), dtype=np.float64)
            if found is None
            else found[:, 0].astype(np.float64)
        )
        return FacePointBatch(pixels)

    mapped = views.map_points(detect, source_id="opencv-gftt")
    fused = mapped.angular_nms(max_angle_deg=0.4)
    shown = rgb.copy()
    for observation in fused.observations.observations:
        if observation.source_erp_xy is None:
            continue
        x, y = np.rint(observation.source_erp_xy).astype(int)
        cv2.circle(shown, (x % shown.shape[1], y), 3, (255, 196, 0), -1)
        cv2.circle(shown, (x % shown.shape[1], y), 4, (15, 23, 42), 1)
    source = _label(Image.fromarray(rgb).resize((600, 300)), "Input ERP")
    result = _label(
        Image.fromarray(shown).resize((600, 300)), "Bearings after angular NMS"
    )
    figure = Image.new("RGB", (1200, 300), (15, 23, 42))
    figure.paste(source, (0, 0))
    figure.paste(result, (600, 0))
    figure.save(output, quality=91, optimize=True, progressive=True)
    return {
        "route": "cube face keypoints -> bearings -> 0.4deg angular NMS",
        "input_count": len(mapped),
        "kept_count": len(fused.observations),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def _save_region_return(rgb: np.ndarray, output: Path) -> dict[str, object]:
    panorama = EquirectangularImage(rgb)
    views = panorama.views("cube", size=192, fov=100)

    def detect(face):
        gray = cv2.cvtColor(face.image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.0), 40, 120)
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            cv2.dilate(edges, np.ones((5, 5), dtype=np.uint8))
        )
        candidates = []
        for left, top, width, height, area in stats[1:count]:
            if 100 <= area <= int(0.35 * edges.size):
                candidates.append((area, [left, top, left + width, top + height]))
        boxes = np.asarray(
            [box for _, box in sorted(candidates, reverse=True)[:3]],
            dtype=np.float64,
        ).reshape(-1, 4)
        scores = np.asarray(
            [area / edges.size for area, _ in sorted(candidates, reverse=True)[:3]],
            dtype=np.float64,
        )
        return FaceRegionBatch.from_boxes(boxes, scores=scores)

    mapped = views.map_regions(detect, source_id="opencv-components")
    fused = mapped.nms(iou_threshold=0.45, evaluation_shape_hw=(128, 256))
    shown = rgb.astype(np.float32)
    palette = np.asarray(
        ((14, 165, 233), (244, 114, 182), (250, 204, 21), (74, 222, 128)),
        dtype=np.float32,
    )
    for index, observation in enumerate(fused.observations.observations[:12]):
        mask = observation.to_mask(rgb.shape[:2]).to_array()
        color = palette[index % len(palette)]
        shown[mask] = 0.55 * shown[mask] + 0.45 * color
        contour, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(shown, contour, -1, tuple(float(v) for v in color), 1)
    source = _label(Image.fromarray(rgb).resize((600, 300)), "Input ERP")
    result = _label(
        Image.fromarray(np.rint(shown).astype(np.uint8)).resize((600, 300)),
        "Spherical regions after IoU NMS",
        width=250,
    )
    figure = Image.new("RGB", (1200, 300), (15, 23, 42))
    figure.paste(source, (0, 0))
    figure.paste(result, (600, 0))
    figure.save(output, quality=91, optimize=True, progressive=True)
    return {
        "route": "cube face boxes -> spherical footprints -> solid-angle IoU NMS",
        "input_count": len(mapped),
        "kept_count": len(fused.observations),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def _save_view_return(rgb: np.ndarray, output: Path) -> dict[str, object]:
    panorama = EquirectangularImage(rgb)
    views = panorama.views("cube", size=192, fov=100)
    evidence = views.map_views(
        lambda face: FaceViewOutput(
            scores=np.asarray(cv2.mean(face.image)[:3], dtype=np.float64) / 255.0,
            class_ids=np.asarray(("red", "green", "blue"), dtype=object),
        ),
        source_id="mean-rgb",
    )
    aggregated = evidence.aggregate("coverage_mean")
    faces = Image.new("RGB", (800, 300), (15, 23, 42))
    thumb_size = (260, 130)
    for index, face in enumerate(views):
        thumb = _label(
            Image.fromarray(face.image.astype(np.uint8)).resize(thumb_size),
            face.face_id,
            width=90,
        )
        faces.paste(thumb, ((index % 3) * 266, (index // 3) * 150))
    summary = Image.new("RGB", (400, 300), (15, 23, 42))
    summary = _label(summary, "coverage_mean", width=180)
    draw = ImageDraw.Draw(summary)
    scores = np.asarray(aggregated.scores)
    for index, (name, color) in enumerate(
        (("red", (239, 68, 68)), ("green", (74, 222, 128)), ("blue", (59, 130, 246)))
    ):
        y = 88 + index * 58
        draw.text((28, y), name, fill=(226, 232, 240))
        draw.rectangle((105, y, 350, y + 22), fill=(30, 41, 59))
        draw.rectangle(
            (105, y, 105 + int(245 * float(scores[index])), y + 22), fill=color
        )
        draw.text((358, y), f"{scores[index]:.2f}", fill=(226, 232, 240))
    figure = Image.new("RGB", (1200, 300), (15, 23, 42))
    figure.paste(faces, (0, 0))
    figure.paste(summary, (800, 0))
    figure.save(output, quality=91, optimize=True, progressive=True)
    return {
        "route": "one score vector per cube face -> coverage-mean aggregation",
        "face_count": len(views),
        "aggregate_scores": scores.tolist(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def main() -> int:
    args = _arguments()
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != EXPECTED_INPUT_SHA256:
        raise ValueError("input ERP checksum does not match the documented CC0 asset")

    rgb = np.asarray(Image.open(args.input).convert("RGB"))
    rgb = np.asarray(Image.fromarray(rgb).resize((512, 256), Image.Resampling.LANCZOS))
    gray = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2GRAY)
    output_dir = args.output_dir
    if output_dir.suffix:
        output_dir = output_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    canny_path = output_dir / "projected-canny.jpg"
    gradient_path = output_dir / "spherical-gradient-components.jpg"
    dog_path = output_dir / "spherical-dog-pyramid.jpg"
    points_path = output_dir / "return-path-points.jpg"
    regions_path = output_dir / "return-path-regions.jpg"
    views_path = output_dir / "return-path-view-evidence.jpg"
    results = {
        "projected_canny": _save_projected_canny(rgb, canny_path),
        "spherical_gradient": _save_gradient(gray, gradient_path),
        "spherical_dog_pyramid": _save_dog(rgb, gray, dog_path),
        "point_return": _save_point_return(rgb, points_path),
        "region_return": _save_region_return(rgb, regions_path),
        "view_return": _save_view_return(rgb, views_path),
    }

    print(
        json.dumps(
            results,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
