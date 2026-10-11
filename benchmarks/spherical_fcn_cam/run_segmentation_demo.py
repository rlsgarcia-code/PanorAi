#!/usr/bin/env python3
"""Generate a bounded class-agnostic spherical segmentation demonstration."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from panorai.experimental.deep_learning.segmentation import (
    SAM21_HIERA_LARGE_LICENSE,
    SAM21_HIERA_LARGE_REPOSITORY,
    SAM21_HIERA_LARGE_REVISION,
    SphericalSegmentationConfig,
    SphericalSemanticSegmenter,
    load_sam21_hiera_large,
)

SCHEMA = "panorai-spherical-segmentation-demo/v1"
PALETTE = np.asarray(
    [
        (0, 0, 0),
        (0, 191, 255),
        (255, 111, 97),
        (117, 223, 139),
        (255, 206, 84),
        (190, 126, 255),
        (255, 139, 61),
        (66, 214, 198),
        (240, 99, 164),
        (138, 173, 255),
        (193, 225, 91),
        (255, 157, 181),
    ],
    dtype=np.uint8,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resize_erp(image: Image.Image, height: int) -> np.ndarray:
    if image.width != 2 * image.height:
        raise ValueError("input must be a complete 2:1 equirectangular panorama")
    return np.asarray(
        image.resize((2 * height, height), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


def _panoptic_boundaries(panoptic: np.ndarray) -> np.ndarray:
    horizontal = panoptic != np.roll(panoptic, 1, axis=1)
    vertical = np.zeros_like(horizontal)
    vertical[1:] = panoptic[1:] != panoptic[:-1]
    return (horizontal | vertical) & (panoptic > 0)


def render_overlay(rgb: np.ndarray, panoptic: np.ndarray) -> np.ndarray:
    """Render deterministic colors and seam-aware outlines for a panoptic map."""

    source = np.asarray(rgb, dtype=np.uint8)
    labels = np.asarray(panoptic, dtype=np.uint32)
    if source.shape[:2] != labels.shape or source.shape[2:] != (3,):
        raise ValueError("rgb and panoptic arrays must be aligned HWC/HW values")
    colors = PALETTE[labels % len(PALETTE)]
    result = source.copy()
    foreground = labels > 0
    blended = np.rint(0.52 * source + 0.48 * colors).astype(np.uint8)
    result[foreground] = blended[foreground]
    result[_panoptic_boundaries(labels)] = 255
    return result


def _config() -> SphericalSegmentationConfig:
    """Return the frozen bounded documentation preset."""

    return SphericalSegmentationConfig(
        coverage_direction_count=16,
        discovery_face_size=512,
        prompt_grid_size=4,
        prompt_batch_size=16,
        maximum_expansion_faces=6,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--processing-height", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--accept-upstream-terms",
        action="store_true",
        help="Confirm review of the pinned SAM 2.1 model and dataset terms.",
    )
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--source-license", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not args.accept_upstream_terms:
        raise SystemExit("--accept-upstream-terms is required before model acquisition")
    if args.processing_height < 64:
        raise SystemExit("--processing-height must be at least 64")

    source_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        source_size = [image.height, image.width]
        rgb = _resize_erp(image.convert("RGB"), args.processing_height)
    support = np.ones(rgb.shape[:2], dtype=bool)
    config = _config()

    def progress(event: dict[str, object]) -> None:
        print(json.dumps(event, sort_keys=True), flush=True)

    started = time.perf_counter()
    backend = load_sam21_hiera_large(
        accept_upstream_terms=True,
        cache_dir=args.cache_dir,
        device=args.device,
    )
    try:
        result = SphericalSemanticSegmenter(backend, config).segment(
            rgb,
            support,
            evidence=None,
            output_shape_hw=rgb.shape[:2],
            progress=progress,
        )
        assets = backend.assets.to_dict()
        assets.pop("snapshot_path", None)
        assets.pop("previously_cached", None)
    finally:
        backend.release()
        del backend
    elapsed = time.perf_counter() - started
    if not result.segments:
        raise RuntimeError("the frozen demonstration preset produced no segments")

    input_name = "industrial-piping-erp.jpg"
    overlay_name = "industrial-piping-segmentation.jpg"
    Image.fromarray(rgb, mode="RGB").save(
        output_dir / input_name, quality=92, optimize=True, subsampling=0
    )
    Image.fromarray(render_overlay(rgb, result.panoptic_map), mode="RGB").save(
        output_dir / overlay_name, quality=92, optimize=True, subsampling=0
    )

    metadata = {
        "schema": SCHEMA,
        "scope": (
            "bounded class-agnostic documentation demonstration; not semantic "
            "class accuracy, benchmark, or ground truth"
        ),
        "input": {
            "filename": source_path.name,
            "sha256": _sha256(source_path),
            "source_shape_hw": source_size,
            "processing_shape_hw": list(rgb.shape[:2]),
            "projection": "equirectangular",
            "source_url": args.source_url,
            "license": args.source_license,
        },
        "model": {
            "repository": SAM21_HIERA_LARGE_REPOSITORY,
            "revision": SAM21_HIERA_LARGE_REVISION,
            "license": SAM21_HIERA_LARGE_LICENSE,
            "assets": assets,
        },
        "method": {
            "interface": result.interface,
            "semantic_evidence": None,
            "configuration": asdict(config),
            "elapsed_seconds": elapsed,
        },
        "result": {
            "segment_count": len(result.segments),
            "unknown_segment_count": len(result.unknown_segments),
            "diagnostics": result.diagnostics,
            "segments": [
                {
                    "segment_id": segment.segment_id,
                    "concept_id": segment.concept_id,
                    "quality_score": segment.quality_score,
                    "proposal_ids": list(segment.proposal_ids),
                }
                for segment in result.segments
            ],
        },
        "outputs": {
            input_name: _sha256(output_dir / input_name),
            overlay_name: _sha256(output_dir / overlay_name),
        },
    }
    (output_dir / "result.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata["result"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
