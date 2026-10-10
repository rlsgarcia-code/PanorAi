#!/usr/bin/env python3
"""Run exhaustive Experimental spherical segmentation on one P74 panorama.

InternImage evidence is consumed on its native spherical lattice.  SAM 2.1
sees only one transient gnomonic face and embedding at a time.  Proposal masks
are fused on a compact working ERP before accepted instances are expanded with
nearest sampling to the native-density canonical ERP and original P74 raster.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.spherical_multiview_depth.p74 import (  # noqa: E402
    P74_POLAR_LIMIT_RAD,
    load_native_angular_rgb,
)
from panorai.experimental.deep_learning.segmentation import (  # noqa: E402
    SphericalBinaryMask,
    SphericalSegment,
    SphericalSegmentationConfig,
    SphericalSemanticSegmenter,
    concept_evidence_from_imagenet,
    load_sam21_hiera_large,
    merge_proposals,
    partition_proposals_by_origin,
    resolve_panoptic_map,
)


def _nearest_resize(values: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    source = np.asarray(values)
    source_height, source_width = source.shape[:2]
    height, width = shape_hw
    rows = np.minimum(
        ((np.arange(height) + 0.5) / height * source_height).astype(np.int64),
        source_height - 1,
    )
    columns = np.minimum(
        ((np.arange(width) + 0.5) / width * source_width).astype(np.int64),
        source_width - 1,
    )
    return source[rows[:, None], columns[None, :]]


def _load_evidence(path: Path):
    with np.load(path, allow_pickle=False) as stored:
        schema = stored["schema"].item()
        if schema != "panorai-internimage-proxy-evidence/v1":
            raise ValueError(f"unsupported evidence schema {schema!r}")
        return concept_evidence_from_imagenet(
            stored["logits"],
            stored["class_indices"],
            stored["support"],
            provenance={
                "schema": schema,
                "source": str(path),
                "model": "spherical InternImage-G",
            },
        )


def _native_from_canonical(
    values: np.ndarray, raw_shape_hw: tuple[int, int]
) -> np.ndarray:
    raw_height, raw_width = raw_shape_hw
    canonical_height, canonical_width = values.shape[:2]
    x = np.arange(raw_width, dtype=np.float64)
    y = np.arange(raw_height, dtype=np.float64)
    longitude = (
        np.mod(-x / (raw_width - 1) * 2 * math.pi + math.pi, 2 * math.pi) - math.pi
    )
    polar = y / (raw_height - 1) * P74_POLAR_LIMIT_RAD
    map_x = np.mod(
        np.floor((longitude + math.pi) / (2 * math.pi) * canonical_width),
        canonical_width,
    ).astype(np.int64)
    map_y = np.clip(
        np.floor(polar / math.pi * canonical_height), 0, canonical_height - 1
    ).astype(np.int64)
    return values[map_y[:, None], map_x[None, :]]


def _palette(count: int) -> np.ndarray:
    colors = np.zeros((count + 1, 3), dtype=np.uint8)
    golden = 0.61803398875
    import colorsys

    for index in range(1, count + 1):
        colors[index] = (
            np.asarray(colorsys.hsv_to_rgb((index * golden) % 1.0, 0.72, 1.0)) * 255
        )
    return colors


def _overlay(rgb: np.ndarray, labels: np.ndarray, colors: np.ndarray) -> Image.Image:
    image = np.asarray(rgb, dtype=np.float32)
    color = colors[np.asarray(labels, dtype=np.int64)].astype(np.float32)
    foreground = labels > 0
    output = image.copy()
    output[foreground] = 0.48 * image[foreground] + 0.52 * color[foreground]
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode="RGB")


def _semantic_proxy_overlay(
    rgb: np.ndarray,
    labels: np.ndarray,
    segments: tuple[SphericalSegment, ...],
) -> Image.Image:
    concept_ids = sorted(
        {segment.concept_id for segment in segments if segment.concept_id is not None}
    )
    concept_colors = _palette(len(concept_ids))
    color_lookup = {
        concept_id: concept_colors[index + 1]
        for index, concept_id in enumerate(concept_ids)
    }
    colors = np.zeros((len(segments) + 1, 3), dtype=np.uint8)
    semantic_labels = np.asarray(labels).copy()
    for segment in segments:
        if segment.concept_id is None:
            semantic_labels[semantic_labels == segment.segment_id] = 0
        else:
            colors[segment.segment_id] = color_lookup[segment.concept_id]
    return _overlay(rgb, semantic_labels, colors)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _annotate_instances(image: Image.Image, labels: np.ndarray) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    font = _font(max(14, image.height // 130))
    radius = max(11, image.height // 155)
    for identifier in sorted(int(value) for value in np.unique(labels) if value):
        rows, columns = np.nonzero(labels == identifier)
        if not len(rows):
            continue
        x = int(np.median(columns))
        y = int(np.median(rows))
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill="black",
            outline="white",
            width=max(1, radius // 5),
        )
        text = str(identifier)
        box = draw.textbbox((0, 0), text, font=font)
        draw.text(
            (x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2 - 1),
            text,
            fill="white",
            font=font,
        )
    return result


def _write_legend(
    output_dir: Path, segments: tuple[SphericalSegment, ...], colors: np.ndarray
) -> None:
    row_height = 34
    width = 1500
    canvas = Image.new("RGB", (width, 36 + row_height * len(segments)), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(20)
    font = _font(15)
    draw.text(
        (12, 8),
        "Instancias esfericas — conceito | proxies ImageNet",
        fill="black",
        font=title_font,
    )
    for row, segment in enumerate(segments):
        y = 40 + row * row_height
        color = tuple(int(value) for value in colors[segment.segment_id])
        draw.rectangle((12, y, 42, y + 24), fill=color, outline="black")
        proxies = ", ".join(segment.proxy_classes) or "sem evidencia suficiente"
        draw.text(
            (52, y + 3),
            f"{segment.segment_id:03d}  {segment.concept_name} | {proxies}",
            fill="black",
            font=font,
        )
    canvas.save(output_dir / "instance-legend.png")


def _write_concept_panels(
    output_dir: Path,
    segments: tuple[SphericalSegment, ...],
    canonical_rgb: np.ndarray,
) -> None:
    preview_shape = (768, 1536)
    preview_rgb = _nearest_resize(canonical_rgb, preview_shape)
    concept_dir = output_dir / "concepts"
    unknown_dir = output_dir / "unknown"
    concept_dir.mkdir(parents=True, exist_ok=True)
    unknown_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[SphericalSegment]] = {}
    for segment in segments:
        key = segment.concept_id or "unknown"
        groups.setdefault(key, []).append(segment)
        if segment.concept_id is None:
            mask = _nearest_resize(segment.consensus.to_array(), preview_shape)
            _overlay(
                preview_rgb,
                mask.astype(np.uint16),
                np.asarray([[0, 0, 0], [255, 30, 120]], dtype=np.uint8),
            ).save(unknown_dir / f"unknown-{segment.segment_id:04d}.jpg", quality=92)
    for concept_id, members in groups.items():
        union = np.zeros(preview_shape, dtype=bool)
        for segment in members:
            union |= _nearest_resize(segment.consensus.to_array(), preview_shape)
        panel = _overlay(
            preview_rgb,
            union.astype(np.uint16),
            np.asarray([[0, 0, 0], [255, 30, 120]], dtype=np.uint8),
        )
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, 900, 42), fill="black")
        draw.text(
            (10, 10),
            f"{concept_id}: {len(members)} instancia(s)",
            fill="white",
            font=_font(16),
        )
        panel.save(concept_dir / f"{concept_id}.jpg", quality=92)


def _write_origin_comparison(
    output_dir: Path,
    result,
    evidence,
    config: SphericalSegmentationConfig,
    canonical_rgb: np.ndarray,
) -> None:
    by_origin = partition_proposals_by_origin(result.proposals)
    source_segments = {
        "CAM esferico": merge_proposals(by_origin["cam"], config, evidence=evidence),
        "Cobertura exaustiva": merge_proposals(
            by_origin["coverage"], config, evidence=evidence
        ),
        "Hibrido + expansao": result.segments,
    }
    preview_shape = (512, 1024)
    preview_rgb = _nearest_resize(canonical_rgb, preview_shape)
    canvas = Image.new("RGB", (preview_shape[1] * 3, preview_shape[0] + 44), "white")
    for index, (title, segments) in enumerate(source_segments.items()):
        labels = resolve_panoptic_map(segments, result.support)
        labels = _nearest_resize(labels, preview_shape)
        panel = _overlay(preview_rgb, labels, _palette(len(segments)))
        canvas.paste(panel, (index * preview_shape[1], 44))
        ImageDraw.Draw(canvas).text(
            (index * preview_shape[1] + 12, 12),
            f"{title} — {len(segments)} segmentos",
            fill="black",
            font=_font(17),
        )
    canvas.save(output_dir / "comparison-cam-exhaustive-hybrid.jpg", quality=93)


def _write_previous_comparison(
    output_dir: Path, current: Image.Image, previous_path: Path | None
) -> None:
    if previous_path is None or not previous_path.is_file():
        return
    with Image.open(previous_path) as source:
        previous = source.convert("RGB")
    target_height = 720
    previous.thumbnail((1600, target_height), Image.Resampling.LANCZOS)
    current_copy = current.copy()
    current_copy.thumbnail((1600, target_height), Image.Resampling.LANCZOS)
    canvas = Image.new(
        "RGB",
        (previous.width + current_copy.width, target_height + 48),
        "white",
    )
    canvas.paste(previous, (0, 48))
    canvas.paste(current_copy, (previous.width, 48))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 14), "Estudo anterior: coluna e tubulacao", fill="black", font=_font(17)
    )
    draw.text(
        (previous.width + 12, 14),
        "Segmentacao panoptica automatica",
        fill="black",
        font=_font(17),
    )
    canvas.save(output_dir / "comparison-column-pipe.jpg", quality=93)


def _peak_rss_megabytes() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return value / divisor


def _full_resolution_segments(
    segments: tuple[SphericalSegment, ...], shape_hw: tuple[int, int]
) -> tuple[SphericalSegment, ...]:
    result = []
    for segment in segments:
        result.append(
            SphericalSegment(
                segment_id=segment.segment_id,
                concept_id=segment.concept_id,
                concept_name=segment.concept_name,
                proxy_classes=segment.proxy_classes,
                consensus=SphericalBinaryMask.from_array(
                    _nearest_resize(segment.consensus.to_array(), shape_hw).astype(bool)
                ),
                envelope=SphericalBinaryMask.from_array(
                    _nearest_resize(segment.envelope.to_array(), shape_hw).astype(bool)
                ),
                quality_score=segment.quality_score,
                semantic_score=segment.semantic_score,
                proposal_ids=segment.proposal_ids,
            )
        )
    return tuple(result)


def _write_segment_artifacts(
    output_dir: Path,
    segments: tuple[SphericalSegment, ...],
    proposals,
    canonical_rgb: np.ndarray,
) -> None:
    segment_dir = output_dir / "segments"
    proposal_dir = output_dir / "proposals"
    segment_dir.mkdir(parents=True, exist_ok=True)
    proposal_dir.mkdir(parents=True, exist_ok=True)
    preview_shape = (1024, 2048)
    preview_rgb = _nearest_resize(canonical_rgb, preview_shape)
    for segment in segments:
        np.savez_compressed(
            segment_dir / f"segment-{segment.segment_id:04d}.npz",
            schema=np.asarray("panorai-spherical-segment/v1"),
            shape_hw=np.asarray(segment.consensus.shape_hw),
            consensus_packed=segment.consensus.packed_bits,
            envelope_packed=segment.envelope.packed_bits,
        )
        mask = _nearest_resize(segment.consensus.to_array(), preview_shape).astype(bool)
        color = np.zeros(preview_shape, dtype=np.uint16)
        color[mask] = 1
        overlay = _overlay(preview_rgb, color, np.asarray([[0, 0, 0], [255, 30, 120]]))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle((0, 0, 850, 48), fill="black")
        draw.text(
            (12, 14),
            f"{segment.segment_id:03d} {segment.concept_name} | proxies: "
            + ", ".join(segment.proxy_classes),
            fill="white",
            font=ImageFont.load_default(),
        )
        overlay.save(segment_dir / f"segment-{segment.segment_id:04d}.jpg", quality=92)
    retained_ids = {
        identifier for segment in segments for identifier in segment.proposal_ids
    }
    for proposal in proposals:
        if proposal.proposal_id not in retained_ids:
            continue
        canvas = Image.new("RGB", (1536, 320), "white")
        for index, mask in enumerate(proposal.alternatives):
            local = _nearest_resize(mask.to_array(), (256, 512)).astype(bool)
            panel = _overlay(
                _nearest_resize(canonical_rgb, (256, 512)),
                local.astype(np.uint16),
                np.asarray([[0, 0, 0], [255, 30, 120]]),
            )
            ImageDraw.Draw(panel).text(
                (8, 8),
                f"mask {index + 1}/3",
                fill="white",
                font=ImageFont.load_default(),
            )
            canvas.paste(panel, (512 * index, 40))
        canvas.save(
            proposal_dir / f"{proposal.proposal_id}-three-masks.jpg", quality=90
        )
        np.savez_compressed(
            proposal_dir / f"{proposal.proposal_id}-masks.npz",
            schema=np.asarray("panorai-spherical-mask-proposal/v1"),
            shape_hw=np.asarray(proposal.consensus.shape_hw),
            alternatives_packed=np.stack(
                [mask.packed_bits for mask in proposal.alternatives]
            ),
            consensus_packed=proposal.consensus.packed_bits,
            envelope_packed=proposal.envelope.packed_bits,
        )


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    record = json.loads(args.record.read_text(encoding="utf-8"))
    source = Path(record["source_path"])
    with Image.open(source) as raw_image:
        raw_rgb = np.asarray(raw_image.convert("RGB"))
    raw_shape = raw_rgb.shape[:2]
    canonical_height = round((raw_shape[0] - 1) * math.pi / P74_POLAR_LIMIT_RAD)
    canonical_shape = (canonical_height, 2 * canonical_height)
    canonical_rgb, canonical_support = load_native_angular_rgb(source, canonical_shape)
    evidence = _load_evidence(args.evidence)
    backend = load_sam21_hiera_large(
        accept_upstream_terms=args.accept_upstream_terms,
        cache_dir=args.cache_dir,
        device=args.device,
    )
    config = SphericalSegmentationConfig(
        semantic_peaks_per_concept=args.semantic_peaks_per_concept,
        coverage_direction_count=args.coverage_directions,
        prompt_grid_size=args.prompt_grid_size,
        prompt_batch_size=args.prompt_batch_size,
        maximum_expansion_faces=args.maximum_expansion_faces,
        discovery_face_size=args.face_size,
    )
    events_path = output_dir / "progress.jsonl"

    def progress(event: dict[str, object]) -> None:
        line = json.dumps({"time": time.time(), **event}, sort_keys=True)
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)

    started = time.perf_counter()
    result = SphericalSemanticSegmenter(backend, config).segment(
        canonical_rgb,
        canonical_support,
        evidence=evidence,
        output_shape_hw=(args.working_height, 2 * args.working_height),
        progress=progress,
    )
    elapsed = time.perf_counter() - started
    full_segments = _full_resolution_segments(result.segments, canonical_shape)
    full_panoptic = resolve_panoptic_map(full_segments, canonical_support)
    native_panoptic = _native_from_canonical(full_panoptic, raw_shape)
    colors = _palette(len(full_segments))
    canonical_clean = _overlay(canonical_rgb, full_panoptic, colors)
    native_clean = _overlay(raw_rgb, native_panoptic, colors)
    canonical_overlay = _annotate_instances(canonical_clean, full_panoptic)
    native_overlay = _annotate_instances(native_clean, native_panoptic)
    canonical_clean.save(output_dir / "panoptic-canonical-overlay.jpg", quality=94)
    native_clean.save(output_dir / "panoptic-native-p74-overlay.jpg", quality=94)
    canonical_overlay.save(
        output_dir / "panoptic-canonical-overlay-numbered.jpg", quality=94
    )
    native_overlay.save(
        output_dir / "panoptic-native-p74-overlay-numbered.jpg", quality=94
    )
    _semantic_proxy_overlay(canonical_rgb, full_panoptic, full_segments).save(
        output_dir / "semantic-proxy-canonical-overlay.jpg", quality=94
    )
    _semantic_proxy_overlay(raw_rgb, native_panoptic, full_segments).save(
        output_dir / "semantic-proxy-native-p74-overlay.jpg", quality=94
    )
    Image.fromarray(full_panoptic.astype(np.uint16)).save(
        output_dir / "panoptic-canonical.png"
    )
    Image.fromarray(native_panoptic.astype(np.uint16)).save(
        output_dir / "panoptic-native-p74.png"
    )
    _write_segment_artifacts(output_dir, full_segments, result.proposals, canonical_rgb)
    _write_legend(output_dir, full_segments, colors)
    _write_concept_panels(output_dir, result.segments, canonical_rgb)
    _write_origin_comparison(output_dir, result, evidence, config, canonical_rgb)
    _write_previous_comparison(output_dir, native_clean, args.previous_proof)
    retained_proposal_ids = {
        identifier for segment in full_segments for identifier in segment.proposal_ids
    }
    catalog = {
        "schema": "panorai-p74-exhaustive-segmentation/v1",
        "source_path": str(source),
        "source_native_shape_hw": list(raw_shape),
        "canonical_shape_hw": list(canonical_shape),
        "working_shape_hw": [args.working_height, 2 * args.working_height],
        "interface": result.interface,
        "models": {
            "semantic": "spherical InternImage-G evidence",
            "segmentation": backend.assets.to_dict(),
            "device": str(backend.device),
        },
        "config": {name: getattr(config, name) for name in config.__dataclass_fields__},
        "diagnostics": {
            **result.diagnostics,
            "elapsed_seconds": elapsed,
            "peak_resident_memory_megabytes": _peak_rss_megabytes(),
            "retained_proposal_count": len(retained_proposal_ids),
        },
        "segments": [
            {
                "segment_id": item.segment_id,
                "concept_id": item.concept_id,
                "concept_name": item.concept_name,
                "proxy_classes": list(item.proxy_classes),
                "quality_score": item.quality_score,
                "semantic_score": item.semantic_score,
                "proposal_ids": list(item.proposal_ids),
                "pixel_count_canonical": item.consensus.pixel_count,
            }
            for item in full_segments
        ],
        "proposals": [
            {
                "proposal_id": item.proposal_id,
                "parent_proposal_id": item.parent_proposal_id,
                "seed_id": item.seed.seed_id,
                "seed_source": item.seed.source,
                "center_longitude_degrees": item.seed.longitude_degrees,
                "center_latitude_degrees": item.seed.latitude_degrees,
                "concept_id": item.concept_id,
                "proxy_classes": list(item.proxy_classes),
                "pairwise_iou": item.pairwise_iou,
                "predicted_iou": list(item.predicted_iou),
                "stability_score": item.stability_score,
                "quality_score": item.quality_score,
                "retained": item.proposal_id in retained_proposal_ids,
                "mask_archive": (
                    f"proposals/{item.proposal_id}-masks.npz"
                    if item.proposal_id in retained_proposal_ids
                    else None
                ),
                "audit_panel": (
                    f"proposals/{item.proposal_id}-three-masks.jpg"
                    if item.proposal_id in retained_proposal_ids
                    else None
                ),
            }
            for item in result.proposals
        ],
        "artifacts": {
            "canonical_overlay": "panoptic-canonical-overlay.jpg",
            "native_overlay": "panoptic-native-p74-overlay.jpg",
            "canonical_numbered_overlay": "panoptic-canonical-overlay-numbered.jpg",
            "native_numbered_overlay": "panoptic-native-p74-overlay-numbered.jpg",
            "canonical_semantic_proxy_overlay": "semantic-proxy-canonical-overlay.jpg",
            "native_semantic_proxy_overlay": "semantic-proxy-native-p74-overlay.jpg",
            "canonical_labels": "panoptic-canonical.png",
            "native_labels": "panoptic-native-p74.png",
            "legend": "instance-legend.png",
            "origin_comparison": "comparison-cam-exhaustive-hybrid.jpg",
            "previous_column_pipe_comparison": (
                "comparison-column-pipe.jpg"
                if args.previous_proof is not None and args.previous_proof.is_file()
                else None
            ),
            "concept_panels": "concepts/",
            "unknown_panels": "unknown/",
        },
        "limitations": [
            "ImageNet concepts are visual proxies rather than industrial ground truth",
            "SAM agreement measures mask repeatability rather than semantic correctness",
            "P74 G100 has no complete instance or panoptic annotation",
        ],
    }
    (output_dir / "result.json").write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--face-size", type=int, default=1024)
    parser.add_argument("--working-height", type=int, default=1024)
    parser.add_argument("--semantic-peaks-per-concept", type=int, default=12)
    parser.add_argument("--coverage-directions", type=int, default=96)
    parser.add_argument("--prompt-grid-size", type=int, default=8)
    parser.add_argument("--prompt-batch-size", type=int, default=16)
    parser.add_argument("--maximum-expansion-faces", type=int, default=24)
    parser.add_argument("--previous-proof", type=Path, default=None)
    parser.add_argument("--accept-upstream-terms", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args())["diagnostics"], indent=2, sort_keys=True))
