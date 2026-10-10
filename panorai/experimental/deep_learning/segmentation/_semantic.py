"""Generic proxy vocabulary and spherical semantic seed selection."""

from __future__ import annotations

import math

import numpy as np

from ._models import DenseSemanticEvidence, SemanticProxyConcept, SphericalSeed

IMAGENET_PROXY_VOCABULARY = (
    SemanticProxyConcept(
        "circular-object",
        "circular object",
        (
            (409, "analog clock", 0.50),
            (545, "electric fan", 0.25),
            (897, "washer", 0.25),
        ),
    ),
    SemanticProxyConcept(
        "railing",
        "railing or banister",
        ((421, "bannister", 1.0),),
    ),
    SemanticProxyConcept(
        "screen-like-object",
        "screen-like object",
        ((556, "fire screen", 1.0),),
    ),
    SemanticProxyConcept(
        "repeated-vertical-structure",
        "repeated vertical structure",
        ((687, "organ", 1.0),),
    ),
    SemanticProxyConcept(
        "grid-like-structure",
        "grid-like structure",
        ((743, "prison", 1.0),),
    ),
    SemanticProxyConcept(
        "elongated-object",
        "elongated object",
        ((466, "bullet train", 1.0),),
    ),
    SemanticProxyConcept(
        "cylindrical-container",
        "cylindrical container",
        (
            (653, "milk can", 0.35),
            (822, "steel drum", 0.25),
            (427, "barrel", 0.20),
            (438, "beaker", 0.20),
        ),
    ),
    SemanticProxyConcept(
        "curved-structure",
        "curved structure",
        ((506, "coil", 0.35), (821, "steel arch bridge", 0.35), (758, "reel", 0.30)),
    ),
)


def _normalize_channel(values: np.ndarray, support: np.ndarray) -> np.ndarray:
    positive = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    valid = positive[support]
    output = np.zeros_like(positive, dtype=np.float64)
    if len(valid):
        low = float(valid.min())
        scale = float(valid.max() - low)
        if scale > np.finfo(np.float64).eps:
            output[support] = (valid - low) / scale
    return output


def concept_evidence_from_imagenet(
    logits: np.ndarray,
    class_indices: np.ndarray,
    support: np.ndarray,
    *,
    concepts: tuple[SemanticProxyConcept, ...] = IMAGENET_PROXY_VOCABULARY,
    provenance: dict[str, object] | None = None,
) -> DenseSemanticEvidence:
    """Convert selected ImageNet dense logits into normalized proxy concepts."""

    values = np.asarray(logits)
    indices = np.asarray(class_indices, dtype=np.int64)
    valid = np.asarray(support, dtype=bool)
    if values.ndim != 3 or len(indices) != values.shape[0]:
        raise ValueError("logits must be CHW and align with class_indices")
    if valid.shape != values.shape[-2:]:
        raise ValueError("support and logits must share spatial shape")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("class_indices must not contain duplicates")
    lookup = {int(index): position for position, index in enumerate(indices)}
    channels: list[np.ndarray] = []
    for concept in concepts:
        combined = np.zeros(valid.shape, dtype=np.float64)
        for class_index, _, weight in concept.class_weights:
            if class_index not in lookup:
                raise ValueError(
                    f"ImageNet class {class_index} required by {concept.concept_id!r} "
                    "is absent"
                )
            combined += weight * _normalize_channel(values[lookup[class_index]], valid)
        channels.append(np.where(valid, np.clip(combined, 0.0, 1.0), 0.0))
    return DenseSemanticEvidence(
        tuple(concept.concept_id for concept in concepts),
        np.stack(channels).astype(np.float32),
        valid,
        provenance={} if provenance is None else provenance,
    )


def _direction(row: int, column: int, shape_hw: tuple[int, int]) -> tuple[float, float]:
    height, width = shape_hw
    longitude = (column + 0.5) / width * 360.0 - 180.0
    latitude = 90.0 - (row + 0.5) / height * 180.0
    return longitude, latitude


def angular_distance_degrees(
    first: tuple[float, float], second: tuple[float, float]
) -> float:
    lon1, lat1 = map(math.radians, first)
    lon2, lat2 = map(math.radians, second)
    cosine = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(
        lat2
    ) * math.cos(lon1 - lon2)
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def select_semantic_seeds(
    evidence: DenseSemanticEvidence,
    *,
    maximum_per_concept: int = 12,
    minimum_separation_degrees: float = 12.0,
    relative_threshold: float = 0.30,
) -> tuple[SphericalSeed, ...]:
    """Select geodesically separated local maxima from every concept channel."""

    if maximum_per_concept < 1:
        raise ValueError("maximum_per_concept must be positive")
    if not 0 <= relative_threshold <= 1:
        raise ValueError("relative_threshold must be within [0,1]")
    seeds: list[SphericalSeed] = []
    for channel_index, concept_id in enumerate(evidence.concept_ids):
        channel = np.where(evidence.support, evidence.scores[channel_index], -np.inf)
        maximum = float(np.max(channel))
        if not math.isfinite(maximum) or maximum <= 0:
            continue
        selected: list[tuple[float, float]] = []
        for flat_index in np.argsort(channel.reshape(-1))[::-1]:
            score = float(channel.reshape(-1)[flat_index])
            if score < relative_threshold * maximum:
                break
            row, column = np.unravel_index(flat_index, channel.shape)
            direction = _direction(int(row), int(column), channel.shape)
            if any(
                angular_distance_degrees(direction, previous)
                < minimum_separation_degrees
                for previous in selected
            ):
                continue
            number = len(selected) + 1
            selected.append(direction)
            seeds.append(
                SphericalSeed(
                    seed_id=f"semantic:{concept_id}:{number:02d}",
                    longitude_degrees=direction[0],
                    latitude_degrees=direction[1],
                    source="semantic",
                    score=score,
                    concept_id=concept_id,
                )
            )
            if len(selected) >= maximum_per_concept:
                break
    return tuple(seeds)


def fibonacci_coverage_seeds(
    support: np.ndarray,
    *,
    direction_count: int = 96,
) -> tuple[SphericalSeed, ...]:
    """Return equal-area full-sphere directions whose centers lie in support."""

    valid = np.asarray(support, dtype=bool)
    if valid.ndim != 2 or direction_count < 1:
        raise ValueError("support must be HW and direction_count must be positive")
    height, width = valid.shape
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    seeds: list[SphericalSeed] = []
    for index in range(direction_count):
        y = 1.0 - 2.0 * (index + 0.5) / direction_count
        latitude = math.degrees(math.asin(y))
        longitude = math.degrees(
            (index * golden_angle + math.pi) % (2 * math.pi) - math.pi
        )
        row = int(np.clip((90.0 - latitude) / 180.0 * height, 0, height - 1))
        column = int(np.floor((longitude + 180.0) / 360.0 * width)) % width
        if valid[row, column]:
            seeds.append(
                SphericalSeed(
                    seed_id=f"coverage:{index:03d}",
                    longitude_degrees=longitude,
                    latitude_degrees=latitude,
                    source="coverage",
                    score=1.0,
                )
            )
    return tuple(seeds)


def concept_display_names(
    concepts: tuple[SemanticProxyConcept, ...] = IMAGENET_PROXY_VOCABULARY,
) -> dict[str, str]:
    return {concept.concept_id: concept.display_name for concept in concepts}


def concept_proxy_names(
    concepts: tuple[SemanticProxyConcept, ...] = IMAGENET_PROXY_VOCABULARY,
) -> dict[str, tuple[str, ...]]:
    return {
        concept.concept_id: tuple(item[1] for item in concept.class_weights)
        for concept in concepts
    }
