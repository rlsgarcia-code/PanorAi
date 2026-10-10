from __future__ import annotations

import math
import os
import subprocess
import sys

import numpy as np
import pytest

from panorai.experimental.deep_learning.segmentation import (
    SPHERICAL_INDUSTRIAL_SEGMENTATION_INTERFACE,
    DenseSemanticEvidence,
    SamMultimaskOutput,
    SphericalBinaryMask,
    SphericalIndustrialSegmenter,
    SphericalMaskProposal,
    SphericalSeed,
    SphericalSegmentationConfig,
    SphericalSemanticSegmenter,
    backproject_face_mask_sets,
    backproject_face_masks,
    concept_evidence_from_imagenet,
    consensus_frontiers,
    fibonacci_coverage_seeds,
    majority_and_envelope,
    merge_proposals,
    non_maximum_suppression,
    partition_proposals_by_origin,
    prompt_grid,
    resolve_panoptic_map,
    select_semantic_seeds,
    weighted_intersection_over_union,
)
from panorai.geometry import GnomonicSpec, equirectangular_to_gnomonic


def test_segmentation_import_keeps_model_runtimes_lazy() -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "KMP_USE_SHM": "0",
            "KMP_INIT_AT_FORK": "FALSE",
            "OMP_NUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import panorai.experimental.deep_learning.segmentation; "
            "assert 'transformers' not in sys.modules; "
            "assert 'huggingface_hub' not in sys.modules",
        ],
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr


def test_packed_mask_round_trip_and_full_sphere_area() -> None:
    values = np.ones((37, 73), dtype=bool)
    packed = SphericalBinaryMask.from_array(values)

    np.testing.assert_array_equal(packed.to_array(), values)
    assert packed.solid_angle_area() == pytest.approx(4 * math.pi, rel=4e-4)


def test_solid_angle_iou_downweights_polar_pixels() -> None:
    equator = np.zeros((18, 36), dtype=bool)
    pole = np.zeros_like(equator)
    equator[8:10, :4] = True
    pole[:2, :4] = True
    combined = equator | pole

    weighted = weighted_intersection_over_union(equator, combined)
    pixel_iou = np.count_nonzero(equator) / np.count_nonzero(combined)

    assert weighted > pixel_iou


def test_proxy_evidence_keeps_support_and_explicit_class_provenance() -> None:
    required = [
        409,
        545,
        897,
        421,
        556,
        687,
        743,
        466,
        653,
        822,
        427,
        438,
        506,
        821,
        758,
    ]
    logits = np.zeros((len(required), 3, 6), dtype=np.float32)
    for index in range(len(required)):
        logits[index, 1, index % 6] = index + 1
    support = np.ones((3, 6), dtype=bool)
    support[2] = False

    evidence = concept_evidence_from_imagenet(
        logits, np.asarray(required), support, provenance={"model": "test"}
    )

    assert evidence.scores.shape == (8, 3, 6)
    assert np.all(evidence.scores[:, 2] == 0)
    assert evidence.provenance["model"] == "test"


def test_semantic_seed_nms_is_geodesic_across_longitude_seam() -> None:
    scores = np.zeros((1, 4, 8), dtype=np.float32)
    scores[0, 1, 0] = 1.0
    scores[0, 1, -1] = 0.9
    scores[0, 2, 4] = 0.8
    evidence = DenseSemanticEvidence(("pipes",), scores, np.ones((4, 8), dtype=bool))

    seeds = select_semantic_seeds(
        evidence,
        maximum_per_concept=3,
        minimum_separation_degrees=60,
        relative_threshold=0.1,
    )

    assert len(seeds) == 2
    assert seeds[0].longitude_degrees < -150
    assert abs(seeds[1].longitude_degrees) < 30


def test_fibonacci_coverage_respects_support() -> None:
    support = np.zeros((18, 36), dtype=bool)
    support[:9] = True

    seeds = fibonacci_coverage_seeds(support, direction_count=40)

    assert seeds
    assert all(seed.latitude_degrees > 0 for seed in seeds)


def test_multimask_majority_and_frontiers_do_not_follow_union_only() -> None:
    masks = np.zeros((3, 20, 20), dtype=bool)
    masks[:2, :5, 8:12] = True
    masks[0, -5:, 8:12] = True

    consensus, envelope = majority_and_envelope(masks)
    frontiers = consensus_frontiers(consensus, band_pixels=5, minimum_pixels=10)

    assert [side for side, _ in frontiers] == ["top"]
    assert envelope[-5:].any()


def test_mask_round_trip_through_gnomonic_chart_exceeds_declared_iou() -> None:
    shape = (180, 360)
    source = np.zeros(shape, dtype=np.uint8)
    source[70:110, 160:200] = 1
    spec = GnomonicSpec(
        center_lon_deg=0,
        center_lat_deg=0,
        hfov_deg=60,
        vfov_deg=60,
        output_shape_hw=(256, 256),
    )
    face = equirectangular_to_gnomonic(source, spec, interpolation="nearest").data
    alternatives = np.repeat(face[None].astype(bool), 3, axis=0)
    reconstructed = backproject_face_masks(alternatives, spec, shape)[0]

    assert weighted_intersection_over_union(source > 0, reconstructed) >= 0.95


def test_batched_mask_backprojection_matches_individual_oracle() -> None:
    shape = (90, 180)
    masks = np.zeros((2, 3, 64, 64), dtype=bool)
    masks[0, :, 12:30, 18:42] = True
    masks[1, :, 28:52, 4:36] = True
    spec = GnomonicSpec(
        center_lon_deg=35,
        center_lat_deg=-15,
        hfov_deg=70,
        vfov_deg=70,
        output_shape_hw=(64, 64),
    )

    batched = backproject_face_mask_sets(masks, spec, shape)

    assert batched.shape == (2, 3, *shape)
    for index in range(2):
        np.testing.assert_array_equal(
            batched[index], backproject_face_masks(masks[index], spec, shape)
        )


def _proposal(
    identifier: str,
    mask: np.ndarray,
    *,
    score: float,
    concept: str | None = None,
) -> SphericalMaskProposal:
    packed = SphericalBinaryMask.from_array(mask)
    seed = SphericalSeed(identifier, 0, 0, "coverage", 1.0, concept)
    return SphericalMaskProposal(
        identifier,
        seed,
        (packed, packed, packed),
        packed,
        packed,
        (score, score, score),
        1.0,
        1.0,
        0.8 if concept else 0.0,
        concept,
        (),
    )


def test_merge_and_panoptic_resolution_are_deterministic() -> None:
    first = np.zeros((20, 40), dtype=bool)
    second = np.zeros_like(first)
    first[4:12, 3:12] = True
    second[5:13, 4:13] = True
    config = SphericalSegmentationConfig(
        discovery_face_size=64,
        minimum_pixels=4,
        expansion_frontier_band_pixels=4,
        expansion_frontier_minimum_pixels=2,
    )

    segments = merge_proposals(
        (_proposal("one", first, score=0.9), _proposal("two", second, score=0.8)),
        config,
    )
    panoptic = resolve_panoptic_map(segments, np.ones_like(first))

    assert len(segments) == 1
    assert set(np.unique(panoptic)) == {0, 1}
    assert np.all(panoptic[first | second] == 1)


def test_fused_segment_proxies_follow_the_final_evidence_concept() -> None:
    mask = np.zeros((20, 40), dtype=bool)
    mask[6:14, 10:20] = True
    packed = SphericalBinaryMask.from_array(mask)
    seed = SphericalSeed("access", 0, 0, "semantic", 0.8, "access-structure")
    proposal = SphericalMaskProposal(
        "proposal-access",
        seed,
        (packed, packed, packed),
        packed,
        packed,
        (0.95, 0.94, 0.93),
        1.0,
        1.0,
        0.8,
        "access-structure",
        ("bannister",),
    )
    scores = np.zeros((1, 20, 40), dtype=np.float32)
    scores[0, mask] = 0.9
    evidence = DenseSemanticEvidence(
        ("pipe-bank",), scores, np.ones((20, 40), dtype=bool)
    )
    config = SphericalSegmentationConfig(
        discovery_face_size=64,
        minimum_pixels=4,
        expansion_frontier_band_pixels=4,
        expansion_frontier_minimum_pixels=2,
    )

    segment = merge_proposals((proposal,), config, evidence=evidence)[0]

    assert segment.concept_id == "pipe-bank"
    assert segment.proxy_classes == ("organ",)


def test_unknown_and_semantic_duplicates_remain_competing_proposals() -> None:
    mask = np.zeros((20, 40), dtype=bool)
    mask[6:14, 10:20] = True
    unknown = _proposal("unknown", mask, score=0.95)
    semantic = _proposal("semantic", mask, score=0.94, concept="pipe-bank")

    retained = non_maximum_suppression((unknown, semantic), threshold=0.8)

    assert {item.proposal_id for item in retained} == {"unknown", "semantic"}


def test_expansion_proposals_are_partitioned_by_root_origin() -> None:
    mask = np.zeros((20, 40), dtype=bool)
    mask[6:14, 10:20] = True
    root = _proposal("root", mask, score=0.95, concept="pipe-bank")
    packed = SphericalBinaryMask.from_array(mask)
    child = SphericalMaskProposal(
        "child",
        SphericalSeed("frontier", 10, 0, "frontier", 0.8, "pipe-bank"),
        (packed, packed, packed),
        packed,
        packed,
        (0.95, 0.94, 0.93),
        1.0,
        1.0,
        0.8,
        "pipe-bank",
        ("organ",),
        "root",
    )

    groups = partition_proposals_by_origin((root, child))

    assert [item.proposal_id for item in groups["coverage"]] == ["root", "child"]
    assert groups["cam"] == ()


class _DeterministicMaskBackend:
    def __init__(self) -> None:
        self.release_count = 0

    def encode(self, rgb: np.ndarray) -> tuple[int, int]:
        return rgb.shape[:2]

    def predict(self, embeddings, shape_hw, prompt) -> SamMultimaskOutput:
        height, width = shape_hw
        x, y = prompt.points_xy[0]
        yy, xx = np.ogrid[:height, :width]
        base = (xx - x) ** 2 + (yy - y) ** 2 <= (min(shape_hw) * 0.08) ** 2
        masks = np.stack((base, base, base))
        logits = np.where(masks, 10.0, -10.0).astype(np.float32)
        return SamMultimaskOutput(masks, logits, (0.95, 0.94, 0.93), 5.0, (1, 1, 1))

    def release(self, embeddings=None) -> None:
        if embeddings is None:
            self.release_count += 1


def test_pipeline_combines_semantic_and_coverage_seeds() -> None:
    rgb = np.zeros((64, 128, 3), dtype=np.uint8)
    support = np.ones((64, 128), dtype=bool)
    scores = np.zeros((1, 8, 16), dtype=np.float32)
    scores[0, 4, 8] = 1
    evidence = DenseSemanticEvidence(("round-fitting",), scores, np.ones((8, 16), bool))
    config = SphericalSegmentationConfig(
        semantic_peaks_per_concept=1,
        coverage_direction_count=4,
        discovery_face_size=64,
        prompt_grid_size=2,
        prompt_batch_size=2,
        minimum_pixels=4,
        expansion_frontier_band_pixels=4,
        expansion_frontier_minimum_pixels=2,
        maximum_expansion_faces=1,
    )

    result = SphericalSemanticSegmenter(_DeterministicMaskBackend(), config).segment(
        rgb, support, evidence=evidence
    )

    assert result.interface == "panorai-spherical-semantic-segmentation/v1"
    assert result.diagnostics["semantic_seed_count"] == 1
    assert result.diagnostics["coverage_seed_count"] == 4
    assert result.segments
    assert result.panoptic_map.shape == support.shape
    assert np.all(result.panoptic_map[~support] == 0)


def test_pretrained_facade_runs_models_sequentially_and_reports_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    scores = np.zeros((1, 8, 16), dtype=np.float32)
    scores[0, 4, 8] = 1
    evidence = DenseSemanticEvidence(("round-fitting",), scores, np.ones((8, 16), bool))
    semantic_provenance = {"model": "test spherical classifier"}
    backend = _DeterministicMaskBackend()

    def infer(self, rgb, support, *, progress):
        assert self._mask_backend is None
        events.append("semantic-released")
        return evidence, semantic_provenance

    def load_mask(self, *, progress):
        assert events == ["semantic-released"]
        events.append("sam-loaded")
        self._mask_backend = backend
        return backend

    monkeypatch.setattr(SphericalIndustrialSegmenter, "_infer_semantic_evidence", infer)
    monkeypatch.setattr(SphericalIndustrialSegmenter, "_load_mask_backend", load_mask)
    config = SphericalSegmentationConfig(
        semantic_peaks_per_concept=1,
        coverage_direction_count=4,
        discovery_face_size=32,
        prompt_grid_size=2,
        prompt_batch_size=2,
        minimum_pixels=2,
        expansion_frontier_band_pixels=2,
        expansion_frontier_minimum_pixels=2,
        maximum_expansion_faces=1,
    )
    facade = SphericalIndustrialSegmenter.from_pretrained(
        accept_upstream_terms=True,
        device="cpu",
        config=config,
        semantic_input_height=32,
        working_height=32,
    )
    panorama = np.zeros((64, 128, 3), dtype=np.uint8)

    result = facade.predict(panorama)

    assert events == ["semantic-released", "sam-loaded"]
    assert result.interface == "panorai-spherical-semantic-segmentation/v1"
    assert result.panoptic_map.shape == (32, 64)
    assert result.diagnostics["facade"] == {
        "interface": SPHERICAL_INDUSTRIAL_SEGMENTATION_INTERFACE,
        "semantic_model": "internimage-g",
        "mask_model": "sam2.1-hiera-large",
        "source_shape_hw": [64, 128],
        "output_shape_hw": [32, 64],
        "source_resolution_output": False,
        "support_source": "implicit-full-sphere",
        "semantic": semantic_provenance,
        "mask_assets": None,
        "model_lifecycle": (
            "InternImage and SAM loaded sequentially and released after each stage"
        ),
    }
    assert facade._mask_backend is None
    assert backend.release_count == 1
    facade.close()
    with pytest.raises(RuntimeError, match="closed"):
        facade.predict(panorama)


def test_pretrained_facade_rejects_unsupported_models_projection_and_cuda() -> None:
    with pytest.raises(ValueError, match="semantic_model"):
        SphericalIndustrialSegmenter.from_pretrained(
            semantic_model="resnet18", accept_upstream_terms=True
        )
    with pytest.raises(ValueError, match="CUDA"):
        SphericalIndustrialSegmenter.from_pretrained(
            device="cuda", accept_upstream_terms=True
        )

    facade = SphericalIndustrialSegmenter.from_pretrained(accept_upstream_terms=False)
    panorama = np.zeros((8, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="equirectangular"):
        facade.predict(panorama, projection="cubemap")


def test_prompt_grid_has_declared_count_and_margin() -> None:
    prompts = prompt_grid((100, 200), 8)

    assert len(prompts) == 64
    assert prompts[0].points_xy[0][0] == pytest.approx(0.05 * 199)
    assert prompts[-1].points_xy[0][1] == pytest.approx(0.95 * 99)
