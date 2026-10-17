#!/usr/bin/env python3
"""Run the paper Mask R-CNN wrapper with real CAM and SAM checkpoints.

This runner is intentionally development evidence.  Its default oracle comes
from the previously documented Matterport3D example-scene experiment and is
therefore not relabelled as prospective held-out evidence.  It closes the
real-checkpoint execution question while leaving the frozen held-out promotion
gate to the protocol evaluator.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panorai import EquirectangularImage  # noqa: E402
from panorai.applications import InstanceHeuristicDecision  # noqa: E402
from panorai.data import SphericalBinaryMask  # noqa: E402
from panorai.experimental.deep_learning import mask_rcnn  # noqa: E402
from panorai.experimental.deep_learning.segmentation import (  # noqa: E402
    SphericalSegmentationConfig,
    backproject_face_masks,
    load_sam21_hiera_large,
)
from panorai.pipeline import CAMRegionProposals  # noqa: E402


SCHEMA = "panorai-mask-rcnn-paper-real-checkpoints/v1"
DEFAULT_PROTOCOL = ROOT / "contracts/spherical-application-promotion-v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def parse_house_taxonomy(
    path: Path,
) -> tuple[dict[int, tuple[str, ...]], dict[int, int]]:
    """Return raw/MPcat category aliases and object assignments."""

    categories: dict[int, tuple[str, ...]] = {}
    objects: dict[int, int] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "C":
                aliases = tuple(
                    dict.fromkeys(
                        (
                            fields[3].replace("#", " "),
                            fields[5].replace("#", " "),
                        )
                    )
                )
                categories[int(fields[1])] = aliases
            elif fields[0] == "O":
                objects[int(fields[1])] = int(fields[3])
    if not categories or not objects:
        raise ValueError(f"{path}: missing Matterport category/object records")
    return categories, objects


def _nearest(values: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    return cv2.resize(
        values,
        shape_hw[::-1],
        interpolation=cv2.INTER_NEAREST,
    )


def oracle_instances(
    object_ids: np.ndarray,
    support: np.ndarray,
    categories: dict[int, tuple[str, ...]],
    objects: dict[int, int],
    target_ids: tuple[str, ...],
    *,
    minimum_pixels: int = 16,
) -> dict[str, list[np.ndarray]]:
    """Build per-object masks from independently ray-cast semantic IDs."""

    result = {target: [] for target in target_ids}
    for object_id in np.unique(object_ids[support]):
        numeric_id = int(object_id)
        category_id = objects.get(numeric_id)
        if category_id is None:
            continue
        mask = support & (object_ids == numeric_id)
        if int(mask.sum()) < minimum_pixels:
            continue
        for alias in categories.get(category_id, ()):
            if alias in result:
                result[alias].append(mask)
    return result


def solid_angle_iou(left: np.ndarray, right: np.ndarray) -> float:
    return SphericalBinaryMask.from_array(left).solid_angle_iou(
        SphericalBinaryMask.from_array(right)
    )


class ImageNetCAMAdapter:
    """Direct-spherical ResNet18 CAM callable for the public pipeline."""

    def __init__(self, class_indices: tuple[int, ...], *, threads: int) -> None:
        import torch
        from panorai.experimental.deep_learning import (
            ImageNetFCN,
            class_activation_map,
            load_pretrained_imagenet_model,
            port_module_with_report,
        )

        torch.set_num_threads(threads)
        loaded = load_pretrained_imagenet_model("resnet18", progress=False)
        self.weights = loaded.weights
        self.checkpoint = loaded.checkpoint
        self.model = ImageNetFCN(loaded.model, "resnet18").eval()
        self.class_activation_map = class_activation_map
        self.port_report = port_module_with_report(self.model)
        if self.port_report.remaining_planar_spatial_layers:
            raise RuntimeError(
                "planar spatial layers remain: "
                + ", ".join(self.port_report.remaining_planar_spatial_layers)
            )
        self.class_indices = class_indices

    def __call__(self, image: np.ndarray) -> np.ndarray:
        import torch

        values = torch.from_numpy(np.asarray(image)).permute(2, 0, 1)[None].float()
        values /= 255.0
        transform = self.weights.transforms()
        mean = torch.asarray(transform.mean)[None, :, None, None]
        std = torch.asarray(transform.std)[None, :, None, None]
        normalized = (values - mean) / std
        with torch.inference_mode():
            logits = self.model.forward_dense(normalized).logits
            maps = [
                self.class_activation_map(
                    logits, class_index, output_shape=image.shape[:2]
                )[0]
                for class_index in self.class_indices
            ]
        return torch.stack(maps).cpu().numpy().astype(np.float32, copy=False)


def _development_view(report: dict[str, Any], panorama_index: int) -> dict[str, Any]:
    matches = [
        item
        for item in report.get("views", [])
        if item.get("panorama_index") == panorama_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"CAM report must contain exactly one panorama_index={panorama_index}"
        )
    return matches[0]


def evaluate_masks(
    proposals: list[tuple[str, np.ndarray]],
    segments: list[tuple[str, np.ndarray]],
    ground_truth: dict[str, list[np.ndarray]],
) -> dict[str, float]:
    ground_truth_items = [
        (concept, mask) for concept, masks in ground_truth.items() for mask in masks
    ]
    proposal_best = []
    segment_best = []
    for concept, target in ground_truth_items:
        proposal_best.append(
            max(
                (
                    solid_angle_iou(candidate, target)
                    for candidate_concept, candidate in proposals
                    if candidate_concept == concept
                ),
                default=0.0,
            )
        )
        segment_best.append(
            max(
                (
                    solid_angle_iou(candidate, target)
                    for candidate_concept, candidate in segments
                    if candidate_concept == concept
                ),
                default=0.0,
            )
        )
    correct_predictions = sum(
        max(
            (
                solid_angle_iou(candidate, target)
                for target in ground_truth.get(concept, [])
            ),
            default=0.0,
        )
        >= 0.5
        for concept, candidate in segments
    )
    recovered = sum(value >= 0.5 for value in segment_best)
    return {
        "proposal_recall_spherical_iou_0_5": (
            sum(value >= 0.5 for value in proposal_best) / len(proposal_best)
            if proposal_best
            else 0.0
        ),
        "median_fused_mask_spherical_iou": (
            float(np.median(segment_best)) if segment_best else 0.0
        ),
        "non_abstained_instance_precision": (
            correct_predictions / len(segments) if segments else 0.0
        ),
        "non_abstained_instance_coverage": (
            recovered / len(segment_best) if segment_best else 0.0
        ),
    }


def support_is_subset(candidate: np.ndarray, source: np.ndarray) -> bool:
    """Return whether candidate support is contained in source support."""

    candidate_mask = np.asarray(candidate, dtype=bool)
    source_mask = np.asarray(source, dtype=bool)
    if candidate_mask.shape != source_mask.shape:
        raise ValueError("candidate and source support must have matching shapes")
    return bool(np.all(~candidate_mask | source_mask))


def run(
    *,
    cam_report_path: Path,
    house_path: Path,
    panorama_index: int,
    output_shape_hw: tuple[int, int],
    threads: int,
    sam_cache_dir: Path | None,
    device: str,
) -> dict[str, Any]:
    report = _load_object(cam_report_path)
    view = _development_view(report, panorama_index)
    literal = [
        item for item in view["evaluations"] if item.get("alignment") == "literal"
    ]
    if not literal:
        raise ValueError("selected development view has no literal target")
    concepts = tuple(str(item["target_id"]) for item in literal)
    class_indices = tuple(int(item["imagenet_class_index"]) for item in literal)
    concept_labels = {
        str(item["target_id"]): tuple(str(label) for label in item["matterport_labels"])
        for item in literal
    }
    rgb_path = Path(view["rgb"])
    oracle_path = rgb_path.with_name("semantic-oracle.npz")
    encoded = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if encoded is None:
        raise ValueError(f"cannot decode {rgb_path}")
    rgb = cv2.cvtColor(encoded, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, output_shape_hw[::-1], interpolation=cv2.INTER_AREA)
    with np.load(oracle_path, allow_pickle=False) as oracle:
        object_ids = _nearest(oracle["object_ids"], output_shape_hw).astype(np.int32)
        support = _nearest(oracle["support"].astype(np.uint8), output_shape_hw).astype(
            bool
        )
    categories, objects = parse_house_taxonomy(house_path)
    raw_labels = tuple(
        sorted({label for labels in concept_labels.values() for label in labels})
    )
    raw_ground_truth = oracle_instances(
        object_ids, support, categories, objects, raw_labels, minimum_pixels=8
    )
    ground_truth = {
        concept: [
            mask
            for label in concept_labels[concept]
            for mask in raw_ground_truth[label]
        ]
        for concept in concepts
    }
    if not any(ground_truth.values()):
        raise ValueError(
            "selected view has no target instances at evaluation resolution"
        )

    cam = ImageNetCAMAdapter(class_indices, threads=threads)
    sam = load_sam21_hiera_large(
        accept_upstream_terms=True,
        cache_dir=sam_cache_dir,
        device=device,
    )
    concept_to_class = {concept: index + 1 for index, concept in enumerate(concepts)}

    def heuristic(segment: Any) -> InstanceHeuristicDecision:
        concept = str(segment.concept_id)
        class_id = concept_to_class.get(concept)
        if class_id is None:
            return InstanceHeuristicDecision(None)
        return InstanceHeuristicDecision(
            class_id, concept, max(0.0, min(1.0, float(segment.quality_score)))
        )

    config = SphericalSegmentationConfig(
        semantic_peaks_per_concept=1,
        coverage_direction_count=8,
        discovery_face_size=256,
        prompt_grid_size=2,
        prompt_batch_size=2,
        minimum_pairwise_iou=0.50,
        minimum_predicted_iou=0.70,
        minimum_stability=0.85,
        minimum_pixels=16,
        maximum_expansion_faces=1,
    )
    method = mask_rcnn.workflow(
        proposal_mode="cam",
        proposal=cam,
        mask=sam,
        heuristic=heuristic,
        concepts=concepts,
        num_classes=len(concepts) + 1,
        config=config,
        cam_proposals=CAMRegionProposals(
            maximum_per_concept=1,
            minimum_separation_degrees=12.0,
            relative_threshold=0.30,
            minimum_score=0.25,
            fov_degrees=70.0,
            face_size=256,
        ),
        output_shape_hw=output_shape_hw,
        expand=False,
    )
    started = time.perf_counter()
    pipeline_run = method.as_pipeline().run(
        EquirectangularImage(rgb, support_mask=support, valid=support),
        retain=("proposal_generator", "mask_head", "fusion"),
    )
    elapsed = time.perf_counter() - started
    region_set = pipeline_run["proposal_generator"]
    mask_batch = pipeline_run["mask_head"]
    fused = pipeline_run["fusion"]
    final = pipeline_run.output
    cam_region_masks = [
        (
            str(item.concept_id),
            backproject_face_masks(
                np.ones((3, *item.spec.output_shape_hw), dtype=bool),
                item.spec,
                output_shape_hw,
                support=support,
            )[0],
        )
        for item in region_set.proposals
    ]
    sam_proposals = [
        (str(item.concept_id), item.consensus.to_array())
        for item in mask_batch.masks.proposals
        if item.concept_id is not None
    ]
    segments = [
        (str(item.concept_id), item.consensus.to_array())
        for item in fused.segments
        if item.concept_id is not None
    ]
    metrics = evaluate_masks(cam_region_masks, segments, ground_truth)
    metrics.update(
        {
            "real_cam_checkpoint": True,
            "real_sam_checkpoint": True,
            "no_valid_pixels_outside_support": support_is_subset(
                final.support, support
            ),
            "thresholds_frozen_on_development_only": True,
            "weights_datasets_outputs_external": True,
        }
    )
    import huggingface_hub
    import torch
    import torchvision
    import transformers

    return {
        "schema": SCHEMA,
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "evaluation": "external",
        "split": "development",
        "promotion_eligibility": (
            "real-checkpoint execution only; this previously inspected view is "
            "not eligible for the prospective held-out gate"
        ),
        "application": "mask-rcnn-paper",
        "corpus": "Matterport3D official example scene",
        "panorama_index": panorama_index,
        "concepts": list(concepts),
        "metrics": metrics,
        "details": {
            "ground_truth_instance_counts": {
                key: len(value) for key, value in ground_truth.items()
            },
            "concept_to_matterport_raw_labels": {
                key: list(value) for key, value in concept_labels.items()
            },
            "cam_region_count": len(region_set.proposals),
            "sam_proposal_count": len(sam_proposals),
            "fused_segment_count": len(segments),
            "final_segment_count": len(final.segments),
            "panoptic_values_zero_outside_final_support": bool(
                np.all(final.panoptic_map[~final.support] == 0)
            ),
            "elapsed_seconds": elapsed,
            "output_shape_hw": list(output_shape_hw),
            "config": asdict(config),
        },
        "assets": {
            "rgb": {"path": str(rgb_path), "sha256": sha256_file(rgb_path)},
            "semantic_oracle": {
                "path": str(oracle_path),
                "sha256": sha256_file(oracle_path),
            },
            "house": {"path": str(house_path), "sha256": sha256_file(house_path)},
            "cam_checkpoint": cam.checkpoint.to_dict(),
            "sam_checkpoint": sam.assets.to_dict(),
            "distribution_boundary": "all datasets, weights, and outputs external",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "transformers": transformers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "device": str(sam.device),
            "threads": threads,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam-report", type=Path, required=True)
    parser.add_argument("--house", type=Path, required=True)
    parser.add_argument("--panorama-index", type=int, default=18)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--sam-cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--accept-sam-terms", action="store_true")
    args = parser.parse_args()
    if not args.accept_sam_terms:
        parser.error("--accept-sam-terms is required for the external SAM checkpoint")
    if args.height < 64:
        parser.error("--height must be at least 64")
    result = run(
        cam_report_path=args.cam_report.resolve(),
        house_path=args.house.resolve(),
        panorama_index=args.panorama_index,
        output_shape_hw=(args.height, 2 * args.height),
        threads=args.threads,
        sam_cache_dir=(
            None if args.sam_cache_dir is None else args.sam_cache_dir.resolve()
        ),
        device=args.device,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = args.output.resolve()
    if output.exists() and output.read_text(encoding="utf-8") != payload:
        raise FileExistsError(f"refusing to replace different evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
