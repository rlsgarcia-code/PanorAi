#!/usr/bin/env python3
"""Installed-wheel interface smoke for the spherical Mask R-CNN workflow."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np


def _assert_installed_origin(package_file: str, source_root: Path) -> Path:
    origin = Path(package_file).resolve()
    try:
        origin.relative_to(source_root.resolve())
    except ValueError:
        return origin
    raise AssertionError(f"panorai resolved inside the source checkout: {origin}")


class DeterministicSAM:
    def encode(self, rgb: np.ndarray) -> tuple[int, int]:
        return rgb.shape[:2]

    def predict(self, embeddings, shape_hw, prompt):
        from panorai.experimental.deep_learning.segmentation import SamMultimaskOutput

        height, width = shape_hw
        x, y = prompt.points_xy[0]
        yy, xx = np.ogrid[:height, :width]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= (min(shape_hw) * 0.12) ** 2
        masks = np.stack((mask, mask, mask))
        logits = np.where(masks, 10.0, -10.0).astype(np.float32)
        return SamMultimaskOutput(
            masks, logits, (0.95, 0.94, 0.93), 5.0, (1.0, 1.0, 1.0)
        )

    def release(self, embeddings=None) -> None:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()

    import panorai
    from panorai.applications import (
        InstanceHeuristicDecision,
        SphericalMaskRCNNWorkflow,
    )
    from panorai.experimental.deep_learning import mask_rcnn
    from panorai.experimental.deep_learning.segmentation import (
        SphericalSegmentationConfig,
        SphericalSegmentationResult,
    )

    origin = _assert_installed_origin(panorai.__file__, args.source_root)
    rows, columns = np.indices((24, 48))
    image = np.stack(
        (columns * 5 % 256, rows * 11 % 256, (rows + columns) * 3 % 256),
        axis=-1,
    ).astype(np.uint8)
    support = np.ones((24, 48), dtype=bool)
    support[:, :3] = False
    config = SphericalSegmentationConfig(
        semantic_peaks_per_concept=1,
        coverage_direction_count=2,
        discovery_face_size=24,
        prompt_grid_size=1,
        prompt_batch_size=2,
        minimum_pixels=2,
        expansion_frontier_band_pixels=3,
        expansion_frontier_minimum_pixels=2,
        maximum_expansion_faces=1,
    )

    def accept(segment) -> InstanceHeuristicDecision:
        return InstanceHeuristicDecision(
            1, "object", max(0.5, float(segment.quality_score))
        )

    method = mask_rcnn.workflow(
        proposal_mode="sam",
        proposal=DeterministicSAM(),
        heuristic=accept,
        reuse_proposal_masks=True,
        num_classes=2,
        config=config,
        output_shape_hw=(24, 48),
    )
    result = method(image, support_mask=support)

    assert isinstance(method, SphericalMaskRCNNWorkflow)
    assert isinstance(result, SphericalSegmentationResult)
    assert method.architecture.proposal_generator.mode == "sam"
    assert (
        method.architecture.roi_heads.box_head.responsibility
        == "instance-classification-heuristic"
    )
    assert result.support.shape == support.shape
    assert not np.any(result.support[~support])
    assert np.all(result.panoptic_map[~support] == 0)
    assert result.diagnostics["box_head"]["kind"] == "instance-heuristic"

    print(
        json.dumps(
            {
                "status": "ok",
                "consumer": "spherical-mask-rcnn-workflow-interface-smoke",
                "evidence_scope": "deterministic-adapter-interface-only",
                "version": version("panorai"),
                "origin": str(origin),
                "proposal_mode": method.architecture.proposal_generator.mode,
                "segment_count": len(result.segments),
                "supported_pixels": int(result.support.sum()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
