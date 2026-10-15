from __future__ import annotations

import json

import numpy as np
import pytest

from panorai import EquirectangularImage
from panorai.applications import (
    Application,
    ApplicationConfigurationError,
    InstanceHeuristicDecision,
    SphericalDoGFeatures,
    SphericalMaskRCNNPaper,
    SphericalSLIC,
    SphericalStereoDepth,
    SphericalStereoInput,
)
from panorai.experimental.deep_learning import mask_rcnn
from panorai.experimental.deep_learning.mask_rcnn import (
    SphericalMaskRCNN,
    UnboundMaskRCNNError,
)
from panorai.experimental.deep_learning.segmentation import (
    SamMultimaskOutput,
    SphericalSegmentationConfig,
)
from panorai.features import SphericalDoGSIFTConfig, SphericalFeatureSet
from panorai.pipeline import CAMRegionProposals, RasterField
from panorai.stereo import SphericalStereoOptions, SphericalStereoResult


def _rgb() -> np.ndarray:
    rows, columns = np.indices((24, 48))
    return np.stack(
        (columns * 5 % 256, rows * 11 % 256, (rows + columns) * 3 % 256), axis=-1
    ).astype(np.uint8)


class _DeterministicSAM:
    def encode(self, rgb: np.ndarray) -> tuple[int, int]:
        return rgb.shape[:2]

    def predict(self, embeddings, shape_hw, prompt) -> SamMultimaskOutput:
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


def _segmentation_config() -> SphericalSegmentationConfig:
    return SphericalSegmentationConfig(
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


def _accept(segment) -> InstanceHeuristicDecision:
    return InstanceHeuristicDecision(1, "object", max(0.5, segment.quality_score))


def test_slic_application_matches_its_exposed_pipeline() -> None:
    image = _rgb().astype(np.float32) / 255.0
    application = SphericalSLIC(n_segments=8, max_num_iter=2)

    direct = application(image)
    low_level = application.as_pipeline()(EquirectangularImage(image))

    assert isinstance(direct, RasterField)
    np.testing.assert_array_equal(direct.values, low_level.values)
    assert direct.grid.is_canonical_erp
    assert type(application.with_params(n_segments=9)) is SphericalSLIC


def test_registered_application_config_round_trip_is_safe_and_atomic(tmp_path) -> None:
    application = SphericalSLIC(n_segments=7, compactness=4.0)
    destination = application.save(tmp_path / "nested" / "slic.json")

    loaded = Application.load(destination)

    assert loaded == application
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["application"] == application.application_id


def test_dog_application_reuses_existing_feature_extractor_and_round_trips() -> None:
    config = SphericalDoGSIFTConfig(
        octaves=1,
        levels_per_octave=2,
        max_features=5,
        patch_size=24,
        convolution_backend="numpy",
    )
    application = SphericalDoGFeatures(config)

    result = application(_rgb(), panorama_id="application-test")
    restored = SphericalDoGFeatures.from_config(application.to_config())

    assert isinstance(result, SphericalFeatureSet)
    assert result.panorama_id == "application-test"
    assert restored == application
    assert type(application.with_params(config__max_features=3)) is SphericalDoGFeatures


def test_stereo_wrapper_and_pipeline_share_typed_input() -> None:
    reference = _rgb().astype(np.float32) / 255.0
    source = np.roll(reference, 1, axis=1)
    rotation = np.eye(3)
    translation = np.asarray((0.05, 0.0, 0.0))
    application = SphericalStereoDepth(
        SphericalStereoOptions(
            min_range=0.5,
            max_range=2.0,
            num_hypotheses=3,
            window_size=3,
            bidirectional_consistency=False,
            filter_backend="numpy",
        )
    )

    direct = application(reference, source, rotation, translation)
    composed = application.as_pipeline()(
        SphericalStereoInput(reference, source, rotation, translation)
    )

    assert isinstance(direct, SphericalStereoResult)
    np.testing.assert_array_equal(direct.range, composed.range)
    np.testing.assert_array_equal(direct.validity_mask, composed.validity_mask)
    assert (
        type(application.with_params(options__num_hypotheses=5)) is SphericalStereoDepth
    )


def test_sam_paper_factory_exposes_mask_rcnn_roles_and_runs() -> None:
    method = mask_rcnn.paper(
        proposal_mode="sam",
        proposal=_DeterministicSAM(),
        heuristic=_accept,
        reuse_proposal_masks=True,
        num_classes=2,
        config=_segmentation_config(),
        output_shape_hw=(24, 48),
    )

    result = method(_rgb())

    assert isinstance(method, SphericalMaskRCNNPaper)
    assert method.architecture.proposal_generator.mode == "sam"
    assert method.architecture.roi_heads.box_head.responsibility == (
        "instance-classification-heuristic"
    )
    assert result.support.shape == (24, 48)
    assert np.all(result.panoptic_map[~result.support] == 0)
    assert result.diagnostics["box_head"]["kind"] == "instance-heuristic"


def test_cam_paper_factory_uses_cam_for_regions_and_sam_for_masks() -> None:
    def cam(_image: np.ndarray) -> np.ndarray:
        scores = np.zeros((1, 6, 12), dtype=np.float32)
        scores[0, 3, 6] = 1.0
        return scores

    method = mask_rcnn.paper(
        proposal_mode="cam",
        proposal=cam,
        mask=_DeterministicSAM(),
        heuristic=_accept,
        concepts=("object",),
        num_classes=2,
        config=_segmentation_config(),
        cam_proposals=CAMRegionProposals(
            maximum_per_concept=1,
            face_size=24,
            fov_degrees=70.0,
            minimum_score=0.1,
        ),
        output_shape_hw=(24, 48),
        expand=False,
    )

    run = method.as_pipeline().run(_as_erp(_rgb()), retain=("proposal_generator",))

    assert len(run["proposal_generator"].proposals) == 1
    assert run.output.diagnostics["box_head"]["kind"] == "instance-heuristic"
    assert method.architecture.roi_heads.mask_head.component is method.mask


def _as_erp(image: np.ndarray):
    return EquirectangularImage(image)


def test_paper_method_rejects_unregistered_model_persistence_with_exact_path() -> None:
    method = mask_rcnn.paper(
        proposal_mode="sam",
        proposal=_DeterministicSAM(),
        heuristic=_accept,
        reuse_proposal_masks=True,
        num_classes=2,
        config=_segmentation_config(),
    )

    with pytest.raises(ApplicationConfigurationError, match=r"parameters\.proposal"):
        method.to_config()


def test_trainable_mask_rcnn_remains_explicitly_unbound() -> None:
    architecture = SphericalMaskRCNN(num_classes=3)

    assert architecture.architecture.proposal_generator.mode == "rpn"
    with pytest.raises(UnboundMaskRCNNError, match="mask_rcnn.paper"):
        architecture(_rgb())


def test_legacy_cam_convenience_with_params_preserves_wrapper_type() -> None:
    from panorai.pipeline import CAMGuidedSphericalInstances

    wrapper = CAMGuidedSphericalInstances(
        cam=lambda image: np.zeros((1, *image.shape[:2]), dtype=np.float32),
        sam=_DeterministicSAM(),
        concept_ids=("object",),
    )

    variant = wrapper.with_params(proposals__minimum_score=0.5)

    assert isinstance(variant, CAMGuidedSphericalInstances)
    assert variant.named_steps["proposals"].minimum_score == 0.5


def test_sam_proposal_mode_requires_explicit_mask_reuse() -> None:
    with pytest.raises(ValueError, match="reuse_proposal_masks=True"):
        mask_rcnn.paper(
            proposal_mode="sam",
            proposal=_DeterministicSAM(),
            heuristic=_accept,
            reuse_proposal_masks=False,
            num_classes=2,
        )
