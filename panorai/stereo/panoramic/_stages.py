"""Inspectable stages for panoramic dense stereo."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Iterator

import numpy as np

from panorai.data import EquirectangularImage
from panorai.estimators import SphericalRelativePoseEstimator
from panorai.features import SphericalFeaturePipeline
from panorai.geometry import (
    GnomonicProjector,
    gnomonic_intrinsics,
    rays_to_gnomonic_pixels,
)
from panorai.pipeline import StageResult
from panorai.stereo.classical import RectifiedStereoGeometry

from .._dense import SphericalDenseStereo, SphericalStereoOptions

from ._fusion import compose_fusion_confidence
from ._planning import RectifiedStereoViewPlanner
from ._types import (
    PanoramicStereoPose,
    PanoramicStereoResult,
    StereoPairDiagnostic,
    StereoViewSelectionDiagnostic,
    StereoViewPlan,
)


@dataclass(frozen=True, slots=True)
class PanoramicStereoInput:
    """Two canonical ERPs and optional relative pose/scale information."""

    reference: Any
    source: Any
    rotation_source_from_reference: np.ndarray | None = None
    translation_source_from_reference: np.ndarray | None = None
    baseline: float | None = None
    units: str | None = None
    reference_validity: np.ndarray | None = None
    source_validity: np.ndarray | None = None
    scale_status: str | None = None
    pose_provenance: str | None = None


@dataclass(frozen=True, slots=True)
class _StereoContext:
    request: PanoramicStereoInput
    reference: np.ndarray | None = None
    source: np.ndarray | None = None
    reference_validity: np.ndarray | None = None
    source_validity: np.ndarray | None = None
    pose: PanoramicStereoPose | None = None
    matches: Any | None = None
    pose_result: Any | None = None
    plan: StereoViewPlan | None = None
    stream: Any | None = None
    accumulation: Any | None = None
    selection_audit: Any | None = None


def _panorama_payload(
    value: Any, explicit_validity: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(value, EquirectangularImage):
        image = np.asarray(value.image)
        validity = np.asarray(value.validity("image"), dtype=bool)
        support = np.asarray(value.support("image"), dtype=bool)
        validity &= support
    else:
        image = np.asarray(value)
        if image.ndim not in {2, 3}:
            raise ValueError("panorama must use HW or HWC layout")
        validity = np.ones(image.shape[:2], dtype=bool)
    if image.shape[1] != 2 * image.shape[0]:
        raise ValueError("canonical ERP input must have width == 2 * height")
    if explicit_validity is not None:
        provided = np.asarray(explicit_validity)
        if provided.dtype != np.bool_ or provided.shape != image.shape[:2]:
            raise ValueError("explicit validity must be a boolean HW mask")
        validity &= provided
    if not np.all(np.isfinite(image[validity])):
        raise ValueError("valid panorama samples must be finite")
    return image, validity


@dataclass(frozen=True, slots=True)
class ResolveStereoPose:
    """Use a supplied relative pose or estimate direction with spherical features."""

    pose_mode: str = "auto"
    require_quality_acceptance: bool = True

    input_kind = "panoramic_stereo_input"
    output_kind = "panoramic_stereo_context"
    stage_name = "resolve_pose"

    def __post_init__(self) -> None:
        if self.pose_mode not in {"auto", "provided", "estimate"}:
            raise ValueError("pose_mode must be auto, provided, or estimate")

    def execute(self, value: Any) -> StageResult:
        if not isinstance(value, PanoramicStereoInput):
            raise TypeError("ResolveStereoPose requires PanoramicStereoInput")
        reference, reference_validity = _panorama_payload(
            value.reference, value.reference_validity
        )
        source, source_validity = _panorama_payload(value.source, value.source_validity)
        if reference.shape[:2] != source.shape[:2]:
            raise ValueError("reference and source ERP shapes must match")
        has_rotation = value.rotation_source_from_reference is not None
        has_translation = value.translation_source_from_reference is not None
        if has_rotation != has_translation:
            raise ValueError("rotation and translation must be supplied together")
        use_provided = has_rotation and self.pose_mode != "estimate"
        if self.pose_mode == "provided" and not use_provided:
            raise ValueError("pose_mode='provided' requires R and t")
        matches = None
        pose_result = None
        if use_provided:
            rotation = np.asarray(
                value.rotation_source_from_reference, dtype=np.float64
            )
            translation = np.asarray(
                value.translation_source_from_reference, dtype=np.float64
            )
            provenance = value.pose_provenance or "provided"
        else:
            feature_pipeline = SphericalFeaturePipeline.for_relative_pose()
            matches = feature_pipeline.extract_and_match(
                reference,
                source,
                panorama_id_a="reference",
                panorama_id_b="source",
                validity_mask_a=reference_validity,
                validity_mask_b=source_validity,
            )
            pose_result = SphericalRelativePoseEstimator().estimate(
                matches.to_bearing_correspondences()
            )
            if pose_result is None:
                raise RuntimeError(
                    "relative pose estimation found no accepted hypothesis"
                )
            if (
                self.require_quality_acceptance
                and not pose_result.quality_report.accepted
            ):
                reasons = ", ".join(pose_result.quality_report.rejection_reasons)
                raise RuntimeError(f"relative pose quality gate rejected: {reasons}")
            rotation = pose_result.rotation
            translation = pose_result.translation_direction
            provenance = "estimated-spherical-features"
        # A temporary unit-scale pose is finalized by ResolveStereoScale.
        translation_norm = float(np.linalg.norm(translation))
        if not math.isfinite(translation_norm) or translation_norm <= 1e-12:
            raise ValueError("relative translation must be finite and non-zero")
        direction = translation / translation_norm
        preliminary = PanoramicStereoPose(
            rotation,
            direction,
            1.0,
            "baseline-units",
            "scale-free",
            provenance,
        )
        context = _StereoContext(
            value,
            reference,
            source,
            reference_validity,
            source_validity,
            preliminary,
            matches,
            pose_result,
        )
        return StageResult(
            context,
            {
                "pose_provenance": provenance,
                "match_count": 0 if matches is None else len(matches),
                "pose_inliers": 0 if pose_result is None else pose_result.num_inliers,
                "pose_quality_accepted": (
                    None if pose_result is None else pose_result.quality_report.accepted
                ),
                "pose_quality_rejection_reasons": (
                    ()
                    if pose_result is None
                    else pose_result.quality_report.rejection_reasons
                ),
            },
        )


@dataclass(frozen=True, slots=True)
class ResolveStereoScale:
    """Resolve metric baseline or preserve an explicit scale-free contract."""

    input_kind = "panoramic_stereo_context"
    output_kind = "panoramic_stereo_context"
    stage_name = "resolve_scale"

    def execute(self, value: Any) -> StageResult:
        if not isinstance(value, _StereoContext) or value.pose is None:
            raise TypeError("ResolveStereoScale requires a resolved pose context")
        request = value.request
        original = request.translation_source_from_reference
        if request.scale_status == "scale-free" and request.baseline is None:
            baseline = 1.0
            units = "baseline-units"
            status = "scale-free"
            provenance = f"{value.pose.provenance}+preserved-unit-baseline"
        elif request.baseline is not None:
            baseline = float(request.baseline)
            units = request.units or "m"
            status = "metric"
            provenance = f"{value.pose.provenance}+provided-baseline"
        elif original is not None:
            baseline = float(np.linalg.norm(np.asarray(original, dtype=np.float64)))
            units = request.units or "pose-units"
            status = "metric"
            provenance = f"{value.pose.provenance}+translation-norm"
        else:
            baseline = 1.0
            units = "baseline-units"
            status = "scale-free"
            provenance = f"{value.pose.provenance}+unit-baseline"
        if not math.isfinite(baseline) or baseline <= 0.0:
            raise ValueError("baseline must be positive and finite")
        direction = value.pose.translation_source_from_reference
        direction = direction / np.linalg.norm(direction)
        pose = PanoramicStereoPose(
            value.pose.rotation_source_from_reference,
            baseline * direction,
            baseline,
            units,
            status,
            provenance,
        )
        return StageResult(
            replace(value, pose=pose),
            {"baseline": baseline, "units": units, "scale_status": status},
        )


@dataclass(frozen=True, slots=True)
class PlanRectifiedStereoViews:
    planner: RectifiedStereoViewPlanner

    input_kind = "panoramic_stereo_context"
    output_kind = "panoramic_stereo_context"
    stage_name = "plan_rectified_views"

    def execute(self, value: Any) -> StageResult:
        if not isinstance(value, _StereoContext) or value.pose is None:
            raise TypeError("PlanRectifiedStereoViews requires a resolved pose")
        plan = self.planner.plan(value.pose)
        return StageResult(
            replace(value, plan=plan),
            {
                "pair_count": len(plan.pairs),
                "estimated_angular_coverage": plan.estimated_angular_coverage,
                "omitted_epipolar_cone_deg": plan.omitted_epipolar_cone_deg,
            },
        )


@dataclass(frozen=True, slots=True)
class _ProjectedPair:
    pair_spec: Any
    reference: np.ndarray
    source: np.ndarray
    reference_support: np.ndarray
    source_support: np.ndarray


@dataclass(frozen=True, slots=True)
class ProjectStereoViewPairs:
    """Create a lazy stream: only one RGB pair is materialized at a time."""

    input_kind = "panoramic_stereo_context"
    output_kind = "panoramic_stereo_context"
    stage_name = "project_view_pairs"

    def execute(self, value: Any) -> StageResult:
        if not isinstance(value, _StereoContext) or value.plan is None:
            raise TypeError("ProjectStereoViewPairs requires a view plan")

        # Validity-normalized interpolation is intentionally restricted to
        # floating-point samples by the canonical projector.  RGB panoramas
        # commonly arrive as uint8, so normalize once at this application
        # boundary rather than weakening the projectors' typed contract.
        reference_input = _projection_input(value.reference)
        source_input = _projection_input(value.source)

        def iterator() -> Iterator[_ProjectedPair]:
            for item in value.plan.pairs:
                ref_projector = GnomonicProjector(
                    item.reference_spec,
                    interpolation="bilinear",
                    invalid_policy="renormalize",
                    min_valid_weight=0.5,
                )
                src_projector = GnomonicProjector(
                    item.source_spec,
                    interpolation="bilinear",
                    invalid_policy="renormalize",
                    min_valid_weight=0.5,
                )
                ref = ref_projector.project(
                    reference_input, validity_mask=value.reference_validity
                )
                src = src_projector.project(
                    source_input, validity_mask=value.source_validity
                )
                ref_valid = np.array(ref.support_mask, dtype=bool, copy=True)
                src_valid = np.array(src.support_mask, dtype=bool, copy=True)
                if ref.validity_mask is not None:
                    ref_valid &= np.asarray(ref.validity_mask, dtype=bool)
                if src.validity_mask is not None:
                    src_valid &= np.asarray(src.validity_mask, dtype=bool)
                ref_data = np.asarray(ref.data)
                src_data = np.asarray(src.data)
                ref_finite = (
                    np.isfinite(ref_data)
                    if ref_data.ndim == 2
                    else np.all(np.isfinite(ref_data), axis=-1)
                )
                src_finite = (
                    np.isfinite(src_data)
                    if src_data.ndim == 2
                    else np.all(np.isfinite(src_data), axis=-1)
                )
                ref_valid &= ref_finite
                src_valid &= src_finite
                # Classical matchers require finite rasters and consume the
                # masks independently. Fill storage outside validity without
                # changing its semantic state.
                ref_data = np.where(
                    ref_valid if ref_data.ndim == 2 else ref_valid[..., None],
                    ref_data,
                    0.0,
                )
                src_data = np.where(
                    src_valid if src_data.ndim == 2 else src_valid[..., None],
                    src_data,
                    0.0,
                )
                yield _ProjectedPair(
                    item,
                    ref_data,
                    src_data,
                    ref_valid,
                    src_valid,
                )

        return StageResult(
            replace(value, stream=iterator()),
            {"streaming": True, "simultaneous_materialized_pairs": 1},
        )


def _projection_input(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.float32) / float(np.iinfo(array.dtype).max)
    return array


@dataclass(frozen=True, slots=True)
class SelectStereoViewPairs:
    min_joint_support: float = 0.10
    min_texture_std: float = 0.005
    min_pose_inliers: int = 0
    max_epipolar_row_error_px: float = 2.0
    matcher: Any | None = None

    input_kind = "panoramic_stereo_context"
    output_kind = "panoramic_stereo_context"
    stage_name = "select_view_pairs"

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_joint_support <= 1.0:
            raise ValueError("min_joint_support must be in [0, 1]")
        if self.min_texture_std < 0.0:
            raise ValueError("min_texture_std must be non-negative")
        if self.min_pose_inliers < 0:
            raise ValueError("min_pose_inliers must be non-negative")
        if self.max_epipolar_row_error_px < 0.0:
            raise ValueError("max_epipolar_row_error_px must be non-negative")

    def execute(self, value: Any) -> StageResult:
        if not isinstance(value, _StereoContext) or value.stream is None:
            raise TypeError("SelectStereoViewPairs requires projected pair stream")

        audit: list[StereoViewSelectionDiagnostic] = []

        def iterator() -> Iterator[_ProjectedPair]:
            for pair in value.stream:
                joint = pair.reference_support & pair.source_support
                joint_fraction = float(joint.mean())
                texture = _joint_texture_std(pair.reference, pair.source, joint)
                pose_inliers, row_error = _pair_pose_evidence(value, pair)
                disparity_viable = _disparity_interval_viable(
                    self.matcher, pair.reference.shape[1]
                )
                reasons = []
                if joint_fraction < self.min_joint_support:
                    reasons.append("insufficient-joint-support")
                if texture < self.min_texture_std:
                    reasons.append("insufficient-texture")
                if pose_inliers < self.min_pose_inliers:
                    reasons.append("insufficient-pose-inliers")
                if (
                    np.isfinite(row_error)
                    and row_error > self.max_epipolar_row_error_px
                ):
                    reasons.append("epipolar-row-residual")
                if not disparity_viable:
                    reasons.append("infeasible-disparity-interval")
                accepted = not reasons
                audit.append(
                    StereoViewSelectionDiagnostic(
                        pair.pair_spec.pair_id,
                        accepted,
                        joint_fraction,
                        texture,
                        pose_inliers,
                        row_error,
                        disparity_viable,
                        tuple(reasons),
                    )
                )
                if accepted:
                    yield pair

        return StageResult(
            replace(value, stream=iterator(), selection_audit=audit),
            {
                "min_joint_support": self.min_joint_support,
                "min_texture_std": self.min_texture_std,
                "min_pose_inliers": self.min_pose_inliers,
                "max_epipolar_row_error_px": self.max_epipolar_row_error_px,
                "selection": "lazy-before-cost-volume",
            },
        )


def _gray_unit(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim == 3:
        value = value[..., :3].astype(np.float32) @ np.asarray(
            (0.2126, 0.7152, 0.0722), dtype=np.float32
        )
    else:
        value = value.astype(np.float32)
    if np.issubdtype(np.asarray(image).dtype, np.integer):
        value = value / float(np.iinfo(np.asarray(image).dtype).max)
    return value


def _joint_texture_std(
    reference: np.ndarray, source: np.ndarray, joint: np.ndarray
) -> float:
    if not np.any(joint):
        return 0.0
    reference_gray = _gray_unit(reference)
    source_gray = _gray_unit(source)
    return float(min(np.std(reference_gray[joint]), np.std(source_gray[joint])))


def _pair_pose_evidence(
    context: _StereoContext, pair: _ProjectedPair
) -> tuple[int, float]:
    if context.matches is None or context.pose_result is None:
        return 0, float("nan")
    matches = context.matches
    selected = np.asarray(matches.valid, dtype=bool) & np.asarray(
        context.pose_result.inlier_mask, dtype=bool
    )
    if not np.any(selected):
        return 0, float("nan")
    projected_a = rays_to_gnomonic_pixels(
        np.asarray(matches.bearings_a[selected]), pair.pair_spec.reference_spec
    )
    projected_b = rays_to_gnomonic_pixels(
        np.asarray(matches.bearings_b[selected]), pair.pair_spec.source_spec
    )
    common = np.asarray(projected_a.valid) & np.asarray(projected_b.valid)
    count = int(common.sum())
    if count == 0:
        return 0, float("nan")
    rows_a = np.asarray(projected_a.pixels_xy)[common, 1]
    rows_b = np.asarray(projected_b.pixels_xy)[common, 1]
    return count, float(np.median(np.abs(rows_a - rows_b)))


def _disparity_interval_viable(matcher: Any | None, width: int) -> bool:
    config = getattr(matcher, "config", None)
    minimum = getattr(config, "min_disparity", None)
    maximum = getattr(config, "max_disparity", None)
    if minimum is None or maximum is None:
        return True
    return float(maximum) > 0.0 and float(minimum) < float(width - 1)


@dataclass(frozen=True, slots=True)
class _DensePair:
    pair_spec: Any
    result: Any


@dataclass(frozen=True, slots=True)
class DenseStereoPerPair:
    """Run one rectified matcher at a time and release its cost volume."""

    matcher: Any

    input_kind = "panoramic_stereo_context"
    output_kind = "panoramic_stereo_context"
    stage_name = "dense_stereo_per_pair"

    def execute(self, value: Any) -> StageResult:
        if (
            not isinstance(value, _StereoContext)
            or value.stream is None
            or value.pose is None
        ):
            raise TypeError("DenseStereoPerPair requires selected pairs and pose")

        def iterator() -> Iterator[_DensePair]:
            for projected in value.stream:
                intrinsic = np.asarray(
                    gnomonic_intrinsics(projected.pair_spec.reference_spec),
                    dtype=np.float64,
                )
                geometry = RectifiedStereoGeometry(
                    focal_length_x=float(intrinsic[0, 0]),
                    focal_length_y=float(intrinsic[1, 1]),
                    principal_point_x=float(intrinsic[0, 2]),
                    principal_point_y=float(intrinsic[1, 2]),
                    baseline=value.pose.baseline,
                )
                result = self.matcher.run(
                    projected.reference,
                    projected.source,
                    reference_support=projected.reference_support,
                    source_support=projected.source_support,
                    geometry=geometry,
                )
                yield _DensePair(projected.pair_spec, result)

        return StageResult(
            replace(value, stream=iterator()),
            {
                "matcher": type(self.matcher).__name__,
                "streaming_cost_volume": True,
            },
        )


def _edge_taper(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    return np.clip((1.0 - xx * xx) * (1.0 - yy * yy), 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class _Accumulation:
    sum_weight: np.ndarray
    sum_taper: np.ndarray
    sum_range: np.ndarray
    sum_range_squared: np.ndarray
    count: np.ndarray
    diagnostics: tuple[StereoPairDiagnostic, ...]
    selection: tuple[StereoViewSelectionDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class BackProjectStereoRange:
    """Backproject pair ranges and accumulate confidence in ERP space online."""

    input_kind = "panoramic_stereo_context"
    output_kind = "panoramic_stereo_context"
    stage_name = "backproject_stereo_range"

    def execute(self, value: Any) -> StageResult:
        if not isinstance(value, _StereoContext) or value.stream is None:
            raise TypeError("BackProjectStereoRange requires dense pair stream")
        shape = value.reference.shape[:2]
        sum_weight = np.zeros(shape, dtype=np.float64)
        sum_taper = np.zeros(shape, dtype=np.float64)
        sum_range = np.zeros(shape, dtype=np.float64)
        sum_squared = np.zeros(shape, dtype=np.float64)
        count = np.zeros(shape, dtype=np.uint16)
        diagnostics: list[StereoPairDiagnostic] = []
        for dense in value.stream:
            result = dense.result
            if result.radial_range is None:
                raise RuntimeError("rectified matcher did not return radial range")
            valid = np.asarray(result.validity, dtype=bool)
            confidence = np.asarray(result.confidence, dtype=np.float32)
            taper = _edge_taper(valid.shape)
            weight_face = np.where(valid, confidence * taper, 0.0).astype(np.float32)
            projector = GnomonicProjector(
                dense.pair_spec.reference_spec,
                interpolation="bilinear",
                invalid_policy="renormalize",
                min_valid_weight=0.5,
            )
            range_erp = projector.back_project(
                np.asarray(result.radial_range, dtype=np.float32),
                shape,
                validity_mask=valid,
            )
            weight_erp = projector.back_project(
                weight_face,
                shape,
                validity_mask=valid,
            )
            taper_erp = projector.back_project(
                np.where(valid, taper, 0.0).astype(np.float32),
                shape,
                validity_mask=valid,
            )
            supported = np.array(range_erp.support_mask, dtype=bool, copy=True)
            if range_erp.validity_mask is not None:
                supported &= np.asarray(range_erp.validity_mask, dtype=bool)
            supported &= np.asarray(weight_erp.support_mask, dtype=bool)
            supported &= np.asarray(taper_erp.support_mask, dtype=bool)
            radial = np.asarray(range_erp.data, dtype=np.float64)
            weight = np.asarray(weight_erp.data, dtype=np.float64)
            taper_weight = np.asarray(taper_erp.data, dtype=np.float64)
            supported &= (
                np.isfinite(radial)
                & np.isfinite(weight)
                & np.isfinite(taper_weight)
                & (weight > 0.0)
                & (taper_weight > 0.0)
            )
            sum_weight[supported] += weight[supported]
            sum_taper[supported] += taper_weight[supported]
            sum_range[supported] += weight[supported] * radial[supported]
            sum_squared[supported] += weight[supported] * radial[supported] ** 2
            count[supported] = np.minimum(
                np.iinfo(np.uint16).max, count[supported].astype(np.uint32) + 1
            ).astype(np.uint16)
            selected_confidence = confidence[valid]
            selected_range = np.asarray(result.radial_range)[valid]
            diagnostics.append(
                StereoPairDiagnostic(
                    dense.pair_spec.pair_id,
                    float(valid.mean()),
                    float(np.median(selected_confidence))
                    if selected_confidence.size
                    else 0.0,
                    float(np.median(selected_range))
                    if selected_range.size
                    else float("nan"),
                    result.method,
                )
            )
        accumulation = _Accumulation(
            sum_weight,
            sum_taper,
            sum_range,
            sum_squared,
            count,
            tuple(diagnostics),
            tuple(value.selection_audit or ()),
        )
        return StageResult(
            replace(value, accumulation=accumulation, stream=None),
            {
                "processed_pairs": len(diagnostics),
                "retained_cost_volumes": 0,
                "quantity": "radial_range",
            },
        )


@dataclass(frozen=True, slots=True)
class FuseStereoRange:
    min_weight: float = 1e-6

    input_kind = "panoramic_stereo_context"
    output_kind = "panoramic_stereo_result"
    stage_name = "fuse_stereo_range"

    def execute(self, value: Any) -> StageResult:
        if not isinstance(value, _StereoContext) or value.accumulation is None:
            raise TypeError("FuseStereoRange requires accumulated ERP evidence")
        if value.pose is None or value.plan is None:
            raise RuntimeError("pose and plan disappeared before fusion")
        acc = value.accumulation
        valid = acc.sum_weight > self.min_weight
        radial = np.full(valid.shape, np.nan, dtype=np.float32)
        radial[valid] = (acc.sum_range[valid] / acc.sum_weight[valid]).astype(
            np.float32
        )
        variance = np.zeros(valid.shape, dtype=np.float64)
        variance[valid] = np.maximum(
            0.0,
            acc.sum_range_squared[valid] / acc.sum_weight[valid]
            - radial[valid].astype(np.float64) ** 2,
        )
        disagreement = np.full(valid.shape, np.nan, dtype=np.float32)
        disagreement[valid] = np.sqrt(variance[valid]).astype(np.float32)
        matching_confidence, consistency_confidence, confidence = (
            compose_fusion_confidence(
                acc.sum_weight,
                acc.sum_taper,
                radial,
                disagreement,
                valid,
            )
        )
        result = PanoramicStereoResult(
            radial,
            valid,
            confidence,
            disagreement,
            acc.count,
            value.pose,
            value.plan.pairs,
            acc.selection,
            acc.diagnostics,
            {
                "route": "rectified-views",
                "range_units": value.pose.units,
                "scale_status": value.pose.scale_status,
                "valid_fraction": float(valid.mean()),
                "pair_count": len(acc.diagnostics),
                "rejected_pair_count": sum(not item.accepted for item in acc.selection),
                "fusion": "confidence-and-edge-taper weighted radial range",
                "confidence": (
                    "taper-normalized matching margin multiplied by "
                    "exp(-range_disagreement/radial_range)"
                ),
            },
            matching_confidence=matching_confidence,
            consistency_confidence=consistency_confidence,
        )
        return StageResult(
            result,
            {
                "valid_fraction": float(valid.mean()),
                "range_units": value.pose.units,
                "fused_quantity": "radial_range",
            },
        )


@dataclass(frozen=True, slots=True)
class SphereNativeStereo:
    options: SphericalStereoOptions

    input_kind = "panoramic_stereo_context"
    output_kind = "panoramic_stereo_result"
    stage_name = "sphere_native_stereo"

    def execute(self, value: Any) -> StageResult:
        if not isinstance(value, _StereoContext) or value.pose is None:
            raise TypeError("SphereNativeStereo requires resolved pose and scale")
        if not np.all(value.source_validity):
            raise ValueError(
                "the sphere-native route does not yet accept partial source validity; "
                "use route='rectified-views' for explicitly masked source support"
            )
        direct = SphericalDenseStereo(self.options).estimate(
            value.reference,
            value.source,
            value.pose.rotation_source_from_reference,
            value.pose.translation_source_from_reference,
        )
        valid = np.asarray(direct.validity_mask, dtype=bool).copy()
        valid &= np.asarray(value.reference_validity, dtype=bool)
        disagreement = np.full(valid.shape, np.nan, dtype=np.float32)
        count = valid.astype(np.uint16)
        radial = np.where(valid, direct.range, np.nan).astype(np.float32)
        direct_confidence = np.where(valid, direct.confidence, 0.0).astype(np.float32)
        result = PanoramicStereoResult(
            radial,
            valid,
            direct_confidence,
            disagreement,
            count,
            value.pose,
            (),
            (),
            (),
            {
                "route": "sphere-native",
                "range_units": value.pose.units,
                "scale_status": value.pose.scale_status,
                "valid_fraction": float(valid.mean()),
            },
            matching_confidence=direct_confidence,
            consistency_confidence=np.where(valid, 1.0, 0.0),
        )
        return StageResult(result, dict(result.diagnostics))


@dataclass(frozen=True, slots=True)
class ValidatePanoramicStereo:
    """Enforce the common panoramic result contract for either execution route."""

    min_valid_fraction: float = 0.0

    input_kind = "panoramic_stereo_result"
    output_kind = "panoramic_stereo_result"
    stage_name = "validate_panoramic_stereo"

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be in [0, 1]")

    def execute(self, value: Any) -> StageResult:
        if not isinstance(value, PanoramicStereoResult):
            raise TypeError("ValidatePanoramicStereo requires PanoramicStereoResult")
        valid_fraction = float(value.validity.mean())
        if valid_fraction < self.min_valid_fraction:
            raise RuntimeError(
                "panoramic stereo valid fraction "
                f"{valid_fraction:.6f} is below {self.min_valid_fraction:.6f}"
            )
        return StageResult(
            value,
            {
                "valid_fraction": valid_fraction,
                "finite_valid_range": bool(
                    np.all(np.isfinite(value.radial_range[value.validity]))
                ),
                "explicit_invalidity": True,
            },
        )


__all__ = [
    "BackProjectStereoRange",
    "DenseStereoPerPair",
    "FuseStereoRange",
    "PanoramicStereoInput",
    "PlanRectifiedStereoViews",
    "ProjectStereoViewPairs",
    "ResolveStereoPose",
    "ResolveStereoScale",
    "SelectStereoViewPairs",
    "SphereNativeStereo",
    "ValidatePanoramicStereo",
]
