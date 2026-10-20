"""End-to-end panoramic dense-stereo application façade."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from panorai.pipeline import Pipeline
from panorai.stereo import SphericalStereoOptions
from panorai.stereo.panoramic import (
    BackProjectStereoRange,
    DenseStereoPerPair,
    FuseStereoRange,
    FullSphereMultiViewStereo,
    FullSphereStereoOptions,
    FullSphereStereoResult,
    PanoramicStereoInput,
    PanoramicStereoPose,
    PanoramicStereoResult,
    RegisteredPanoramaSet,
    PlanRectifiedStereoViews,
    ProjectStereoViewPairs,
    RectifiedStereoViewPlanner,
    ResolveStereoPose,
    ResolveStereoScale,
    SelectStereoViewPairs,
    SphereNativeStereo,
    ValidatePanoramicStereo,
)

from ._base import Application, PipelineApplication, application_config
from ._classical_stereo import (
    ADCensusStereoDepth,
    CensusMGMStereoDepth,
    ELASStereoDepth,
    PatchMatchStereoDepth,
    StereoBMDepth,
    StereoSGBMDepth,
)


_REVIEWED_FULL_SPHERE_INTERFACE = "panorai-registered-full-sphere-stereo/v1"


class _ReviewedFullSphereMultiViewStereo(FullSphereMultiViewStereo):
    """Bind the exact reviewed execution to the Stable facade contract."""

    interface = _REVIEWED_FULL_SPHERE_INTERFACE
    stability = "stable"


_MATCHERS = {
    item.application_id: item
    for item in (
        StereoBMDepth,
        ADCensusStereoDepth,
        StereoSGBMDepth,
        CensusMGMStereoDepth,
        ELASStereoDepth,
        PatchMatchStereoDepth,
    )
}


@dataclass(frozen=True, slots=True)
class PanoramicDenseStereo(PipelineApplication):
    """Estimate reference-camera radial range from a pair of ERP panoramas.

    ``route='rectified-views'`` applies a complete classical matcher to a
    streamed ring of epipolar-rectified pinhole pairs. ``route='sphere-native'``
    runs PanorAi's direct ERP inverse-range plane sweep.  Both routes can use a
    supplied pose or estimate its direction from spherical feature matches.
    """

    matcher: Any = StereoSGBMDepth()
    route: str = "rectified-views"
    pose_mode: str = "auto"
    require_pose_quality_acceptance: bool = True
    planner: RectifiedStereoViewPlanner = RectifiedStereoViewPlanner()
    direct_options: SphericalStereoOptions = SphericalStereoOptions()
    min_joint_support: float = 0.10
    min_texture_std: float = 0.005
    min_pose_inliers_per_view: int = 0
    max_epipolar_row_error_px: float = 2.0
    min_valid_fraction: float = 0.0

    application_id = "panorai.applications.PanoramicDenseStereo"

    def __post_init__(self) -> None:
        if self.route not in {"rectified-views", "sphere-native"}:
            raise ValueError("route must be 'rectified-views' or 'sphere-native'")
        if self.pose_mode not in {"auto", "provided", "estimate"}:
            raise ValueError("pose_mode must be auto, provided, or estimate")
        if self.route == "rectified-views" and not callable(
            getattr(self.matcher, "run", None)
        ):
            raise TypeError("rectified-views matcher must expose run()")

    def as_pipeline(self) -> Pipeline:
        prefix: tuple[Any, ...] = (
            (
                "resolve_pose",
                ResolveStereoPose(self.pose_mode, self.require_pose_quality_acceptance),
            ),
            ("resolve_scale", ResolveStereoScale()),
        )
        if self.route == "sphere-native":
            return Pipeline(
                *prefix,
                ("sphere_native_stereo", SphereNativeStereo(self.direct_options)),
                (
                    "validate_panoramic_stereo",
                    ValidatePanoramicStereo(self.min_valid_fraction),
                ),
            )
        return Pipeline(
            *prefix,
            ("plan_rectified_views", PlanRectifiedStereoViews(self.planner)),
            ("project_view_pairs", ProjectStereoViewPairs()),
            (
                "select_view_pairs",
                SelectStereoViewPairs(
                    self.min_joint_support,
                    self.min_texture_std,
                    self.min_pose_inliers_per_view,
                    self.max_epipolar_row_error_px,
                    self.matcher,
                ),
            ),
            ("dense_stereo_per_pair", DenseStereoPerPair(self.matcher)),
            ("backproject_stereo_range", BackProjectStereoRange()),
            ("fuse_stereo_range", FuseStereoRange()),
            (
                "validate_panoramic_stereo",
                ValidatePanoramicStereo(self.min_valid_fraction),
            ),
        )

    def run(
        self,
        reference: Any,
        source: Any,
        *,
        pose: PanoramicStereoPose | None = None,
        rotation_source_from_reference: Any | None = None,
        translation_source_from_reference: Any | None = None,
        baseline: float | None = None,
        units: str | None = None,
        reference_validity: Any | None = None,
        source_validity: Any | None = None,
    ) -> PanoramicStereoResult:
        scale_status = None
        pose_provenance = None
        if pose is not None:
            if (
                rotation_source_from_reference is not None
                or translation_source_from_reference is not None
            ):
                raise ValueError("pass either pose or separate R,t, not both")
            rotation_source_from_reference = pose.rotation_source_from_reference
            translation_source_from_reference = pose.translation_source_from_reference
            if pose.scale_status == "metric" and baseline is None:
                baseline = pose.baseline
            units = pose.units if units is None else units
            scale_status = pose.scale_status
            pose_provenance = pose.provenance
        request = PanoramicStereoInput(
            reference,
            source,
            rotation_source_from_reference,
            translation_source_from_reference,
            baseline,
            units,
            reference_validity,
            source_validity,
            scale_status,
            pose_provenance,
        )
        run = self.as_pipeline().run(request)
        result = run.output
        if not isinstance(result, PanoramicStereoResult):
            raise RuntimeError(
                "panoramic stereo pipeline returned an unexpected result"
            )
        diagnostics = dict(result.diagnostics)
        diagnostics["pipeline_stages"] = tuple(record.name for record in run.records)
        diagnostics["stage_diagnostics"] = {
            record.name: dict(record.details) for record in run.records
        }
        return replace(result, diagnostics=diagnostics)

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "stability": self.stability,
            "application": self.application_id,
            "input": "two canonical ERPs; optional R,t and baseline",
            "output": "PanoramicStereoResult with radial range, confidence, disagreement, pose and diagnostics",
            "route": self.route,
            "pose_mode": self.pose_mode,
            "require_pose_quality_acceptance": self.require_pose_quality_acceptance,
            "scale": "metric with supplied baseline; otherwise explicit baseline-units",
            "streaming": "one projected pair and one dense cost volume at a time",
            "pipeline": self.as_pipeline().describe(),
        }

    def to_config(self) -> dict[str, Any]:
        matcher_config = self.matcher.to_config()
        return application_config(
            self.application_id,
            {
                "route": self.route,
                "pose_mode": self.pose_mode,
                "require_pose_quality_acceptance": self.require_pose_quality_acceptance,
                "planner": asdict(self.planner),
                "direct_options": self.direct_options.to_dict(),
                "min_joint_support": self.min_joint_support,
                "min_texture_std": self.min_texture_std,
                "min_pose_inliers_per_view": self.min_pose_inliers_per_view,
                "max_epipolar_row_error_px": self.max_epipolar_row_error_px,
                "min_valid_fraction": self.min_valid_fraction,
                "matcher": matcher_config,
            },
        )

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> "PanoramicDenseStereo":
        if payload.get("application") != cls.application_id:
            raise ValueError("configuration does not describe PanoramicDenseStereo")
        parameters = payload.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError("panoramic stereo parameters are missing")
        raw_matcher = parameters.get("matcher")
        if not isinstance(raw_matcher, dict):
            raise ValueError("panoramic stereo matcher is missing")
        matcher_type = _MATCHERS.get(raw_matcher.get("application"))
        if matcher_type is None:
            raise ValueError("panoramic stereo matcher is not registered")
        planner = parameters.get("planner")
        options = parameters.get("direct_options")
        if not isinstance(planner, dict) or not isinstance(options, dict):
            raise ValueError("panoramic stereo planner/options are missing")
        return cls(
            matcher=matcher_type.from_config(raw_matcher),
            route=str(parameters.get("route")),
            pose_mode=str(parameters.get("pose_mode")),
            require_pose_quality_acceptance=bool(
                parameters.get("require_pose_quality_acceptance", True)
            ),
            planner=RectifiedStereoViewPlanner(**planner),
            direct_options=SphericalStereoOptions(**options),
            min_joint_support=float(parameters.get("min_joint_support", 0.10)),
            min_texture_std=float(parameters.get("min_texture_std", 0.005)),
            min_pose_inliers_per_view=int(
                parameters.get("min_pose_inliers_per_view", 0)
            ),
            max_epipolar_row_error_px=float(
                parameters.get("max_epipolar_row_error_px", 2.0)
            ),
            min_valid_fraction=float(parameters.get("min_valid_fraction", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class FullSphereDenseStereo(Application):
    """Estimate radial range over a cube atlas from a registered view set.

    :class:`~panorai.stereo.panoramic.RegisteredPanoramaSet` owns the reference
    and every directed source relation. The result keeps sampling support,
    observability, and numerical validity as separate masks.
    """

    options: FullSphereStereoOptions = FullSphereStereoOptions()

    application_id = "panorai.applications.FullSphereDenseStereo"

    @property
    def interface(self) -> str:
        """Return the contract selected by the exact option profile."""

        return (
            _REVIEWED_FULL_SPHERE_INTERFACE
            if self.options.profile_id is not None
            else FullSphereMultiViewStereo.interface
        )

    @property
    def stability(self) -> str:
        """Return Stable only for the exact reviewed supplied-pose profile."""

        return "stable" if self.options.profile_id is not None else "experimental"

    @classmethod
    def reviewed_profile(
        cls,
        *,
        filter_backend: str = "auto",
    ) -> "FullSphereDenseStereo":
        """Construct the frozen supplied-pose profile used for promotion work."""

        return cls(
            FullSphereStereoOptions.reviewed_profile(filter_backend=filter_backend)
        )

    def run(
        self,
        registered: RegisteredPanoramaSet,
    ) -> FullSphereStereoResult:
        is_reviewed = self.options.profile_id is not None
        if is_reviewed and isinstance(registered, RegisteredPanoramaSet):
            if any(
                source.pose.scale_status != "metric" for source in registered.sources
            ):
                raise ValueError(
                    "the Stable reviewed profile requires supplied metric poses"
                )
        engine_type = (
            _ReviewedFullSphereMultiViewStereo
            if is_reviewed
            else FullSphereMultiViewStereo
        )
        return engine_type(self.options).estimate(registered)

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "stability": self.stability,
            "application": self.application_id,
            "input": "RegisteredPanoramaSet with one reference ERP and one or more relative source views",
            "output": "FullSphereStereoResult with radial range, sampling support, observability and validity",
            "domain": "overlapping cube-oriented gnomonic atlas with spherical multibaseline warping",
            "quantity": "radial_range",
            "streaming": "one reference face and one hypothesis source stack at a time",
            "pose_policy": "provided-only",
            "profile_id": self.options.profile_id,
            "experimental_extensions": list(self.options.experimental_extensions),
            "confidence_semantics": "within-result ranking score; not a calibrated probability",
        }

    def to_config(self) -> dict[str, Any]:
        return application_config(
            self.application_id,
            {"options": asdict(self.options)},
        )

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> "FullSphereDenseStereo":
        if payload.get("application") != cls.application_id:
            raise ValueError("configuration does not describe FullSphereDenseStereo")
        parameters = payload.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError("full-sphere stereo parameters are missing")
        raw_options = parameters.get("options")
        if not isinstance(raw_options, dict):
            raise ValueError("full-sphere stereo options are missing")
        raw_planner = raw_options.get("planner")
        raw_stereo = raw_options.get("stereo")
        if not isinstance(raw_planner, dict) or not isinstance(raw_stereo, dict):
            raise ValueError("planner and stereo options are required")
        from panorai.stereo.panoramic import OverlappingCubemapPlanner

        options = FullSphereStereoOptions(
            planner=OverlappingCubemapPlanner(**raw_planner),
            stereo=SphericalStereoOptions(**raw_stereo),
            min_parallax_deg=float(raw_options.get("min_parallax_deg", 0.25)),
            min_source_evidence=int(raw_options.get("min_source_evidence", 1)),
            max_sources_per_hypothesis=raw_options.get(
                "max_sources_per_hypothesis", None
            ),
            source_aggregation=str(raw_options.get("source_aggregation", "median")),
            matching_cost=str(raw_options.get("matching_cost", "intensity-gradient")),
            local_aggregation=str(raw_options.get("local_aggregation", "box")),
            regularizer=str(raw_options.get("regularizer", "sgm4")),
            census_radius=int(raw_options.get("census_radius", 2)),
            ad_weight=float(raw_options.get("ad_weight", 0.25)),
            census_weight=float(raw_options.get("census_weight", 0.75)),
            ad_truncation=float(raw_options.get("ad_truncation", 0.20)),
            erp_feature_tap_deg=(
                None
                if raw_options.get("erp_feature_tap_deg") is None
                else float(raw_options["erp_feature_tap_deg"])
            ),
            face_cost_tap_deg=(
                None
                if raw_options.get("face_cost_tap_deg") is None
                else float(raw_options["face_cost_tap_deg"])
            ),
            range_consensus_tolerance_steps=int(
                raw_options.get("range_consensus_tolerance_steps", 2)
            ),
            min_range_consensus_sources=int(
                raw_options.get("min_range_consensus_sources", 0)
            ),
        )
        return cls(options)


__all__ = ["FullSphereDenseStereo", "PanoramicDenseStereo"]
