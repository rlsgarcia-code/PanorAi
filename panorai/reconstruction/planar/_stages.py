"""Inspectable registered-scene stages for the existing Pipeline runtime."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from panorai.geometry import erp_pixels_to_rays
from ._models import (
    PlaneCorrespondence,
    PlanarPointObservation,
    PlanarReconstructionOptions,
    PlanarWorkspace,
    RegisteredPlanarScene,
)


def _out(value, **details):
    from panorai.pipeline import StageResult

    return StageResult(value, details)


class _PlanarStage:
    """Bridge the planar stages to the public ``Stage.execute`` contract."""

    def execute(self, value):
        return self._run(value)


@dataclass(frozen=True)
class ValidatePlanarScene(_PlanarStage):
    input_kind = "registered-planar-scene"
    output_kind = "planar-workspace"

    def _run(self, value):
        if not isinstance(value, RegisteredPlanarScene):
            raise TypeError("expected RegisteredPlanarScene")
        return _out(
            PlanarWorkspace(value, correspondences=value.correspondences),
            views=len(value.views),
            revision=value.revision,
            units="m",
            pose_convention="x_camera=R@(X_world-C)",
        )


@dataclass(frozen=True)
class SegmentPlaneRegions(_PlanarStage):
    method: str = "manual"
    face_size: int = 512
    sam_backend: Any = None
    input_kind = output_kind = "planar-workspace"

    def _run(self, value):
        from ._segmentation import segment_planar_region

        views = {v.view_id: v for v in value.scene.views}
        regions, alternatives = [], []
        for region in value.scene.regions:
            if region.interpretation == "seed" and region.state != "rejected":
                result, masks = segment_planar_region(
                    views[region.view_id],
                    region,
                    method=self.method,
                    face_size=self.face_size,
                    sam_backend=self.sam_backend,
                )
                regions.append(result)
                alternatives.append((region.plane_id, region.view_id, masks))
            else:
                regions.append(region)
        scene = replace(value.scene, regions=tuple(regions))
        return _out(
            replace(value, scene=scene, segmentation_alternatives=tuple(alternatives)),
            method=self.method,
            needs_review=len(alternatives),
        )


@dataclass(frozen=True)
class MatchPlaneRegions(_PlanarStage):
    options: PlanarReconstructionOptions = PlanarReconstructionOptions()
    input_kind = output_kind = "planar-workspace"

    def _run(self, value):
        if not self.options.automatic_matching:
            return _out(
                value, automatic=False, correspondences=len(value.correspondences)
            )
        from panorai.features import SphericalFeaturePipeline
        from ._geometry import sample_mask

        pipeline = SphericalFeaturePipeline.from_preset(
            "sift-bf",
            face_sampler="cube",
            face_shape_hw=self.options.feature_face_size,
            face_fov_deg=100,
            ratio_test=0.75,
            cross_check=True,
            max_features=self.options.max_features,
        )
        features = {
            v.view_id: pipeline.extract(
                v.image, panorama_id=v.view_id, validity_mask=v.support
            )
            for v in value.scene.views
        }
        views = {v.view_id: v for v in value.scene.views}
        tracks = list(value.correspondences)
        for plane_id in sorted({r.plane_id for r in value.scene.regions}):
            regions = {
                r.view_id: r for r in value.scene.regions if r.plane_id == plane_id
            }
            accepted = [
                r for r in regions.values() if r.state == "accepted" and r.mask.any()
            ]
            if not accepted:
                continue
            reference_region = accepted[0]
            reference = views[reference_region.view_id]
            source_features = features[reference.view_id]
            groups = {}
            for target in value.scene.views:
                region = regions.get(target.view_id)
                if target.view_id == reference.view_id or (
                    region is not None and region.state == "rejected"
                ):
                    continue
                matches = pipeline.match(source_features, features[target.view_id])
                for index in range(len(matches)):
                    if not matches.valid[index]:
                        continue
                    ia, ib = (
                        int(matches.feature_indices_a[index]),
                        int(matches.feature_indices_b[index]),
                    )
                    a = source_features.features[ia].source_erp_xy
                    b = features[target.view_id].features[ib].source_erp_xy
                    if (
                        a is None
                        or b is None
                        or not sample_mask(reference_region.mask, np.asarray(a)[None])[
                            0
                        ]
                    ):
                        continue
                    if (
                        region is not None
                        and region.state == "accepted"
                        and not sample_mask(region.mask, np.asarray(b)[None])[0]
                    ):
                        continue
                    # Fixed registered geometry, no essential-matrix fitting on
                    # a planar subset (which would be a degenerate pose sample).
                    ba = (
                        erp_pixels_to_rays(
                            np.asarray(a)[None], reference.support.shape
                        )[0]
                        @ reference.pose.R
                    )
                    bb = (
                        erp_pixels_to_rays(np.asarray(b)[None], target.support.shape)[0]
                        @ target.pose.R
                    )
                    normal = np.cross(target.pose.center - reference.pose.center, ba)
                    norm = np.linalg.norm(normal)
                    error = (
                        90
                        if norm < 1e-10
                        else np.degrees(
                            np.arcsin(np.clip(abs(normal @ bb) / norm, 0, 1))
                        )
                    )
                    if error > self.options.maximum_reprojection_deg:
                        continue
                    observations = groups.setdefault(
                        ia,
                        {
                            reference.view_id: PlanarPointObservation(
                                reference.view_id, tuple(a)
                            )
                        },
                    )
                    observations.setdefault(
                        target.view_id, PlanarPointObservation(target.view_id, tuple(b))
                    )
            for feature_index, observations in sorted(groups.items()):
                if len(observations) >= 2:
                    tracks.append(
                        PlaneCorrespondence(
                            plane_id,
                            f"sift:{reference.view_id}:{feature_index}",
                            tuple(observations.values()),
                            "sift",
                        )
                    )
        return _out(
            replace(value, correspondences=tuple(tracks)),
            correspondences=len(tracks),
            automatic=True,
            profile=pipeline.describe(),
        )


@dataclass(frozen=True)
class TriangulatePlaneSupport(_PlanarStage):
    options: PlanarReconstructionOptions = PlanarReconstructionOptions()
    input_kind = output_kind = "planar-workspace"

    def _run(self, value):
        from ._geometry import triangulate_workspace

        result = triangulate_workspace(value, self.options)
        return _out(
            result,
            accepted=sum(t.reason is None for t in result.tracks),
            rejected=sum(t.reason is not None for t in result.tracks),
        )


@dataclass(frozen=True)
class FitPlanes(_PlanarStage):
    options: PlanarReconstructionOptions = PlanarReconstructionOptions()
    input_kind = output_kind = "planar-workspace"

    def _run(self, value):
        from ._geometry import fit_workspace

        result = fit_workspace(value, self.options)
        return _out(
            result,
            planes=[
                {"plane_id": p.plane_id, "state": p.state, **p.diagnostics}
                for p in result.planes
            ],
        )


@dataclass(frozen=True)
class PropagatePlaneRegions(_PlanarStage):
    options: PlanarReconstructionOptions = PlanarReconstructionOptions()
    input_kind = output_kind = "planar-workspace"

    def _run(self, value):
        from ._geometry import propagate_workspace

        result = propagate_workspace(value, self.options)
        return _out(result, proposals=len(result.proposals), review_required=True)


@dataclass(frozen=True)
class RasterizePlaneRanges(_PlanarStage):
    options: PlanarReconstructionOptions = PlanarReconstructionOptions()
    input_kind = output_kind = "planar-workspace"

    def _run(self, value):
        from ._geometry import rasterize_workspace

        result = rasterize_workspace(value, self.options)
        return _out(
            result,
            quantity="radial_range",
            units="m",
            native_resolution=True,
            valid_pixels={r.view_id: int(r.validity.sum()) for r in result.ranges},
        )


PLANAR_STAGE_TYPES = (
    ValidatePlanarScene,
    SegmentPlaneRegions,
    MatchPlaneRegions,
    TriangulatePlaneSupport,
    FitPlanes,
    PropagatePlaneRegions,
    RasterizePlaneRanges,
)
