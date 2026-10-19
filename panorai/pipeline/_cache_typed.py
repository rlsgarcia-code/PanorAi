"""Safe typed-result cache codec backed by NumPy archives and JSON state."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np


def _is_reload_equivalent(
    value: Any,
    expected: type,
    *,
    module: str,
    name: str,
) -> bool:
    """Recognize a public PanorAi value created before its module was reloaded.

    Plugin discovery and isolated consumers may legitimately reload a module.
    Python then gives the same public class a new identity even though existing
    instances still satisfy the same versioned codec contract.  Restrict the
    fallback to PanorAi's exact module and class name; arbitrary lookalikes are
    still rejected.
    """

    value_type = type(value)
    return isinstance(value, expected) or (
        value_type.__module__ == module and value_type.__name__ == name
    )


def _safe_json(value: Any, *, path: str) -> Any:
    if isinstance(value, np.generic):
        return _safe_json(value.item(), path=path)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{path}: metadata must be finite")
        return value
    if isinstance(value, (tuple, list)):
        return [
            _safe_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{path}: metadata keys must be strings")
        return {
            key: _safe_json(item, path=f"{path}.{key}") for key, item in value.items()
        }
    raise ValueError(f"{path}: unsupported metadata type {type(value).__name__}")


class _Encoder:
    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}

    def array(self, value: Any, *, path: str) -> dict[str, Any]:
        module = type(value).__module__
        backend = "numpy"
        device = "cpu"
        if module == "torch" or module.startswith("torch."):
            if bool(value.requires_grad):
                raise ValueError(
                    f"{path}: tensors requiring gradients cannot be cached; "
                    "cache reuse must never disconnect autograd"
                )
            if str(value.dtype) == "torch.bfloat16":
                raise ValueError(f"{path}: torch.bfloat16 has no safe NumPy codec")
            if not bool(value.is_contiguous()):
                raise ValueError(
                    f"{path}: non-contiguous Torch layout has no lossless cache codec"
                )
            backend = "torch"
            device = str(value.device)
            value = value.detach().cpu().numpy()
        if not isinstance(value, np.ndarray) or value.dtype.hasobject:
            raise ValueError(f"{path}: expected a numeric NumPy array or tensor")
        if not value.flags.c_contiguous and not value.flags.f_contiguous:
            raise ValueError(
                f"{path}: strided non-contiguous NumPy layout has no lossless cache codec"
            )
        key = f"array_{len(self.arrays):06d}"
        self.arrays[key] = np.asarray(value)
        return {
            "key": key,
            "backend": backend,
            "device": device,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "strides": list(value.strides),
            "c_contiguous": bool(value.flags.c_contiguous),
            "f_contiguous": bool(value.flags.f_contiguous),
        }

    def maybe_array(self, value: Any, *, path: str) -> Any:
        if value is None:
            return None
        return self.array(value, path=path)

    def grid(self, value: Any) -> dict[str, Any]:
        return _safe_json(value.describe(), path="grid")

    def spec(self, value: Any) -> dict[str, Any]:
        return {
            "center_lat_deg": value.center_lat_deg,
            "center_lon_deg": value.center_lon_deg,
            "hfov_deg": value.hfov_deg,
            "vfov_deg": value.vfov_deg,
            "roll_deg": value.roll_deg,
            "output_shape_hw": list(value.output_shape_hw),
        }

    def view_result(self, value: Any, *, path: str) -> dict[str, Any]:
        return {
            "face_id": value.face_id,
            "spec": self.spec(value.spec),
            "value": self.array(value.value, path=f"{path}.value"),
            "support": self.array(value.support, path=f"{path}.support"),
            "validity": self.array(value.validity, path=f"{path}.validity"),
        }

    def encode(self, value: Any, *, path: str = "output") -> dict[str, Any]:
        from panorai.data import (
            EquirectangularImage,
            GnomonicFaceSet,
            RegionAnalysisResult,
        )
        from panorai.estimators import RelativePoseResult
        from panorai.reconstruction import (
            MultiViewRangeMap,
            MultiViewRangeSet,
            RegisteredSphericalScene,
            RegisteredSphericalView,
            SphericalDenseReconstructionResult,
            SurfaceMesh,
            TSDFVolume,
        )

        from ._fields import RasterField, ResultBundle
        from ._observations import (
            ClassificationResult,
            EmbeddingResult,
            PerViewResults,
            SphericalObservations,
            ViewObservations,
        )

        if value is None:
            return {"type": "None"}
        if isinstance(value, RegisteredSphericalView):
            return {
                "type": "RegisteredSphericalView",
                "image": self.encode(value.image, path=f"{path}.image"),
                "panorama_id": value.panorama_id,
                "rotation": self.array(value.pose.R, path=f"{path}.pose.rotation"),
                "center": self.array(value.pose.center, path=f"{path}.pose.center"),
                "validity": self.maybe_array(value.validity, path=f"{path}.validity"),
            }
        if isinstance(value, RegisteredSphericalScene):
            return {
                "type": "RegisteredSphericalScene",
                "views": [
                    self.encode(item, path=f"{path}.views[{index}]")
                    for index, item in enumerate(value.views)
                ],
                "units": value.units,
                "frame": value.frame,
            }
        if isinstance(value, MultiViewRangeMap):
            return {
                "type": "MultiViewRangeMap",
                "reference_id": value.reference_id,
                "pose": {
                    "panorama_id": value.pose.panorama_id,
                    "rotation": self.array(value.pose.R, path=f"{path}.pose.rotation"),
                    "center": self.array(value.pose.center, path=f"{path}.pose.center"),
                },
                "radial_range": self.array(
                    value.radial_range, path=f"{path}.radial_range"
                ),
                "validity": self.array(value.validity, path=f"{path}.validity"),
                "sampling_support": self.array(
                    value.sampling_support, path=f"{path}.sampling_support"
                ),
                "observable": self.array(value.observable, path=f"{path}.observable"),
                "confidence": self.array(value.confidence, path=f"{path}.confidence"),
                "observable_source_count": self.array(
                    value.observable_source_count,
                    path=f"{path}.observable_source_count",
                ),
                "source_count": self.array(
                    value.source_count, path=f"{path}.source_count"
                ),
                "source_consensus_count": self.array(
                    value.source_consensus_count,
                    path=f"{path}.source_consensus_count",
                ),
                "contribution_count": self.array(
                    value.contribution_count,
                    path=f"{path}.contribution_count",
                ),
                "source_ids": list(value.source_ids),
                "diagnostics": _safe_json(
                    dict(value.diagnostics), path=f"{path}.diagnostics"
                ),
            }
        if isinstance(value, MultiViewRangeSet):
            return {
                "type": "MultiViewRangeSet",
                "scene": self.encode(value.scene, path=f"{path}.scene"),
                "maps": [
                    self.encode(item, path=f"{path}.maps[{index}]")
                    for index, item in enumerate(value.maps)
                ],
                "diagnostics": _safe_json(
                    dict(value.diagnostics), path=f"{path}.diagnostics"
                ),
            }
        if isinstance(value, TSDFVolume):
            return {
                "type": "TSDFVolume",
                "tsdf_zyx": self.array(value.tsdf_zyx, path=f"{path}.tsdf_zyx"),
                "weight_zyx": self.array(value.weight_zyx, path=f"{path}.weight_zyx"),
                "view_count_zyx": self.array(
                    value.view_count_zyx, path=f"{path}.view_count_zyx"
                ),
                "origin_xyz": self.array(value.origin_xyz, path=f"{path}.origin_xyz"),
                "voxel_size": value.voxel_size,
                "truncation_distance": value.truncation_distance,
                "units": value.units,
                "frame": value.frame,
                "min_surface_views": value.min_surface_views,
                "diagnostics": _safe_json(
                    dict(value.diagnostics), path=f"{path}.diagnostics"
                ),
            }
        if isinstance(value, SurfaceMesh):
            return {
                "type": "SurfaceMesh",
                "vertices_xyz": self.array(
                    value.vertices_xyz, path=f"{path}.vertices_xyz"
                ),
                "faces": self.array(value.faces, path=f"{path}.faces"),
                "normals_xyz": self.array(
                    value.normals_xyz, path=f"{path}.normals_xyz"
                ),
                "vertex_weight": self.array(
                    value.vertex_weight, path=f"{path}.vertex_weight"
                ),
                "vertex_view_count": self.array(
                    value.vertex_view_count, path=f"{path}.vertex_view_count"
                ),
                "colors_rgb": self.maybe_array(
                    value.colors_rgb, path=f"{path}.colors_rgb"
                ),
                "color_validity": self.maybe_array(
                    value.color_validity, path=f"{path}.color_validity"
                ),
                "color_contribution_count": self.maybe_array(
                    value.color_contribution_count,
                    path=f"{path}.color_contribution_count",
                ),
                "units": value.units,
                "frame": value.frame,
                "provenance": _safe_json(
                    dict(value.provenance), path=f"{path}.provenance"
                ),
            }
        if isinstance(value, SphericalDenseReconstructionResult):
            return {
                "type": "SphericalDenseReconstructionResult",
                "status": value.status,
                "reasons": list(value.reasons),
                "diagnostics": _safe_json(
                    dict(value.diagnostics), path=f"{path}.diagnostics"
                ),
                "surface": self.encode(value.surface, path=f"{path}.surface"),
                "ranges": self.encode(value.ranges, path=f"{path}.ranges"),
                "volume": self.encode(value.volume, path=f"{path}.volume"),
            }
        if isinstance(value, RelativePoseResult):
            return {
                "type": "RelativePoseResult",
                "rotation": self.array(value.rotation, path=f"{path}.rotation"),
                "translation_direction": self.array(
                    value.translation_direction,
                    path=f"{path}.translation_direction",
                ),
                "essential_matrix": self.array(
                    value.essential_matrix, path=f"{path}.essential_matrix"
                ),
                "inlier_mask": self.array(
                    value.inlier_mask, path=f"{path}.inlier_mask"
                ),
                "residuals_rad": self.array(
                    value.residuals_rad, path=f"{path}.residuals_rad"
                ),
                "metadata": _safe_json(
                    {
                        "num_inliers": value.num_inliers,
                        "num_trials": value.num_trials,
                        "median_parallax_deg": value.median_parallax_deg,
                        "cheirality_ratio": value.cheirality_ratio,
                        "degenerate": value.degenerate,
                        "degeneracy_reasons": value.degeneracy_reasons,
                        "sampling_diagnostics": (value.sampling_diagnostics.to_dict()),
                        "quality_report": value.quality_report.to_dict(),
                        "acceptance_policy": value.acceptance_policy.to_dict(),
                        "options": value.options.to_dict(),
                        "compute_backend": value.compute_backend,
                        "interface": value.interface,
                        "minimal_solver": value.minimal_solver,
                        "robust_estimator": value.robust_estimator,
                        "consensus_refit_steps": value.consensus_refit_steps,
                        "decoupled_pose_report": (
                            None
                            if value.decoupled_pose_report is None
                            else value.decoupled_pose_report.to_dict()
                        ),
                    },
                    path=f"{path}.metadata",
                ),
            }
        if isinstance(value, RegionAnalysisResult):
            measurements = value.measurements
            adjacency = value.adjacency
            boundaries = value.boundaries
            contours = value.contours
            return {
                "type": "RegionAnalysisResult",
                "interface": value.interface,
                "measurements": {
                    "shape_hw": list(measurements.shape_hw),
                    "support_solid_angle_sr": measurements.support_solid_angle_sr,
                    "regions": [
                        {
                            "label": item.label,
                            "pixel_count": item.pixel_count,
                            "solid_angle_sr": item.solid_angle_sr,
                            "support_fraction": item.support_fraction,
                            "angular_perimeter_rad": item.angular_perimeter_rad,
                            "unsupported_boundary_rad": (item.unsupported_boundary_rad),
                            "centroid_bearing_xyz": self.maybe_array(
                                item.centroid_bearing_xyz,
                                path=f"{path}.measurements[{index}].centroid",
                            ),
                            "enclosing_cap": {
                                "center_bearing_xyz": self.array(
                                    item.enclosing_cap.center_bearing_xyz,
                                    path=(
                                        f"{path}.measurements[{index}]"
                                        ".enclosing_cap.center"
                                    ),
                                ),
                                "radius_deg": item.enclosing_cap.radius_deg,
                            },
                        }
                        for index, item in enumerate(measurements.regions)
                    ],
                },
                "adjacency": {
                    "labels": list(adjacency.labels),
                    "edges": [
                        {
                            "labels": list(item.labels),
                            "shared_boundary_rad": item.shared_boundary_rad,
                        }
                        for item in adjacency.edges
                    ],
                },
                "boundaries": (
                    None
                    if boundaries is None
                    else {
                        "shape_hw": list(boundaries.shape_hw),
                        "items": [
                            {
                                "label": item.label,
                                "bearings_xyz": self.array(
                                    item.bearings_xyz,
                                    path=f"{path}.boundaries[{index}].bearings_xyz",
                                ),
                                "angular_length_rad": item.angular_length_rad,
                            }
                            for index, item in enumerate(boundaries.boundaries)
                        ],
                    }
                ),
                "contours": (
                    None
                    if contours is None
                    else {
                        "shape_hw": list(contours.shape_hw),
                        "items": [
                            {
                                "label": item.label,
                                "bearings_xyz": self.array(
                                    item.bearings_xyz,
                                    path=f"{path}.contours[{index}].bearings_xyz",
                                ),
                                "closed": item.closed,
                                "angular_length_rad": item.angular_length_rad,
                            }
                            for index, item in enumerate(contours.contours)
                        ],
                    }
                ),
            }
        if isinstance(value, RasterField):
            is_views = value.grid.domain == "views"
            values = value.values if is_views else (value.values,)
            supports = value.support if is_views else (value.support,)
            validities = value.validity if is_views else (value.validity,)
            return {
                "type": "RasterField",
                "name": value.name,
                "kind": value.kind,
                "units": value.units,
                "invalid_policy": value.invalid_policy,
                "grid": self.grid(value.grid),
                "values": [
                    self.array(item, path=f"{path}.values[{index}]")
                    for index, item in enumerate(values)
                ],
                "support": [
                    self.array(item, path=f"{path}.support[{index}]")
                    for index, item in enumerate(supports)
                ],
                "validity": [
                    self.array(item, path=f"{path}.validity[{index}]")
                    for index, item in enumerate(validities)
                ],
            }
        if isinstance(value, ResultBundle):
            return {
                "type": "ResultBundle",
                "values": {
                    name: self.encode(item, path=f"{path}.{name}")
                    for name, item in value.items()
                },
                "descriptions": _safe_json(
                    dict(value._descriptions), path=f"{path}.descriptions"
                ),
            }
        if isinstance(value, (ClassificationResult, EmbeddingResult)):
            return {
                "type": type(value).__name__,
                "value": (
                    None
                    if value.is_per_view
                    else self.array(value.value, path=f"{path}.value")
                ),
                "items": [
                    self.view_result(item, path=f"{path}.items[{index}]")
                    for index, item in enumerate(value.items)
                ],
                "source_field": value.source_field,
                "erp_shape_hw": (
                    None if value.erp_shape_hw is None else list(value.erp_shape_hw)
                ),
            }
        if isinstance(value, PerViewResults):
            return {
                "type": "PerViewResults",
                "kind": value.kind,
                "items": [
                    self.view_result(item, path=f"{path}.items[{index}]")
                    for index, item in enumerate(value.items)
                ],
                "source_field": value.source_field,
                "erp_shape_hw": list(value.erp_shape_hw),
            }
        if isinstance(value, ViewObservations):
            return {
                "type": "ViewObservations",
                "geometry": value.geometry,
                "pixels_xy": [
                    self.array(item, path=f"{path}.pixels_xy[{index}]")
                    for index, item in enumerate(value.pixels_xy)
                ],
                "scores": self.array(value.scores, path=f"{path}.scores"),
                "labels": self.maybe_array(value.labels, path=f"{path}.labels"),
                "features": self.maybe_array(value.features, path=f"{path}.features"),
                "valid": self.array(value.valid, path=f"{path}.valid"),
                "metadata": _safe_json(
                    [dict(item) for item in value.metadata], path=f"{path}.metadata"
                ),
            }
        if isinstance(value, SphericalObservations):
            face_ids = [str(item) for item in value.source_face_ids]
            return {
                "type": "SphericalObservations",
                "bearings_xyz": self.array(
                    value.bearings_xyz, path=f"{path}.bearings_xyz"
                ),
                "footprints_xyz": [
                    self.array(item, path=f"{path}.footprints_xyz[{index}]")
                    for index, item in enumerate(value.footprints_xyz)
                ],
                "scores": self.array(value.scores, path=f"{path}.scores"),
                "support": self.array(value.support, path=f"{path}.support"),
                "data_validity": self.array(
                    value.data_validity, path=f"{path}.data_validity"
                ),
                "valid": self.array(value.valid, path=f"{path}.valid"),
                "source_face_ids": face_ids,
                "source_indices": self.array(
                    value.source_indices, path=f"{path}.source_indices"
                ),
                "source_pixels_xy": [
                    self.array(item, path=f"{path}.source_pixels_xy[{index}]")
                    for index, item in enumerate(value.source_pixels_xy)
                ],
                "geometry": value.geometry,
                "labels": self.maybe_array(value.labels, path=f"{path}.labels"),
                "features": self.maybe_array(value.features, path=f"{path}.features"),
                "metadata": _safe_json(
                    [dict(item) for item in value.metadata], path=f"{path}.metadata"
                ),
            }
        if _is_reload_equivalent(
            value,
            EquirectangularImage,
            module="panorai.data.equirectangular_image",
            name="EquirectangularImage",
        ):
            data = value._workflow_data()
            return {
                "type": "EquirectangularImage",
                "shadow_angle": value.shadow_angle,
                "shadow_padded": value.shadow_padded,
                "lat": value.lat,
                "lon": value.lon,
                "data": {
                    name: self.array(item, path=f"{path}.{name}.values")
                    for name, item in data.items()
                },
                "support": {
                    name: self.array(item, path=f"{path}.{name}.support")
                    for name, item in value._workflow_support.items()
                },
                "metadata": {
                    name: self.field_metadata(item, path=f"{path}.{name}.metadata")
                    for name, item in value._workflow_metadata.items()
                },
            }
        if _is_reload_equivalent(
            value,
            GnomonicFaceSet,
            module="panorai.data.gnomonic_imageset",
            name="GnomonicFaceSet",
        ):
            return self.face_set(value, path=path)
        module = type(value).__module__
        if (
            isinstance(value, np.ndarray)
            or module == "torch"
            or module.startswith("torch.")
        ):
            return {"type": "array", "value": self.array(value, path=path)}
        raise ValueError(
            f"{path}: no registered typed cache codec for {type(value).__name__}"
        )

    def field_metadata(self, metadata: dict[str, Any], *, path: str) -> dict[str, Any]:
        spec = metadata["field_spec"]
        return {
            "kind": metadata.get("kind"),
            "units": metadata.get("units"),
            "interpolation": metadata.get("interpolation"),
            "validity": self.array(metadata["validity"], path=f"{path}.validity"),
            "field_spec": _safe_json(asdict(spec), path=f"{path}.field_spec"),
        }

    def face_set(self, value: Any, *, path: str) -> dict[str, Any]:
        workflow = value._require_workflow()
        faces = []
        for face_index, face in enumerate(value):
            data = face._workflow_data()
            faces.append(
                {
                    "face_id": face.face_id,
                    "spec": self.spec(face.spec),
                    "erp_shape_hw": list(face._erp_shape_hw),
                    "data": {
                        name: self.array(
                            item, path=f"{path}.faces[{face_index}].{name}.values"
                        )
                        for name, item in data.items()
                    },
                    "support": {
                        name: self.array(
                            item, path=f"{path}.faces[{face_index}].{name}.support"
                        )
                        for name, item in face._workflow_support.items()
                    },
                    "metadata": {
                        name: self.field_metadata(
                            item,
                            path=f"{path}.faces[{face_index}].{name}.metadata",
                        )
                        for name, item in face._workflow_metadata.items()
                    },
                }
            )
        return {
            "type": "GnomonicFaceSet",
            "channel_name": value.channel_name,
            "faces": faces,
            "workflow": {
                "layout": workflow["layout"],
                "order": list(workflow["order"]),
                "erp_shape": list(workflow["erp_shape"]),
                "specs": [self.spec(item) for item in workflow["specs"]],
                "depth_policy": workflow["depth_policy"],
                "min_valid_weight": workflow["min_valid_weight"],
                "source_metadata": {
                    name: self.field_metadata(
                        item, path=f"{path}.workflow.source_metadata.{name}"
                    )
                    for name, item in workflow["source_metadata"].items()
                },
            },
        }


class _Decoder:
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.arrays = arrays
        self.used: set[str] = set()

    def array(self, state: dict[str, Any]) -> Any:
        key = state["key"]
        if key in self.used or key not in self.arrays:
            raise ValueError("typed cache array inventory is missing or reused")
        self.used.add(key)
        value = self.arrays[key]
        if list(value.shape) != state["shape"] or str(value.dtype) != state["dtype"]:
            raise ValueError("typed cache array metadata mismatch")
        if state["f_contiguous"] and not state["c_contiguous"]:
            value = np.asfortranarray(value)
        else:
            value = np.ascontiguousarray(value)
        if state["backend"] == "numpy":
            return value
        if state["backend"] != "torch":
            raise ValueError("unknown typed cache array backend")
        try:
            import torch
        except ImportError as exc:
            raise ValueError(
                "cached Torch result requires the optional Torch extra"
            ) from exc
        try:
            return torch.from_numpy(value).to(state["device"])
        except (RuntimeError, TypeError) as exc:
            raise ValueError(
                f"cached Torch result cannot be restored on {state['device']!r}"
            ) from exc

    def maybe_array(self, state: Any) -> Any:
        return None if state is None else self.array(state)

    @staticmethod
    def spec(state: dict[str, Any]) -> Any:
        from panorai.geometry import GnomonicSpec

        values = dict(state)
        values["output_shape_hw"] = tuple(values["output_shape_hw"])
        return GnomonicSpec(**values)

    @staticmethod
    def grid(state: dict[str, Any]) -> Any:
        from ._fields import GridTransform, RasterGrid

        def deep_tuple(value: Any) -> Any:
            return (
                tuple(deep_tuple(item) for item in value)
                if isinstance(value, list)
                else value
            )

        transforms = tuple(
            GridTransform(
                item["operation"],
                tuple(item["input_shape_hw"]),
                tuple(item["output_shape_hw"]),
                tuple(item["parameters"].items()),
            )
            for item in state["transforms"]
        )
        return RasterGrid(
            domain=state["domain"],
            shape_hw=tuple(state["shape_hw"]),
            erp_shape_hw=tuple(state["erp_shape_hw"]),
            alignment=state["alignment"],
            view_ids=tuple(state["view_ids"]),
            view_specs=tuple(deep_tuple(item) for item in state["view_specs"]),
            first_center_px_yx=tuple(state["first_center_px_yx"]),
            pixel_stride_yx=tuple(state["pixel_stride_yx"]),
            angular_stride_rad=(
                None
                if state["angular_stride_rad"] is None
                else tuple(state["angular_stride_rad"])
            ),
            longitude_periodic=state["longitude_periodic"],
            transforms=transforms,
        )

    def view_result(self, state: dict[str, Any]) -> Any:
        from ._observations import ViewResult

        return ViewResult(
            state["face_id"],
            self.spec(state["spec"]),
            self.array(state["value"]),
            self.array(state["support"]),
            self.array(state["validity"]),
        )

    def decode(self, state: dict[str, Any]) -> Any:
        from panorai.reconstruction import (
            MultiViewRangeMap,
            MultiViewRangeSet,
            RegisteredSphericalScene,
            RegisteredSphericalView,
            SphericalCameraPose,
            SphericalDenseReconstructionResult,
            SurfaceMesh,
            TSDFVolume,
        )
        from ._fields import RasterField, ResultBundle
        from ._observations import (
            ClassificationResult,
            EmbeddingResult,
            PerViewResults,
            SphericalObservations,
            ViewObservations,
        )

        kind = state["type"]
        if kind == "None":
            return None
        if kind == "RegisteredSphericalView":
            return RegisteredSphericalView(
                image=self.decode(state["image"]),
                pose=SphericalCameraPose(
                    state["panorama_id"],
                    self.array(state["rotation"]),
                    self.array(state["center"]),
                ),
                validity=self.maybe_array(state["validity"]),
            )
        if kind == "RegisteredSphericalScene":
            return RegisteredSphericalScene(
                views=tuple(self.decode(item) for item in state["views"]),
                units=state["units"],
                frame=state["frame"],
            )
        if kind == "MultiViewRangeMap":
            pose = state["pose"]
            return MultiViewRangeMap(
                reference_id=state["reference_id"],
                pose=SphericalCameraPose(
                    pose["panorama_id"],
                    self.array(pose["rotation"]),
                    self.array(pose["center"]),
                ),
                radial_range=self.array(state["radial_range"]),
                validity=self.array(state["validity"]),
                sampling_support=self.array(state["sampling_support"]),
                observable=self.array(state["observable"]),
                confidence=self.array(state["confidence"]),
                observable_source_count=self.array(state["observable_source_count"]),
                source_count=self.array(state["source_count"]),
                source_consensus_count=self.array(state["source_consensus_count"]),
                contribution_count=self.array(state["contribution_count"]),
                source_ids=tuple(state["source_ids"]),
                diagnostics=state["diagnostics"],
            )
        if kind == "MultiViewRangeSet":
            return MultiViewRangeSet(
                scene=self.decode(state["scene"]),
                maps=tuple(self.decode(item) for item in state["maps"]),
                diagnostics=state["diagnostics"],
            )
        if kind == "TSDFVolume":
            return TSDFVolume(
                tsdf_zyx=self.array(state["tsdf_zyx"]),
                weight_zyx=self.array(state["weight_zyx"]),
                view_count_zyx=self.array(state["view_count_zyx"]),
                origin_xyz=self.array(state["origin_xyz"]),
                voxel_size=state["voxel_size"],
                truncation_distance=state["truncation_distance"],
                units=state["units"],
                frame=state["frame"],
                min_surface_views=state["min_surface_views"],
                diagnostics=state["diagnostics"],
            )
        if kind == "SurfaceMesh":
            return SurfaceMesh(
                vertices_xyz=self.array(state["vertices_xyz"]),
                faces=self.array(state["faces"]),
                normals_xyz=self.array(state["normals_xyz"]),
                vertex_weight=self.array(state["vertex_weight"]),
                vertex_view_count=self.array(state["vertex_view_count"]),
                colors_rgb=self.maybe_array(state["colors_rgb"]),
                color_validity=self.maybe_array(state["color_validity"]),
                color_contribution_count=self.maybe_array(
                    state["color_contribution_count"]
                ),
                units=state["units"],
                frame=state["frame"],
                provenance=state["provenance"],
            )
        if kind == "SphericalDenseReconstructionResult":
            return SphericalDenseReconstructionResult(
                status=state["status"],
                reasons=tuple(state["reasons"]),
                diagnostics=state["diagnostics"],
                surface=self.decode(state["surface"]),
                ranges=self.decode(state["ranges"]),
                volume=self.decode(state["volume"]),
            )
        if kind == "RelativePoseResult":
            return self.relative_pose_result(state)
        if kind == "RegionAnalysisResult":
            return self.region_analysis_result(state)
        if kind == "array":
            return self.array(state["value"])
        if kind == "RasterField":
            grid = self.grid(state["grid"])
            values = tuple(self.array(item) for item in state["values"])
            support = tuple(self.array(item) for item in state["support"])
            validity = tuple(self.array(item) for item in state["validity"])
            if grid.domain != "views":
                values, support, validity = values[0], support[0], validity[0]
            return RasterField(
                state["name"],
                values,
                state["kind"],
                support,
                validity,
                grid,
                state["units"],
                state["invalid_policy"],
            )
        if kind == "ResultBundle":
            return ResultBundle(
                {name: self.decode(item) for name, item in state["values"].items()},
                descriptions=state["descriptions"],
            )
        if kind in {"ClassificationResult", "EmbeddingResult"}:
            result_type = (
                ClassificationResult
                if kind == "ClassificationResult"
                else EmbeddingResult
            )
            if state["value"] is not None:
                return result_type(self.array(state["value"]))
            return result_type(
                items=tuple(self.view_result(item) for item in state["items"]),
                source_field=state["source_field"],
                erp_shape_hw=tuple(state["erp_shape_hw"]),
            )
        if kind == "PerViewResults":
            return PerViewResults(
                state["kind"],
                tuple(self.view_result(item) for item in state["items"]),
                source_field=state["source_field"],
                erp_shape_hw=tuple(state["erp_shape_hw"]),
            )
        if kind == "ViewObservations":
            return ViewObservations(
                state["geometry"],
                tuple(self.array(item) for item in state["pixels_xy"]),
                self.array(state["scores"]),
                labels=self.maybe_array(state["labels"]),
                features=self.maybe_array(state["features"]),
                valid=self.array(state["valid"]),
                metadata=tuple(state["metadata"]),
            )
        if kind == "SphericalObservations":
            return SphericalObservations(
                self.array(state["bearings_xyz"]),
                footprints_xyz=tuple(
                    self.array(item) for item in state["footprints_xyz"]
                ),
                scores=self.array(state["scores"]),
                support=self.array(state["support"]),
                data_validity=self.array(state["data_validity"]),
                valid=self.array(state["valid"]),
                source_face_ids=np.asarray(state["source_face_ids"], dtype=object),
                source_indices=self.array(state["source_indices"]),
                source_pixels_xy=tuple(
                    self.array(item) for item in state["source_pixels_xy"]
                ),
                geometry=state["geometry"],
                labels=self.maybe_array(state["labels"]),
                features=self.maybe_array(state["features"]),
                metadata=tuple(state["metadata"]),
            )
        if kind == "EquirectangularImage":
            from panorai.data import EquirectangularImage

            data = {name: self.array(value) for name, value in state["data"].items()}
            result = EquirectangularImage(
                data,
                shadow_angle=state["shadow_angle"],
                shadow_padded=state["shadow_padded"],
                lat=state["lat"],
                lon=state["lon"],
            )
            result._workflow_support = {
                name: self.array(value) for name, value in state["support"].items()
            }
            result._workflow_metadata = {
                name: self.field_metadata(value)
                for name, value in state["metadata"].items()
            }
            first_support = next(iter(result._workflow_support.values()))
            result.support_mask = (
                first_support.clone()
                if hasattr(first_support, "clone")
                else first_support.copy()
            )
            return result
        if kind == "GnomonicFaceSet":
            return self.face_set(state)
        raise ValueError(f"unknown typed cache result {kind!r}")

    def region_analysis_result(self, state: dict[str, Any]) -> Any:
        from panorai.data import (
            EnclosingSphericalCap,
            RegionAdjacencyEdge,
            RegionAdjacencyGraph,
            RegionAnalysisResult,
            RegionMeasurement,
            RegionMeasurements,
            SphericalBoundary,
            SphericalBoundarySet,
            SphericalContour,
            SphericalContourSet,
        )

        measurement_state = state["measurements"]
        measurements = RegionMeasurements(
            shape_hw=tuple(measurement_state["shape_hw"]),
            support_solid_angle_sr=measurement_state["support_solid_angle_sr"],
            regions=tuple(
                RegionMeasurement(
                    label=item["label"],
                    pixel_count=item["pixel_count"],
                    solid_angle_sr=item["solid_angle_sr"],
                    support_fraction=item["support_fraction"],
                    angular_perimeter_rad=item["angular_perimeter_rad"],
                    unsupported_boundary_rad=item["unsupported_boundary_rad"],
                    centroid_bearing_xyz=self.maybe_array(item["centroid_bearing_xyz"]),
                    enclosing_cap=EnclosingSphericalCap(
                        self.array(item["enclosing_cap"]["center_bearing_xyz"]),
                        item["enclosing_cap"]["radius_deg"],
                    ),
                )
                for item in measurement_state["regions"]
            ),
        )
        adjacency_state = state["adjacency"]
        adjacency = RegionAdjacencyGraph(
            labels=tuple(adjacency_state["labels"]),
            edges=tuple(
                RegionAdjacencyEdge(tuple(item["labels"]), item["shared_boundary_rad"])
                for item in adjacency_state["edges"]
            ),
        )
        boundary_state = state["boundaries"]
        boundaries = (
            None
            if boundary_state is None
            else SphericalBoundarySet(
                shape_hw=tuple(boundary_state["shape_hw"]),
                boundaries=tuple(
                    SphericalBoundary(
                        item["label"],
                        self.array(item["bearings_xyz"]),
                        item["angular_length_rad"],
                    )
                    for item in boundary_state["items"]
                ),
            )
        )
        contour_state = state["contours"]
        contours = (
            None
            if contour_state is None
            else SphericalContourSet(
                shape_hw=tuple(contour_state["shape_hw"]),
                contours=tuple(
                    SphericalContour(
                        item["label"],
                        self.array(item["bearings_xyz"]),
                        item["closed"],
                        item["angular_length_rad"],
                    )
                    for item in contour_state["items"]
                ),
            )
        )
        return RegionAnalysisResult(
            measurements=measurements,
            adjacency=adjacency,
            boundaries=boundaries,
            contours=contours,
            interface=state["interface"],
        )

    def relative_pose_result(self, state: dict[str, Any]) -> Any:
        from panorai.estimators import (
            DecoupledPoseReport,
            FivePointSamplingDiagnostics,
            ModelCompetitionReport,
            ModelEvidence,
            PoseStabilityReport,
            RelativePoseAcceptancePolicy,
            RelativePoseOptions,
            RelativePoseQualityReport,
            RelativePoseResult,
            TranslationOrientationReport,
        )

        metadata = state["metadata"]
        quality_values = dict(metadata["quality_report"])
        orientation = TranslationOrientationReport(
            **quality_values.pop("translation_orientation")
        )
        stability = PoseStabilityReport(**quality_values.pop("stability"))
        competition_values = dict(quality_values.pop("model_competition"))
        competition = ModelCompetitionReport(
            essential=ModelEvidence(**competition_values.pop("essential")),
            rotation_only=ModelEvidence(**competition_values.pop("rotation_only")),
            spherical_homography=ModelEvidence(
                **competition_values.pop("spherical_homography")
            ),
            **competition_values,
        )
        quality_values["rejection_reasons"] = tuple(quality_values["rejection_reasons"])
        quality = RelativePoseQualityReport(
            translation_orientation=orientation,
            stability=stability,
            model_competition=competition,
            **quality_values,
        )
        sampling_values = dict(metadata["sampling_diagnostics"])
        sampling_values["sampler_configuration"] = tuple(
            sorted(sampling_values["sampler_configuration"].items())
        )
        decoupled = metadata["decoupled_pose_report"]
        return RelativePoseResult(
            rotation=self.array(state["rotation"]),
            translation_direction=self.array(state["translation_direction"]),
            essential_matrix=self.array(state["essential_matrix"]),
            inlier_mask=self.array(state["inlier_mask"]),
            residuals_rad=self.array(state["residuals_rad"]),
            num_inliers=metadata["num_inliers"],
            num_trials=metadata["num_trials"],
            median_parallax_deg=metadata["median_parallax_deg"],
            cheirality_ratio=metadata["cheirality_ratio"],
            degenerate=metadata["degenerate"],
            degeneracy_reasons=tuple(metadata["degeneracy_reasons"]),
            sampling_diagnostics=FivePointSamplingDiagnostics(**sampling_values),
            quality_report=quality,
            acceptance_policy=RelativePoseAcceptancePolicy(
                **metadata["acceptance_policy"]
            ),
            options=RelativePoseOptions(**metadata["options"]),
            compute_backend=metadata["compute_backend"],
            interface=metadata["interface"],
            minimal_solver=metadata["minimal_solver"],
            robust_estimator=metadata["robust_estimator"],
            consensus_refit_steps=metadata["consensus_refit_steps"],
            decoupled_pose_report=(
                None if decoupled is None else DecoupledPoseReport(**decoupled)
            ),
        )

    def field_metadata(self, state: dict[str, Any]) -> dict[str, Any]:
        from panorai.data.fields import RasterFieldSpec

        spec = dict(state["field_spec"])
        spec["component_ids"] = tuple(spec["component_ids"])
        return {
            "kind": state["kind"],
            "units": state["units"],
            "interpolation": state["interpolation"],
            "validity": self.array(state["validity"]),
            "field_spec": RasterFieldSpec(**spec),
        }

    def face_set(self, state: dict[str, Any]) -> Any:
        from panorai.data import GnomonicFace, GnomonicFaceSet

        faces = []
        for item in state["faces"]:
            spec = self.spec(item["spec"])
            data = {name: self.array(value) for name, value in item["data"].items()}
            face = GnomonicFace(
                data,
                spec.center_lat_deg,
                spec.center_lon_deg,
                spec.hfov_deg,
                face_id=item["face_id"],
            )
            face.spec = spec
            face._erp_shape_hw = tuple(item["erp_shape_hw"])
            face._workflow_support = {
                name: self.array(value) for name, value in item["support"].items()
            }
            face._workflow_metadata = {
                name: self.field_metadata(value)
                for name, value in item["metadata"].items()
            }
            face.support_mask = next(iter(face._workflow_support.values())).copy()
            faces.append(face)
        result = GnomonicFaceSet(faces, channel_name=state["channel_name"])
        workflow = state["workflow"]
        result._workflow = {
            "layout": workflow["layout"],
            "order": list(workflow["order"]),
            "erp_shape": tuple(workflow["erp_shape"]),
            "specs": [self.spec(item) for item in workflow["specs"]],
            "depth_policy": workflow["depth_policy"],
            "min_valid_weight": workflow["min_valid_weight"],
            "source_metadata": {
                name: self.field_metadata(value)
                for name, value in workflow["source_metadata"].items()
            },
            "projector": None,
        }
        return result


def encode_typed(value: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    encoder = _Encoder()
    state = encoder.encode(value)
    return encoder.arrays, state


def decode_typed(arrays: dict[str, np.ndarray], state: dict[str, Any]) -> Any:
    decoder = _Decoder(arrays)
    result = decoder.decode(state)
    if decoder.used != set(arrays):
        raise ValueError("typed cache array inventory contains unreferenced arrays")
    return result


__all__ = ["decode_typed", "encode_typed"]
