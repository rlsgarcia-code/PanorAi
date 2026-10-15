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
        from panorai.data import EquirectangularImage, GnomonicFaceSet

        from ._fields import RasterField, ResultBundle
        from ._observations import (
            ClassificationResult,
            EmbeddingResult,
            PerViewResults,
            SphericalObservations,
            ViewObservations,
        )

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
        from ._fields import RasterField, ResultBundle
        from ._observations import (
            ClassificationResult,
            EmbeddingResult,
            PerViewResults,
            SphericalObservations,
            ViewObservations,
        )

        kind = state["type"]
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
