"""
gnomonic_imageset.py
====================

Implements the GnomonicFaceSet class, which represents a collection of
gnomonic faces (GnomonicFace objects) and allows easy batch operations
and blending back into an equirectangular image.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import TYPE_CHECKING, List, Callable, Iterator, Tuple, Union
import inspect
import numpy as np

if TYPE_CHECKING:
    from .equirectangular_image import EquirectangularImage
    from .gnomonic_image import GnomonicFace


class GnomonicFaceSet(Iterator):
    """
    Represents a collection of GnomonicFace objects.

    - Supports multi-channel data among the faces.
    - Can attach blending methods to reconstruct an equirectangular image.
    - Allows easy iteration, addition, and transformation of faces.
    """

    def __init__(
        self,
        faces: Union[List["GnomonicFace"], None] = None,
        channel_name: str = "default",
    ):
        """Initialize a :class:`GnomonicFaceSet`.

        Args:
            faces: Optional list of :class:`GnomonicFace` objects.
            channel_name: Label used to identify the data channel.

        Examples:
            >>> fs = GnomonicFaceSet([])
        """
        self._faces: List[GnomonicFace] = faces if faces else []
        self.channel_name = channel_name
        self._index = 0
        self._workflow = None
        self._batch_back_plans = OrderedDict()
        self._batch_back_plan_lock = RLock()

        # Attach a default blender
        self.blender = None
        self.attach_blender("average")

    def __iter__(self):
        """Return an independent iterator, allowing nested/repeated iteration."""
        return iter(self._faces)

    def __next__(self) -> "GnomonicFace":
        """Preserve the legacy direct ``next(face_set)`` protocol."""

        if self._index >= len(self._faces):
            raise StopIteration
        face = self._faces[self._index]
        self._index += 1
        return face

    def __len__(self):
        """Returns number of faces in the set."""
        return len(self._faces)

    def __getitem__(self, idx: int) -> "GnomonicFace":
        """Enable indexed access to faces."""
        return self._faces[idx]

    def __repr__(self):
        if self._workflow is not None:
            modalities = tuple(self._faces[0]._workflow_metadata) if self._faces else ()
            return (
                f"GnomonicFaceSet(layout={self._workflow['layout']!r}, "
                f"faces={len(self._faces)}, modalities={modalities})"
            )
        return f"GnomonicFaceSet(channel={self.channel_name}, faces={len(self._faces)})"

    def _require_workflow(self):
        if self._workflow is None or not self._faces:
            raise TypeError(
                "This face set was not created by EquirectangularImage.views()"
            )
        return self._workflow

    def map(
        self,
        model,
        *,
        input=None,
        output=None,
        units=None,
        replace=False,
    ):
        """Apply ``model`` independently to one modality of every face.

        The model receives an array/tensor and may return either an array/tensor
        or ``(array, boolean_validity_mask)``. The original set is unchanged.
        """

        from ._workflow import (
            MODALITIES,
            clone_array,
            copy_metadata,
            support_for_value,
            validate_model_result,
        )
        from .gnomonic_image import GnomonicFace

        workflow = self._require_workflow()
        available = list(self._faces[0]._workflow_metadata)
        if input is None:
            if len(available) != 1:
                raise ValueError(
                    "input is required when mapping a multimodal face set; "
                    f"choose one of {available}"
                )
            input_name = available[0]
        else:
            input_name = input
        if input_name not in available:
            raise KeyError(f"unknown input modality {input_name!r}")
        output_name = input_name if output is None else output
        if output_name not in MODALITIES:
            raise ValueError(f"output must be one of {MODALITIES}")
        if units is not None and output_name != "depth":
            raise ValueError("units is only valid for depth output")
        if output_name != input_name and output_name in available and not replace:
            raise ValueError(
                f"output modality {output_name!r} already exists; pass replace=True"
            )

        new_faces = []
        for face in self._faces:
            source_data = face._workflow_data()
            result_value, result_validity = validate_model_result(
                model(source_data[input_name]),
                source_data[input_name],
                output=output_name,
                support=face._workflow_support[input_name],
            )
            support = support_for_value(
                face._workflow_support[input_name], result_value
            )
            result_validity = result_validity & support
            face_data = {
                name: clone_array(value) for name, value in source_data.items()
            }
            face_metadata = copy_metadata(face._workflow_metadata)
            face_support = {
                name: clone_array(mask) for name, mask in face._workflow_support.items()
            }
            face_data[output_name] = result_value
            face_metadata[output_name] = {
                "kind": output_name,
                "units": (
                    face._workflow_metadata[input_name]["units"]
                    if output is None and units is None
                    else (units if output_name == "depth" else None)
                ),
                "validity": result_validity,
                "interpolation": "nearest" if output_name == "labels" else "bilinear",
            }
            face_support[output_name] = clone_array(support)
            new_face = GnomonicFace(
                face_data,
                face.lat,
                face.lon,
                face.fov,
                hfov_deg=face.hfov_deg,
                vfov_deg=face.vfov_deg,
                roll_deg=face.roll_deg,
                face_id=face.face_id,
            )
            new_face.spec = face.spec
            new_face._erp_shape_hw = face._erp_shape_hw
            new_face._workflow_metadata = face_metadata
            new_face._workflow_support = face_support
            new_face.support_mask = clone_array(next(iter(face_support.values())))
            new_faces.append(new_face)
        result_set = GnomonicFaceSet(new_faces, channel_name=self.channel_name)
        result_set._workflow = dict(workflow)
        result_set._workflow["source_metadata"] = copy_metadata(
            new_faces[0]._workflow_metadata
        )
        return result_set

    def reconstruct(self, eq_shape=None, *, blend=None, modalities="all"):
        """Reconstruct selected modalities into a new panorama.

        Workflow-created sets infer their original ERP shape. Manually created
        sets must supply ``eq_shape`` and use the legacy single-bundle path.
        """

        if self._workflow is None:
            if eq_shape is None:
                raise ValueError("eq_shape is required for manually created face sets")
            return self.to_equirectangular(eq_shape, blend_method=blend)

        from ._workflow import (
            back_project_modality,
            blend_reprojected,
            clone_array,
            copy_metadata,
            resolved_blends,
            union_masks,
        )
        from .equirectangular_image import EquirectangularImage

        workflow = self._require_workflow()
        target_shape = workflow["erp_shape"] if eq_shape is None else tuple(eq_shape)
        if tuple(target_shape) != tuple(workflow["erp_shape"]):
            raise ValueError(
                "workflow-created sets reconstruct to their recorded ERP shape; "
                "create a manual set for a different eq_shape"
            )
        metadata = self._faces[0]._workflow_metadata
        if modalities == "all":
            selected = list(metadata)
        elif isinstance(modalities, str):
            selected = [modalities]
        else:
            selected = list(modalities)
        if not selected:
            raise ValueError("modalities cannot be empty")
        unknown = [name for name in selected if name not in metadata]
        if unknown:
            raise KeyError(f"unknown modalities: {unknown}")
        blends = resolved_blends(selected, metadata, blend)
        output_data = {}
        output_meta = {}
        for name in selected:
            face_values = [face._workflow_data()[name] for face in self._faces]
            face_validity = [
                face._workflow_metadata[name]["validity"] for face in self._faces
            ]
            specs = [face.spec for face in self._faces]
            if isinstance(blends[name], str) and blends[name] in {
                "average",
                "closest",
                "gaussian",
            }:
                from ._workflow import (
                    back_project_modality_sparse,
                    blend_sparse_reprojected,
                    sparse_support_union,
                )

                plan = self._get_batch_back_plan(target_shape, face_values, specs)
                contributions = back_project_modality_sparse(
                    face_values,
                    face_validity,
                    plan,
                    kind=metadata[name]["kind"],
                    depth_policy=workflow["depth_policy"],
                    min_valid_weight=workflow["min_valid_weight"],
                )
                fused, fused_validity = blend_sparse_reprojected(
                    contributions,
                    face_values[0],
                    target_shape,
                    blends[name],
                )
                fused_support = sparse_support_union(plan, face_values[0])
            else:
                values = []
                masks = []
                supports = []
                for face, value, valid in zip(
                    self._faces, face_values, face_validity, strict=True
                ):
                    projected, support, projected_valid = back_project_modality(
                        value,
                        valid,
                        face.spec,
                        target_shape,
                        kind=metadata[name]["kind"],
                        depth_policy=workflow["depth_policy"],
                        min_valid_weight=workflow["min_valid_weight"],
                        projector_template=workflow["projector"],
                    )
                    values.append(projected)
                    masks.append(projected_valid)
                    supports.append(support)
                fused, fused_validity = blend_reprojected(
                    values, masks, specs, target_shape, blends[name]
                )
                fused_support = union_masks(supports)
            output_data[name] = fused
            output_meta[name] = dict(metadata[name])
            output_meta[name]["validity"] = fused_validity
            output_meta[name]["support"] = fused_support
            output_meta[name]["blend"] = (
                blends[name]
                if isinstance(blends[name], str)
                else type(blends[name]).__name__
            )
        result = EquirectangularImage(output_data)
        result._workflow_metadata = copy_metadata(output_meta)
        for name in output_meta:
            if "blend" in output_meta[name]:
                result._workflow_metadata[name]["blend"] = output_meta[name]["blend"]
        result._workflow_support = {
            name: clone_array(output_meta[name]["support"]) for name in output_meta
        }
        result.support_mask = clone_array(output_meta[selected[0]]["support"])
        return result

    def _get_batch_back_plan(self, output_shape, values, specs):
        """Return a four-entry per-set LRU plan for an arbitrary N-view batch."""

        from panorai.geometry._engine import (
            _gnomonic_batch_back_plan,
            _is_torch,
        )

        first = values[0]
        torch_backend = _is_torch(first)
        face_shapes = tuple(
            tuple(value.shape[-2:] if torch_backend else value.shape[:2])
            for value in values
        )
        device = str(first.device) if torch_backend else "cpu"
        key = (
            tuple(specs),
            tuple(output_shape),
            face_shapes,
            "torch" if torch_backend else "numpy",
            device,
        )
        with self._batch_back_plan_lock:
            try:
                plan = self._batch_back_plans.pop(key)
            except KeyError:
                plan = _gnomonic_batch_back_plan(
                    specs, tuple(output_shape), face_shapes, first
                )
                if len(self._batch_back_plans) >= 4:
                    self._batch_back_plans.popitem(last=False)
            self._batch_back_plans[key] = plan
            return plan

    def describe(self):
        """Return a JSON-friendly record of the exact workflow choices."""

        from ._workflow import (
            WORKFLOW_CONTRACT,
            WORKFLOW_INTERFACE,
            WORKFLOW_STABILITY,
            array_description,
            resolved_blends,
        )

        workflow = self._require_workflow()
        metadata = self._faces[0]._workflow_metadata
        data = self._faces[0]._workflow_data()
        blends = resolved_blends(list(metadata), metadata, None)
        specs = [
            {
                "center_lat_deg": spec.center_lat_deg,
                "center_lon_deg": spec.center_lon_deg,
                "hfov_deg": spec.hfov_deg,
                "vfov_deg": spec.vfov_deg,
                "roll_deg": spec.roll_deg,
                "output_shape_hw": spec.output_shape_hw,
            }
            for spec in workflow["specs"]
        ]
        return {
            "contract": WORKFLOW_CONTRACT,
            "interface": WORKFLOW_INTERFACE,
            "stability": WORKFLOW_STABILITY,
            "layout": workflow["layout"],
            "view_count": len(self._faces),
            "view_order": list(workflow["order"]),
            "erp_shape_hw": tuple(workflow["erp_shape"]),
            "view_shape_hw": specs[0]["output_shape_hw"],
            "specs": specs,
            "backend": array_description(next(iter(data.values())))["backend"],
            "device": array_description(next(iter(data.values())))["device"],
            "depth_policy": workflow["depth_policy"],
            "min_valid_weight": workflow["min_valid_weight"],
            "projector": (
                "GnomonicProjector"
                if workflow["projector"] is None
                else type(workflow["projector"]).__name__
            ),
            "modalities": {
                name: {
                    **array_description(data[name]),
                    "kind": item["kind"],
                    "units": item["units"],
                    "interpolation": item["interpolation"],
                    "blend": blends[name],
                    "validity_shape": tuple(item["validity"].shape),
                    "support_shape": tuple(
                        self._faces[0]._workflow_support[name].shape
                    ),
                }
                for name, item in metadata.items()
            },
        }

    def add_face(self, face: "GnomonicFace"):
        """
        Add a gnomonic face to this set.

        Args:
            face: The :class:`GnomonicFace` to add.

        Examples:
            >>> fs.add_face(face)
        """
        self._faces.append(face.clone())
        with self._batch_back_plan_lock:
            self._batch_back_plans.clear()

    def get_faces(self) -> List["GnomonicFace"]:
        """
        Returns the list of all gnomonic faces.

        Returns:
            List[GnomonicFace]

        Examples:
            >>> faces = fs.get_faces()
        """
        return self._faces

    def apply_to_all(self, func: Callable[["GnomonicFace"], None]):
        """
        Applies a given function to each face in the set.

        Args:
            func: Function that accepts a :class:`GnomonicFace`.

        Examples:
            >>> fs.apply_to_all(lambda f: f.show())
        """
        for face in self._faces:
            func(face)

    def attach_blender(self, name: str, **kwargs):
        """
        Dynamically attach a named blender for reconstructing an equirectangular image.

        Args:
            name: Blender name such as ``"average"``.
            **kwargs: Additional blender configuration.

        Examples:
            >>> fs.attach_blender("average")
        """
        try:
            from panorai.factory.panorai_factory import PanoraiFactory
        except ModuleNotFoundError:
            # Optional dependency missing during lightweight unit tests.
            self.blender = None
            return
        self.blender = PanoraiFactory.get_blender(name, **kwargs)

    def to_equirectangular(
        self,
        eq_shape: Tuple[int, int],
        preserve_dtype: bool = True,
        blend_method: Union[str, None] = None,
    ) -> "EquirectangularImage":
        """
        Convert the GnomonicFaceSet back into a single EquirectangularImage.

        - If multiple faces, uses a blender to merge them.
        - If only one face, returns the single converted face.

        Args:
            eq_shape: Target panorama shape ``(height, width)``.
            preserve_dtype: Keep the original dtype if ``True``.
            blend_method: If provided, attach or switch to this blender.

        Returns:
            EquirectangularImage

        Examples:
            >>> pano = fs.to_equirectangular((512, 1024))
        """

        if not self._faces:
            raise ValueError("No gnomonic faces available for back-projection.")

        if blend_method:
            self.attach_blender(blend_method)

        selective = self._try_legacy_selective_average(eq_shape, preserve_dtype)
        if selective is not None:
            return selective

        # Convert each face and retain geometric support independently from
        # numeric pixel values (black and zero may be valid data).
        projected = [
            face.to_equirectangular(eq_shape, return_mask=True) for face in self._faces
        ]
        eq_faces = [item[0] for item in projected]
        support_masks = [item[1] for item in projected]

        if len(eq_faces) == 1:
            eq_faces[0].support_mask = support_masks[0]
            return eq_faces[0]

        # Blend multiple equirectangular images
        return self.blend_channels(
            eq_faces, preserve_dtype, self.blender, masks=support_masks
        )

    def _legacy_batch_back_plan(self, eq_shape):
        """Build compact OpenCV maps matching the frozen 3.x projection path."""

        configs = [face.projection.config for face in self._faces]
        key = (
            "legacy",
            tuple(eq_shape),
            tuple(
                (
                    config.phi1_deg,
                    config.lam0_deg,
                    config.fov_deg,
                    config.R,
                    config.x_points,
                    config.y_points,
                    config.lon_min,
                    config.lon_max,
                    config.lat_min,
                    config.lat_max,
                )
                for config in configs
            ),
        )
        with self._batch_back_plan_lock:
            try:
                plan = self._batch_back_plans.pop(key)
            except KeyError:
                height, width = eq_shape
                grids = {}
                faces = []
                for config in configs:
                    bounds = (
                        config.lon_min,
                        config.lon_max,
                        config.lat_min,
                        config.lat_max,
                    )
                    try:
                        phi, lam = grids[bounds]
                    except KeyError:
                        lon_grid, lat_grid = np.meshgrid(
                            np.linspace(config.lon_min, config.lon_max, width),
                            np.linspace(config.lat_max, config.lat_min, height),
                        )
                        phi = np.deg2rad(lat_grid).ravel()
                        lam = np.deg2rad(lon_grid).ravel()
                        grids[bounds] = (phi, lam)
                    phi1 = np.deg2rad(config.phi1_deg)
                    lam0 = np.deg2rad(config.lam0_deg)
                    cos_c = np.sin(phi1) * np.sin(phi) + np.cos(phi1) * np.cos(
                        phi
                    ) * np.cos(lam - lam0)
                    safe = np.where(cos_c == 0, 1e-10, cos_c)
                    plane_x = config.R * np.cos(phi) * np.sin(lam - lam0) / safe
                    plane_y = (
                        config.R
                        * (
                            np.cos(phi1) * np.sin(phi)
                            - np.sin(phi1) * np.cos(phi) * np.cos(lam - lam0)
                        )
                        / safe
                    )
                    limit = np.tan(np.deg2rad(config.fov_deg) / 2.0) * config.R
                    map_x = (plane_x + limit) / (2.0 * limit) * (config.x_points - 1)
                    map_y = (limit - plane_y) / (2.0 * limit) * (config.y_points - 1)
                    support = (
                        (cos_c > 0)
                        & np.isfinite(map_x)
                        & np.isfinite(map_y)
                        & (map_x >= -0.5)
                        & (map_x <= config.x_points - 0.5)
                        & (map_y >= -0.5)
                        & (map_y <= config.y_points - 0.5)
                    )
                    indices = np.flatnonzero(support).astype(np.int32, copy=False)
                    compact_x = map_x[indices].astype(np.float32, copy=False)
                    compact_y = map_y[indices].astype(np.float32, copy=False)
                    indices.flags.writeable = False
                    compact_x.flags.writeable = False
                    compact_y.flags.writeable = False
                    faces.append((indices, compact_x, compact_y))
                plan = tuple(faces)
                if len(self._batch_back_plans) >= 4:
                    self._batch_back_plans.popitem(last=False)
            self._batch_back_plans[key] = plan
            return plan

    @staticmethod
    def _legacy_sparse_sample(face, value, face_plan):
        indices, map_x, map_y = face_plan
        value = np.asarray(value)
        if len(indices) == 0:
            return indices, np.empty((0, *value.shape[2:]), dtype=value.dtype)
        # OpenCV remap uses signed-short internal dimensions. Compact vectors
        # can exceed that limit even though both source rasters are valid.
        chunks = []
        for start in range(0, len(indices), 32_766):
            stop = min(start + 32_766, len(indices))
            part = face.projection.interpolation.interpolate(
                value,
                map_x[start:stop, None],
                map_y[start:stop, None],
            )
            if part.shape[:2] != (stop - start, 1):
                # Some compatibility tests inject an interpolation double that
                # does not implement remap geometry. Preserve that legacy path.
                return None
            part = np.squeeze(part, axis=1)
            if value.ndim == 3 and part.ndim == 1:
                part = part[:, None]
            chunks.append(part)
        sampled = np.concatenate(chunks, axis=0)
        return indices, sampled

    def _try_legacy_selective_average(self, eq_shape, preserve_dtype):
        """Fast path for the default N-view legacy sampler reconstruction."""

        from .equirectangular_image import EquirectangularImage

        eq_shape = tuple(eq_shape[:2])
        if len(self._faces) < 2 or self.blender is None:
            return None
        try:
            from panorai.blenders.average import AverageBlender
        except ImportError:
            return None
        if type(self.blender) is not AverageBlender:
            return None
        if any(
            face.projection is None
            or getattr(face.projection, "_canonical_projector", None) is not None
            for face in self._faces
        ):
            return None
        first = self._faces[0]
        multi_channel = first.is_multi_channel()
        if any(face.is_multi_channel() != multi_channel for face in self._faces):
            return None
        if multi_channel:
            channel_names = first.get_channels()
            if any(face.get_channels() != channel_names for face in self._faces[1:]):
                return None
        else:
            channel_names = [None]

        plan = self._legacy_batch_back_plan(tuple(eq_shape))
        pixel_count = eq_shape[0] * eq_shape[1]
        union = np.zeros(pixel_count, dtype=bool)
        for indices, _, _ in plan:
            union[indices] = True

        outputs = {}
        for channel in channel_names:
            first_value = first.data if channel is None else first.data[channel]
            trailing = tuple(first_value.shape[2:])
            combined = np.zeros((pixel_count, *trailing), dtype=np.float32)
            counts = np.zeros(pixel_count, dtype=np.float32)
            for face_index, (face, face_plan) in enumerate(
                zip(self._faces, plan, strict=True)
            ):
                value = face.data if channel is None else face.data[channel]
                sampled_result = self._legacy_sparse_sample(face, value, face_plan)
                if sampled_result is None:
                    return None
                indices, sampled = sampled_result
                if not np.isfinite(sampled).all():
                    raise ValueError(
                        f"Image {face_index} has non-finite values marked as valid."
                    )
                combined[indices] += sampled
                counts[indices] += 1.0
            denominator = counts if not trailing else counts[..., None]
            np.divide(combined, denominator, out=combined, where=denominator > 0)
            if preserve_dtype:
                combined = combined.astype(first_value.dtype)
            outputs[channel] = combined.reshape((*eq_shape, *trailing))

        data = outputs[None] if channel_names == [None] else outputs
        image = EquirectangularImage(data, support_mask=union.reshape(eq_shape))
        image.multi_channel_handler.squeeze_singleton_channels()
        return image

    def blend_channels(
        self,
        projected_faces: List["EquirectangularImage"],
        preserve_dtype: bool,
        blender: Callable,
        masks: Union[List[np.ndarray], None] = None,
    ) -> "EquirectangularImage":
        """
        Blend multiple equirectangular images channel-wise.

        Args:
            projected_faces: The images to blend.
            preserve_dtype: Whether to cast back to the original dtype.
            blender: Blender object with a ``blend`` method.
            masks: Explicit spatial validity mask for each projected face.

        Returns:
            EquirectangularImage: The blended panorama.

        Examples:
            >>> blended = fs.blend_channels(faces, True, blender)
        """
        from .equirectangular_image import EquirectangularImage

        first_face = projected_faces[0]
        if masks is None:
            masks = [getattr(face, "support_mask", None) for face in projected_faces]
        if len(masks) != len(projected_faces) or any(mask is None for mask in masks):
            raise ValueError(
                "Each projected face must provide an explicit support mask."
            )

        output_mask = np.logical_or.reduce(
            np.stack([np.asarray(mask, dtype=bool) for mask in masks], axis=0),
            axis=0,
        )

        def blend(arrays):
            parameters = inspect.signature(blender.blend).parameters.values()
            supports_return_mask = any(
                parameter.name == "return_mask"
                or parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
            result = (
                blender.blend(arrays, masks, return_mask=True)
                if supports_return_mask
                else blender.blend(arrays, masks)
            )
            if isinstance(result, tuple) and len(result) == 2:
                return result
            return result, output_mask

        # Single-channel vs multi-channel
        if first_face.is_multi_channel():
            # Multi-channel blending
            blended_dict = {}
            all_channels = first_face.get_channels()
            for ch in all_channels:
                # Gather arrays for the same channel from each face
                channel_arrays = [pf.data[ch] for pf in projected_faces]
                # Blend them
                blended, channel_mask = blend(channel_arrays)
                output_mask &= np.asarray(channel_mask, dtype=bool)
                # Preserve dtype if needed
                if preserve_dtype:
                    blended = blended.astype(channel_arrays[0].dtype)
                blended_dict[ch] = blended
            img = EquirectangularImage(blended_dict, support_mask=output_mask)
            img.multi_channel_handler.squeeze_singleton_channels()
            return img
        else:
            # Single-channel data
            inputs = [face.data for face in projected_faces]
            blended_data, output_mask = blend(inputs)
            if preserve_dtype:
                blended_data = blended_data.astype(inputs[0].dtype)
            img = EquirectangularImage(blended_data, support_mask=output_mask)
            img.multi_channel_handler.squeeze_singleton_channels()
            return img

    def to_pcd(
        self,
        model=None,
        depth: np.ndarray = None,
        eq_shape: Tuple[int, int] = (512, 1024),
        grad_threshold: float = 0.1,
        min_radius: float = 0.0,
        max_radius: float = 10.0,
        blender_name: str = "simple",
    ):
        """
        Convert this entire GnomonicFaceSet into a single, merged PCD object.

        Args:
            model: Model used to infer depth.
            depth: Optional depth array used instead of ``model``.
            eq_shape: Resolution for back-projection prior to PCD.
            grad_threshold: Gradient threshold in PCD creation.
            min_radius: Minimum radius in the PCD.
            max_radius: Maximum radius in the PCD.
            blender_name: Which blender to use before creating the PCD.

        Returns:
            A PCD object from PCDHandler (adjust for real code).

        Examples:
            >>> pcd = fs.to_pcd(model=my_model)
        """
        if (not model) & (not isinstance(depth, np.ndarray)):
            raise ValueError(
                'You need to pass either a monocular depth estimation model as "model" or a numpy array as depth.'
            )
        else:
            if len(eq_shape) < 2:
                raise ValueError("eq_shape must have at least two dimensions")
            eq_shape = eq_shape[:2]

            from ..pcd.handler import PCDHandler

            return PCDHandler.gnomonic_faceset_to_pcd(
                model=model,
                depth=depth,
                faceset=self,
                eq_shape=eq_shape,
                grad_threshold=grad_threshold,
                min_radius=min_radius,
                max_radius=max_radius,
                blender_name=blender_name,
            )

    def clone(self) -> "GnomonicFaceSet":
        """
        Create a deep copy of this GnomonicFaceSet, including attached blender.

        Returns:
            GnomonicFaceSet

        Examples:
            >>> fs_copy = fs.clone()
        """
        new_set = GnomonicFaceSet(
            faces=[face.clone() for face in self._faces], channel_name=self.channel_name
        )
        new_set.blender = self.blender
        if self._workflow is not None:
            from ._workflow import copy_metadata

            new_set._workflow = dict(self._workflow)
            new_set._workflow["source_metadata"] = copy_metadata(
                self._workflow["source_metadata"]
            )
        return new_set
