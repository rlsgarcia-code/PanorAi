"""
equirectangular_image.py
========================

Implements the EquirectangularImage class, which represents a panoramic
equirectangular image (potentially multi-channel).
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
from PIL import Image  # only if needed for internal usage

from .spherical_data import SphericalData

from panorai.preprocessing.preprocessor import (
    Preprocessor,
)  # assumed import in original code

if TYPE_CHECKING:
    from .gnomonic_image import GnomonicFace
    from .gnomonic_imageset import GnomonicFaceSet


def _validate_shadow_state(
    shadow_angle: float, shadow_padded: bool
) -> tuple[float, bool]:
    if isinstance(shadow_angle, bool) or not isinstance(shadow_angle, Real):
        raise TypeError("shadow_angle must be a finite real number")
    angle = float(shadow_angle)
    if not math.isfinite(angle) or not 0.0 <= angle < 180.0:
        raise ValueError("shadow_angle must be finite and in the interval [0, 180)")
    if not isinstance(shadow_padded, (bool, np.bool_)):
        raise TypeError("shadow_padded must be boolean")
    padded = bool(shadow_padded)
    if padded and angle == 0.0:
        raise ValueError("shadow_padded=True requires shadow_angle > 0")
    return angle, padded


def _apply_shadow_support(mask, shadow_angle: float):
    """Return ``mask`` with the materialized south-polar cap unsupported."""

    result = (
        mask.clone()
        if type(mask).__module__.startswith("torch")
        else np.asarray(mask).copy()
    )
    observed_height = int(round(result.shape[-2] * (1.0 - float(shadow_angle) / 180.0)))
    result[..., observed_height:, :] = False
    return result


class EquirectangularImage(SphericalData):
    """
    Represents an equirectangular image with optional multi-channel support.

    - Inherits from SphericalData, which extends multi-channel capabilities.
    - Allows attaching samplers and projection transforms dynamically.
    - Can be converted into one or multiple GnomonicFaces.
    """

    def __init__(
        self,
        data: Union[np.ndarray, dict],
        shadow_angle: float = 0.0,
        lat: float = 0.0,
        lon: float = 0.0,
        support_mask: Optional[np.ndarray] = None,
        *,
        valid=None,
        shadow_padded: bool = False,
    ) -> None:
        """Initialize an :class:`EquirectangularImage`.

        Args:
            data: Input array or dictionary of channel arrays.
            shadow_angle: Angle used for shadow correction.
            lat: Latitude of the image centre in degrees.
            lon: Longitude of the image centre in degrees.
            support_mask: Optional explicit geometric source support. NumPy
                uses ``(H, W)``; Torch also accepts ``(N, H, W)`` for NCHW.
            valid: Optional explicit image validity for the Experimental
                workflow. Absence means valid throughout source support.
            shadow_padded: Whether the south-polar region described by
                ``shadow_angle`` is already present as rows in ``data``.

        Examples:
            >>> img = EquirectangularImage(np.zeros((512, 1024, 3)))
        """
        shadow_angle, shadow_padded = _validate_shadow_state(
            shadow_angle, shadow_padded
        )
        super().__init__(data, lat, lon)
        self.shadow_angle = shadow_angle
        self.shadow_padded = shadow_padded
        self._workflow_metadata = None
        self._workflow_support = None
        is_array = isinstance(self.data, np.ndarray) or (
            type(self.data).__module__ == "torch"
            or type(self.data).__module__.startswith("torch.")
        )
        if support_mask is None:
            self.support_mask = None
        else:
            if not is_array:
                legacy_support = np.asarray(support_mask, dtype=bool)
                expected = self.shape[:2]
                if legacy_support.shape != expected:
                    raise ValueError(
                        f"support_mask must have shape {expected}; "
                        f"got {legacy_support.shape}"
                    )
                self.support_mask = legacy_support.copy()
            else:
                from ._workflow import validate_support_mask

                self.support_mask = validate_support_mask(self.data, support_mask)

        if self.shadow_padded:
            if self.support_mask is None:
                if is_array:
                    from ._workflow import ones_mask

                    self.support_mask = ones_mask(self.data)
                else:
                    self.support_mask = np.ones(self.shape[:2], dtype=bool)
            self.support_mask = _apply_shadow_support(
                self.support_mask, self.shadow_angle
            )

        if is_array:
            from ._workflow import build_primary_metadata, clone_array, ones_mask

            self._workflow_metadata = build_primary_metadata(self.data, valid=valid)
            source_support = (
                ones_mask(self.data) if self.support_mask is None else self.support_mask
            )
            self._workflow_support = {"image": clone_array(source_support)}
        elif valid is not None:
            raise TypeError(
                "valid requires array data; legacy dictionaries have no declared semantics"
            )

        # Attach default sampler and projection
        self.sampler = None
        self.projection = None
        self.attach_sampler("cube")
        self.attach_projection("gnomonic")

    def _require_workflow(self):
        if self._workflow_metadata is None:
            raise TypeError(
                "The ergonomic workflow requires semantically typed modalities. "
                "Legacy dictionaries remain supported by the existing to_* methods; "
                "start with EquirectangularImage(image) and add with_depth()/with_labels()."
            )
        return self._workflow_metadata

    def _workflow_data(self):
        metadata = self._require_workflow()
        if isinstance(self.data, dict):
            return self.data
        return {next(iter(metadata)): self.data}

    @property
    def image(self):
        """Return the image modality, or ``None`` when it is absent."""

        return self._workflow_data().get("image") if self._workflow_metadata else None

    @property
    def depth(self):
        """Return radial-range depth, or ``None`` when it is absent."""

        return self._workflow_data().get("depth") if self._workflow_metadata else None

    @property
    def labels(self):
        """Return categorical labels, or ``None`` when they are absent."""

        return self._workflow_data().get("labels") if self._workflow_metadata else None

    def validity(self, modality: str):
        """Return a copy of the explicit validity mask for ``modality``."""

        from ._workflow import clone_array

        metadata = self._require_workflow()
        if modality not in metadata:
            raise KeyError(f"unknown modality {modality!r}")
        return clone_array(metadata[modality]["validity"])

    def with_depth(self, depth, *, valid=None, units="m"):
        """Return a new panorama with explicit radial-range depth attached."""

        from ._workflow import add_modality, clone_array, copy_metadata

        data, metadata = add_modality(
            self._workflow_data(),
            self._require_workflow(),
            "depth",
            depth,
            valid=valid,
            units=units,
        )
        result = EquirectangularImage(
            data,
            shadow_angle=self.shadow_angle,
            lat=self.lat,
            lon=self.lon,
            shadow_padded=self.shadow_padded,
        )
        result.support_mask = (
            None if self.support_mask is None else clone_array(self.support_mask)
        )
        result._workflow_metadata = copy_metadata(metadata)
        result._workflow_support = {
            name: clone_array(
                self._workflow_support.get(name, self._workflow_support["image"])
            )
            for name in metadata
        }
        result.sampler = self.sampler
        result.projection = self.projection
        return result

    def with_labels(self, labels):
        """Return a new panorama with integer or boolean labels attached."""

        from ._workflow import add_modality, clone_array, copy_metadata

        data, metadata = add_modality(
            self._workflow_data(),
            self._require_workflow(),
            "labels",
            labels,
            valid=None,
            units=None,
        )
        result = EquirectangularImage(
            data,
            shadow_angle=self.shadow_angle,
            lat=self.lat,
            lon=self.lon,
            shadow_padded=self.shadow_padded,
        )
        result.support_mask = (
            None if self.support_mask is None else clone_array(self.support_mask)
        )
        result._workflow_metadata = copy_metadata(metadata)
        result._workflow_support = {
            name: clone_array(
                self._workflow_support.get(name, self._workflow_support["image"])
            )
            for name in metadata
        }
        result.sampler = self.sampler
        result.projection = self.projection
        return result

    def views(
        self,
        layout="cube",
        *,
        size=None,
        fov=90.0,
        count=None,
        subdivisions=0,
        rotations=(),
        depth_policy="propagate",
        min_valid_weight=None,
        projector=None,
    ):
        """Create an immutable, modality-aware set of canonical gnomonic views.

        This method is part of the Stable ``panorai-object-workflow/v1``
        surface. ``min_valid_weight`` is mandatory when
        ``depth_policy='renormalize'`` and invalid otherwise.
        """

        from panorai.geometry import GnomonicSpec
        from ._workflow import (
            clone_array,
            copy_metadata,
            is_torch,
            normalize_fov,
            normalize_size,
            project_modality_batch,
            resolve_tangent_points,
            spatial_shape,
        )
        from .gnomonic_image import GnomonicFace
        from .gnomonic_imageset import GnomonicFaceSet

        metadata = self._require_workflow()
        self._require_shadow_padding()
        source_data = self._workflow_data()
        if depth_policy not in {"propagate", "renormalize"}:
            raise ValueError("depth_policy must be 'propagate' or 'renormalize'")
        if depth_policy == "renormalize" and min_valid_weight is None:
            raise ValueError(
                "min_valid_weight is required when depth_policy='renormalize'"
            )
        if depth_policy == "renormalize" and min_valid_weight is not None:
            from numbers import Real
            import math

            if isinstance(min_valid_weight, bool) or not isinstance(
                min_valid_weight, Real
            ):
                raise TypeError("min_valid_weight must be a finite real number")
            min_valid_weight = float(min_valid_weight)
            if not math.isfinite(min_valid_weight) or not 0.0 < min_valid_weight <= 1.0:
                raise ValueError(
                    "min_valid_weight must be finite and in the interval (0, 1]"
                )
        if depth_policy == "propagate" and min_valid_weight is not None:
            raise ValueError(
                "min_valid_weight is only valid when depth_policy='renormalize'"
            )
        output_shape = normalize_size(
            size, spatial_shape(next(iter(source_data.values())))
        )
        hfov, vfov = normalize_fov(fov)
        layout_name, tangent_points, order = resolve_tangent_points(
            layout,
            count=count,
            subdivisions=subdivisions,
            rotations=rotations,
        )
        specs = [
            GnomonicSpec(
                center_lat_deg=lat,
                center_lon_deg=lon,
                hfov_deg=hfov,
                vfov_deg=vfov,
                output_shape_hw=output_shape,
            )
            for lat, lon in tangent_points
        ]
        batch_plan = None
        first_source = next(iter(source_data.values()))
        if projector is None and not is_torch(first_source):
            from panorai.geometry._engine import _gnomonic_batch_forward_plan

            batch_plan = _gnomonic_batch_forward_plan(
                specs, spatial_shape(first_source), first_source
            )
        projected_modalities = {
            name: project_modality_batch(
                value,
                metadata[name],
                specs,
                depth_policy=depth_policy,
                min_valid_weight=min_valid_weight,
                projector_template=projector,
                source_support=self._workflow_support[name],
                plan=batch_plan,
            )
            for name, value in source_data.items()
        }
        faces = []
        for face_index, spec in enumerate(specs):
            face_data = {}
            face_meta = copy_metadata(metadata)
            supports = {}
            for name in source_data:
                projected, support, validity = projected_modalities[name][face_index]
                face_data[name] = projected
                face_meta[name]["validity"] = validity
                supports[name] = support
            face = GnomonicFace(
                face_data,
                spec.center_lat_deg,
                spec.center_lon_deg,
                spec.hfov_deg,
                hfov_deg=spec.hfov_deg,
                vfov_deg=spec.vfov_deg,
                roll_deg=spec.roll_deg,
                face_id=order[face_index],
            )
            face.spec = spec
            face._erp_shape_hw = spatial_shape(next(iter(source_data.values())))
            face._workflow_metadata = face_meta
            face._workflow_support = {
                name: clone_array(mask) for name, mask in supports.items()
            }
            face.support_mask = clone_array(next(iter(supports.values())))
            faces.append(face)
        face_set = GnomonicFaceSet(faces)
        face_set._workflow = {
            "layout": layout_name,
            "order": list(order),
            "erp_shape": spatial_shape(next(iter(source_data.values()))),
            "specs": list(specs),
            "depth_policy": depth_policy,
            "min_valid_weight": min_valid_weight,
            "source_metadata": copy_metadata(metadata),
            "projector": projector,
        }
        return face_set

    def process_views(
        self,
        model,
        layout="cube",
        *,
        size=None,
        fov=90.0,
        count=None,
        subdivisions=0,
        rotations=(),
        depth_policy="propagate",
        min_valid_weight=None,
        projector=None,
        input=None,
        output=None,
        units=None,
        replace=False,
        blend=None,
        modalities="all",
    ):
        """Run exactly ``views().map().reconstruct()`` with one call."""

        return (
            self.views(
                layout,
                size=size,
                fov=fov,
                count=count,
                subdivisions=subdivisions,
                rotations=rotations,
                depth_policy=depth_policy,
                min_valid_weight=min_valid_weight,
                projector=projector,
            )
            .map(
                model,
                input=input,
                output=output,
                units=units,
                replace=replace,
            )
            .reconstruct(blend=blend, modalities=modalities)
        )

    def attach_sampler(self, name: str, **kwargs):
        """
        Attach a named sampler for tangent points or other sampling strategies.

        Args:
            name: The sampler name (for example ``"cube"``).
            **kwargs: Additional sampler configuration.

        Examples:
            >>> img = EquirectangularImage(np.zeros((2, 4, 3)))
            >>> img.attach_sampler("cube")
        """
        try:
            from panorai.factory.panorai_factory import PanoraiFactory
        except Exception:
            # Optional dependency missing during lightweight unit tests
            # or additional import errors when the full package is not
            # available (e.g. during isolated unit tests).
            self.sampler = None
            return
        try:
            import panorai.samplers  # noqa: F401 - registers default samplers
        except Exception:
            pass
        try:
            self.sampler = PanoraiFactory.get_sampler(name, **kwargs)
        except Exception:
            # If the factory cannot provide the sampler (e.g., registries
            # haven't been populated), fall back to ``CubeSampler`` when
            # requesting the default 'cube' sampler.
            if name == "cube":
                try:
                    from panorai.samplers.default_samplers import CubeSampler

                    self.sampler = CubeSampler(**kwargs)
                except Exception:
                    self.sampler = None
            else:
                self.sampler = None

    def attach_projection(
        self, name: str, lat: float = 0.0, lon: float = 0.0, fov: float = 90.0, **kwargs
    ):
        """
        Attach a projection method used for converting equirectangular data
        to gnomonic or other coordinate systems.

        Args:
            name: Projection name such as ``"gnomonic"``.
            lat: Latitude of the projection centre.
            lon: Longitude of the projection centre.
            fov: Field of view in degrees.
            **kwargs: Additional projection configuration.

        Examples:
            >>> img = EquirectangularImage(np.zeros((2, 4, 3)))
            >>> img.attach_projection("gnomonic", lat=0.0, lon=0.0, fov=90)
        """
        try:
            from panorai.factory.panorai_factory import PanoraiFactory
        except Exception:
            # Optional dependency missing during lightweight unit tests or
            # additional import errors when the full package is unavailable.
            self.projection = None
            return
        self.projection = PanoraiFactory.get_projection(
            name, lat=lat, lon=lon, fov=fov, **kwargs
        )

    def preprocess(
        self,
        delta_lat: float = 0.0,
        delta_lon: float = 0.0,
        shadow_angle: Optional[float] = None,
        shadow_padded: Optional[bool] = None,
        resize_factor: Union[float, None] = None,
        preprocessing_config: dict = None,
    ):
        """
        Applies preprocessing transformations to the equirectangular image.

        - Adjust lat/lon
        - Update shadow angle
        - Resize, if requested
        - Additional custom preprocessing steps (from config)

        Args:
            delta_lat: Shift in latitude in degrees.
            delta_lon: Shift in longitude in degrees.
            shadow_angle: South-polar blind angle. ``None`` reuses the value
                already stored on the panorama.
            shadow_padded: Whether the current raster already contains the
                rows for ``shadow_angle``. ``None`` reuses the stored state.
            resize_factor: Factor by which to resize the image.
            preprocessing_config: Additional :class:`Preprocessor` configuration.

        Examples:
            >>> img = EquirectangularImage(np.zeros((2, 4, 3)))
            >>> img.preprocess(delta_lat=1.0, delta_lon=1.0)
        """

        requested_angle = self.shadow_angle if shadow_angle is None else shadow_angle
        input_padded = self.shadow_padded if shadow_padded is None else shadow_padded
        requested_angle, input_padded = _validate_shadow_state(
            requested_angle, input_padded
        )
        if self.shadow_padded and not input_padded:
            raise ValueError(
                "the panorama is already shadow-padded; refusing to apply padding twice"
            )
        if self.shadow_padded and not math.isclose(
            requested_angle, self.shadow_angle, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError(
                "cannot change shadow_angle after its padding has been materialized"
            )

        def _preprocess_func(x):
            return Preprocessor.preprocess_eq(
                x,
                delta_lat=delta_lat,
                delta_lon=delta_lon,
                shadow_angle=requested_angle,
                shadow_padded=input_padded,
                resize_factor=resize_factor,
                config=preprocessing_config,
            )

        def _preprocess_mask(mask):
            module = type(mask).__module__
            if module == "torch" or module.startswith("torch."):
                raise TypeError("preprocess currently supports NumPy panoramas only")
            import cv2

            processed = Preprocessor.preprocess_eq(
                np.asarray(mask, dtype=np.uint8),
                delta_lat=delta_lat,
                delta_lon=delta_lon,
                shadow_angle=requested_angle,
                shadow_padded=input_padded,
                resize_factor=resize_factor,
                resize_method="cv2",
                interpolation=cv2.INTER_NEAREST,
                config=preprocessing_config,
            )
            return processed.astype(bool, copy=False)

        support = self.support_mask
        if support is None and requested_angle > 0.0:
            support = np.ones(self.shape[:2], dtype=bool)
            if input_padded:
                support = _apply_shadow_support(support, requested_angle)
        workflow_support = None
        workflow_validity = None
        if self._workflow_support is not None:
            workflow_support = {
                name: (
                    _apply_shadow_support(mask, requested_angle)
                    if input_padded and requested_angle > 0.0
                    else mask
                )
                for name, mask in self._workflow_support.items()
            }
        if self._workflow_metadata is not None:
            workflow_validity = {
                name: item["validity"] for name, item in self._workflow_metadata.items()
            }

        if isinstance(self.data, dict):
            self.data = {
                name: _preprocess_func(value) for name, value in self.data.items()
            }
        else:
            self.data = _preprocess_func(self.data)
        if support is not None:
            self.support_mask = _preprocess_mask(support)
        if workflow_support is not None:
            self._workflow_support = {
                name: _preprocess_mask(mask) for name, mask in workflow_support.items()
            }
        if workflow_validity is not None:
            for name, mask in workflow_validity.items():
                self._workflow_metadata[name]["validity"] = _preprocess_mask(mask)
        self.lat += delta_lat
        self.lon += delta_lon
        self.shadow_angle = requested_angle
        self.shadow_padded = requested_angle > 0.0

    def _require_shadow_padding(self) -> None:
        if self.shadow_angle > 0.0 and not self.shadow_padded:
            raise ValueError(
                "shadow_angle describes an unmaterialized south-polar cap; "
                "call preprocess() before projecting or extracting features"
            )

    def to_gnomonic(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        fov: Optional[float] = None,
        *,
        spec=None,
        interpolation=None,
        **kwargs,
    ) -> "GnomonicFace":
        """
        Projects the equirectangular image to a single gnomonic face.

        Args:
            lat: Latitude of the tangent point in degrees.
            lon: Longitude of the tangent point in degrees.
            fov: Field of view in degrees.
            **kwargs: Additional projection parameters.

        Returns:
            GnomonicFace: The resulting gnomonic face object.

        Examples:
            >>> face = img.to_gnomonic(lat=0.0, lon=0.0, fov=90)
        """
        from .gnomonic_image import GnomonicFace

        self._require_shadow_padding()

        if spec is not None:
            if any(value is not None for value in (lat, lon, fov)) or kwargs:
                raise ValueError(
                    "spec is mutually exclusive with legacy geometry parameters"
                )
            from panorai.geometry import GnomonicProjector, GnomonicSpec

            if not isinstance(spec, GnomonicSpec):
                raise TypeError("spec must be a panorai.geometry.GnomonicSpec")
            projector = GnomonicProjector(
                spec,
                interpolation="bilinear" if interpolation is None else interpolation,
            )
            from .multi_handler import MultiChannelHandler

            handler = MultiChannelHandler(self.data_clone())
            projected_data = handler.apply_projection(
                lambda data: projector.project(data).data
            )
            face = GnomonicFace(
                projected_data,
                spec.center_lat_deg,
                spec.center_lon_deg,
                spec.hfov_deg,
                hfov_deg=spec.hfov_deg,
                vfov_deg=spec.vfov_deg,
                roll_deg=spec.roll_deg,
            )
            face.spec = spec
            face.support_mask = np.ones(spec.output_shape_hw, dtype=bool)
            return face
        if lat is None or lon is None or fov is None:
            raise TypeError("lat, lon and fov are required when spec is not provided")
        if interpolation is not None:
            kwargs["interpolation"] = interpolation
        # 1) Possibly update or use attached projection
        projection, (lat, lon, fov) = self.dynamic_projection(lat, lon, fov, **kwargs)
        # 2) Apply projection on a clone to avoid mutating this object's data
        from .multi_handler import MultiChannelHandler

        handler = MultiChannelHandler(self.data_clone())
        projected_data = handler.apply_projection(lambda d: projection.project(d))
        return GnomonicFace(projected_data, lat, lon, fov)

    def to_gnomonic_face_set(
        self,
        fov: float = 90.0,
        sampling_method: Union[str, None] = None,
        rotations: List[Tuple[float, float]] = [],
    ) -> "GnomonicFaceSet":
        """
        Samples multiple gnomonic faces from the equirectangular image.

        - Attaches a sampler if none is set or if `sampling_method` is specified.
        - Applies a list of (lat, lon) rotations for additional sampling.

        Args:
            fov: Field of view for each face in degrees.
            sampling_method: Sampler name, such as ``"cube"``.
            rotations: Additional ``(lat, lon)`` rotations to sample.

        Returns:
            GnomonicFaceSet: A collection (set) of gnomonic faces.

        Examples:
            >>> faces = img.to_gnomonic_face_set(fov=90)
        """
        from .gnomonic_imageset import GnomonicFaceSet

        if self.sampler is None:
            self.attach_sampler(sampling_method or "cube")

        if self.sampler is None:
            raise AttributeError("Sampler is not attached")

        tangent_points = self.sampler.get_tangent_points()
        if rotations:
            tangent_points = self.augment_with_rotations(tangent_points, rotations)

        faces = [
            self.to_gnomonic(lat=tp[0], lon=tp[1], fov=fov) for tp in tangent_points
        ]
        return GnomonicFaceSet(faces)

    def augment_with_rotations(
        self,
        tangent_points: List[Tuple[float, float]],
        rotations: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """
        Augments each existing tangent point with a list of additional rotations.

        Args:
            tangent_points: Original ``(lat, lon)`` pairs.
            rotations: Each entry is ``(delta_lat, delta_lon)``.

        Returns:
            List[Tuple[float, float]]: Combined original and rotated tangent points.

        Examples:
            >>> img.augment_with_rotations([(0.0, 0.0)], [(10.0, 0.0)])
        """
        augmented = []
        for point in tangent_points:
            for dlat, dlon in rotations:
                augmented.append((point[0] + dlat, point[1] + dlon))
        return tangent_points + augmented

    def to_pcd(
        self,
        grad_threshold: float = 1.0,
        min_radius: float = 0.5,
        max_radius: float = 30.0,
    ):
        """
        Convert this EquirectangularImage into a Point Cloud (PCD).

        Uses the PCDHandler (assumed external code) for the conversion.

        Args:
            grad_threshold: Gradient threshold for depth estimation.
            min_radius: Minimum valid radius.
            max_radius: Maximum valid radius.

        Returns:
            Some form of PCD object from the PCDHandler.

        Examples:
            >>> pcd = img.to_pcd()
        """
        from ..pcd.handler import PCDHandler  # Keep consistent with your project

        return PCDHandler.equirectangular_image_to_pcd(
            self,
            grad_threshold=grad_threshold,
            min_radius=min_radius,
            max_radius=max_radius,
        )

    def clone(self) -> "EquirectangularImage":
        """
        Creates a deep copy of this object, preserving data
        and core attributes (lat, lon, shadow_angle, etc.).

        Returns:
            EquirectangularImage

        Examples:
            >>> img_copy = img.clone()
        """
        from ._workflow import clone_array

        new_obj = EquirectangularImage(
            data=self.data_clone(),
            shadow_angle=self.shadow_angle,
            lat=self.lat,
            lon=self.lon,
            shadow_padded=self.shadow_padded,
        )
        new_obj.support_mask = (
            None if self.support_mask is None else clone_array(self.support_mask)
        )
        new_obj.sampler = self.sampler
        new_obj.projection = self.projection
        if self._workflow_metadata is not None:
            from ._workflow import copy_metadata

            new_obj._workflow_metadata = copy_metadata(self._workflow_metadata)
            if self._workflow_support is not None:
                new_obj._workflow_support = {
                    name: clone_array(mask)
                    for name, mask in self._workflow_support.items()
                }
        return new_obj

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the underlying data.

        Examples:
            >>> img.shape
        """
        return self.get_shape()

    def show(self) -> None:
        """Display the image using :mod:`PIL.Image` for a quick preview.

        Examples:
            >>> img.show()
        """
        arr = np.asarray(self.get_data())
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        Image.fromarray(arr).show()

    def __repr__(self):
        return (
            f"EquirectangularImage("
            f"lat={self.lat}, lon={self.lon}, shadow_angle={self.shadow_angle}, "
            f"shadow_padded={self.shadow_padded})"
        )
