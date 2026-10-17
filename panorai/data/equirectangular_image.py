"""
equirectangular_image.py
========================

Implements the EquirectangularImage class, which represents a panoramic
equirectangular image (potentially multi-channel).
"""

from __future__ import annotations

import math
from enum import Enum
from numbers import Real
from os import PathLike

import numpy as np
from typing import TYPE_CHECKING, Any, BinaryIO, List, Optional, Tuple, Union
from PIL import Image  # only if needed for internal usage

from .spherical_data import SphericalData

from panorai.preprocessing.preprocessor import (
    Preprocessor,
)  # assumed import in original code

if TYPE_CHECKING:
    from .gnomonic_image import GnomonicFace
    from .gnomonic_imageset import GnomonicFaceSet


class SphericalTranspose(str, Enum):
    """Named orientation changes that retain a well-defined spherical meaning.

    Unlike a raw array transpose, these operations act on longitude/latitude
    coordinates.  They never exchange the ERP height and width axes.
    """

    FLIP_LONGITUDE = "flip-longitude"
    FLIP_LATITUDE = "flip-latitude"
    ANTIPODAL = "antipodal"


def _first_value(data: Any) -> Any:
    return next(iter(data.values())) if isinstance(data, dict) else data


def _spatial_shape(value: Any) -> tuple[int, int]:
    from ._workflow import spatial_shape

    return spatial_shape(value)


def _validate_pillow_size(size: Any, *, name: str = "size") -> tuple[int, int]:
    if (
        not isinstance(size, (tuple, list))
        or len(size) != 2
        or any(isinstance(item, bool) for item in size)
        or any(not isinstance(item, (int, np.integer)) for item in size)
    ):
        raise TypeError(f"{name} must be a (width, height) integer pair")
    width, height = (int(item) for item in size)
    if width < 2 or height < 2:
        raise ValueError(f"{name} width and height must both be at least 2")
    return width, height


def _validate_interpolation(value: str) -> str:
    if value not in {"nearest", "bilinear"}:
        raise ValueError("interpolation must be 'nearest' or 'bilinear'")
    return value


def _euler_rotation(
    *, yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0
) -> np.ndarray:
    """Return the documented active yaw/pitch/roll rotation.

    Yaw is positive from ``+Z`` toward ``+X``; pitch is positive toward
    ``+Y``; roll is positive clockwise in the ERP-facing image plane.  The
    rotations are composed about the fixed canonical Y, X and Z axes as
    ``Rz(-roll) @ Rx(-pitch) @ Ry(yaw)``.
    """

    values = []
    for name, value in (
        ("yaw_deg", yaw_deg),
        ("pitch_deg", pitch_deg),
        ("roll_deg", roll_deg),
    ):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a finite real number")
        resolved = float(value)
        if not math.isfinite(resolved):
            raise ValueError(f"{name} must be finite")
        values.append(math.radians(resolved))
    yaw, pitch, roll = values
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    yaw_matrix = np.asarray(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)))
    pitch_matrix = np.asarray(((1.0, 0.0, 0.0), (0.0, cp, sp), (0.0, -sp, cp)))
    roll_matrix = np.asarray(((cr, sr, 0.0), (-sr, cr, 0.0), (0.0, 0.0, 1.0)))
    return roll_matrix @ pitch_matrix @ yaw_matrix


def _restore_integer_dtype(result: Any, source: Any) -> Any:
    """Restore image-like integer dtypes after interpolation."""

    from ._workflow import is_torch

    if is_torch(source):
        import torch

        if source.dtype == torch.bool:
            return result >= 0.5
        if not (source.dtype.is_floating_point or source.dtype.is_complex):
            bounds = torch.iinfo(source.dtype)
            finite = torch.nan_to_num(
                result, nan=0.0, posinf=float(bounds.max), neginf=float(bounds.min)
            )
            return finite.round().clamp(bounds.min, bounds.max).to(source.dtype)
        return result.to(source.dtype)
    if source.dtype == np.bool_:
        return result >= 0.5
    if np.issubdtype(source.dtype, np.integer):
        bounds = np.iinfo(source.dtype)
        finite = np.nan_to_num(
            result, nan=0.0, posinf=float(bounds.max), neginf=float(bounds.min)
        )
        return np.rint(np.clip(finite, bounds.min, bounds.max)).astype(
            source.dtype, copy=False
        )
    return result.astype(source.dtype, copy=False)


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

    interface = "panorai-object-workflow/v1"
    stability = "stable"
    Transpose = SphericalTranspose

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
        self._source_format: str | None = None
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

    @classmethod
    def open(
        cls,
        source: str | PathLike[str] | BinaryIO,
        *,
        mode: str | None = "RGB",
        shadow_angle: float = 0.0,
        lat: float = 0.0,
        lon: float = 0.0,
        support_mask: Any = None,
        valid: Any = None,
        shadow_padded: bool = False,
    ) -> "EquirectangularImage":
        """Open an ERP image eagerly and release the source file immediately.

        ``mode=None`` preserves the decoded Pillow mode.  The default ``RGB``
        gives a predictable three-channel image.  File-backed lazy decoding is
        intentionally not exposed: the returned panorama owns its array.
        """

        with Image.open(source) as decoded:
            source_format = decoded.format
            loaded = decoded if mode is None else decoded.convert(mode)
            loaded.load()
            result = cls.from_pil(
                loaded,
                shadow_angle=shadow_angle,
                lat=lat,
                lon=lon,
                support_mask=support_mask,
                valid=valid,
                shadow_padded=shadow_padded,
            )
        result._source_format = source_format
        return result

    @classmethod
    def sample(
        cls,
        pattern: str = "spherical-grid",
        size: tuple[int, int] = (512, 256),
        *,
        shadow_angle: float = 0.0,
        lat: float = 0.0,
        lon: float = 0.0,
        support_mask: Any = None,
        valid: Any = None,
        shadow_padded: bool = False,
    ) -> "EquirectangularImage":
        """Create a deterministic synthetic RGB ERP for examples and tests.

        Args:
            pattern: One of ``"spherical-grid"``, ``"directions"`` or
                ``"checkerboard"``.  ``"directions"`` maps canonical
                ``(X, Y, Z)`` rays to RGB; the other patterns mark angular
                cells on the same pixel-centre lattice.
            size: Pillow-style ``(width, height)`` output size.
            shadow_angle: Optional south-polar shadow angle in degrees.
            lat: Latitude metadata in degrees.
            lon: Longitude metadata in degrees.
            support_mask: Optional geometric source support.
            valid: Optional data-validity mask.
            shadow_padded: Whether shadow rows are already materialized.

        This is a documentation fixture, not a dataset or a scientific oracle.
        Pixel values are deterministic, but callers should rely on the stated
        geometry and modalities rather than treating the artwork as evidence.
        """

        if not isinstance(pattern, str):
            raise TypeError("pattern must be a string")
        choices = ("spherical-grid", "directions", "checkerboard")
        if pattern not in choices:
            expected = ", ".join(repr(choice) for choice in choices)
            raise ValueError(f"pattern must be one of {expected}")

        width, height = _validate_pillow_size(size)
        from panorai.geometry import erp_pixels_to_rays

        rows, columns = np.indices((height, width), dtype=np.float64)
        pixels_xy = np.stack((columns, rows), axis=-1)
        rays = erp_pixels_to_rays(pixels_xy, (height, width))
        direction_rgb = np.rint((rays + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

        if pattern == "directions":
            image = direction_rgb
        else:
            longitude = np.arctan2(rays[..., 0], rays[..., 2])
            latitude = np.arcsin(np.clip(rays[..., 1], -1.0, 1.0))
            longitude_step = math.pi / 6.0
            latitude_step = math.pi / 12.0

            if pattern == "checkerboard":
                longitude_cell = np.floor(
                    (longitude + math.pi) / longitude_step
                ).astype(np.int64)
                latitude_cell = np.floor(
                    (latitude + math.pi / 2.0) / latitude_step
                ).astype(np.int64)
                parity = (longitude_cell + latitude_cell) & 1
                palette = np.asarray(((35, 100, 165), (232, 181, 62)), dtype=np.uint8)
                image = palette[parity]
            else:
                background = np.asarray((35.0, 45.0, 65.0))
                image = (
                    np.rint(direction_rgb.astype(np.float64) * 0.55 + background * 0.45)
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                longitude_distance = np.abs(
                    (longitude + longitude_step / 2.0) % longitude_step
                    - longitude_step / 2.0
                )
                latitude_distance = np.abs(
                    (latitude + latitude_step / 2.0) % latitude_step
                    - latitude_step / 2.0
                )
                longitude_pixel = 2.0 * math.pi / width
                latitude_pixel = math.pi / height
                grid = (longitude_distance <= 0.65 * longitude_pixel) | (
                    latitude_distance <= 0.65 * latitude_pixel
                )
                image[grid] = (225, 232, 240)
                image[np.abs(latitude) <= 0.65 * latitude_pixel] = (244, 178, 35)
                prime_meridian = (
                    np.abs((longitude + math.pi) % (2.0 * math.pi) - math.pi)
                    <= 0.65 * longitude_pixel
                )
                image[prime_meridian] = (190, 55, 55)

        return cls.fromarray(
            image,
            shadow_angle=shadow_angle,
            lat=lat,
            lon=lon,
            support_mask=support_mask,
            valid=valid,
            shadow_padded=shadow_padded,
        )

    @classmethod
    def constant(
        cls,
        value: Any,
        *,
        height: int,
        width: int,
        dtype: Any = np.float32,
        shadow_angle: float = 0.0,
        lat: float = 0.0,
        lon: float = 0.0,
        support_mask: Any = None,
        valid: Any = None,
        shadow_padded: bool = False,
    ) -> "EquirectangularImage":
        """Create a constant NumPy ERP with explicit ``height`` and ``width``.

        A scalar ``value`` produces an ``(H, W)`` field. A one-dimensional
        value such as ``(0.1, 0.2, 0.3)`` produces an ``(H, W, C)`` image.
        The default ``float32`` dtype is intended for numerical tutorials;
        pass ``dtype=np.uint8`` when an encoded 8-bit image is required.

        Unlike :meth:`new`, dimensions use canonical geometry order and are
        keyword-only, so tutorial code cannot silently exchange width and
        height.
        """

        resolved_width, resolved_height = _validate_pillow_size(
            (width, height), name="width and height"
        )
        try:
            resolved_dtype = np.dtype(dtype)
        except TypeError as exc:
            raise TypeError("dtype must be a NumPy numeric or boolean dtype") from exc
        if not (
            np.issubdtype(resolved_dtype, np.number)
            or np.issubdtype(resolved_dtype, np.bool_)
        ) or np.issubdtype(resolved_dtype, np.complexfloating):
            raise TypeError("dtype must be a real numeric or boolean dtype")

        source = np.asarray(value)
        if source.ndim > 1 or (source.ndim == 1 and source.size == 0):
            raise ValueError("value must be a scalar or a non-empty channel vector")
        if not (
            np.issubdtype(source.dtype, np.number)
            or np.issubdtype(source.dtype, np.bool_)
        ) or np.issubdtype(source.dtype, np.complexfloating):
            raise TypeError("value must contain real numeric or boolean values")
        if not np.isfinite(source).all():
            raise ValueError("value must contain only finite values")

        converted = source.astype(resolved_dtype, copy=False)
        shape = (resolved_height, resolved_width)
        if converted.ndim == 0:
            image = np.full(shape, converted.item(), dtype=resolved_dtype)
        else:
            image = np.empty((*shape, int(converted.size)), dtype=resolved_dtype)
            image[...] = converted
        return cls.fromarray(
            image,
            layout="HW" if image.ndim == 2 else "HWC",
            shadow_angle=shadow_angle,
            lat=lat,
            lon=lon,
            support_mask=support_mask,
            valid=valid,
            shadow_padded=shadow_padded,
        )

    @classmethod
    def new(
        cls,
        mode: str,
        size: tuple[int, int],
        color: Any = 0,
        **kwargs: Any,
    ) -> "EquirectangularImage":
        """Create a constant ERP using Pillow-style ``(width, height)`` size."""

        width, height = _validate_pillow_size(size)
        return cls.from_pil(Image.new(mode, (width, height), color), **kwargs)

    @classmethod
    def fromarray(
        cls,
        array: np.ndarray,
        *,
        layout: str = "HWC",
        shadow_angle: float = 0.0,
        lat: float = 0.0,
        lon: float = 0.0,
        support_mask: Any = None,
        valid: Any = None,
        shadow_padded: bool = False,
    ) -> "EquirectangularImage":
        """Create an owned ERP from a NumPy ``HW``, ``HWC`` or ``CHW`` array."""

        if not isinstance(array, np.ndarray):
            raise TypeError("array must be a numpy.ndarray")
        if layout not in {"HW", "HWC", "CHW"}:
            raise ValueError("layout must be 'HW', 'HWC', or 'CHW'")
        if layout == "HW":
            if array.ndim != 2:
                raise ValueError("layout='HW' requires a two-dimensional array")
            data = array
        elif layout == "HWC":
            if array.ndim != 3:
                raise ValueError("layout='HWC' requires a three-dimensional array")
            data = array
        else:
            if array.ndim != 3:
                raise ValueError("layout='CHW' requires a three-dimensional array")
            data = np.moveaxis(array, 0, -1)
        return cls(
            data,
            shadow_angle=shadow_angle,
            lat=lat,
            lon=lon,
            support_mask=support_mask,
            valid=valid,
            shadow_padded=shadow_padded,
        )

    @classmethod
    def from_pil(
        cls,
        image: Image.Image,
        *,
        shadow_angle: float = 0.0,
        lat: float = 0.0,
        lon: float = 0.0,
        support_mask: Any = None,
        valid: Any = None,
        shadow_padded: bool = False,
    ) -> "EquirectangularImage":
        """Create an owned ERP from a Pillow image without retaining its file."""

        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        data = np.asarray(image).copy()
        result = cls(
            data,
            shadow_angle=shadow_angle,
            lat=lat,
            lon=lon,
            support_mask=support_mask,
            valid=valid,
            shadow_padded=shadow_padded,
        )
        result._source_format = image.format
        return result

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

    def _new_workflow_result(
        self,
        data: dict[str, Any],
        metadata: dict[str, dict[str, Any]],
        supports: dict[str, Any],
        *,
        preserve_shadow_state: bool,
        preserve_source_format: bool = False,
    ) -> "EquirectangularImage":
        from ._workflow import clone_array, copy_metadata

        if list(data) == ["image"]:
            constructor_data: Any = data["image"]
        else:
            constructor_data = data
        result = EquirectangularImage(
            constructor_data,
            shadow_angle=self.shadow_angle if preserve_shadow_state else 0.0,
            lat=self.lat,
            lon=self.lon,
            shadow_padded=self.shadow_padded if preserve_shadow_state else False,
        )
        result._workflow_metadata = copy_metadata(metadata)
        result._workflow_support = {
            name: clone_array(mask) for name, mask in supports.items()
        }
        first_support = supports[next(iter(data))]
        result.support_mask = clone_array(first_support)
        result.sampler = self.sampler
        result.projection = self.projection
        result._source_format = self._source_format if preserve_source_format else None
        return result

    def _transform_all_fields(
        self,
        *,
        operation: str,
        output_shape_hw: tuple[int, int],
        rotation: np.ndarray | None = None,
        interpolation: str | None = None,
        invalid_policy: str = "propagate",
        min_valid_weight: float | None = None,
        preserve_dtype: bool = True,
    ) -> "EquirectangularImage":
        """Apply one canonical spatial transform to every aligned field."""

        from ._workflow import clone_array, copy_metadata, is_torch
        from panorai.ops.spherical import Resize, Rotate

        if invalid_policy not in {"propagate", "renormalize"}:
            raise ValueError("invalid_policy must be 'propagate' or 'renormalize'")
        if invalid_policy == "renormalize":
            if (
                min_valid_weight is None
                or isinstance(min_valid_weight, bool)
                or not isinstance(min_valid_weight, Real)
            ):
                raise TypeError(
                    "renormalize requires min_valid_weight as a finite real in (0, 1]"
                )
            threshold = float(min_valid_weight)
            if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
                raise ValueError("min_valid_weight must be in the interval (0, 1]")
        elif min_valid_weight is not None:
            raise ValueError("min_valid_weight requires invalid_policy='renormalize'")

        metadata = self._require_workflow()
        data: dict[str, Any] = {}
        supports: dict[str, Any] = {}
        output_metadata = copy_metadata(metadata)
        for name, source in self._workflow_data().items():
            field_interpolation = (
                "nearest"
                if metadata[name]["field_spec"].kind == "categorical"
                else interpolation or metadata[name]["interpolation"]
            )
            field_policy = (
                "propagate"
                if metadata[name]["field_spec"].kind == "categorical"
                else invalid_policy
            )
            field_threshold = (
                min_valid_weight if field_policy == "renormalize" else None
            )
            support = self._workflow_support[name]
            validity = metadata[name]["validity"]
            if is_torch(source):
                from panorai.ops.spherical import nn as spherical_nn

                if field_policy != "propagate":
                    raise ValueError(
                        "Torch container transforms currently support only strict "
                        "validity propagation"
                    )
                if source.ndim == 2:
                    model_input = source[None, None]
                elif source.ndim == 3:
                    model_input = source[None]
                elif source.ndim == 4:
                    model_input = source
                else:  # validated by the workflow; defensive for clearer failures
                    raise ValueError("Torch fields must use HW, CHW, or NCHW layout")
                working = model_input
                if not working.dtype.is_floating_point:
                    import torch

                    working = working.to(dtype=torch.float32)
                if operation == "resize":
                    layer = spherical_nn.Resize(
                        output_shape_hw, interpolation=field_interpolation
                    )
                else:
                    layer = spherical_nn.Rotate(
                        rotation,
                        output_shape_hw=output_shape_hw,
                        interpolation=field_interpolation,
                    )
                transformed, field_support, field_validity = (
                    spherical_nn.execute_with_masks(layer, working, support, validity)
                )
                if source.ndim == 2:
                    transformed = transformed[0, 0]
                elif source.ndim == 3:
                    transformed = transformed[0]
            else:
                configured = (
                    Resize(output_shape_hw, interpolation=field_interpolation)
                    if operation == "resize"
                    else Rotate(
                        rotation,
                        output_shape_hw=output_shape_hw,
                        interpolation=field_interpolation,
                    )
                )
                transformed, field_support, field_validity = (
                    configured.apply_with_masks(
                        source,
                        support,
                        validity,
                        invalid_policy=field_policy,
                        min_valid_weight=field_threshold,
                    )
                )
            if preserve_dtype:
                transformed = _restore_integer_dtype(transformed, source)
            data[name] = transformed
            supports[name] = clone_array(field_support)
            output_metadata[name]["validity"] = clone_array(field_validity)

        return self._new_workflow_result(
            data,
            output_metadata,
            supports,
            preserve_shadow_state=operation == "resize",
        )

    @property
    def size(self) -> tuple[int, int]:
        """Return Pillow-style ``(width, height)`` for the ERP raster."""

        height, width = _spatial_shape(_first_value(self.data))
        return width, height

    @property
    def width(self) -> int:
        """ERP raster width in pixels."""

        return self.size[0]

    @property
    def height(self) -> int:
        """ERP raster height in pixels."""

        return self.size[1]

    @property
    def shape_hw(self) -> tuple[int, int]:
        """Canonical ``(height, width)`` spatial shape."""

        return self.height, self.width

    @property
    def dtype(self):
        """Dtype of the primary image field (or first typed field)."""

        return _first_value(self.data).dtype

    @property
    def backend(self) -> str:
        """Return ``"numpy"`` or ``"torch"`` for the owned raster fields."""

        from ._workflow import backend

        return backend(_first_value(self.data))

    @property
    def mode(self) -> str | None:
        """Return a Pillow-like logical mode for the image field, if present."""

        image = self.image
        if image is None:
            return None
        from ._workflow import is_torch

        channels = 1
        if image.ndim == 3:
            channels = int(image.shape[-3] if is_torch(image) else image.shape[-1])
        elif image.ndim == 4:
            channels = int(image.shape[-3])
        if channels == 3:
            return "RGB"
        if channels == 4:
            return "RGBA"
        if channels != 1:
            return f"C{channels}"
        dtype = image.dtype
        if str(dtype).endswith("bool"):
            return "1"
        if "float" in str(dtype):
            return "F"
        if str(dtype).endswith("int32"):
            return "I"
        return "L"

    @property
    def format(self) -> str | None:
        """Decoded source format, or ``None`` for generated/transformed ERPs."""

        return self._source_format

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

    @classmethod
    def from_field(
        cls,
        name,
        value,
        *,
        spec,
        valid=None,
        support=None,
    ):
        """Create an ERP container whose first field has an explicit contract."""

        from ._workflow import (
            clone_array,
            normalize_field_name,
            ones_mask,
            validate_finite_where_valid,
            validate_mask,
            validate_modality,
            validate_support_mask,
        )
        from .fields import RasterFieldSpec

        name = normalize_field_name(name)
        if not isinstance(spec, RasterFieldSpec):
            raise TypeError("spec must be a RasterFieldSpec")
        validate_modality(value, name, name=name, field_spec=spec)
        source_support = (
            ones_mask(value)
            if support is None
            else validate_support_mask(value, support)
        )
        validity = ones_mask(value) if valid is None else validate_mask(value, valid)
        validity = validity & source_support
        validate_finite_where_valid(value, validity, name=name)
        result = cls({name: clone_array(value)})
        result.support_mask = clone_array(source_support)
        result._workflow_metadata = {
            name: {
                "kind": name,
                "units": spec.units,
                "validity": clone_array(validity),
                "interpolation": spec.interpolation,
                "field_spec": spec,
            }
        }
        result._workflow_support = {name: clone_array(source_support)}
        return result

    def field(self, name):
        """Return the named raster field without inferring meaning from values."""

        data = self._workflow_data()
        if name not in data:
            raise KeyError(f"unknown field {name!r}")
        return data[name]

    def field_spec(self, name):
        """Return the immutable :class:`RasterFieldSpec` for ``name``."""

        metadata = self._require_workflow()
        if name not in metadata:
            raise KeyError(f"unknown field {name!r}")
        return metadata[name]["field_spec"]

    def validity(self, modality: str):
        """Return a copy of the explicit validity mask for ``modality``."""

        from ._workflow import clone_array

        metadata = self._require_workflow()
        if modality not in metadata:
            raise KeyError(f"unknown modality {modality!r}")
        return clone_array(metadata[modality]["validity"])

    def support(self, modality: str):
        """Return a copy of the geometric support for ``modality``."""

        from ._workflow import clone_array

        metadata = self._require_workflow()
        if modality not in metadata:
            raise KeyError(f"unknown modality {modality!r}")
        return clone_array(self._workflow_support[modality])

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

    def with_field(self, name, value, *, spec, valid=None, support=None):
        """Return a copy with an explicitly typed raster field attached."""

        from ._workflow import (
            add_field,
            clone_array,
            copy_metadata,
            normalize_field_name,
        )

        name = normalize_field_name(name)
        inherited_support = next(iter(self._workflow_support.values()))
        data, metadata, field_support = add_field(
            self._workflow_data(),
            self._require_workflow(),
            name,
            value,
            spec=spec,
            valid=valid,
            support=inherited_support if support is None else support,
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
            field_name: (
                clone_array(field_support)
                if field_name == name
                else clone_array(self._workflow_support[field_name])
            )
            for field_name in metadata
        }
        result.sampler = self.sampler
        result.projection = self.projection
        return result

    def copy(self) -> "EquirectangularImage":
        """Return an owned deep copy; all image-style transforms are immutable."""

        return self.clone()

    def to_pil(
        self,
        *,
        field: str = "image",
        mode: str | None = None,
        value_range: tuple[float, float] | None = None,
        invalid_fill: float | int = 0,
    ) -> Image.Image:
        """Convert one field to an owned Pillow image for display or encoding.

        Floating multi-channel rasters use ``value_range=(0, 1)`` by default
        and are quantized to 8-bit for Pillow.  Floating single-channel data
        stays in Pillow ``F`` mode unless a different mode is requested.
        Invalid samples are filled explicitly for the exported preview; the
        PanorAi container remains unchanged.
        """

        from ._workflow import expand_mask, is_torch

        value = self.field(field)
        valid = self.validity(field)
        if is_torch(value):
            if value.requires_grad:
                raise ValueError(
                    "to_pil() is non-differentiable; detach the tensor explicitly "
                    "before constructing the panorama"
                )
            array = value.detach().cpu().numpy()
            valid_array = valid.detach().cpu().numpy()
            if array.ndim == 4:
                if array.shape[0] != 1:
                    raise ValueError("to_pil() requires an unbatched field")
                array = array[0]
                valid_array = valid_array[0]
            if array.ndim == 3:
                array = np.moveaxis(array, 0, -1)
        else:
            array = np.asarray(value).copy()
            valid_array = np.asarray(valid)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        exported = np.where(expand_mask(valid_array, array), array, invalid_fill)
        if np.issubdtype(exported.dtype, np.floating):
            if exported.ndim == 2 and mode in {None, "F"} and value_range is None:
                image = Image.fromarray(exported.astype(np.float32), mode="F")
            else:
                lower, upper = (0.0, 1.0) if value_range is None else value_range
                if (
                    isinstance(lower, bool)
                    or isinstance(upper, bool)
                    or not math.isfinite(float(lower))
                    or not math.isfinite(float(upper))
                    or float(upper) <= float(lower)
                ):
                    raise ValueError("value_range must be finite and increasing")
                finite = np.isfinite(exported)
                if not finite.all():
                    raise ValueError(
                        "exported values must be finite after invalid_fill"
                    )
                if np.any(exported < lower) or np.any(exported > upper):
                    raise ValueError(
                        f"floating values fall outside value_range={(lower, upper)}"
                    )
                scaled = (exported - lower) / (upper - lower)
                encoded = np.rint(scaled * 255.0).astype(np.uint8)
                image = Image.fromarray(encoded)
        else:
            image = Image.fromarray(exported)
        return image if mode is None or image.mode == mode else image.convert(mode)

    def save(
        self,
        destination: str | PathLike[str] | BinaryIO,
        format: str | None = None,
        *,
        field: str = "image",
        mode: str | None = None,
        value_range: tuple[float, float] | None = None,
        invalid_fill: float | int = 0,
        **params: Any,
    ) -> None:
        """Encode one raster field through Pillow without serializing metadata."""

        image = self.to_pil(
            field=field,
            mode=mode,
            value_range=value_range,
            invalid_fill=invalid_fill,
        )
        image.save(destination, format=format, **params)

    def convert(self, mode: str) -> "EquirectangularImage":
        """Return a copy whose image field uses a different Pillow color mode."""

        if self.backend != "numpy":
            raise TypeError("convert() currently requires a NumPy-backed panorama")
        converted = np.asarray(self.to_pil(mode=mode)).copy()
        data = {
            name: (converted if name == "image" else value.copy())
            for name, value in self._workflow_data().items()
        }
        from ._workflow import copy_metadata

        return self._new_workflow_result(
            data,
            copy_metadata(self._require_workflow()),
            self._workflow_support,
            preserve_shadow_state=True,
        )

    def resize(
        self,
        size: tuple[int, int],
        *,
        interpolation: str | None = None,
        invalid_policy: str = "propagate",
        min_valid_weight: float | None = None,
        preserve_dtype: bool = True,
    ) -> "EquirectangularImage":
        """Resample every aligned field onto a new canonical ERP lattice.

        ``size`` follows Pillow's ``(width, height)`` order.  Continuous fields
        use bilinear sampling unless overridden; categorical fields always use
        nearest sampling.  Longitude remains periodic and masks are transformed
        independently from numeric values.
        """

        width, height = _validate_pillow_size(size)
        if interpolation is not None:
            _validate_interpolation(interpolation)
        return self._transform_all_fields(
            operation="resize",
            output_shape_hw=(height, width),
            interpolation=interpolation,
            invalid_policy=invalid_policy,
            min_valid_weight=min_valid_weight,
            preserve_dtype=preserve_dtype,
        )

    def thumbnail(
        self,
        size: tuple[int, int],
        *,
        interpolation: str | None = None,
        preserve_dtype: bool = True,
    ) -> "EquirectangularImage":
        """Return an immutable aspect-preserving ERP bounded by ``size``.

        Unlike Pillow's in-place ``thumbnail()``, this method never mutates and
        never enlarges the source.
        """

        max_width, max_height = _validate_pillow_size(size)
        scale = min(max_width / self.width, max_height / self.height, 1.0)
        if scale == 1.0:
            return self.copy()
        target = (
            int(round(self.width * scale)),
            int(round(self.height * scale)),
        )
        if min(target) < 2:
            raise ValueError(
                "thumbnail bounds cannot preserve this aspect ratio while keeping "
                "both spherical dimensions at least 2 pixels"
            )
        return self.resize(
            target,
            interpolation=interpolation,
            preserve_dtype=preserve_dtype,
        )

    def rotate(
        self,
        rotation: Any = None,
        *,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        size: tuple[int, int] | None = None,
        interpolation: str | None = None,
        invalid_policy: str = "propagate",
        min_valid_weight: float | None = None,
        preserve_dtype: bool = True,
    ) -> "EquirectangularImage":
        """Actively rotate rays in 3D, never the rectangular ERP canvas.

        Supply either a canonical ``3x3`` rotation matrix or named Euler angles.
        A positional scalar such as Pillow's ``rotate(45)`` is deliberately
        rejected because a planar angle does not identify a spherical axis.
        """

        if rotation is not None:
            if any(float(value) != 0.0 for value in (yaw_deg, pitch_deg, roll_deg)):
                raise ValueError("rotation is mutually exclusive with Euler angles")
            matrix = np.asarray(rotation, dtype=np.float64)
            if matrix.shape != (3, 3):
                raise ValueError("rotation must have shape (3, 3)")
        else:
            matrix = _euler_rotation(
                yaw_deg=yaw_deg, pitch_deg=pitch_deg, roll_deg=roll_deg
            )
        output_shape = self.shape_hw
        if size is not None:
            width, height = _validate_pillow_size(size)
            output_shape = (height, width)
        if interpolation is not None:
            _validate_interpolation(interpolation)
        return self._transform_all_fields(
            operation="rotate",
            output_shape_hw=output_shape,
            rotation=matrix,
            interpolation=interpolation,
            invalid_policy=invalid_policy,
            min_valid_weight=min_valid_weight,
            preserve_dtype=preserve_dtype,
        )

    def transpose(self, method: SphericalTranspose | str) -> "EquirectangularImage":
        """Apply a named spherical reflection without swapping raster axes."""

        try:
            resolved = SphericalTranspose(method)
        except ValueError as exc:
            choices = ", ".join(item.value for item in SphericalTranspose)
            raise ValueError(f"transpose method must be one of: {choices}") from exc
        if resolved is SphericalTranspose.ANTIPODAL and self.width % 2:
            return self.rotate(yaw_deg=180.0, interpolation="nearest").transpose(
                SphericalTranspose.FLIP_LATITUDE
            )

        from ._workflow import copy_metadata, is_torch

        def transformed(value: Any) -> Any:
            if is_torch(value):
                if resolved is SphericalTranspose.FLIP_LONGITUDE:
                    return value.flip(-1)
                if resolved is SphericalTranspose.FLIP_LATITUDE:
                    return value.flip(-2)
                return value.flip(-2).roll(self.width // 2, dims=-1)
            if resolved is SphericalTranspose.FLIP_LONGITUDE:
                return np.flip(value, axis=1).copy()
            if resolved is SphericalTranspose.FLIP_LATITUDE:
                return np.flip(value, axis=0).copy()
            return np.roll(np.flip(value, axis=0), self.width // 2, axis=1).copy()

        def transformed_mask(mask: Any) -> Any:
            if is_torch(mask):
                if resolved is SphericalTranspose.FLIP_LONGITUDE:
                    return mask.flip(-1)
                if resolved is SphericalTranspose.FLIP_LATITUDE:
                    return mask.flip(-2)
                return mask.flip(-2).roll(self.width // 2, dims=-1)
            if resolved is SphericalTranspose.FLIP_LONGITUDE:
                return np.flip(mask, axis=-1).copy()
            if resolved is SphericalTranspose.FLIP_LATITUDE:
                return np.flip(mask, axis=-2).copy()
            return np.roll(np.flip(mask, axis=-2), self.width // 2, axis=-1).copy()

        metadata = copy_metadata(self._require_workflow())
        data = {
            name: transformed(value) for name, value in self._workflow_data().items()
        }
        supports = {
            name: transformed_mask(mask)
            for name, mask in self._workflow_support.items()
        }
        for name in metadata:
            metadata[name]["validity"] = transformed_mask(metadata[name]["validity"])
        return self._new_workflow_result(
            data,
            metadata,
            supports,
            preserve_shadow_state=resolved is SphericalTranspose.FLIP_LONGITUDE,
        )

    def crop(
        self,
        spec: Any,
        *,
        depth_policy: str = "propagate",
        min_valid_weight: float | None = None,
    ) -> "GnomonicFace":
        """Project a tangent crop and return a ``GnomonicFace``.

        A rectangular pixel box is intentionally not accepted: the crop is
        defined by a :class:`panorai.geometry.GnomonicSpec` with centre, angular
        field of view, roll and output lattice.
        """

        from panorai.geometry import GnomonicSpec
        from ._workflow import clone_array, copy_metadata, project_modality
        from .gnomonic_image import GnomonicFace

        if not isinstance(spec, GnomonicSpec):
            raise TypeError("spec must be a panorai.geometry.GnomonicSpec")
        metadata = self._require_workflow()
        self._require_shadow_padding()
        if depth_policy not in {"propagate", "renormalize"}:
            raise ValueError("depth_policy must be 'propagate' or 'renormalize'")
        if depth_policy == "renormalize" and min_valid_weight is None:
            raise ValueError(
                "min_valid_weight is required when depth_policy='renormalize'"
            )
        if depth_policy == "propagate" and min_valid_weight is not None:
            raise ValueError(
                "min_valid_weight is only valid when depth_policy='renormalize'"
            )
        face_data: dict[str, Any] = {}
        face_support: dict[str, Any] = {}
        face_metadata = copy_metadata(metadata)
        for name, value in self._workflow_data().items():
            projected, support, validity = project_modality(
                value,
                metadata[name],
                spec,
                depth_policy=depth_policy,
                min_valid_weight=min_valid_weight,
                source_support=self._workflow_support[name],
            )
            face_data[name] = projected
            face_support[name] = clone_array(support)
            face_metadata[name]["validity"] = clone_array(validity)
        constructor_data: Any = (
            face_data["image"] if list(face_data) == ["image"] else face_data
        )
        face = GnomonicFace(
            constructor_data,
            spec.center_lat_deg,
            spec.center_lon_deg,
            spec.hfov_deg,
            hfov_deg=spec.hfov_deg,
            vfov_deg=spec.vfov_deg,
            roll_deg=spec.roll_deg,
        )
        face.spec = spec
        face._erp_shape_hw = self.shape_hw
        face._workflow_metadata = face_metadata
        face._workflow_support = face_support
        face.support_mask = clone_array(face_support[next(iter(face_data))])
        return face

    def filter(self, operation: Any, *, field: str = "image") -> "EquirectangularImage":
        """Apply one registered sphere-aware operation to a named ERP field."""

        from panorai.pipeline import OnSphere

        result = OnSphere(operation, input=field, output=field).execute(self).value
        if not isinstance(result, EquirectangularImage):
            raise TypeError("filter operation must return an ERP raster field")
        return result

    def composite(
        self,
        face: "GnomonicFace",
        *,
        blend: str = "replace",
        alpha: float | None = None,
    ) -> "EquirectangularImage":
        """Reproject and combine one tangent face with this complete ERP.

        ``replace`` makes face support authoritative, including its validity.
        ``alpha`` blends continuous values only where both sources are valid;
        the blended region is invalid otherwise.  A raw planar array is not
        accepted because it has no projection back to the sphere.
        """

        from ._workflow import clone_array, copy_metadata, expand_mask, is_torch
        from .gnomonic_image import GnomonicFace
        from .gnomonic_imageset import GnomonicFaceSet

        if not isinstance(face, GnomonicFace):
            raise TypeError("face must be a GnomonicFace with projection metadata")
        if face.spec is None or face._workflow_metadata is None:
            raise ValueError(
                "face must carry a GnomonicSpec and typed fields; use panorama.crop(spec)"
            )
        if blend not in {"replace", "alpha"}:
            raise ValueError("blend must be 'replace' or 'alpha'")
        if blend == "alpha":
            if alpha is None or isinstance(alpha, bool) or not isinstance(alpha, Real):
                raise TypeError("blend='alpha' requires a finite scalar alpha")
            alpha_value = float(alpha)
            if not math.isfinite(alpha_value) or not 0.0 <= alpha_value <= 1.0:
                raise ValueError("alpha must be finite and in the interval [0, 1]")
        elif alpha is not None:
            raise ValueError("alpha is only valid with blend='alpha'")

        base_metadata = self._require_workflow()
        if tuple(face._workflow_metadata) != tuple(base_metadata):
            raise ValueError(
                "face fields must exactly match the destination fields; "
                f"expected {list(base_metadata)}, got {list(face._workflow_metadata)}"
            )
        face_set = GnomonicFaceSet([face.clone()])
        face_set._workflow = {
            "layout": "single-face-composite",
            "order": [str(face.face_id)],
            "erp_shape": self.shape_hw,
            "specs": [face.spec],
            "depth_policy": "propagate",
            "min_valid_weight": None,
            "source_metadata": copy_metadata(face._workflow_metadata),
            "projector": None,
        }
        overlay = face_set.reconstruct()
        output_data: dict[str, Any] = {}
        output_support: dict[str, Any] = {}
        output_metadata = copy_metadata(base_metadata)
        for name, base in self._workflow_data().items():
            over = _restore_integer_dtype(overlay.field(name), base)
            region = overlay.support(name)
            base_valid = base_metadata[name]["validity"]
            over_valid = overlay.validity(name)
            if is_torch(base):
                import torch

                if is_torch(over) is False or over.device != base.device:
                    raise TypeError(
                        "face and destination must share backend and device"
                    )
                expanded = expand_mask(region, base)
                if blend == "replace" or alpha_value == 1.0:
                    combined = torch.where(expanded, over, base)
                    validity = torch.where(region, over_valid, base_valid)
                elif alpha_value == 0.0:
                    combined = base.clone()
                    validity = base_valid.clone()
                else:
                    mixed = base * (1.0 - alpha_value) + over * alpha_value
                    combined = torch.where(expanded, mixed, base)
                    validity = torch.where(region, base_valid & over_valid, base_valid)
            else:
                if is_torch(over):
                    raise TypeError("face and destination must share one backend")
                expanded = expand_mask(region, base)
                if blend == "replace" or alpha_value == 1.0:
                    combined = np.where(expanded, over, base)
                    validity = np.where(region, over_valid, base_valid)
                elif alpha_value == 0.0:
                    combined = base.copy()
                    validity = base_valid.copy()
                else:
                    mixed = base * (1.0 - alpha_value) + over * alpha_value
                    combined = np.where(expanded, mixed, base)
                    validity = np.where(region, base_valid & over_valid, base_valid)
                combined = _restore_integer_dtype(combined, base)
            if (
                blend == "alpha"
                and base_metadata[name]["field_spec"].kind == "categorical"
            ):
                raise TypeError("alpha blending is undefined for categorical fields")
            output_data[name] = combined
            output_support[name] = clone_array(self._workflow_support[name] | region)
            output_metadata[name]["validity"] = clone_array(validity)
        return self._new_workflow_result(
            output_data,
            output_metadata,
            output_support,
            preserve_shadow_state=False,
        )

    def paste(
        self,
        face: "GnomonicFace",
        *,
        blend: str = "replace",
        alpha: float | None = None,
    ) -> "EquirectangularImage":
        """Alias for :meth:`composite`; returns a new spherical image."""

        return self.composite(face, blend=blend, alpha=alpha)

    def apply_spherical(
        self,
        operator,
        *,
        input,
        output,
        spec=None,
        replace=False,
    ):
        """Apply a shape-preserving operator directly to one ERP field."""

        from ._workflow import (
            clone_array,
            copy_metadata,
            normalize_field_name,
            validate_model_result,
        )

        metadata = self._require_workflow()
        input = normalize_field_name(input)
        output = normalize_field_name(output)
        if input not in metadata:
            raise KeyError(f"unknown input field {input!r}")
        if output in metadata and output != input and not replace:
            raise ValueError(
                f"output field {output!r} already exists; pass replace=True"
            )
        output_spec = metadata[input]["field_spec"] if spec is None else spec
        value, validity, support = validate_model_result(
            operator(self.field(input)),
            self.field(input),
            output=output,
            support=self._workflow_support[input],
            spec=output_spec,
            input_validity=metadata[input]["validity"],
        )
        data = {name: clone_array(item) for name, item in self._workflow_data().items()}
        result_metadata = copy_metadata(metadata)
        result_support = {
            name: clone_array(item) for name, item in self._workflow_support.items()
        }
        data[output] = value
        result_metadata[output] = {
            "kind": output,
            "units": output_spec.units,
            "validity": validity,
            "interpolation": output_spec.interpolation,
            "field_spec": output_spec,
        }
        result_support[output] = support
        result = EquirectangularImage(
            data,
            shadow_angle=self.shadow_angle,
            lat=self.lat,
            lon=self.lon,
            shadow_padded=self.shadow_padded,
        )
        result._workflow_metadata = result_metadata
        result._workflow_support = result_support
        result.support_mask = (
            None if self.support_mask is None else clone_array(self.support_mask)
        )
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

    def iter_views(
        self,
        specs,
        *,
        face_ids=None,
        depth_policy="propagate",
        min_valid_weight=None,
    ):
        """Yield explicit :class:`GnomonicSpec` views incrementally.

        Unlike :meth:`views`, this route accepts independently rolled and
        oriented specifications.  Projection is performed only when each item
        is requested, so callers can process and release one view at a time.
        Support and data validity remain separate and follow the same policy as
        the regular object workflow.
        """

        from panorai.geometry import GnomonicSpec
        from ._workflow import (
            clone_array,
            copy_metadata,
            project_modality_batch,
            spatial_shape,
        )
        from .gnomonic_image import GnomonicFace

        metadata = self._require_workflow()
        self._require_shadow_padding()
        if depth_policy not in {"propagate", "renormalize"}:
            raise ValueError("depth_policy must be 'propagate' or 'renormalize'")
        if depth_policy == "renormalize" and min_valid_weight is None:
            raise ValueError(
                "min_valid_weight is required when depth_policy='renormalize'"
            )
        if depth_policy == "propagate" and min_valid_weight is not None:
            raise ValueError(
                "min_valid_weight is only valid when depth_policy='renormalize'"
            )
        if min_valid_weight is not None:
            if isinstance(min_valid_weight, bool) or not isinstance(
                min_valid_weight, Real
            ):
                raise TypeError("min_valid_weight must be a finite real number")
            min_valid_weight = float(min_valid_weight)
            if not math.isfinite(min_valid_weight) or not 0.0 < min_valid_weight <= 1.0:
                raise ValueError(
                    "min_valid_weight must be finite and in the interval (0, 1]"
                )
        source_data = self._workflow_data()
        source_shape = spatial_shape(next(iter(source_data.values())))
        id_iterator = None if face_ids is None else iter(face_ids)
        for index, spec in enumerate(specs):
            if not isinstance(spec, GnomonicSpec):
                raise TypeError("every explicit view spec must be a GnomonicSpec")
            if id_iterator is None:
                face_id = f"explicit-{index:04d}"
            else:
                try:
                    face_id = next(id_iterator)
                except StopIteration as exc:
                    raise ValueError(
                        "face_ids contains fewer entries than specs"
                    ) from exc
                if not isinstance(face_id, str) or not face_id:
                    raise ValueError("face_ids entries must be non-empty strings")
            face_data = {}
            face_meta = copy_metadata(metadata)
            supports = {}
            for name, source in source_data.items():
                projected, support, validity = project_modality_batch(
                    source,
                    metadata[name],
                    [spec],
                    depth_policy=depth_policy,
                    min_valid_weight=min_valid_weight,
                    projector_template=None,
                    source_support=self._workflow_support[name],
                    plan=None,
                )[0]
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
                face_id=face_id,
            )
            face.spec = spec
            face._erp_shape_hw = source_shape
            face._workflow_metadata = face_meta
            face._workflow_support = {
                name: clone_array(mask) for name, mask in supports.items()
            }
            face.support_mask = clone_array(next(iter(supports.values())))
            yield face
        if id_iterator is not None:
            try:
                next(id_iterator)
            except StopIteration:
                pass
            else:
                raise ValueError("face_ids contains more entries than specs")

    def project_views(
        self,
        specs,
        *,
        face_ids=None,
        depth_policy="propagate",
        min_valid_weight=None,
    ):
        """Materialize an explicit sequence of arbitrary gnomonic views."""

        from ._workflow import copy_metadata, spatial_shape
        from .gnomonic_imageset import GnomonicFaceSet

        specifications = tuple(specs)
        if not specifications:
            raise ValueError("specs must contain at least one GnomonicSpec")
        ids = None if face_ids is None else tuple(face_ids)
        faces = tuple(
            self.iter_views(
                specifications,
                face_ids=ids,
                depth_policy=depth_policy,
                min_valid_weight=min_valid_weight,
            )
        )
        face_set = GnomonicFaceSet(faces)
        order = [face.face_id for face in faces]
        face_set._workflow = {
            "layout": "explicit",
            "order": order,
            "erp_shape": spatial_shape(next(iter(self._workflow_data().values()))),
            "specs": list(specifications),
            "depth_policy": depth_policy,
            "min_valid_weight": min_valid_weight,
            "source_metadata": copy_metadata(self._require_workflow()),
            "projector": None,
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
        spec=None,
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
                spec=spec,
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
        new_obj._source_format = self._source_format
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
        """Display the image field using :mod:`PIL.Image` for a quick preview.

        Examples:
            >>> img.show()
        """
        self.to_pil().show()

    def __repr__(self):
        return (
            f"EquirectangularImage("
            f"size={self.size}, mode={self.mode!r}, backend={self.backend!r}, "
            f"lat={self.lat}, lon={self.lon}, shadow_angle={self.shadow_angle}, "
            f"shadow_padded={self.shadow_padded})"
        )
