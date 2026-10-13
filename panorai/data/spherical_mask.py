"""Compact canonical-ERP binary masks."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _reject_torch(value: object, *, _seen: set[int] | None = None) -> None:
    module = type(value).__module__
    if module == "torch" or module.startswith("torch."):
        raise TypeError(
            "SphericalBinaryMask is NumPy-canonical; convert Torch explicitly "
            "with SphericalBinaryMask.from_torch()"
        )
    seen = set() if _seen is None else _seen
    marker = id(value)
    if marker in seen:
        return
    seen.add(marker)
    if isinstance(value, np.ndarray) and value.dtype == object:
        for item in value.flat:
            _reject_torch(item, _seen=seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_torch(item, _seen=seen)


def _readonly(value: object, *, dtype: np.dtype | type) -> np.ndarray:
    _reject_torch(value)
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class SphericalBinaryMask:
    """Bit-packed boolean mask on a declared canonical ERP lattice."""

    shape_hw: tuple[int, int]
    packed_bits: np.ndarray

    def __post_init__(self) -> None:
        try:
            raw_shape = tuple(self.shape_hw)
        except TypeError as exc:
            raise TypeError("shape_hw must contain exactly two integers") from exc
        if len(raw_shape) != 2 or any(
            not isinstance(value, (int, np.integer)) for value in raw_shape
        ):
            raise TypeError("shape_hw must contain exactly two integers")
        height, width = (int(raw_shape[0]), int(raw_shape[1]))
        if height < 1 or width < 1:
            raise ValueError("shape_hw must contain positive dimensions")
        packed = _readonly(self.packed_bits, dtype=np.uint8).reshape(-1)
        expected = (height * width + 7) // 8
        if len(packed) != expected:
            raise ValueError(
                f"packed_bits has length {len(packed)}, expected {expected}"
            )
        object.__setattr__(self, "shape_hw", (height, width))
        object.__setattr__(self, "packed_bits", packed)

    @classmethod
    def from_array(cls, mask: np.ndarray) -> "SphericalBinaryMask":
        _reject_torch(mask)
        values = np.asarray(mask)
        if values.ndim != 2:
            raise ValueError("mask must have shape (H,W)")
        packed = np.packbits(values.astype(bool).reshape(-1), bitorder="little")
        return cls(tuple(int(value) for value in values.shape), packed)

    @classmethod
    def from_torch(cls, mask: object) -> "SphericalBinaryMask":
        module = type(mask).__module__
        if not (module == "torch" or module.startswith("torch.")):
            raise TypeError("mask must be a torch.Tensor")
        return cls.from_array(mask.detach().cpu().numpy())  # type: ignore[attr-defined]

    def to_array(self) -> np.ndarray:
        height, width = self.shape_hw
        return (
            np.unpackbits(
                self.packed_bits, bitorder="little", count=height * width
            )
            .reshape(height, width)
            .astype(bool, copy=False)
        )

    @property
    def pixel_count(self) -> int:
        return int(np.count_nonzero(self.to_array()))

    def solid_angle_area(self, support: np.ndarray | None = None) -> float:
        mask = self.to_array()
        if support is not None:
            support_array = np.asarray(support, dtype=bool)
            if support_array.shape != self.shape_hw:
                raise ValueError("support and mask shapes must match")
            mask &= support_array
        height, width = self.shape_hw
        latitude_north = math.pi / 2 - np.arange(height) * math.pi / height
        latitude_south = (
            math.pi / 2 - (np.arange(height) + 1) * math.pi / height
        )
        row_area = (2 * math.pi / width) * (
            np.sin(latitude_north) - np.sin(latitude_south)
        )
        return float(np.sum(mask * row_area[:, None]))

    def solid_angle_iou(
        self,
        other: "SphericalBinaryMask",
        *,
        support: np.ndarray | None = None,
    ) -> float:
        if self.shape_hw != other.shape_hw:
            raise ValueError("mask shapes must match")
        first = self.to_array()
        second = other.to_array()
        if support is not None:
            valid = np.asarray(support, dtype=bool)
            if valid.shape != self.shape_hw:
                raise ValueError("support and mask shapes must match")
            first &= valid
            second &= valid
        intersection = SphericalBinaryMask.from_array(
            first & second
        ).solid_angle_area()
        union = SphericalBinaryMask.from_array(first | second).solid_angle_area()
        return intersection / union if union > 0 else 1.0
