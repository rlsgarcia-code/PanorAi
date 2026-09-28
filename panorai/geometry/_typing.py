"""Public, optional-backend-safe typing for :mod:`panorai.geometry`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol, TypeAlias, TypeVar, Union

import numpy as np


class TensorLike(Protocol):
    """Structural minimum shared by supported Torch tensor inputs.

    The protocol keeps Torch optional at import time. Runtime geometry still
    accepts concrete ``numpy.ndarray`` and ``torch.Tensor`` objects only.
    """

    @property
    def shape(self) -> Sequence[int]: ...

    @property
    def ndim(self) -> int: ...

    @property
    def dtype(self) -> object: ...


NumpyArray: TypeAlias = np.ndarray[Any, np.dtype[Any]]
ArrayLike: TypeAlias = Union[NumpyArray, TensorLike]
ArrayT = TypeVar("ArrayT", bound=ArrayLike)
Interpolation: TypeAlias = Literal["nearest", "bilinear"]
InvalidPolicy: TypeAlias = Literal["propagate", "renormalize"]
ShapeHW: TypeAlias = tuple[int, int]


def _validate_interpolation(value: str) -> Interpolation:
    if value not in {"nearest", "bilinear"}:
        raise ValueError("interpolation must be 'nearest' or 'bilinear'")
    return value  # type: ignore[return-value]


def _validate_invalid_policy(value: str) -> InvalidPolicy:
    if value not in {"propagate", "renormalize"}:
        raise ValueError("invalid_policy must be 'propagate' or 'renormalize'")
    return value  # type: ignore[return-value]
