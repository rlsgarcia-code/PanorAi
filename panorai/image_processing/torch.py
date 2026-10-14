"""Deprecated compatibility facade for :mod:`panorai.core.torch`."""

from panorai.core.torch import *  # noqa: F403
from panorai.core.torch import __all__ as _core_all

__all__ = _core_all
