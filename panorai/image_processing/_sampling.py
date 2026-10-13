"""Compatibility alias for the core spherical sampling implementation."""

import sys

from panorai.core import _sampling as _implementation

sys.modules[__name__] = _implementation
