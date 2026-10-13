"""Compatibility alias for the core NumPy/native dispatcher."""

import sys

from panorai.core.numpy import _native as _implementation

sys.modules[__name__] = _implementation
