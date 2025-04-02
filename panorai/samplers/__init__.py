"""
panorai_data_rewritten.pipelines.sampler
========================================

Sampler sub-package containing:
 - Sampler base classes
 - Default samplers (cube, fibonacci, etc.)
 - Sampler registry and configuration
"""

from .registry import SamplerRegistry
from .base_samplers import Sampler
from .default_samplers import (CubeSampler, IcosahedronSampler, FibonacciSampler,
                               BlueNoiseSampler,  SpiralSampler)

__all__ = [
    "SamplerRegistry",
    "Sampler",
    "CubeSampler",
    "IcosahedronSampler",
    "FibonacciSampler",
    "BlueNoiseSampler", 
    #"HEALPixSampler", => deprecated for now
    "SpiralSampler"
]