"""Build the optional first-party PanorAi native kernels."""

from __future__ import annotations

import os

from setuptools import Extension, setup


compile_args = ["/std:c++17"] if os.name == "nt" else ["-std=c++17"]

setup(
    ext_modules=[
        Extension(
            "panorai._native._essential",
            sources=["panorai/_native/essential_kernels.cpp"],
            language="c++",
            extra_compile_args=compile_args,
            optional=True,
        )
    ]
)
