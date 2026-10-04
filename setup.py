"""Build the optional first-party PanorAi native kernels."""

from __future__ import annotations

import os
import sys

from setuptools import Extension, setup

compile_args = ["/std:c++17"] if os.name == "nt" else ["-std=c++17"]
geometry_compile_args = (
    [*compile_args, "/fp:precise"]
    if os.name == "nt"
    else [*compile_args, "-ffp-contract=off"]
)
native_link_args = (
    ["-Wl,-no_uuid", "-Wl,-x"] if sys.platform == "darwin" else []
)
geometry_link_args = [
    *native_link_args,
    *(["-pthread"] if sys.platform.startswith("linux") else []),
]

setup(
    ext_modules=[
        Extension(
            "panorai._native._essential",
            sources=["panorai/_native/essential_kernels.cpp"],
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=native_link_args,
            optional=True,
        ),
        Extension(
            "panorai._native._geometry",
            sources=["panorai/_native/geometry_kernels.cpp"],
            language="c++",
            extra_compile_args=geometry_compile_args,
            extra_link_args=geometry_link_args,
            optional=True,
        ),
    ]
)
