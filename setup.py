#!/usr/bin/env python

import os
from setuptools import setup, find_packages
import glob
import sys

# Load the README file as long_description
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

def read_requirements(path):
    with open(path) as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#") and not line.startswith("-e")
        ]

base_reqs = [
    "numpy>=2.3.0",
    "scipy",
    "joblib",
    "torch",
    "scikit-image",
    "opencv-python-headless",
    "pydantic>=2.0.0",
    "pyyaml",
    "open3d",
]

depth_reqs = set(read_requirements("requirements.txt"))
for req_file in glob.glob("panorai/depth/*/requirements*.txt"):
    depth_reqs.update(read_requirements(req_file))

setup(
    name="panorai",
    version="v3.0.19",
    author="Robinson Luiz Souza Garcia",
    author_email="rlsgarcia@icloud.com",
    description="Panoramic image projection and blending using Gnomonic and other spherical projections.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RobinsonGarcia/PanorAi",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=base_reqs,
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "flake8",
            "black",
            "mypy",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
        "depth": sorted(depth_reqs),
    },
    # If you have console scripts, you can uncomment and configure entry_points:
    # entry_points={
    #     "console_scripts": [
    #         "panorai-cli=panorai.cli.projection_pipeline_cli:main",
    #     ],
    # },
    # Include depth modules only when the extra is requested
    packages=[p for p in find_packages(include=['panorai', 'panorai.*'])
              if 'depth' in p and any('depth' in arg for arg in sys.argv)
              or not p.startswith('panorai.depth')],
    include_package_data=True,
    license="MIT",
    project_urls={
        "Bug Tracker": "https://github.com/RobinsonGarcia/PanorAi/issues",
        "Source Code": "https://github.com/RobinsonGarcia/PanorAi",
        "Documentation": "https://github.com/RobinsonGarcia/PanorAi/wiki",
    },
    keywords=[
        "panorama",
        "projection",
        "gnomonic",
        "spherical images",
        "3D reconstruction",
        "computer vision",
    ],
)
