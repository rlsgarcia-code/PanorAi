#!/usr/bin/env python

import os
from setuptools import setup, find_packages

# Load the README as long_description
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="panorai",  # Package name
    version="v1.4.5",  # Semantic versioning
    author="Robinson Luiz Souza Garcia",
    author_email="rlsgarcia@icloud.com",
    description="A Python package for panoramic image projection and blending using Gnomonic (and other) projections.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RobinsonGarcia/PanorAi",  
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),  # Adjust exclusions as needed
    classifiers=[
        "Development Status :: 3 - Alpha",  # Update based on project maturity
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Specify the minimum Python version
    install_requires=[
        "numpy",
        "opencv-python-headless",  # Use headless if no GUI is needed
        "scikit-image",
        "scipy",
        "joblib",
        "pydantic>=2.0.0",
        "spherical-projections==0.1.2b0"
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "flake8",  # Linter
            "black",  # Code formatter
            "mypy",  # Type checker
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    entry_points={
        "console_scripts": [
            "panorai-cli=panorai.cli.projection_pipeline_cli:main",
        ],
    },
    include_package_data=True,  # Includes non-code files specified in MANIFEST.in
    license="MIT",  # Or whichever license you use
    project_urls={
        "Bug Tracker": "https://github.com/RobinsonGarcia/PanorAi/issues",
        "Source Code": "https://github.com/RobinsonGarcia/PanorAi",
        "Documentation": "https://github.com/RobinsonGarcia/PanorAi/wiki",  # Adjust if you have docs
    },
    keywords=[
        "panorama",
        "image processing",
        "projection",
        "gnomonic",
        "computer vision",
    ],
)
