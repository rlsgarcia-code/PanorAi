#!/usr/bin/env python3
"""Audit wheel and sdist contents against the PanorAi release policy."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import sys
import tarfile
from typing import Callable
import zipfile

BANNED_SUFFIXES = {
    ".ckpt",
    ".ipynb",
    ".jpeg",
    ".jpg",
    ".mp4",
    ".npz",
    ".onnx",
    ".pcd",
    ".png",
    ".pt",
    ".pth",
}
BANNED_PARTS = {".idea", "__pycache__", "ZoeDepth_not_used"}
BANNED_PART_PREFIXES = {"panorai_models"}
BANNED_ROOTS = {"artifacts", "datasets", "notebooks", "reports", "tests"}
LIMITS = {".whl": 2 * 1024 * 1024, ".gz": 5 * 1024 * 1024}
APPROVED_DOCUMENTATION_MEDIA_SHA256 = {
    "docs/_static/tutorials/feature-detectors.jpg": (
        "3acc770c058e171570225b052ff2cb3e02d32ecb43a8b37f0ff560e30ed91a82"
    ),
    "docs/_static/tutorials/feature-matches.jpg": (
        "78708c15dfbb0802c6250524fba59c8378edb4cc1d2db10a7604e1a1699a3ffc"
    ),
    "docs/_static/tutorials/nature-reserve-forest-erp.jpg": (
        "5333c2cecc55468fcbc64a252f4bc097fab064cae70bead4f76d6c2e6101c524"
    ),
    "docs/_static/tutorials/poly-haven-studio-erp.jpg": (
        "9b6e2b7521e2cf35d998f4087840c4bf98b480467bd753d719533228c984be8a"
    ),
    "docs/_static/tutorials/spherical-stereo-synthetic.png": (
        "fc0b5b1b94f507874178b4ee89403e31b1a1e20e8c68d4662de724184c0a14c6"
    ),
    "docs/_static/tutorials/spherical-dog-sift.jpg": (
        "2f50b9a0275c519e196c1af9e6fd836885e9df452e360a38fc4b3086cb46eadb"
    ),
    "docs/_static/tutorials/spherical-histogram-equalization.jpg": (
        "bc6c04d6592ced4894ca9863b6e07d5aeeeaf63dc8c28c8c369fe8d2fb826b19"
    ),
    "docs/_static/tutorials/spherical-image-processing.jpg": (
        "70136e1076275798175e358677f098305aadb29c4085d943ecd4507a8b5a48c3"
    ),
}
ADAPTER_ONLY_EXCLUDED_PREFIXES = {
    "panorai/depth/DepthAnythingV2/",
    "panorai/depth/Dust3r/",
    "panorai/depth/Metric3D/",
    "panorai/depth/ZoeDepth_not_used/",
    "panorai/depth/custom_data/",
    "panorai/depth/trainers/",
    "panorai/depth/training/",
}
VENDORED_COMPONENTS = {
    "DepthAnythingV2": {
        "prefix": "panorai/depth/DepthAnythingV2/",
        "legal": ("panorai/depth/DepthAnythingV2/LICENSE",),
    },
    "Dust3r": {
        "prefix": "panorai/depth/Dust3r/",
        "legal": (
            "panorai/depth/Dust3r/LICENSE",
            "panorai/depth/Dust3r/NOTICE",
        ),
    },
    "CroCo": {
        "prefix": "panorai/depth/Dust3r/croco/",
        "legal": (
            "panorai/depth/Dust3r/croco/LICENSE",
            "panorai/depth/Dust3r/croco/NOTICE",
        ),
    },
    "Metric3D": {
        "prefix": "panorai/depth/Metric3D/",
        "legal": ("panorai/depth/Metric3D/LICENSE",),
    },
}
NONCOMMERCIAL_COMPONENTS = {"Dust3r", "CroCo"}


def _members(path: Path) -> tuple[list[str], Callable[[str], bytes]]:
    if path.suffix == ".whl":
        archive = zipfile.ZipFile(path)
        return archive.namelist(), archive.read
    archive = tarfile.open(path, "r:gz")

    def read(name: str) -> bytes:
        member = archive.extractfile(name)
        return b"" if member is None else member.read()

    return archive.getnames(), read


def _normalize(path: Path, name: str) -> str:
    parts = PurePosixPath(name).parts
    return name if path.suffix == ".whl" else "/".join(parts[1:])


def _has_legal_file(normalized: set[str], expected: str) -> bool:
    """Accept source-tree placement or wheel ``dist-info/licenses`` placement."""

    return expected in normalized or any(
        name.endswith(f".dist-info/licenses/{expected}") for name in normalized
    )


def _metadata_text(
    path: Path,
    names_by_normalized: dict[str, str],
    read: Callable[[str], bytes],
) -> str:
    if path.suffix != ".whl" and "PKG-INFO" in names_by_normalized:
        return read(names_by_normalized["PKG-INFO"]).decode("utf-8", errors="replace")
    suffix = ".dist-info/METADATA"
    candidates = [name for name in names_by_normalized if name.endswith(suffix)]
    if len(candidates) != 1:
        return ""
    return read(names_by_normalized[candidates[0]]).decode("utf-8", errors="replace")


def _claims_mit_only(metadata: str) -> bool:
    fields = [
        line.strip()
        for line in metadata.splitlines()
        if line.startswith(
            ("License:", "License-Expression:", "Classifier: License ::")
        )
    ]
    expressions = [
        line.partition(":")[2].strip()
        for line in fields
        if line.startswith("License-Expression:")
    ]
    if expressions:
        return all(expression.upper() == "MIT" for expression in expressions)
    licenses = [
        line.partition(":")[2].strip() for line in fields if line.startswith("License:")
    ]
    if licenses:
        return all(license_name.upper() == "MIT" for license_name in licenses)
    return fields == ["Classifier: License :: OSI Approved :: MIT License"]


def audit(path: Path) -> None:
    failures: list[str] = []
    limit = LIMITS[path.suffix]
    if path.stat().st_size > limit:
        failures.append(f"size:{path.stat().st_size}>{limit}")
    names, read = _members(path)
    for name in names:
        parts = PurePosixPath(name).parts
        relative = parts if path.suffix == ".whl" else parts[1:]
        if PurePosixPath(name).suffix.lower() in BANNED_SUFFIXES:
            normalized_name = _normalize(path, name)
            expected_digest = APPROVED_DOCUMENTATION_MEDIA_SHA256.get(normalized_name)
            approved_sdist_documentation = (
                path.suffix != ".whl"
                and expected_digest is not None
                and hashlib.sha256(read(name)).hexdigest() == expected_digest
            )
            if not approved_sdist_documentation:
                failures.append(name)
        if BANNED_PARTS.intersection(parts):
            failures.append(name)
        if any(
            part.startswith(prefix) for part in parts for prefix in BANNED_PART_PREFIXES
        ):
            failures.append(name)
        if relative and relative[0] in BANNED_ROOTS:
            failures.append(name)
        if "pyarmor" in name.lower():
            failures.append(name)

    names_by_normalized = {_normalize(path, name): name for name in names}
    normalized = set(names_by_normalized)
    for prefix in sorted(ADAPTER_ONLY_EXCLUDED_PREFIXES):
        if any(name.startswith(prefix) for name in normalized):
            failures.append(f"adapter-only-boundary:{prefix}")
    required = {
        "panorai/__init__.py",
        "panorai/_native/__init__.py",
        "panorai/depth/__init__.py",
        "panorai/depth/_adapters.py",
        "panorai/depth/registry.py",
        "panorai/geometry/__init__.py",
        "panorai/geometry/_contracts.py",
        "panorai/geometry/_engine.py",
        "panorai/geometry/_native.py",
        "panorai/geometry/_projectors.py",
        "panorai/estimators/_native.py",
        "panorai/experimental/__init__.py",
        "panorai/experimental/deep_learning/__init__.py",
        "panorai/experimental/deep_learning/depth.py",
        "panorai/experimental/deep_learning/fcn.py",
        "panorai/experimental/deep_learning/pretrained.py",
        "panorai/image_processing/torch.py",
        "panorai/pcd/__init__.py",
        "panorai/pcd/data.py",
        "panorai/pcd/handler.py",
        "panorai/stereo/__init__.py",
        "panorai/stereo/_dense.py",
        "panorai/stereo/_visualization.py",
    }
    failures.extend(f"missing:{name}" for name in sorted(required - normalized))
    if path.suffix == ".whl":
        for kernel in ("essential", "geometry"):
            native_extensions = [
                name
                for name in normalized
                if name.startswith(f"panorai/_native/_{kernel}")
                and PurePosixPath(name).suffix.lower() in {".pyd", ".so"}
            ]
            if len(native_extensions) != 1:
                failures.append(
                    "native-policy:wheel must contain exactly one compiled "
                    f"{kernel} kernel"
                )
    else:
        if "panorai.egg-info/scm_version.json" in normalized:
            failures.append("stale-scm-cache:panorai.egg-info/scm_version.json")
        for source_member in (
            "setup.py",
            "panorai/_native/essential_kernels.cpp",
            "panorai/_native/geometry_kernels.cpp",
            "docs/release-3.6.0-checklist.md",
            "scripts/run_geometry_conformance.py",
            "scripts/verify_geometry_fixture_integrity.py",
        ):
            if source_member not in normalized:
                failures.append(f"missing:{source_member}")
    metadata = _metadata_text(path, names_by_normalized, read)
    if not metadata:
        failures.append("missing:distribution metadata")
    if not _has_legal_file(normalized, "LICENSE"):
        failures.append("missing:root LICENSE")

    included_components: set[str] = set()
    for component, policy in VENDORED_COMPONENTS.items():
        prefix = policy["prefix"]
        has_code = any(
            name.startswith(prefix) and name.endswith(".py") for name in normalized
        )
        if not has_code:
            continue
        included_components.add(component)
        for legal_file in policy["legal"]:
            if not _has_legal_file(normalized, legal_file):
                failures.append(f"missing:{component}:{legal_file}")

    if included_components and not _has_legal_file(
        normalized, "THIRD_PARTY_NOTICES.md"
    ):
        failures.append(
            "missing:THIRD_PARTY_NOTICES.md for vendored third-party implementations"
        )
    incompatible = sorted(included_components & NONCOMMERCIAL_COMPONENTS)
    if incompatible and _claims_mit_only(metadata):
        failures.append(
            "license-policy:MIT-only distribution metadata includes non-commercial/"
            f"share-alike component(s): {', '.join(incompatible)}"
        )

    engine = next(
        (name for name in names if name.endswith("panorai/geometry/_engine.py")),
        None,
    )
    if engine is None or b"def equirectangular_to_gnomonic" not in read(engine):
        failures.append("geometry engine is not readable Python source")
    adapter = next(
        (name for name in names if name.endswith("panorai/depth/_adapters.py")),
        None,
    )
    if adapter is None or any(
        loader not in read(adapter)
        for loader in (
            b"def load_dav2_model",
            b"def load_m3dv2_model",
            b"def load_dust3r_model",
            b"def load_zoe_model",
        )
    ):
        failures.append("depth adapter surface is not readable Python source")
    if failures:
        raise SystemExit(
            f"{path}: forbidden or missing members:\n"
            + "\n".join(sorted(set(failures)))
        )
    print(f"OK {path}: {len(names)} members, {path.stat().st_size} bytes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    failed = False
    for artifact in parser.parse_args().artifacts:
        try:
            audit(artifact)
        except SystemExit as error:
            print(error, file=sys.stderr)
            failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
