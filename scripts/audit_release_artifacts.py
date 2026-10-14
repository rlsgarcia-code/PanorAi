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
BANNED_MEMBERS = {
    "PANORAI-README.md",
    "Untitled.ipynb",
    "configs/train_depth.yaml",
    "docs_audit.txt",
    "lmdb_report.py",
    "requirements.dev.txt",
    "requirements.tmp.txt",
    "requirements.txt",
    "run.sh",
}
LIMITS = {".whl": 2 * 1024 * 1024, ".gz": 5 * 1024 * 1024}
APPROVED_DOCUMENTATION_MEDIA_SHA256 = {
    "docs/_static/tutorials/application-features-mosaic-tunnel.jpg": (
        "54132e985d473013bfdf03a7767e4ca3aadc6981f83039091886e6a143642f2e"
    ),
    "docs/_static/tutorials/application-pose-range.jpg": (
        "7a491dc660b359df2f47611411113149aaafeebab91aa652058692909f8803e9"
    ),
    "docs/_static/tutorials/application-processing-boma.jpg": (
        "b79152afec7776b58cd04c2f9b616c7113333078eb47429c7389b7dc117675b9"
    ),
    "docs/_static/tutorials/application-projection-sundowner.jpg": (
        "51f48a1030e6fc9a15b7707ce918d487c2ab9d0fd1f422f3dec965ab3152d6b3"
    ),
    "docs/_static/tutorials/application-semantics-boiler-room.jpg": (
        "25c949dda66f0a9043ab6cbb80a72308eadd4a8cea82718fee17d6b62ab4b5be"
    ),
    "docs/_static/tutorials/pinhole-workflow-sundowner.jpg": (
        "dee93ce269ba3f95717cd50a31078f9cc0b7a32bedb38b3b15ac0b75b2834a4d"
    ),
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
    "docs/_static/tutorials/projected-canny.jpg": (
        "6f91d7ff0f61672c9769a8ef5ecb03cb8428e9fb88890055e3458c85eb4b9b68"
    ),
    "docs/_static/tutorials/spherical-gradient-components.jpg": (
        "ef6a2eb8472e2e3bacf54b970db74e78bb0a5c83a7a0fa3072ad34b085838ef3"
    ),
    "docs/_static/tutorials/spherical-dog-pyramid.jpg": (
        "41b430745479803c6285071b72a418b7f62012a1fa49c9ed31cbf29fd0391490"
    ),
    "docs/_static/tutorials/depth-anything-views.jpg": (
        "1349cb77a45128ce8b3c7a9c5c91f075e6ecf4479b144dd3e598b3bb773a83ce"
    ),
    "docs/_static/tutorials/return-path-points.jpg": (
        "2637b1269a1e8b4ae18caeb46965509bc7985807de1b6ce3fcdf0adfc17e6b90"
    ),
    "docs/_static/tutorials/return-path-regions.jpg": (
        "6edbe44e27122b94367e1236c0644e64a57a26ff3f7bbf47c66acce5e3f345be"
    ),
    "docs/_static/tutorials/return-path-view-evidence.jpg": (
        "fa026637cf008456abe0684ee51a097a5236f8a87bba6518e8b7611bcfa3df85"
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
    "docs/_static/tutorials/spherical-cam-imagenet-lakeside.jpg": (
        "dd0827f6e6cd5301d5ffe14d086cded0ff7ce02e8955d7a95df9b7cd5c65f43c"
    ),
    "docs/_static/tutorials/spherical-cam-places365-forest.jpg": (
        "bcfb1df74b4f57cfacab38d8695015f4eb3ed70d16734b2afe8311ff5d9e9861"
    ),
    "docs/_static/tutorials/spherical-cam-openclip-path.jpg": (
        "baef726a3330669c2d84e711efb8b96e714890d44771f38a14e6d643eff241df"
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
    failures.extend(sorted(BANNED_MEMBERS & normalized))
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
        "panorai/graph/__init__.py",
        "panorai/graph/_models.py",
        "panorai/graph/association.py",
        "panorai/graph/builder.py",
        "panorai/graph/localization.py",
        "panorai/graph/query.py",
        "panorai/graph/serialization.py",
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
            "docs/release-3.7.0-checklist.md",
            "contracts/public-api-surfaces-v2.json",
            "contracts/public-api-surfaces-v2.schema.json",
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
