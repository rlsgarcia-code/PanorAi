#!/usr/bin/env python3
"""Emit one installed-wheel matrix record for the reviewed DoG profile."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import distribution, version
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
from numpy.typing import NDArray


PROFILE_ID = "panorai-spherical-dog-features/reviewed-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _validate_direct_url(payload: dict[str, Any], wheel: Path) -> None:
    parsed = urlparse(str(payload.get("url", "")))
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise AssertionError("installed PanorAi does not name a local wheel origin")
    installed_from = Path(unquote(parsed.path)).resolve()
    if installed_from != wheel.resolve():
        raise AssertionError("installed PanorAi came from a different wheel")
    archive = payload.get("archive_info")
    if not isinstance(archive, dict):
        raise AssertionError("installed PanorAi has no wheel archive binding")
    expected = _sha256(wheel)
    hashes = archive.get("hashes")
    actual = hashes.get("sha256") if isinstance(hashes, dict) else None
    if actual != expected and archive.get("hash") != f"sha256={expected}":
        raise AssertionError("installed PanorAi wheel SHA-256 does not match")


def _validate_source_checkout(*, source_root: Path, source_commit: str) -> None:
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ValueError("source_commit must be a lowercase 40-character Git SHA")
    if _git_output(source_root, "rev-parse", "HEAD") != source_commit:
        raise AssertionError("source_commit does not match the source checkout HEAD")
    dirty = _git_output(source_root, "status", "--porcelain=v1", "--untracked-files=no")
    if dirty:
        raise AssertionError("source checkout has tracked modifications")
    tracked = set(
        _git_output(
            source_root,
            "ls-files",
            "--",
            "panorai",
            "scripts/run_spherical_application_evidence.py",
        ).splitlines()
    )
    importable_suffixes = {".py", ".pyi", ".so", ".pyd", ".dll", ".dylib"}
    importable: set[str] = set()
    package_root = source_root / "panorai"
    if package_root.is_dir():
        importable.update(
            path.relative_to(source_root).as_posix()
            for path in package_root.rglob("*")
            if path.is_file() and path.suffix.lower() in importable_suffixes
        )
    oracle = source_root / "scripts" / "run_spherical_application_evidence.py"
    if oracle.is_file():
        importable.add(oracle.relative_to(source_root).as_posix())
    unexpected = sorted(importable - tracked)
    if unexpected:
        raise AssertionError(
            "source checkout has untracked importable artifacts: "
            + ", ".join(unexpected)
        )


def _validate_artifact_version(*, source_commit: str, artifact_version: str) -> None:
    expected_marker = f"g{source_commit[:9]}"
    if expected_marker not in artifact_version.split("+")[-1].split("."):
        raise AssertionError("installed artifact version is not bound to source_commit")


def _validate_installed_artifact(
    *, source_root: Path, wheel: Path, source_commit: str
) -> str:
    if not wheel.is_file():
        raise FileNotFoundError(wheel)
    installed_version = version("panorai")
    _validate_source_checkout(
        source_root=source_root,
        source_commit=source_commit,
    )
    _validate_artifact_version(
        source_commit=source_commit,
        artifact_version=installed_version,
    )
    direct_url_text = distribution("panorai").read_text("direct_url.json")
    if direct_url_text is None:
        raise AssertionError("installed PanorAi has no direct_url.json wheel binding")
    direct_url = json.loads(direct_url_text)
    if not isinstance(direct_url, dict):
        raise AssertionError("installed PanorAi direct_url.json is invalid")
    _validate_direct_url(direct_url, wheel)
    wheel_version = wheel.name.split("-", 2)[1]
    if wheel_version != installed_version:
        raise AssertionError("installed PanorAi version differs from the named wheel")
    return installed_version


def _texture() -> NDArray[np.float32]:
    y, x = np.indices((96, 192), dtype=np.float32)
    base = 0.5 + 0.22 * np.sin(x / 7.0) + 0.18 * np.cos(y / 5.0)
    return np.stack((base, np.roll(base, 9, axis=1), base), axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--platform-id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {args.output}")
    if not args.wheel.is_file():
        raise FileNotFoundError(args.wheel)

    import panorai
    from panorai.applications import SphericalDoGFeatures

    origin = Path(panorai.__file__).resolve()
    try:
        origin.relative_to(args.source_root.resolve())
    except ValueError:
        isolated = True
    else:
        isolated = False
    if not isolated:
        raise AssertionError(f"panorai resolved inside source checkout: {origin}")

    installed_version = _validate_installed_artifact(
        source_root=args.source_root,
        wheel=args.wheel,
        source_commit=args.source_commit,
    )

    method = SphericalDoGFeatures.reviewed_profile(convolution_backend="numpy")
    result = method(_texture(), panorama_id="installed-reviewed-dog")
    restored = SphericalDoGFeatures.from_config(method.to_config())
    if not len(result) or result.descriptors.shape != (len(result), 128):
        raise AssertionError("reviewed DoG application produced no real descriptors")
    if restored != method or restored.profile_id != PROFILE_ID:
        raise AssertionError("reviewed DoG profile identity did not round-trip")
    if SphericalDoGFeatures().stability != "experimental":
        raise AssertionError("generic DoG constructor was promoted unexpectedly")

    record = {
        "application": "spherical-dog-reviewed-profile",
        "evaluation": "installed-wheel",
        "profile_id": PROFILE_ID,
        "source_commit": args.source_commit,
        "wheel_sha256": _sha256(args.wheel),
        "artifact_version": installed_version,
        "platform": args.platform_id
        or f"{platform.system().lower()}-{platform.machine().lower()}",
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "metrics": {
            "exact_commit_and_wheel_sha_bound": True,
            "source_checkout_isolated": True,
            "profile_identity_roundtrip": True,
            "real_application_execution": True,
            "generic_constructor_remains_experimental": True,
            "feature_count": len(result),
        },
        "origin": str(origin),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"output={args.output} sha256={_sha256(args.output)}")


if __name__ == "__main__":
    main()
