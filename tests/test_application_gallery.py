from __future__ import annotations

import hashlib
import importlib.util
from io import BytesIO
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/generate_application_gallery.py"
SPEC = importlib.util.spec_from_file_location("generate_application_gallery", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GALLERY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GALLERY
SPEC.loader.exec_module(GALLERY)


def _asset(payload: bytes):
    return GALLERY.SourceAsset(
        slug="fixture",
        title="Fixture",
        filename="fixture.hdr",
        url="https://example.invalid/fixture.hdr",
        page_url="https://example.invalid/fixture",
        authors=("Fixture Author",),
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_acquire_source_reuses_a_verified_cache_without_network(tmp_path: Path) -> None:
    payload = b"verified public source"
    asset = _asset(payload)
    (tmp_path / asset.filename).write_bytes(payload)

    def fail_if_opened(_url: str):
        raise AssertionError("the network must not be used for a valid cache")

    assert GALLERY.acquire_source(asset, tmp_path, opener=fail_if_opened) == (
        tmp_path / asset.filename
    )


def test_acquire_source_rejects_a_corrupt_cached_source(tmp_path: Path) -> None:
    asset = _asset(b"expected")
    (tmp_path / asset.filename).write_bytes(b"corrupt!")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        GALLERY.acquire_source(asset, tmp_path, offline=True)


def test_acquire_source_offline_requires_a_cached_source(tmp_path: Path) -> None:
    asset = _asset(b"expected")

    with pytest.raises(FileNotFoundError, match="offline mode"):
        GALLERY.acquire_source(asset, tmp_path, offline=True)


def test_acquire_source_downloads_and_verifies_missing_bytes(tmp_path: Path) -> None:
    payload = b"downloaded public source"
    asset = _asset(payload)
    opened: list[str] = []

    def opener(url: str):
        opened.append(url)
        return BytesIO(payload)

    path = GALLERY.acquire_source(asset, tmp_path, opener=opener)

    assert opened == [asset.url]
    assert path.read_bytes() == payload
    assert not (tmp_path / f".{asset.filename}.partial").exists()


def test_public_source_manifest_has_unique_checksum_pinned_assets() -> None:
    assets = GALLERY.SOURCE_ASSETS
    assert {asset.slug for asset in assets} == {
        "boiler_room",
        "boma",
        "modern_buildings",
        "sundowner_overlook",
    }
    assert len({asset.filename for asset in assets}) == len(assets)
    assert all(len(asset.sha256) == 64 for asset in assets)
    assert all(asset.url.startswith("https://dl.polyhaven.org/") for asset in assets)
