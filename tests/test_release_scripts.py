from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import tarfile
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load_script("audit_release_artifacts.py")
NORMALIZE = _load_script("normalize_sdist.py")
SMOKE = _load_script("release_smoke.py")
SELECT_WHEEL = _load_script("select_compatible_wheel.py")
VERIFY_INDEX = _load_script("verify_index_artifacts.py")

REQUIRED_MEMBERS = {
    "panorai/__init__.py": b"",
    "panorai/_native/__init__.py": b"",
    "panorai/depth/__init__.py": b"",
    "panorai/depth/_adapters.py": (
        b"def load_dav2_model():\n    pass\n"
        b"def load_m3dv2_model():\n    pass\n"
        b"def load_dust3r_model():\n    pass\n"
        b"def load_zoe_model():\n    pass\n"
    ),
    "panorai/depth/registry.py": b"class ModelRegistry:\n    pass\n",
    "panorai/geometry/__init__.py": b"",
    "panorai/geometry/_contracts.py": b"",
    "panorai/geometry/_engine.py": b"def equirectangular_to_gnomonic():\n    pass\n",
    "panorai/geometry/_native.py": b"",
    "panorai/geometry/_projectors.py": b"",
    "panorai/estimators/_native.py": b"",
    "panorai/pcd/__init__.py": b"",
    "panorai/pcd/data.py": b"class PCD:\n    pass\n",
    "panorai/pcd/handler.py": b"class PCDHandler:\n    pass\n",
    "panorai/stereo/__init__.py": b"",
    "panorai/stereo/_dense.py": b"",
    "panorai/stereo/_visualization.py": b"",
}
METADATA = b"Metadata-Version: 2.4\nName: panorai\nVersion: 3.1.0\nLicense: MIT\n"


def _wheel(tmp_path: Path, extra: dict[str, bytes] | None = None) -> Path:
    members = {
        **REQUIRED_MEMBERS,
        "panorai/_native/_essential.cpython-312-test.so": b"native",
        "panorai/_native/_geometry.cpython-312-test.so": b"native",
        "panorai-3.1.0.dist-info/METADATA": METADATA,
        "panorai-3.1.0.dist-info/licenses/LICENSE": b"MIT\n",
        **(extra or {}),
    }
    path = tmp_path / "panorai-3.1.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


def _sdist(tmp_path: Path, extra: dict[str, bytes] | None = None) -> Path:
    members = {
        **REQUIRED_MEMBERS,
        "setup.py": b"from setuptools import setup\nsetup()\n",
        "panorai/_native/essential_kernels.cpp": b"// native\n",
        "panorai/_native/geometry_kernels.cpp": b"// native\n",
        "docs/release-3.4.0-checklist.md": b"# PanorAi 3.4.0 release checklist\n",
        "scripts/run_geometry_conformance.py": b"# conformance\n",
        "scripts/verify_geometry_fixture_integrity.py": b"# fixture verifier\n",
        "PKG-INFO": METADATA,
        "LICENSE": b"MIT\n",
        **(extra or {}),
    }
    path = tmp_path / "panorai-3.1.0.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(f"panorai-3.1.0/{name}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return path


@pytest.mark.parametrize("factory", [_wheel, _sdist])
def test_clean_minimal_artifact_passes_policy(tmp_path: Path, factory) -> None:
    AUDIT.audit(factory(tmp_path))


def test_only_checksum_pinned_tutorial_media_is_allowed_in_sdist(
    tmp_path: Path,
) -> None:
    assert set(AUDIT.APPROVED_DOCUMENTATION_MEDIA_SHA256) == {
        "docs/_static/tutorials/feature-detectors.jpg",
        "docs/_static/tutorials/feature-matches.jpg",
        "docs/_static/tutorials/nature-reserve-forest-erp.jpg",
        "docs/_static/tutorials/poly-haven-studio-erp.jpg",
        "docs/_static/tutorials/spherical-dog-sift.jpg",
        "docs/_static/tutorials/spherical-histogram-equalization.jpg",
        "docs/_static/tutorials/spherical-image-processing.jpg",
        "docs/_static/tutorials/spherical-stereo-synthetic.png",
    }
    media = {
        name: (ROOT / name).read_bytes()
        for name in AUDIT.APPROVED_DOCUMENTATION_MEDIA_SHA256
    }
    AUDIT.audit(_sdist(tmp_path, media))

    wheel = _wheel(tmp_path, media)
    with pytest.raises(SystemExit, match="feature-detectors.jpg"):
        AUDIT.audit(wheel)

    mutated = dict(media)
    mutated["docs/_static/tutorials/feature-detectors.jpg"] += b"changed"
    with pytest.raises(SystemExit, match="feature-detectors.jpg"):
        AUDIT.audit(_sdist(tmp_path, mutated))


@pytest.mark.parametrize("factory", [_wheel, _sdist])
def test_unapproved_media_remains_rejected(tmp_path: Path, factory) -> None:
    artifact = factory(tmp_path, {"docs/_static/tutorials/private.jpg": b"image"})
    with pytest.raises(SystemExit, match="private.jpg"):
        AUDIT.audit(artifact)


@pytest.mark.parametrize(
    ("kernel", "message"),
    [
        ("essential", "compiled essential kernel"),
        ("geometry", "compiled geometry kernel"),
    ],
)
def test_wheel_without_compiled_native_kernel_fails_policy(
    tmp_path: Path, kernel: str, message: str
) -> None:
    artifact = _wheel(tmp_path)
    rewritten = tmp_path / "without-native.whl"
    with zipfile.ZipFile(artifact) as source, zipfile.ZipFile(rewritten, "w") as target:
        for name in source.namelist():
            if f"panorai/_native/_{kernel}" not in name:
                target.writestr(name, source.read(name))

    with pytest.raises(SystemExit, match=message):
        AUDIT.audit(rewritten)


@pytest.mark.parametrize("factory", [_wheel, _sdist])
@pytest.mark.parametrize(
    "member",
    [
        "docs/reference/panorai_models.Dust3r.dust3r.training.rst",
        "docs/reference/panorai_models_backup.rst",
        "docs/reference/panorai_models-Dust3r.rst",
        "panorai_models/module.py",
    ],
)
def test_namespaced_panorai_models_reference_stub_is_rejected(
    tmp_path: Path, factory, member: str
) -> None:
    artifact = factory(tmp_path, {member: b".. automodule:: panorai_models\n"})

    with pytest.raises(SystemExit) as caught:
        AUDIT.audit(artifact)

    assert member in str(caught.value)


@pytest.mark.parametrize("factory", [_wheel, _sdist])
def test_nonprefixed_panorai_models_text_is_not_overblocked(
    tmp_path: Path, factory
) -> None:
    artifact = factory(
        tmp_path,
        {"docs/reference/my_panorai_models_notes.rst": b"Project notes\n"},
    )

    AUDIT.audit(artifact)


def test_manifest_excludes_generated_panorai_models_reference_stubs() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-exclude docs/reference panorai_models*.rst" in manifest


def test_manifest_keeps_conformance_fixture_verifier_pair_in_sdist() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    assert "include scripts/run_geometry_conformance.py" in manifest
    assert "include scripts/verify_geometry_fixture_integrity.py" in manifest
    assert "include docs/release-3.4.0-checklist.md" in manifest


@pytest.mark.parametrize("factory", [_wheel, _sdist])
def test_vendored_dust3r_is_rejected_even_with_legal_payload(
    tmp_path: Path, factory
) -> None:
    artifact = factory(
        tmp_path,
        {
            "panorai/depth/Dust3r/dust3r/model.py": b"class Model:\n    pass\n",
            "panorai/depth/Dust3r/LICENSE": b"CC BY-NC-SA 4.0\n",
            "panorai/depth/Dust3r/NOTICE": b"Upstream notice\n",
            "THIRD_PARTY_NOTICES.md": b"DUSt3R: CC BY-NC-SA 4.0\n",
        },
    )

    with pytest.raises(SystemExit) as caught:
        AUDIT.audit(artifact)

    message = str(caught.value)
    assert "adapter-only-boundary:panorai/depth/Dust3r/" in message


def test_noncommercial_component_cannot_hide_behind_mit_metadata(
    tmp_path: Path,
) -> None:
    artifact = _wheel(
        tmp_path,
        {
            "panorai/depth/Dust3r/dust3r/model.py": b"pass\n",
            "panorai/depth/Dust3r/LICENSE": b"CC BY-NC-SA 4.0\n",
            "panorai/depth/Dust3r/NOTICE": b"Upstream notice\n",
            "THIRD_PARTY_NOTICES.md": b"DUSt3R: CC BY-NC-SA 4.0\n",
        },
    )

    with pytest.raises(SystemExit, match="adapter-only-boundary"):
        AUDIT.audit(artifact)


@pytest.mark.parametrize(
    "prefix",
    [
        "panorai/depth/DepthAnythingV2/",
        "panorai/depth/Dust3r/",
        "panorai/depth/Metric3D/",
        "panorai/depth/ZoeDepth_not_used/",
        "panorai/depth/custom_data/",
        "panorai/depth/trainers/",
        "panorai/depth/training/",
    ],
)
@pytest.mark.parametrize("factory", [_wheel, _sdist])
def test_adapter_only_boundary_rejects_every_excluded_tree(
    tmp_path: Path, factory, prefix: str
) -> None:
    artifact = factory(tmp_path, {f"{prefix}module.py": b"pass\n"})

    with pytest.raises(SystemExit, match=f"adapter-only-boundary:{prefix}"):
        AUDIT.audit(artifact)


@pytest.mark.parametrize("factory", [_wheel, _sdist])
def test_depth_adapter_and_pcd_surface_is_required(tmp_path: Path, factory) -> None:
    artifact = factory(tmp_path)
    if artifact.suffix == ".whl":
        rewritten = tmp_path / "missing-adapter.whl"
        with (
            zipfile.ZipFile(artifact) as source,
            zipfile.ZipFile(rewritten, "w") as target,
        ):
            for name in source.namelist():
                if name != "panorai/depth/_adapters.py":
                    target.writestr(name, source.read(name))
    else:
        rewritten = tmp_path / "missing-adapter.tar.gz"
        with (
            tarfile.open(artifact, "r:gz") as source,
            tarfile.open(rewritten, "w:gz") as target,
        ):
            for member in source.getmembers():
                if member.name.endswith("panorai/depth/_adapters.py"):
                    continue
                payload = source.extractfile(member)
                target.addfile(member, payload)

    with pytest.raises(SystemExit, match="missing:panorai/depth/_adapters.py"):
        AUDIT.audit(rewritten)


def test_combined_license_expression_is_not_misreported_as_mit_only() -> None:
    metadata = "License-Expression: MIT AND CC-BY-NC-SA-4.0\n"
    assert not AUDIT._claims_mit_only(metadata)


def test_root_license_is_required(tmp_path: Path) -> None:
    artifact = _wheel(tmp_path)
    rewritten = tmp_path / "without-license.whl"
    with zipfile.ZipFile(artifact) as source, zipfile.ZipFile(rewritten, "w") as target:
        for name in source.namelist():
            if not name.endswith("/licenses/LICENSE"):
                target.writestr(name, source.read(name))

    with pytest.raises(SystemExit, match="missing:root LICENSE"):
        AUDIT.audit(rewritten)


def test_installed_origin_rejects_checkout_and_accepts_site_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "checkout"
    source_file = source_root / "panorai" / "__init__.py"
    installed_file = tmp_path / "venv/site-packages/panorai/__init__.py"
    source_file.parent.mkdir(parents=True)
    installed_file.parent.mkdir(parents=True)
    source_file.touch()
    installed_file.touch()
    monkeypatch.setattr(SMOKE, "version", lambda _: "3.1.0")

    with pytest.raises(AssertionError, match="source checkout"):
        SMOKE.assert_installed_origin(
            str(source_file), source_root=source_root, package_version="3.1.0"
        )

    assert (
        SMOKE.assert_installed_origin(
            str(installed_file),
            source_root=source_root,
            package_version="3.1.0",
            expected_version="3.1.0",
        )
        == installed_file.resolve()
    )


def test_installed_origin_rejects_metadata_version_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed_file = tmp_path / "site-packages/panorai/__init__.py"
    installed_file.parent.mkdir(parents=True)
    installed_file.touch()
    monkeypatch.setattr(SMOKE, "version", lambda _: "3.1.0")

    with pytest.raises(AssertionError, match="version mismatch"):
        SMOKE.assert_installed_origin(
            str(installed_file),
            source_root=tmp_path / "checkout",
            package_version="3.0.19",
        )


def test_native_release_smoke_executes_when_backend_is_built() -> None:
    from panorai.estimators import native_kernels_available
    from panorai.geometry._native import native_geometry_available

    if not native_kernels_available() or not native_geometry_available():
        pytest.skip("optional native kernels are not built")
    SMOKE.assert_native_estimator()
    SMOKE.assert_native_geometry()
    SMOKE.assert_spherical_stereo()


def test_sdist_normalization_is_byte_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    for path, mtime, owner in ((first, 11, "alice"), (second, 99, "bob")):
        with tarfile.open(path, "w:gz") as archive:
            directory = tarfile.TarInfo("panorai-3.1.0/")
            directory.type = tarfile.DIRTYPE
            directory.mtime = mtime
            directory.uname = owner
            archive.addfile(directory)
            content = b"version = '3.1.0'\n"
            module = tarfile.TarInfo("panorai-3.1.0/panorai/__init__.py")
            module.size = len(content)
            module.mtime = mtime
            module.uname = owner
            archive.addfile(module, io.BytesIO(content))

    NORMALIZE.normalize_sdist(first, epoch=123456789)
    NORMALIZE.normalize_sdist(second, epoch=123456789)

    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, "r:gz") as archive:
        members = archive.getmembers()
        assert {member.mtime for member in members} == {123456789}
        assert {member.uid for member in members} == {0}
        assert {member.uname for member in members} == {""}
        assert (
            archive.extractfile("panorai-3.1.0/panorai/__init__.py").read()
            == b"version = '3.1.0'\n"
        )


def test_index_artifact_verification_requires_exact_names_and_hashes(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "panorai-3.2.1-cp312-cp312-manylinux_2_28_x86_64.whl"
    sdist = tmp_path / "panorai-3.2.1.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    expected = VERIFY_INDEX.local_digests([wheel, sdist])
    payload = {
        "urls": [
            {"filename": name, "digests": {"sha256": digest}}
            for name, digest in expected.items()
        ]
    }
    actual = VERIFY_INDEX.index_digests(payload)
    VERIFY_INDEX.verify_exact_artifacts(expected, actual)

    with pytest.raises(ValueError, match="unexpected"):
        VERIFY_INDEX.verify_exact_artifacts(
            expected, {**actual, "panorai-3.2.1-py3-none-any.whl": "0" * 64}
        )
    wrong = {**actual, wheel.name: "f" * 64}
    with pytest.raises(ValueError, match="sha256"):
        VERIFY_INDEX.verify_exact_artifacts(expected, wrong)


def test_index_artifact_verification_retries_until_complete() -> None:
    expected = {"panorai-3.2.1.tar.gz": "a" * 64}
    payloads = iter(
        (
            {"urls": []},
            {
                "urls": [
                    {
                        "filename": "panorai-3.2.1.tar.gz",
                        "digests": {"sha256": "a" * 64},
                    }
                ]
            },
        )
    )
    calls = []

    def fetch(url: str):
        calls.append(url)
        return next(payloads)

    VERIFY_INDEX.wait_for_exact_artifacts(
        url="https://example.invalid/pypi/panorai/3.2.1/json",
        expected=expected,
        attempts=2,
        delay_seconds=0.0,
        fetch=fetch,
    )
    assert len(calls) == 2


def test_compatible_wheel_selector_requires_exactly_one_match(tmp_path: Path) -> None:
    tag = next(SELECT_WHEEL.sys_tags())
    compatible = tmp_path / (
        f"panorai-3.2.1-{tag.interpreter}-{tag.abi}-{tag.platform}.whl"
    )
    compatible.touch()
    (tmp_path / "panorai-3.2.1-cp39-cp39-win32.whl").touch()
    assert SELECT_WHEEL.select_compatible_wheel(tmp_path) == compatible

    duplicate = tmp_path / (
        f"panorai_extra-3.2.1-{tag.interpreter}-{tag.abi}-{tag.platform}.whl"
    )
    duplicate.touch()
    with pytest.raises(ValueError, match="exactly one compatible wheel"):
        SELECT_WHEEL.select_compatible_wheel(tmp_path)
