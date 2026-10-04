from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CI_PATH = ROOT / ".github/workflows/ci.yml"
RELEASE_PATH = ROOT / ".github/workflows/python-publish.yml"
PAGES_RECOVERY_PATH = ROOT / ".github/workflows/pages-recovery.yml"


def _workflow(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _runs(job: dict) -> str:
    return "\n".join(
        step.get("run", "") for step in job.get("steps", []) if isinstance(step, dict)
    )


def test_ci_is_premerge_only_and_read_only() -> None:
    workflow = _workflow(CI_PATH)
    triggers = workflow["on"]

    assert set(triggers) == {"pull_request", "push"}
    assert triggers["push"]["branches"] == ["main", "release/**"]
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] == "true"
    assert all("permissions" not in job for job in workflow["jobs"].values())

    raw = CI_PATH.read_text(encoding="utf-8")
    assert "gh-action-pypi-publish" not in raw
    assert "id-token: write" not in raw
    assert "release:" not in raw


def test_ci_encodes_required_matrix_and_independent_gates() -> None:
    jobs = _workflow(CI_PATH)["jobs"]
    assert set(jobs) == {
        "native-kernels",
        "core-tests",
        "torch-cpu",
        "build-wheels",
        "build-sdist",
        "build-artifacts",
        "artifact-policy",
        "installed-wheel",
        "docs",
    }
    expected_versions = ["3.10", "3.11", "3.12"]
    assert (
        jobs["core-tests"]["strategy"]["matrix"]["python-version"] == expected_versions
    )
    assert (
        jobs["installed-wheel"]["strategy"]["matrix"]["python-version"]
        == expected_versions
    )
    assert "needs" not in jobs["core-tests"]
    assert "needs" not in jobs["native-kernels"]
    assert "needs" not in jobs["torch-cpu"]
    assert "needs" not in jobs["docs"]
    assert jobs["artifact-policy"]["needs"] == "build-artifacts"
    assert jobs["installed-wheel"]["needs"] == "build-artifacts"

    wheel_matrix = jobs["build-wheels"]["strategy"]["matrix"]["include"]
    assert wheel_matrix == [
        {"id": "manylinux-x86_64", "os": "ubuntu-latest", "arch": "x86_64"},
        {
            "id": "manylinux-aarch64",
            "os": "ubuntu-24.04-arm",
            "arch": "aarch64",
        },
        {"id": "macos-x86_64", "os": "macos-15-intel", "arch": "x86_64"},
        {"id": "macos-arm64", "os": "macos-14", "arch": "arm64"},
        {"id": "windows-amd64", "os": "windows-latest", "arch": "AMD64"},
    ]
    assert "pypa/cibuildwheel@v4.2.1" in str(jobs["build-wheels"])
    assert jobs["build-artifacts"]["needs"] == ["build-wheels", "build-sdist"]
    artifact_build = _runs(jobs["build-artifacts"])
    assert '" = "15"' in artifact_build
    assert "python -m twine check dist/*" in artifact_build

    native = jobs["native-kernels"]
    assert native["strategy"]["matrix"] == {
        "os": ["ubuntu-latest", "macos-latest", "windows-latest"],
        "python-version": ["3.10", "3.12"],
    }
    native_commands = _runs(native)
    assert "python -m build --wheel" in native_commands
    assert "select_compatible_wheel.py native-wheelhouse" in native_commands
    assert "--force-reinstall --no-deps" in native_commands
    assert "$RUNNER_TEMP" in native_commands
    assert "$GITHUB_WORKSPACE/scripts/release_smoke.py" in native_commands
    assert "--require-installed --require-native" in native_commands
    assert "--import-mode=importlib" in native_commands
    assert "tests/test_native_essential_kernels.py" in native_commands
    assert "tests/test_native_geometry_kernels.py" in native_commands
    assert "build_ext --inplace" not in native_commands

    native_test_source = (ROOT / "tests/test_native_essential_kernels.py").read_text(
        encoding="utf-8"
    )
    assert 'pytest.importorskip(\n    "panorai._native._essential"' in native_test_source
    geometry_test_source = (ROOT / "tests/test_native_geometry_kernels.py").read_text(
        encoding="utf-8"
    )
    assert 'pytest.importorskip(\n    "panorai._native._geometry"' in geometry_test_source

    installed = _runs(jobs["installed-wheel"])
    assert "$RUNNER_TEMP" in installed
    assert "$GITHUB_WORKSPACE/scripts/release_smoke.py" in installed
    assert "$GITHUB_WORKSPACE/scripts/run_geometry_conformance.py" in installed
    assert "$GITHUB_WORKSPACE/scripts/run_geometry_oracle.py" in installed
    assert "$GITHUB_WORKSPACE/scripts/run_documentation_examples.py" in installed
    assert installed.count("--require-installed") == 5
    assert "tests/typing/geometry_api.py" in _runs(jobs["core-tests"])
    assert "tests/typing/geometry_torch_api.py" in _runs(jobs["torch-cpu"])
    assert "py.typed" in installed
    assert "matrix.python-version == '3.12'" in str(jobs["installed-wheel"])
    assert "python -m pip check" in installed
    assert "--require-native" in installed
    assert "select_compatible_wheel.py dist" in installed

    policy = _runs(jobs["artifact-policy"])
    assert "audit_release_artifacts.py dist/*" in policy
    assert "-n -W --keep-going" in _runs(jobs["docs"])


def test_ci_builds_candidate_once_and_reuses_the_same_artifact() -> None:
    workflow = _workflow(CI_PATH)
    raw = CI_PATH.read_text(encoding="utf-8")
    assert raw.count("python -m build --sdist") == 1
    assert raw.count("python -m build --wheel") == 1

    jobs = workflow["jobs"]
    build_uses = [
        step
        for step in jobs["build-artifacts"]["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact")
    ]
    assert build_uses[0]["with"]["name"] == "candidate-dists"
    for job_name in ("artifact-policy", "installed-wheel"):
        downloads = [
            step
            for step in jobs[job_name]["steps"]
            if step.get("uses", "").startswith("actions/download-artifact")
        ]
        assert downloads[0]["with"]["name"] == "candidate-dists"

    matrix_uploads = [
        step
        for step in jobs["build-wheels"]["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact")
    ]
    assert matrix_uploads[0]["with"]["name"] == "candidate-wheels-${{ matrix.id }}"
    assembly_download = next(
        step
        for step in jobs["build-artifacts"]["steps"]
        if step.get("uses", "").startswith("actions/download-artifact")
    )
    assert assembly_download["with"]["pattern"] == "candidate-*"
    assert assembly_download["with"]["merge-multiple"] == "true"


def test_release_workflow_is_the_only_publisher_and_tests_installed_origin() -> None:
    workflow = _workflow(RELEASE_PATH)
    assert workflow["on"] == {"release": {"types": ["published"]}}
    assert workflow["env"]["RELEASE_VERSION"] == "3.2.0"
    raw = RELEASE_PATH.read_text(encoding="utf-8")
    assert raw.count("python -m build") == 1
    assert raw.count("pypa/cibuildwheel@v4.2.1") == 1
    assert raw.count("pypa/gh-action-pypi-publish@release/v1") == 2

    jobs = workflow["jobs"]
    installed = _runs(jobs["install-wheel"])
    testpypi = _runs(jobs["smoke-testpypi"])
    for commands in (installed, testpypi):
        assert "$RUNNER_TEMP" in commands
        assert "$GITHUB_WORKSPACE/scripts/release_smoke.py" in commands
        assert "$GITHUB_WORKSPACE/scripts/run_geometry_conformance.py" in commands
        assert "$GITHUB_WORKSPACE/scripts/run_geometry_oracle.py" in commands
        assert "$GITHUB_WORKSPACE/scripts/run_documentation_examples.py" in commands
        assert commands.count("--require-installed") == 4
        assert "--require-native" in commands
        assert '--expected-version "$RELEASE_VERSION"' in commands
    assert "-n -W --keep-going" in _runs(jobs["build-docs"])

    build = _runs(jobs["build"])
    assert '"panorai-${RELEASE_VERSION}-*.whl"' in build
    assert '"dist/panorai-${RELEASE_VERSION}.tar.gz"' in build
    assert "panorai-3.1.0" not in build
    assert jobs["build"]["needs"] == ["build-wheels", "build-sdist"]
    assert '" = "15"' in build
    assert len(jobs["install-wheel"]["strategy"]["matrix"]["include"]) == 15
    assert "select_compatible_wheel.py dist" in installed


def test_release_verifies_exact_testpypi_files_before_pypi() -> None:
    jobs = _workflow(RELEASE_PATH)["jobs"]
    smoke = jobs["smoke-testpypi"]
    publish = jobs["publish-pypi"]

    assert smoke["needs"] == "publish-testpypi"
    assert publish["needs"] == "smoke-testpypi"

    downloads = [
        step
        for step in smoke["steps"]
        if step.get("uses", "").startswith("actions/download-artifact")
    ]
    assert len(downloads) == 1
    assert downloads[0]["with"] == {"name": "release-dists", "path": "dist/"}

    commands = _runs(smoke)
    assert commands.count("python -m pip download") == 2
    assert "--only-binary=panorai" in commands
    assert "--no-binary=panorai" in commands
    assert '"panorai==${RELEASE_VERSION}"' in commands
    assert 'test "$(find "$RUNNER_TEMP/testpypi-dist"' in commands
    assert '" = "2"' in commands
    assert "verify_index_artifacts.py" in commands
    assert "--index-url https://test.pypi.org/pypi" in commands
    assert "dist/panorai-*" in commands


def test_release_finalizer_targets_the_repository_explicitly() -> None:
    jobs = _workflow(RELEASE_PATH)["jobs"]
    commands = _runs(jobs["finalize-release"])

    assert 'gh release upload "${{ github.event.release.tag_name }}"' in commands
    assert '--repo "${{ github.repository }}"' in commands


def test_pages_recovery_is_manual_tag_exact_and_cannot_publish_packages() -> None:
    workflow = _workflow(PAGES_RECOVERY_PATH)
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}

    job = workflow["jobs"]["deploy-pages"]
    assert job["permissions"] == {
        "contents": "read",
        "pages": "write",
        "id-token": "write",
    }
    assert job["environment"]["name"] == "github-pages"

    raw = PAGES_RECOVERY_PATH.read_text(encoding="utf-8")
    assert "gh-action-pypi-publish" not in raw
    assert "gh release" not in raw
    assert "python -m build" not in raw

    checkout = next(
        step
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/checkout")
    )
    assert checkout["with"]["ref"] == "${{ inputs.release_tag }}"
    assert (
        workflow["on"]["workflow_dispatch"]["inputs"]["release_tag"]["default"]
        == "v3.2.0"
    )
    commands = _runs(job)
    assert "^v[0-9]+\\.[0-9]+\\.[0-9]+$" in commands
    assert 'git cat-file -t "$REQUESTED_TAG"' in commands
    assert "git describe --tags --exact-match HEAD" in commands
    assert 'test "$(python -m setuptools_scm)" = "${REQUESTED_TAG#v}"' in commands
    assert "sphinx-build -b html -n -W --keep-going docs _site" in commands

    uses = [step.get("uses", "") for step in job["steps"]]
    assert any(value.startswith("actions/upload-pages-artifact") for value in uses)
    assert any(value.startswith("actions/deploy-pages") for value in uses)
