from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CI_PATH = ROOT / ".github/workflows/ci.yml"
RELEASE_PATH = ROOT / ".github/workflows/python-publish.yml"


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
        "core-tests",
        "torch-cpu",
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
    assert "needs" not in jobs["torch-cpu"]
    assert "needs" not in jobs["docs"]
    assert jobs["artifact-policy"]["needs"] == "build-artifacts"
    assert jobs["installed-wheel"]["needs"] == "build-artifacts"

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

    policy = _runs(jobs["artifact-policy"])
    assert "audit_release_artifacts.py dist/*" in policy
    assert "-n -W --keep-going" in _runs(jobs["docs"])


def test_ci_builds_once_and_reuses_the_same_artifact() -> None:
    workflow = _workflow(CI_PATH)
    raw = CI_PATH.read_text(encoding="utf-8")
    assert raw.count("python -m build") == 1

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


def test_release_workflow_is_the_only_publisher_and_tests_installed_origin() -> None:
    workflow = _workflow(RELEASE_PATH)
    assert workflow["on"] == {"release": {"types": ["published"]}}
    raw = RELEASE_PATH.read_text(encoding="utf-8")
    assert raw.count("python -m build") == 1
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
        assert '--expected-version "$RELEASE_VERSION"' in commands
    assert "-n -W --keep-going" in _runs(jobs["build-docs"])


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
    assert "for artifact in dist/panorai-*" in commands
    assert '$(basename "$artifact")' in commands
    assert commands.count("sha256sum") >= 2
