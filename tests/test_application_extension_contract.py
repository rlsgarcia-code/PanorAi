from __future__ import annotations

import inspect
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/development/application_extensions.rst"


def test_documented_application_and_stage_signatures_match_public_runtime() -> None:
    from panorai.applications import Application, PipelineApplication
    from panorai.pipeline import Pipeline, StageResult

    run_parameters = inspect.signature(Application.run).parameters
    assert tuple(run_parameters) == ("self", "args", "kwargs")
    assert run_parameters["args"].kind is inspect.Parameter.VAR_POSITIONAL
    assert run_parameters["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
    assert getattr(Application.run, "__isabstractmethod__", False)
    assert getattr(PipelineApplication.as_pipeline, "__isabstractmethod__", False)

    class IdentityStage:
        input_kind = "any"
        output_kind = "any"

        def execute(self, value: object) -> StageResult:
            return StageResult(value, {"domain": "test"})

    pipeline = Pipeline(IdentityStage())
    marker = object()
    assert pipeline(marker) is marker
    assert pipeline.execute(marker).value is marker


def test_application_extension_contract_is_discoverable() -> None:
    development_index = (ROOT / "docs/development/index.rst").read_text(
        encoding="utf-8"
    )
    contributing = (ROOT / ".github/CONTRIBUTING.md").read_text(encoding="utf-8")
    pull_request_template = (ROOT / ".github/pull_request_template.md").read_text(
        encoding="utf-8"
    )

    assert "   application_extensions" in development_index
    assert "docs/development/application_extensions.rst" in contributing
    assert "public API component contract" in contributing
    assert "contracts/public-api-surfaces-v2.json" in pull_request_template
    assert "catalog-only" in pull_request_template


def test_application_extension_contract_covers_every_public_artifact_level() -> None:
    guide = GUIDE.read_text(encoding="utf-8")

    for required in (
        "Domain algorithm",
        "Geometry",
        "Typed data or result",
        "Functional operation",
        "Configured operation",
        "``Stage``",
        "``Pipeline``",
        "``Application``",
        "``PipelineApplication``",
        "Estimator or policy",
        "Torch layer/module",
        "def run(self, *args: Any, **kwargs: Any) -> Any:",
        "def as_pipeline(self) -> Pipeline:",
        "def execute(self, value: Any) -> StageResult:",
        "def transform_grid(self, input_grid: RasterGrid) -> RasterGrid:",
        "def forward(self, values):",
    ):
        assert required in guide


def test_public_component_contract_makes_documentation_fail_closed() -> None:
    guide = GUIDE.read_text(encoding="utf-8")

    for required in (
        "Documentation is part of the public contract",
        "documentation.assignments",
        "application_components",
        "documentation.readme.featured",
        "panorai-api-components",
        "Stable Application lacks a",
        "contracts/**",
        "README.md",
        "GitHub Pages",
    ):
        assert required in guide


def test_application_extension_contract_preserves_spherical_semantics() -> None:
    guide = GUIDE.read_text(encoding="utf-8")

    for required in (
        "support",
        "validity",
        "radial range",
        "nearest-only",
        "RasterGrid",
        "SphericalObservations",
        "ResultBundle",
        "register_torch_module",
        "safetensors",
        "Never use pickle",
        "installed-wheel",
        "independent oracle",
    ):
        assert required in guide


def test_application_extension_contract_names_safe_persistence_boundaries() -> None:
    guide = GUIDE.read_text(encoding="utf-8")

    assert "add the class to the explicit allowlist in ``Application.load``" in guide
    assert "``panorai.pipeline._serialization``" in guide
    assert "Loading never\nimports a class named by the archive." in guide
    assert "must fail with the exact\nconfiguration path" in guide
