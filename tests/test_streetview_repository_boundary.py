from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_streetview_is_an_isolated_application_not_a_panorai_package() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert 'include = ["panorai", "panorai.*"]' in pyproject
    assert "prune panorai-streetview" in manifest.splitlines()
    assert (ROOT / "panorai-streetview" / "pyproject.toml").is_file()
    assert (ROOT / "panorai-streetview" / "package.json").is_file()


def test_root_package_has_no_streetview_runtime_dependency() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    root_project = pyproject.split("[project.optional-dependencies]", 1)[0]

    assert "fastapi" not in root_project.lower()
    assert "uvicorn" not in root_project.lower()
    assert "vite" not in root_project.lower()


def test_streetview_does_not_restore_generic_line_or_vanishing_tools() -> None:
    application_files = [
        ROOT / "panorai-streetview" / "src" / "main.ts",
        ROOT / "panorai-streetview" / "src" / "api.ts",
        ROOT / "panorai-streetview" / "src" / "model.ts",
        ROOT / "panorai-streetview" / "backend" / "panorai_streetview" / "api.py",
        ROOT / "panorai-streetview" / "backend" / "panorai_streetview" / "models.py",
        ROOT / "panorai-streetview" / "backend" / "panorai_streetview" / "planner.py",
    ]
    combined = "\n".join(
        path.read_text(encoding="utf-8").lower() for path in application_files
    )

    for forbidden in (
        "vanishing_point",
        "line_segment",
        "manual-orientation",
        "manual-calibration-evidence",
        "direção de fuga",
        ">orientação<",
    ):
        assert forbidden not in combined


def test_streetview_separates_angular_ruler_from_engine_backed_metric_tools() -> None:
    main = (ROOT / "panorai-streetview" / "src" / "main.ts").read_text(encoding="utf-8")
    model = (ROOT / "panorai-streetview" / "src" / "model.ts").read_text(
        encoding="utf-8"
    )
    api = (ROOT / "panorai-streetview" / "src" / "api.ts").read_text(encoding="utf-8")

    assert 'data-tool="pan"' in main
    assert 'data-tool="measure"' in main
    assert 'id="zoomRange"' in main
    assert 'id="resetViewport"' in main
    assert "erpAngularMeasurement" in model
    assert "distância angular" in main
    assert 'data-tab="metrology"' in main
    assert 'data-tool="floor_a"' in main
    assert 'data-tool="floor_b"' in main
    assert 'data-tool="floor_measure"' in main
    assert 'data-tool="vertical_measure"' in main
    assert "calibrateFloor" in api
    assert 'form.append("pairs", JSON.stringify(input.pairs))' in api
    assert "revision: number" in api
    assert "revision: input.revision" in api
    assert "sphericalPairCandidates" in main
    assert "measureFloorPoint" in api
    assert "measureVertical" in api
    assert "panorai.experimental.single_view_metrology" in (
        ROOT / "panorai-streetview" / "backend" / "panorai_streetview" / "metrology.py"
    ).read_text(encoding="utf-8")


def test_streetview_requires_the_panorai_api_line_that_contains_metrology() -> None:
    pyproject = (ROOT / "panorai-streetview" / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    readme = (ROOT / "panorai-streetview" / "README.md").read_text(encoding="utf-8")

    assert '"panorai>=3.8.1.dev0,<4"' in pyproject
    assert "--with-editable ." in readme


def test_streetview_async_results_are_scoped_to_the_active_workspace() -> None:
    main = (ROOT / "panorai-streetview" / "src" / "main.ts").read_text(encoding="utf-8")

    def function_body(name: str, next_name: str) -> str:
        return main.split(f"async function {name}", 1)[1].split(
            f"async function {next_name}", 1
        )[0]

    assert "imageRequestIsCurrent" in function_body(
        "calibrateActiveFloor", "applyMetrologyPoint"
    )
    assert "imageRequestIsCurrent" in function_body("applyMetrologyPoint", "setTool")
    assert "projectRequestIsCurrent" in function_body(
        "executeStitching", "executeStitchingPromotion"
    )
    assert "projectRequestIsCurrent" in function_body(
        "executeStitchingPromotion", "renderDenseAnnotations"
    )
    assert "projectRequestIsCurrent" in function_body(
        "executeDenseScene", "renderDensePreview"
    )
    for name, next_name in (
        ("saveObjectBox", "saveManualMatchPoint"),
        ("saveManualMatchPoint", "saveDenseEvidence"),
        ("saveDenseEvidence", "updateMatchReview"),
    ):
        body = function_body(name, next_name)
        assert "enqueueEvidenceMutation" in body
        assert "projectLoadSequence" in body
    dense_evidence = function_body("saveDenseEvidence", "updateMatchReview")
    assert dense_evidence.index("const source = sourcePoints(points)") < (
        dense_evidence.index("enqueueEvidenceMutation")
    )
    object_box = function_body("saveObjectBox", "saveManualMatchPoint")
    assert object_box.index(
        'const objectId = element<HTMLInputElement>("#objectId")'
    ) < (object_box.index("enqueueEvidenceMutation"))
    assert object_box.index(
        'const label = element<HTMLInputElement>("#objectLabel")'
    ) < (object_box.index("enqueueEvidenceMutation"))
    match_review = function_body("updateMatchReview", "setTool")
    assert "enqueueMatchReviewMutation" in match_review
    assert "activeRegistration.revision" in match_review
    assert "matchReviewSequence" not in main
    global_map_action = main.split(
        'element<HTMLButtonElement>("#runGlobalMap").onclick', 1
    )[1].split("buildButton.onclick", 1)[0]
    assert "projectRequestIsCurrent" in global_map_action
    pair_selection = function_body("selectPair", "selectPairAndOpenRegistration")
    assert "currentPair()?.id !== pairId" in pair_selection
    pair_navigation = function_body("selectPairAndOpenRegistration", "openDialog")
    assert "if (await selectPair(index))" in pair_navigation
    manifest_action = main.split("buildButton.onclick", 1)[1].split(
        "sourceImage.onload", 1
    )[0]
    assert "projectRequestIsCurrent" in manifest_action
    assert "sourceProject.name" in manifest_action
