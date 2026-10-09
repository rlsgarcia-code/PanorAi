from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "benchmarks/p74_pair_eligibility/run_optimized_public_pair.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("p74_optimized_public_pair", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_serializes_required_optimized_route() -> None:
    profile = _load_runner().profile_configuration()

    assert profile["route"] == {
        "detector_class": "SphericalDoGDetector",
        "detector_method": "detect_batch",
        "batch_size": 2,
        "sequential_detection": False,
        "validity_masks": "explicit-per-panorama",
        "patch_provider_class": "TangentPatchProvider",
        "patch_provider_max_workers": 4,
        "multiface_route": False,
        "private_imports": False,
    }
    assert profile["detector"]["convolution_backend"] == "native"
    assert profile["detector"]["max_keypoints"] == 4096
    assert profile["patches"]["output_shape_hw"] == [48, 48]
    assert profile["patches"]["radius_in_scales"] == 6.0
    assert profile["descriptor"]["keypoint_diameter_in_scales"] == 1.25
    assert profile["descriptor"]["orientation_policy"] == "fixed-zero"
    assert profile["descriptor"]["photometric_normalization"] == (
        "local-standardization"
    )
    assert profile["descriptor"]["scale_multipliers"] == [1.0]
    assert profile["descriptor"]["root_sift"] is True


def test_runner_has_no_private_panorai_imports_or_convenience_pipeline() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    imported_modules = []
    imported_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")
            imported_names.extend(alias.name for alias in node.names)

    assert not any(
        part.startswith("panorai.") and "._" in part for part in imported_modules
    )
    assert "SphericalDoGSIFTPipeline" not in imported_names
    assert "SphericalCoarseDoGDetector" not in imported_names


def test_runner_calls_batch_detection_and_explicit_parallel_patches() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "detector.detect_batch(" in source
    assert "validity_masks=(validity_a, validity_b)" in source
    assert "TangentPatchProvider(max_workers=4)" in source
    for timing in (
        '"detection_pair"',
        '"patches_pair"',
        '"descriptor_pair"',
        '"matching"',
        '"pose"',
        '"pair_total"',
    ):
        assert timing in source


def test_pose_errors_transform_panorai_result_into_p74_frame() -> None:
    runner = _load_runner()
    expected_rotation = np.asarray(
        [
            [0.9917322552, -0.1283240946, -0.0002462912],
            [0.1283215812, 0.9916966129, 0.0084498403],
            [-0.0008400719, -0.0084115836, 0.9999642691],
        ]
    )
    expected_translation = np.asarray([0.1399182589, -0.9900877158, -0.0122145770])
    frame = runner.P74_FROM_PANORAI

    class Pose:
        rotation = frame.T @ expected_rotation @ frame
        translation_direction = frame.T @ expected_translation

    errors = runner._pose_errors(
        Pose(),
        {
            "reference": {
                "R_to_from": expected_rotation.tolist(),
                "t_direction_to_from": expected_translation.tolist(),
            }
        },
    )

    assert errors["rotation_error_deg"] < 1e-3
    assert errors["translation_direction_error_deg"] < 1e-5
    assert errors["evaluation_frame"] == "p74-native-scanner"
    assert errors["strict_valid"] is True
    assert errors["precise_valid"] is True
