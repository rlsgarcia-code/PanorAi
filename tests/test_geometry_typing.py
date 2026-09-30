from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_pep561_marker_is_available_as_package_resource() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from importlib.resources import files; "
            "assert files('panorai').joinpath('py.typed').is_file()",
        ],
        cwd=ROOT,
        check=True,
    )


def test_geometry_typing_example_passes_strict_mypy() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--follow-imports=silent",
            "tests/typing/geometry_api.py",
        ],
        cwd=ROOT,
        check=True,
    )


def test_public_typing_does_not_import_torch() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from panorai.geometry import ArrayLike, Interpolation; "
            "assert 'torch' not in sys.modules; assert ArrayLike; assert Interpolation",
        ],
        cwd=ROOT,
        check=True,
    )
