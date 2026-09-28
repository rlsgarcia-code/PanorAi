from pathlib import Path
import subprocess
import sys


def test_geometry_v1_independent_runner() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "scripts/run_geometry_conformance.py", "--source-checkout"],
        cwd=root,
        check=True,
    )
