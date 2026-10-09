"""Cold-process import checks for the dependency-light experiment package."""

import subprocess
import sys


def test_experimental_package_import_does_not_eagerly_load_torch() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import panorai.experimental; "
            "assert 'torch' not in sys.modules; "
            "assert 'panorai.experimental.deep_learning' not in sys.modules",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_deep_learning_import_does_not_eagerly_load_torchvision() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import panorai.experimental.deep_learning; "
            "assert 'torchvision' not in sys.modules",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
