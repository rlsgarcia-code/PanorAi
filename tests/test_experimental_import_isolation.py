"""Cold-process import checks for the dependency-light experiment package."""

import os
import subprocess
import sys


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["KMP_USE_SHM"] = "0"
    environment["KMP_INIT_AT_FORK"] = "FALSE"
    environment["OMP_NUM_THREADS"] = "1"
    return environment


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
        env=_environment(),
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
        env=_environment(),
    )

    assert completed.returncode == 0, completed.stderr


def test_depth_adapter_import_has_no_network_or_cache_side_effect() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, pathlib, sys, tempfile; "
            "root=tempfile.mkdtemp(); os.environ['PANORAI_CACHE_HOME']=root; "
            "sys.addaudithook(lambda event, args: "
            "(_ for _ in ()).throw(AssertionError('network')) "
            "if event == 'socket.connect' else None); "
            "import panorai.experimental.deep_learning.depth; "
            "assert list(pathlib.Path(root).iterdir()) == []",
        ],
        text=True,
        capture_output=True,
        check=False,
        env=_environment(),
    )

    assert completed.returncode == 0, completed.stderr
