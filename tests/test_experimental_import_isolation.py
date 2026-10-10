"""Cold-process import checks for the dependency-light experiment package."""

from importlib.util import find_spec
import os
import subprocess
import sys

import pytest


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


@pytest.mark.skipif(find_spec("torch") is None, reason="Torch is optional")
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


def test_segmentation_contract_imports_when_torch_is_unavailable() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.abc, sys; "
            "exec('class BlockTorch(importlib.abc.MetaPathFinder):\\n"
            "    def find_spec(self, fullname, path=None, target=None):\\n"
            "        if fullname == \\\"torch\\\" or fullname.startswith(\\\"torch.\\\"):\\n"
            "            raise ModuleNotFoundError(\\\"Torch intentionally unavailable\\\")'); "
            "sys.meta_path.insert(0, BlockTorch()); "
            "import panorai.experimental.deep_learning.segmentation; "
            "assert 'torch' not in sys.modules",
        ],
        text=True,
        capture_output=True,
        check=False,
        env=_environment(),
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.skipif(find_spec("torch") is None, reason="Torch is optional")
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


@pytest.mark.skipif(find_spec("torch") is None, reason="Torch is optional")
def test_da3_adapter_import_has_no_network_cache_or_safetensors_side_effect() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, pathlib, sys, tempfile; "
            "root=tempfile.mkdtemp(); os.environ['PANORAI_CACHE_HOME']=root; "
            "sys.addaudithook(lambda event, args: "
            "(_ for _ in ()).throw(AssertionError('network')) "
            "if event == 'socket.connect' else None); "
            "import panorai.experimental.deep_learning.da3; "
            "assert 'safetensors' not in sys.modules; "
            "assert 'depth_anything_3' not in sys.modules; "
            "assert list(pathlib.Path(root).iterdir()) == []",
        ],
        text=True,
        capture_output=True,
        check=False,
        env=_environment(),
    )

    assert completed.returncode == 0, completed.stderr
