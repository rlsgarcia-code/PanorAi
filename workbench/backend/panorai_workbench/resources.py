"""Explicit, allow-listed provisioning for optional Workbench runtimes."""

from __future__ import annotations

from importlib import metadata, util
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

from packaging.specifiers import SpecifierSet


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_SAM_EXTRA = "deep-learning-sam"
_SAM_REQUIREMENTS = {
    "torch": ("torch", ">=2.2,<3"),
    "torchvision": ("torchvision", ">=0.17,<1"),
    "transformers": ("transformers", ">=4.57,<5"),
    "huggingface_hub": ("huggingface-hub", ">=0.23,<1"),
    "safetensors": ("safetensors", ">=0.4,<1"),
}
_install_lock = threading.Lock()


def _requirement_state(module: str, distribution: str, specifier: str) -> dict[str, Any]:
    try:
        version = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        version = None
    importable = util.find_spec(module) is not None
    compatible = version is not None and version in SpecifierSet(specifier)
    return {
        "module": module,
        "distribution": distribution,
        "required": specifier,
        "version": version,
        "importable": importable,
        "compatible": compatible,
    }


def sam21_runtime_status() -> dict[str, Any]:
    """Describe dependency readiness without loading Torch or acquiring weights."""

    requirements = [
        _requirement_state(module, distribution, specifier)
        for module, (distribution, specifier) in _SAM_REQUIREMENTS.items()
    ]
    missing = [
        f"{item['distribution']}{item['required']}"
        for item in requirements
        if not item["importable"] or not item["compatible"]
    ]
    return {
        "schema": "panorai-workbench-resource/v1",
        "id": "sam21",
        "state": "ready" if not missing else "missing-dependencies",
        "ready": not missing,
        "extra": f"panorai[{_SAM_EXTRA}]",
        "interpreter": sys.executable,
        "requirements": requirements,
        "missing": missing,
        "installation": {
            "supported": (_REPOSITORY_ROOT / "pyproject.toml").is_file(),
            "action": "install-sam21-dependencies",
            "target": f"local checkout + [{_SAM_EXTRA}]",
            "fixed_allowlist": True,
        },
        "checkpoint": {
            "state": "not-checked",
            "acquisition": "only when the user accepts upstream terms and starts SAM",
            "bundled": False,
        },
    }


def require_sam21_runtime() -> None:
    status = sam21_runtime_status()
    if status["ready"]:
        return
    missing = ", ".join(status["missing"])
    raise RuntimeError(
        "SAM 2.1 indisponível neste runtime. Instale o suporte opcional pela "
        f"interface do Workbench. Dependências ausentes ou incompatíveis: {missing}"
    )


def install_sam21_dependencies(*, accept_environment_change: bool) -> dict[str, Any]:
    """Install one fixed local extra; the caller cannot supply a package or command."""

    if not accept_environment_change:
        raise PermissionError(
            "a instalação exige confirmação explícita da alteração do ambiente Python"
        )
    if not (_REPOSITORY_ROOT / "pyproject.toml").is_file():
        raise RuntimeError(
            "instalação pela interface só é suportada em um checkout local do PanorAi"
        )
    with _install_lock:
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            f"{_REPOSITORY_ROOT}[{_SAM_EXTRA}]",
        ]
        completed = subprocess.run(
            command,
            cwd=_REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    if completed.returncode != 0:
        raise RuntimeError(
            "pip não conseguiu instalar o extra fixo panorai[deep-learning-sam]: "
            + output[-4000:]
        )
    status = sam21_runtime_status()
    return {
        **status,
        "installation_result": {
            "returncode": completed.returncode,
            "output_tail": output[-4000:],
            "restart_required": False,
        },
    }
