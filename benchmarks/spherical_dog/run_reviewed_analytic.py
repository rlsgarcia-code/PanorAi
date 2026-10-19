#!/usr/bin/env python3
"""Reproduce the frozen DoG analytic oracle through reviewed-v1."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable, cast

if __package__:
    from .run_installed_profile import (  # type: ignore[import-not-found]
        _validate_installed_artifact,
        _validate_source_checkout,
    )
else:
    from run_installed_profile import (
        _validate_installed_artifact,
        _validate_source_checkout,
    )


ROOT = Path(__file__).resolve().parents[2]
PROFILE_ID = "panorai-spherical-dog-features/reviewed-v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_evaluator(*, source_root: Path | None) -> Callable[..., dict[str, Any]]:
    if source_root is not None:
        resolved_source = source_root.resolve()
        sys.path.insert(0, str(resolved_source))
        script_root = resolved_source
    else:
        import panorai

        origin = Path(panorai.__file__).resolve()
        try:
            origin.relative_to(ROOT)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "native analytic evidence must import an installed wheel outside "
                "the source checkout"
            )
        script_root = ROOT

    script = script_root / "scripts" / "run_spherical_application_evidence.py"
    specification = importlib.util.spec_from_file_location(
        "panorai_reviewed_dog_oracle", script
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not load analytic oracle: {script}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return cast(Callable[..., dict[str, Any]], module.evaluate_dog_analytic)


def build_record(
    *,
    backend: str,
    source_commit: str,
    evaluator: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    if backend not in {"native", "numpy"}:
        raise ValueError("backend must be 'native' or 'numpy'")
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ValueError("source_commit must be a lowercase 40-character Git SHA")

    first = evaluator(convolution_backend=backend)
    second = evaluator(convolution_backend=backend)
    deterministic = _canonical_bytes(first) == _canonical_bytes(second)
    metrics = first.get("metrics")
    details = first.get("details")
    if not isinstance(metrics, dict) or not isinstance(details, dict):
        raise ValueError("analytic evaluator returned an invalid record")
    return {
        "application": "spherical-dog-reviewed-profile",
        "evaluation": "analytic",
        "profile_id": PROFILE_ID,
        "backend": backend,
        "source_commit": source_commit,
        "evidence_sha256": hashlib.sha256(_canonical_bytes(details)).hexdigest(),
        "metrics": {
            "isolated_blob_detection_rate": metrics["isolated_blob_detection_rate"],
            "maximum_angular_error_deg": metrics["maximum_angular_error_deg"],
            "maximum_scale_level_error": metrics["maximum_scale_level_error"],
            "seam_and_high_latitude_corpus": metrics["seam_and_high_latitude_corpus"],
            "reviewed_factory_matches_frozen_profile": metrics[
                "reviewed_factory_matches_frozen_profile"
            ],
            "deterministic_replay": deterministic,
        },
        "details": details,
    }


def _validate_execution_binding(
    *, source_root: Path | None, wheel: Path | None, source_commit: str
) -> dict[str, Any]:
    if source_root is not None:
        if wheel is not None:
            raise ValueError("--wheel and --source-root are mutually exclusive")
        _validate_source_checkout(
            source_root=source_root,
            source_commit=source_commit,
        )
        return {
            "mode": "clean-source-checkout",
            "source_root": str(source_root.resolve()),
        }
    if wheel is None:
        raise ValueError("installed analytic evidence requires --wheel")
    artifact_version = _validate_installed_artifact(
        source_root=ROOT,
        wheel=wheel,
        source_commit=source_commit,
    )
    return {
        "mode": "installed-wheel",
        "artifact_version": artifact_version,
        "wheel_sha256": _sha256(wheel),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("native", "numpy"), required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="explicit source checkout for development-only NumPy reproduction",
    )
    parser.add_argument(
        "--wheel",
        type=Path,
        help="exact installed wheel; required when --source-root is omitted",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {args.output}")
    provenance = _validate_execution_binding(
        source_root=args.source_root,
        wheel=args.wheel,
        source_commit=args.source_commit,
    )
    evaluator = _load_evaluator(source_root=args.source_root)
    record = build_record(
        backend=args.backend,
        source_commit=args.source_commit,
        evaluator=evaluator,
    )
    record["provenance"] = provenance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"output={args.output} sha256={_sha256(args.output)}")


if __name__ == "__main__":
    main()
