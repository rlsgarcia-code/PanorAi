"""Reject new Workbench typing debt across every public executable surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "workbench" / "contracts" / "typing-debt-v1.json"
sys.path.insert(0, str(ROOT / "workbench" / "backend"))
sys.path.insert(0, str(ROOT))

from panorai_workbench.discovery import discover_components  # noqa: E402


def _debt_record(component: dict[str, Any]) -> dict[str, str]:
    return {
        "surface_id": component["surface_id"],
        "symbol": component["symbol"],
        "public_api": component["public_api"],
        "diagnostic": component["diagnostic"] or component["status"],
    }


def _key(record: dict[str, str]) -> tuple[str, str]:
    return record["surface_id"], record["symbol"]


def _current_debt(components: list[dict[str, Any]]) -> list[dict[str, str]]:
    return sorted(
        (
            _debt_record(item)
            for item in components
            if item["execution_kind"] != "auxiliary" and item["status"] == "blocked"
        ),
        key=_key,
    )


def audit_components(
    components: list[dict[str, Any]],
    baseline: list[dict[str, str]],
    automatic_surface_ids: set[str],
) -> tuple[str, ...]:
    """Return violations without grandfathering new blocked symbols."""

    current = _current_debt(components)
    current_by_key = {_key(item): item for item in current}
    baseline_by_key = {_key(item): item for item in baseline}
    issues: list[str] = []

    for key in sorted(current_by_key.keys() - baseline_by_key.keys()):
        item = current_by_key[key]
        issues.append(f"new typing debt: {item['public_api']}: {item['diagnostic']}")
    for key in sorted(baseline_by_key.keys() - current_by_key.keys()):
        item = baseline_by_key[key]
        issues.append(
            "resolved typing debt still recorded: "
            f"{item['public_api']} (run this checker with --update)"
        )
    for key in sorted(current_by_key.keys() & baseline_by_key.keys()):
        if current_by_key[key] != baseline_by_key[key]:
            issues.append(
                f"typing debt diagnostic changed: {current_by_key[key]['public_api']}"
            )

    for item in current:
        if item["surface_id"] in automatic_surface_ids:
            issues.append(
                "automatic catchall cannot carry typing debt: "
                f"{item['public_api']}: {item['diagnostic']}"
            )
    return tuple(dict.fromkeys(issues))


def _load_manifest() -> dict[str, Any]:
    return json.loads(
        (ROOT / "contracts" / "public-api-surfaces-v2.json").read_text(encoding="utf-8")
    )


def _load_baseline() -> list[dict[str, str]]:
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    if payload.get("schema") != "panorai-workbench-typing-debt/v1":
        raise ValueError("unsupported Workbench typing-debt baseline")
    blocked = payload.get("blocked")
    if not isinstance(blocked, list):
        raise ValueError("typing-debt baseline must contain a blocked list")
    return blocked


def _update_baseline(components: list[dict[str, Any]]) -> None:
    current = _current_debt(components)
    if BASELINE.exists():
        previous = _load_baseline()
        previous_keys = {_key(item) for item in previous}
        additions = [item for item in current if _key(item) not in previous_keys]
        if additions:
            lines = ["refusing to grow the approved typing-debt baseline:"]
            lines.extend(f"- {item['public_api']}" for item in additions)
            raise SystemExit("\n".join(lines))
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(
        json.dumps(
            {
                "schema": "panorai-workbench-typing-debt/v1",
                "blocked": current,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"updated typing-debt baseline: {len(current)} blocked components")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="remove resolved entries; refuses to add new debt",
    )
    arguments = parser.parse_args(argv)
    components = discover_components()
    if arguments.update:
        _update_baseline(components)
        return 0

    manifest = _load_manifest()
    automatic_surfaces = set(
        manifest["policy"]["public_namespace_discovery"]["default_surfaces"].values()
    )
    baseline = _load_baseline()
    issues = audit_components(components, baseline, automatic_surfaces)
    if issues:
        raise SystemExit(
            "Workbench automatic-component contract failed:\n"
            + "\n".join(f"- {item}" for item in issues)
        )
    status_counts = {
        status: sum(item["status"] == status for item in components)
        for status in ("ready", "inspect", "blocked", "unavailable")
    }
    print(
        "automatic-component audit passed: "
        + ", ".join(f"{value} {name}" for name, value in status_counts.items())
        + f"; {len(baseline)} approved debt entries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
