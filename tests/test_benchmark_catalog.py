from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "benchmarks" / "catalog.json"


def test_benchmark_catalog_has_complete_reproducibility_fields() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert catalog["catalog_id"] == "panorai-benchmarks/v1"
    assert set(catalog["categories"]) == {
        "conformance",
        "regression",
        "performance",
        "scientific",
    }
    ids = [item["id"] for item in catalog["entries"]]
    assert len(ids) == len(set(ids))
    required = {
        "family",
        "runner",
        "datasets",
        "license",
        "evidence_tier",
        "category",
        "output",
        "last_validated_commit",
    }
    for entry in catalog["entries"]:
        assert required.issubset(entry)
        assert entry["category"] in catalog["categories"]
        assert (ROOT / entry["runner"]).is_file()
        assert len(entry["last_validated_commit"]) == 40
        assert entry["datasets"]


def test_importing_panorai_does_not_import_benchmark_modules() -> None:
    code = """
import json
import sys
import panorai
import panorai.graph
print(json.dumps(sorted(name for name in sys.modules if name == 'benchmarks' or name.startswith('benchmarks.'))))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []
