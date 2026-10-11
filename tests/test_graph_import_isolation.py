from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_graph_import_does_not_load_optional_perception_backends() -> None:
    root = Path(__file__).resolve().parents[1]
    code = """
import json
import sys
import panorai.graph
print(json.dumps(sorted(name for name in ('torch', 'open3d', 'pycolmap') if name in sys.modules)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []
