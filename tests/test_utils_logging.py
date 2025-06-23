import importlib.util
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "panorai.utils.logging_config",
    ROOT / "panorai" / "utils" / "logging_config.py",
)
logging_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logging_config)
sys.modules["panorai.utils.logging_config"] = logging_config
setup_logging = logging_config.setup_logging


def test_setup_logging_sets_debug(monkeypatch):
    captured = {}

    def fake_basicConfig(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(logging, "basicConfig", fake_basicConfig)
    logger = logging.getLogger("panorai")
    logger.setLevel(logging.NOTSET)

    setup_logging(logging.DEBUG)

    assert captured.get("level") == logging.DEBUG
    assert logger.level == logging.DEBUG
