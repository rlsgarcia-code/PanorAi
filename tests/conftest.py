"""Pytest fixtures used across the PanorAi test suite.

This module patches :class:`pydantic.BaseModel` so that tests work on both
pydantic v1 and v2. In pydantic>=2 the ``copy`` method was renamed to
``model_copy``. The PanorAi code expects ``model_copy`` to be present, so
when running under pydantic<2 we provide it before importing modules that
rely on it.
"""

import pytest
from pydantic import BaseModel


@pytest.fixture(autouse=True)
def ensure_model_copy(monkeypatch):
    """Ensure ``BaseModel.model_copy`` exists for older pydantic versions."""
    if not hasattr(BaseModel, "model_copy"):
        monkeypatch.setattr(BaseModel, "model_copy", BaseModel.copy, raising=False)
    yield
