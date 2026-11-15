from __future__ import annotations

import pytest

from panorai_workbench.operational import OperationalMirror, OperationalSettings


def test_local_mode_is_ready_without_optional_clients() -> None:
    mirror = OperationalMirror(OperationalSettings())
    assert mirror.readiness() == {
        "mode": "local",
        "ready": True,
        "metadata": "filesystem",
        "objects": "filesystem",
    }


def test_replicated_mode_requires_a_closed_complete_configuration() -> None:
    with pytest.raises(ValueError, match="replicated persistence is incomplete"):
        OperationalSettings(mode="replicated").validate()


def test_replicated_mode_rejects_non_postgres_metadata() -> None:
    settings = OperationalSettings(
        mode="replicated",
        database_url="sqlite:///tmp/workbench.db",
        s3_endpoint_url="http://127.0.0.1:9000",
        s3_bucket="workbench",
        s3_access_key_id="local",
        s3_secret_access_key="local-secret",
    )
    with pytest.raises(ValueError, match="must use PostgreSQL"):
        settings.validate()


def test_replicated_mode_accepts_explicit_postgres_and_s3() -> None:
    OperationalSettings(
        mode="replicated",
        database_url="postgresql://workbench:secret@127.0.0.1/workbench",
        s3_endpoint_url="http://127.0.0.1:9000",
        s3_bucket="workbench",
        s3_access_key_id="local",
        s3_secret_access_key="local-secret",
    ).validate()
