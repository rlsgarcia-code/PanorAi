"""Optional operational persistence for a loopback-only Workbench deployment.

The local filesystem remains the execution cache and the default single-user
store.  ``replicated`` mode durably mirrors run metadata to PostgreSQL and
binary inputs/artifacts to an S3-compatible object store.  Browser requests
never choose connection strings, buckets, SQL, object keys, or Python imports.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any
from urllib.parse import urlparse


_TRUE = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class OperationalSettings:
    mode: str = "local"
    database_url: str | None = None
    s3_endpoint_url: str | None = None
    s3_bucket: str | None = None
    s3_region: str = "us-east-1"
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_secure: bool = False
    bootstrap: bool = False

    @classmethod
    def from_environment(cls) -> "OperationalSettings":
        mode = os.environ.get("PANORAI_WORKBENCH_PERSISTENCE", "local").strip()
        if mode not in {"local", "replicated"}:
            raise ValueError(
                "PANORAI_WORKBENCH_PERSISTENCE must be 'local' or 'replicated'"
            )
        settings = cls(
            mode=mode,
            database_url=os.environ.get("PANORAI_WORKBENCH_DATABASE_URL"),
            s3_endpoint_url=os.environ.get("PANORAI_WORKBENCH_S3_ENDPOINT_URL"),
            s3_bucket=os.environ.get("PANORAI_WORKBENCH_S3_BUCKET"),
            s3_region=os.environ.get("PANORAI_WORKBENCH_S3_REGION", "us-east-1"),
            s3_access_key_id=os.environ.get("PANORAI_WORKBENCH_S3_ACCESS_KEY_ID"),
            s3_secret_access_key=os.environ.get(
                "PANORAI_WORKBENCH_S3_SECRET_ACCESS_KEY"
            ),
            s3_secure=os.environ.get("PANORAI_WORKBENCH_S3_SECURE", "false").lower()
            in _TRUE,
            bootstrap=os.environ.get(
                "PANORAI_WORKBENCH_BOOTSTRAP_STORAGE", "false"
            ).lower()
            in _TRUE,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.mode == "local":
            return
        missing = [
            name
            for name, value in (
                ("PANORAI_WORKBENCH_DATABASE_URL", self.database_url),
                ("PANORAI_WORKBENCH_S3_ENDPOINT_URL", self.s3_endpoint_url),
                ("PANORAI_WORKBENCH_S3_BUCKET", self.s3_bucket),
                ("PANORAI_WORKBENCH_S3_ACCESS_KEY_ID", self.s3_access_key_id),
                (
                    "PANORAI_WORKBENCH_S3_SECRET_ACCESS_KEY",
                    self.s3_secret_access_key,
                ),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "replicated persistence is incomplete: " + ", ".join(missing)
            )
        database = urlparse(str(self.database_url))
        if database.scheme not in {"postgres", "postgresql"}:
            raise ValueError("the metadata URL must use PostgreSQL")
        endpoint = urlparse(str(self.s3_endpoint_url))
        if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
            raise ValueError("the S3 endpoint must be an absolute HTTP(S) URL")
        if endpoint.scheme == "http" and self.s3_secure:
            raise ValueError("S3 secure mode requires an https endpoint")


class OperationalMirror:
    """Lazy, fail-closed mirror with fixed schema and object-key policy."""

    def __init__(self, settings: OperationalSettings | None = None) -> None:
        self.settings = settings or OperationalSettings.from_environment()
        self._lock = threading.RLock()
        self._initialized = False
        self._postgres: Any = None
        self._s3: Any = None

    @property
    def enabled(self) -> bool:
        return self.settings.mode == "replicated"

    def _initialize(self) -> None:
        if not self.enabled or self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            try:
                import boto3
                import psycopg
            except ImportError as exc:
                raise RuntimeError(
                    "replicated persistence requires panorai-workbench[ops]"
                ) from exc
            self._postgres = psycopg
            self._s3 = boto3.client(
                "s3",
                endpoint_url=self.settings.s3_endpoint_url,
                region_name=self.settings.s3_region,
                aws_access_key_id=self.settings.s3_access_key_id,
                aws_secret_access_key=self.settings.s3_secret_access_key,
                use_ssl=self.settings.s3_secure,
            )
            if self.settings.bootstrap:
                self._bootstrap()
            self._initialized = True

    def _bootstrap(self) -> None:
        with self._postgres.connect(self.settings.database_url) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workbench_runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    document JSONB NOT NULL
                )
                """
            )
        buckets = {item["Name"] for item in self._s3.list_buckets().get("Buckets", [])}
        if self.settings.s3_bucket not in buckets:
            self._s3.create_bucket(Bucket=self.settings.s3_bucket)

    def record_run(self, run: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._initialize()
        document = json.dumps(run, ensure_ascii=False, allow_nan=False)
        with self._postgres.connect(self.settings.database_url) as connection:
            connection.execute(
                """
                INSERT INTO workbench_runs (run_id, status, updated_at, document)
                VALUES (%s, %s, %s, %s::jsonb)
                ON CONFLICT (run_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    updated_at = EXCLUDED.updated_at,
                    document = EXCLUDED.document
                """,
                (run["id"], run["status"], run["updated_at"], document),
            )

    def record_blob(
        self,
        key: str,
        payload: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> None:
        if not self.enabled:
            return
        if key.startswith("/") or ".." in Path(key).parts:
            raise ValueError("invalid operational object key")
        self._initialize()
        self._s3.put_object(
            Bucket=self.settings.s3_bucket,
            Key=key,
            Body=payload,
            ContentType=content_type,
            Metadata={"sha256": hashlib.sha256(payload).hexdigest()},
        )

    def record_file(self, key: str, path: Path) -> None:
        self.record_blob(key, path.read_bytes())

    def readiness(self) -> dict[str, Any]:
        if not self.enabled:
            return {
                "mode": "local",
                "ready": True,
                "metadata": "filesystem",
                "objects": "filesystem",
            }
        try:
            self._initialize()
            with self._postgres.connect(
                self.settings.database_url, connect_timeout=3
            ) as connection:
                connection.execute("SELECT 1").fetchone()
            self._s3.head_bucket(Bucket=self.settings.s3_bucket)
        except Exception as exc:
            return {
                "mode": "replicated",
                "ready": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {
            "mode": "replicated",
            "ready": True,
            "metadata": "postgresql",
            "objects": "s3-compatible",
            "bucket": self.settings.s3_bucket,
        }


MIRROR = OperationalMirror()


__all__ = ["OperationalMirror", "OperationalSettings", "MIRROR"]
