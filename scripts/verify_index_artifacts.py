#!/usr/bin/env python3
"""Verify that a package index exposes exactly the locally built artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable
from urllib.parse import quote
from urllib.request import urlopen


def local_digests(paths: list[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise ValueError(f"artifact does not exist: {path}")
        if path.name in result:
            raise ValueError(f"duplicate local artifact filename: {path.name}")
        result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not result:
        raise ValueError("at least one local artifact is required")
    return result


def index_digests(payload: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in payload.get("urls", ()):
        filename = entry.get("filename")
        digest = entry.get("digests", {}).get("sha256")
        if not isinstance(filename, str) or not isinstance(digest, str):
            raise ValueError("index response contains an artifact without SHA-256")
        if filename in result:
            raise ValueError(f"duplicate index artifact filename: {filename}")
        result[filename] = digest.lower()
    return result


def verify_exact_artifacts(expected: dict[str, str], actual: dict[str, str]) -> None:
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatched = sorted(
        name
        for name in expected.keys() & actual.keys()
        if expected[name] != actual[name]
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "package-index artifact mismatch: "
            f"missing={missing}, unexpected={unexpected}, sha256={mismatched}"
        )


def fetch_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=30) as response:  # noqa: S310 - caller selects index
        return json.load(response)


def wait_for_exact_artifacts(
    *,
    url: str,
    expected: dict[str, str],
    attempts: int,
    delay_seconds: float,
    fetch: Callable[[str], dict[str, Any]] = fetch_json,
) -> None:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            verify_exact_artifacts(expected, index_digests(fetch(url)))
            return
        except (OSError, ValueError, json.JSONDecodeError) as error:
            last_error = error
            if attempt < attempts:
                time.sleep(delay_seconds)
    raise RuntimeError(
        f"index did not expose the exact artifact set after {attempts} attempts: "
        f"{last_error}"
    ) from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-url", required=True)
    parser.add_argument("--distribution", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--delay-seconds", type=float, default=10.0)
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()
    if args.attempts <= 0:
        parser.error("--attempts must be positive")
    if args.delay_seconds < 0:
        parser.error("--delay-seconds must be non-negative")

    url = (
        f"{args.index_url.rstrip('/')}"
        f"/{quote(args.distribution, safe='')}"
        f"/{quote(args.version, safe='')}/json"
    )
    expected = local_digests(args.artifacts)
    wait_for_exact_artifacts(
        url=url,
        expected=expected,
        attempts=args.attempts,
        delay_seconds=args.delay_seconds,
    )
    print(f"index artifact verification: OK ({len(expected)} files, {url})")


if __name__ == "__main__":
    main()
