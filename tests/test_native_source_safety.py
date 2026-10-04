from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_SOURCE = ROOT / "panorai" / "_native" / "geometry_kernels.cpp"


def _function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening : index + 1]
    raise AssertionError(f"unterminated function {signature}")


def test_native_geometry_has_exception_safe_gil_and_thread_ownership() -> None:
    source = GEOMETRY_SOURCE.read_text(encoding="utf-8")

    assert "Py_BEGIN_ALLOW_THREADS" not in source
    assert "Py_END_ALLOW_THREADS" not in source
    assert "class AllowThreads" in source
    assert "class ThreadGroup" in source
    assert re.search(r"~ThreadGroup\(\)\s*\{\s*join\(\);\s*\}", source)
    assert "catch (const std::bad_alloc&)" in source
    assert "catch (const std::system_error& error)" in source

    for entry_point in (
        "PyObject* equirectangular_to_gnomonic_batch(PyObject*, PyObject* args)",
        "PyObject* gnomonic_gaussian_to_equirectangular(PyObject*, PyObject* args)",
        "PyObject* cubemap_to_equirectangular(PyObject*, PyObject* args)",
    ):
        body = _function_body(source, entry_point)
        assert "translate_cpp_exceptions" in body

    worker_body = _function_body(source, "void gaussian_reconstruct_range(")
    assert "std::vector" not in worker_body
    assert (
        "noexcept"
        in source[
            source.index("void gaussian_reconstruct_range(") : source.index(
                "void gaussian_reconstruct_range("
            )
            + 700
        ]
    )
