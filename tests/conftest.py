"""Shared pytest fixtures and unittest-compatible setup.

This file is also import-friendly for `python -m unittest discover`
because it doesn't use any pytest-only APIs at import time.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

# Make the repo root importable when running tests from any cwd. This
# lets `python -m pytest` and `python -m unittest discover` both work
# without an editable install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Quieten the import-time print()s in Agent.py during tests.
os.environ.setdefault("STREAMLIT_RUN", "1")


def make_sample_csv(path: os.PathLike | str) -> str:
    """Write a small, realistic CSV to `path` and return the path."""
    import csv
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "age", "score"])
        w.writerow(["Alice", 30, 88])
        w.writerow(["Bob", 25, 92])
        w.writerow(["Carol", 35, 75])
    return str(p)


def make_sample_txt(path: os.PathLike | str, body: str) -> str:
    """Write `body` to `path` and return the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return str(p)


# Pytest fixtures ----------------------------------------------------------

def pytest_configure(config: Any) -> None:
    """Tell pytest/agents running this suite that this is a test run."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration (deselect with '-m \"not integration\"')"
    )


def _tmp_workdir() -> Iterator[str]:
    """Yield a temp dir as the test's CWD and clean it up after."""
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="dsvrf-") as tmp:
        os.chdir(tmp)
        try:
            yield tmp
        finally:
            os.chdir(old_cwd)


# Pytest fixtures ----------------------------------------------------------
# Importing pytest directly would break `python -m unittest discover`,
# so we put the @pytest.fixture definitions behind a try/import. If
# pytest isn't installed (it's a dev-only dep), the fixtures below are
# still importable as helper functions; they just won't be auto-wired.

try:
    import pytest  # type: ignore
    _HAVE_PYTEST = True
except ImportError:  # pragma: no cover - dev dep
    pytest = None  # type: ignore
    _HAVE_PYTEST = False


def pytest_configure(config: Any) -> None:  # pragma: no cover - pytest hook
    if _HAVE_PYTEST:
        config.addinivalue_line(
            "markers",
            "integration: marks tests as integration (deselect with '-m \"not integration\"')",
        )


if _HAVE_PYTEST:

    @pytest.fixture
    def tmp_workdir() -> Iterator[str]:
        """A per-test CWD inside a temp dir; restored on teardown."""
        yield from _tmp_workdir()

    @pytest.fixture
    def sample_csv_path(tmp_workdir: str) -> str:
        """Path to a real, well-formed CSV inside the test CWD."""
        return make_sample_csv(os.path.join(tmp_workdir, "people.csv"))

    @pytest.fixture
    def sample_txt_path(tmp_workdir: str) -> str:
        """Path to a small .txt file with known content."""
        return make_sample_txt(
            os.path.join(tmp_workdir, "notes.txt"),
            "Chapter 1 introduction.\n\n"
            "The solar system consists of the Sun and the planets.\n\n"
            "Chapter 2 the secret ingredient.\n\n"
            "A rare isotope called unobtanium-238 was found on Europa.\n",
        )
else:  # pragma: no cover - dev dep fallback
    # Plain functions for unittest users.
    tmp_workdir = _tmp_workdir  # type: ignore[assignment]
    sample_csv_path = make_sample_csv  # type: ignore[assignment]
    sample_txt_path = make_sample_txt  # type: ignore[assignment]
