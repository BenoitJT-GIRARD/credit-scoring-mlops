"""The two Streamlit pages hold the screen and nothing else.

A page is the one file of a Python project that no test can run without a browser or a
Streamlit runtime, so whatever is written inside it is the part of the project no test ever sees. Both pages
here used to hold their own HTTP client, their own SQL and their own error handling; all
three now live in `credexp.app`, where `tests/unit/app/` reaches them.

What this file asserts is that they stay that way. It reads the page sources as text: a
page that grows a `create_engine`, a `httpx` call or a `SELECT` has taken logic back out of
the tested modules, and the next reader of this repository will not notice.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from credexp.utils import SRC_DIR

PAGES_DIR = SRC_DIR / "app" / "pages"
PAGES = sorted(PAGES_DIR.glob("*.py"))

#: What belongs in `credexp.app`, never in a page.
FORBIDDEN = ("httpx.", "create_engine(", "read_sql", "SELECT ", "px.")

#: Streamlit numbers its pages by their filename, so the digit is part of the contract.
EXPECTED_PAGES = {"1_Score_an_applicant.py", "2_Recent_decisions.py"}


def test_both_pages_are_where_streamlit_looks_for_them() -> None:
    """`pages/` beside the entry script is the whole of Streamlit's multipage convention."""
    assert {page.name for page in PAGES} == EXPECTED_PAGES
    assert (SRC_DIR / "app" / "Overview.py").is_file()


@pytest.mark.parametrize("page", PAGES, ids=lambda path: path.name)
def test_a_page_holds_no_logic_of_its_own(page: Path) -> None:
    source = page.read_text(encoding="utf-8")
    found = [needle for needle in FORBIDDEN if needle in source]

    assert not found, f"{page.name} carries {found}: it belongs in credexp.app"


@pytest.mark.parametrize("page", PAGES, ids=lambda path: path.name)
def test_a_page_defines_no_function_a_test_cannot_reach(page: Path) -> None:
    """A helper defined inside a page is a helper no test imports."""
    tree = ast.parse(page.read_text(encoding="utf-8"))
    defined = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]

    assert defined == []


@pytest.mark.parametrize("page", PAGES, ids=lambda path: path.name)
def test_a_page_imports_what_it_shows_from_the_package(page: Path) -> None:
    tree = ast.parse(page.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert any(module.startswith("credexp.") for module in imported)


@pytest.mark.parametrize("page", PAGES, ids=lambda path: path.name)
def test_a_page_stays_short_enough_to_read_in_one_screen(page: Path) -> None:
    """Sixty-five lines is the size of a screen, and a screen is what a page describes."""
    assert len(page.read_text(encoding="utf-8").splitlines()) <= 65
