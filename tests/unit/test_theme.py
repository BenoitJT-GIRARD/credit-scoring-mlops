"""The theme file and the figure palette hold the same four colours, or the pages lie.

``.streamlit/config.toml`` is read before the process starts, so it imports nothing and its
values are typed by hand; the charts take theirs from :mod:`credexp.figure_style`. Two copies
of one decision, and this file is what stops them drifting.
"""

from __future__ import annotations

import tomllib

import pytest

from credexp.figure_style import PALETTE
from credexp.utils.paths import INFRA_DIR, ROOT_DIR

THEME = tomllib.loads((ROOT_DIR / ".streamlit" / "config.toml").read_text(encoding="utf-8"))


def test_every_theme_colour_is_a_token_of_the_palette() -> None:
    expected = {
        "primaryColor": PALETTE["primary"],
        "backgroundColor": PALETTE["paper"],
        "secondaryBackgroundColor": PALETTE["surface"],
        "textColor": PALETTE["ink"],
    }
    assert {key: THEME["theme"][key] for key in expected} == expected


def test_the_editor_toolbar_is_hidden() -> None:
    """The Deploy button and the three-dot menu belong to the author, not to the product."""
    assert THEME["client"]["toolbarMode"] == "minimal"


def test_the_page_accepts_no_upload() -> None:
    """One page asks an API and the other reads a log: neither has an upload to accept."""
    assert THEME["server"]["maxUploadSize"] == 1
    assert THEME["server"]["enableXsrfProtection"] is True


@pytest.mark.parametrize(
    "dockerfile", ["api.Dockerfile", "huggingface/Dockerfile"], ids=["compose", "space"]
)
def test_the_images_that_run_the_pages_carry_the_theme(dockerfile: str) -> None:
    """Streamlit reads `.streamlit/` from its working directory, so an image without it serves
    the default colours and the Deploy toolbar whatever this file says."""
    lines = (INFRA_DIR / dockerfile).read_text(encoding="utf-8").splitlines()
    copies = [line.split() for line in lines if line.startswith("COPY ")]

    assert any(".streamlit" in parts[1:-1] for parts in copies)
