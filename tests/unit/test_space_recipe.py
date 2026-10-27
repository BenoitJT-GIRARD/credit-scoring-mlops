"""The start script of the Hugging Face recipe runs files that exist in this repository.

`infra/huggingface/start.sh` used to start Streamlit on `src/credexp/app/app.py`, a file
the repository does not have: the interface's entry page is `Overview.py`. The image copies
`src/` and `scripts/` to `/app` with the same layout, so a path checked here is the path the
container runs.
"""

from __future__ import annotations

import re

from credexp.utils.paths import INFRA_DIR, ROOT_DIR

START = INFRA_DIR / "huggingface" / "start.sh"


def test_every_python_file_the_start_script_runs_exists() -> None:
    named = re.findall(r"\b(?:src|scripts)/[\w/]+\.py\b", START.read_text(encoding="utf-8"))

    assert named, "the start script names no Python file"
    assert [path for path in named if not (ROOT_DIR / path).is_file()] == []
