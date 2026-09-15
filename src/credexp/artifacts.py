"""How a published artefact is written: one line ending, one indent, one final newline.

Git stores what it is given after normalising it, so a file written with the wrong ending
looks identical in a diff and differs on disk. The difference surfaces the day a run is
compared with what is committed. Two writers need telling: the CSV writer, whose default row
terminator is not LF, and `Path.write_text`, which translates on Windows.

Nothing here is about the content of a result. It is about the file being the same file
when it is written twice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: The line ending of every artefact this repository publishes, on every platform.
LINE_TERMINATOR = "\n"

#: Two spaces, sorted keys, one trailing newline. A JSON artefact is read in a diff as often
#: as it is read by a program, and an unsorted dump reorders itself between two runs.
INDENT = 2


def write_json(path: Path, payload: Any, *, sort_keys: bool = False) -> Path:
    """Write a published JSON artefact, and create the directory it lives in."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=INDENT, ensure_ascii=False, sort_keys=sort_keys)
        + LINE_TERMINATOR,
        encoding="utf-8",
        newline=LINE_TERMINATOR,
    )
    return path
