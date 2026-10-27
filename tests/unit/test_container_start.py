"""The containers start on the environment their image built, without syncing it again.

Both images install their dependency groups at build time and leave the `dev` group out. A
container started with a plain `uv run` added that group at every start: it needed the
network to answer, and it downloaded the test and lint tools before serving anything.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from credexp.utils.paths import INFRA_DIR, ROOT_DIR

#: Every file that holds a command run inside a container of this repository.
START_FILES = (
    INFRA_DIR / "api.Dockerfile",
    ROOT_DIR / "docker-compose.yml",
    INFRA_DIR / "huggingface" / "start.sh",
)

#: `uv run` in shell form and `"uv", "run"` in exec form, with the token that follows.
UV_RUN = re.compile(r'\buv"?\s*,?\s*"?run"?\s*,?\s*"?([^\s",\]]+)')


def _uv_runs(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    commands = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
    return UV_RUN.findall(commands)


@pytest.mark.parametrize("path", START_FILES, ids=lambda path: path.name)
def test_every_uv_run_in_a_container_skips_the_sync(path: Path) -> None:
    following = _uv_runs(path)

    assert following, f"{path.name} runs no `uv run`"
    assert all(token == "--no-sync" for token in following), following
