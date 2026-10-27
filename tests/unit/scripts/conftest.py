"""The scripts under ``scripts/`` are entry points, not modules of the package.

They are loaded from their file, under a private name, so that a test can call their
functions and replace what they import, without ``scripts/`` being put on ``sys.path``.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from types import ModuleType

import pytest

from credexp.utils.paths import SCRIPTS_DIR


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_script_{name}", SCRIPTS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None, f"scripts/{name}.py is missing"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def load_script() -> Callable[[str], ModuleType]:
    """``load_script("tune_optuna")`` returns a fresh copy of ``scripts/tune_optuna.py``."""
    return _load
