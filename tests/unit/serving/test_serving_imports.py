"""The serving path runs without MLflow installed, which is why the image leaves it out.

`infra/api.Dockerfile` installs every dependency group but `mlops`. That is only safe while
no module the API imports at startup reaches for MLflow, and an import added at the top of
any of them would break the image without breaking a single test on a developer machine,
where MLflow is installed.

So the check is made the only way it can be: in a subprocess, with MLflow made unimportable.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from credexp.utils.paths import ROOT_DIR

#: Blocks `mlflow` and every submodule of it, then imports the serving path.
PROGRAM = """
import sys

class Refuse:
    def find_module(self, name, path=None):
        if name == "mlflow" or name.startswith("mlflow."):
            raise ImportError(f"{name} is not installed in the serving image")
        return None

    def find_spec(self, name, path=None, target=None):
        if name == "mlflow" or name.startswith("mlflow."):
            raise ImportError(f"{name} is not installed in the serving image")
        return None

sys.meta_path.insert(0, Refuse())

import credexp.serving.api  # noqa: F401
import credexp.serving.model_loader  # noqa: F401
import credexp.serving.audit_log  # noqa: F401
import credexp.app.service  # noqa: F401

print("imported")
"""


def _run(program: str) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", program],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )


def test_the_api_imports_with_mlflow_unavailable() -> None:
    done = _run(PROGRAM)

    assert done.returncode == 0, done.stderr[-3000:]
    assert "imported" in done.stdout


@pytest.mark.parametrize("module", ["credexp.serving.api", "credexp.serving.model_loader"])
def test_no_serving_module_imports_mlflow_at_the_top(module: str) -> None:
    """Imported normally, the serving modules still leave `sys.modules` free of MLflow."""
    program = f"import sys; import {module}; print('mlflow' in sys.modules)"

    done = _run(program)

    assert done.returncode == 0, done.stderr[-3000:]
    assert done.stdout.strip() == "False"
