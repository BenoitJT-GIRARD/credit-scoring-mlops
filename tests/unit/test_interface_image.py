"""The image the interface runs in installs every package the interface imports.

`docker-compose.yml` builds the Streamlit service from `infra/api.Dockerfile`, which installs
some dependency groups and leaves `dev` and `mlops` out. `credexp.app.charts` imports Plotly,
which reached the environment only through Evidently, a package of the `mlops` group. The
decision-log page then stopped on `ModuleNotFoundError` in the compose stack, while the suite
passed on a machine where every group is installed. `credexp.app.service` imports httpx, which
was declared in the `dev` group only.

The check reads the groups from the Dockerfile and follows `uv.lock` from them, so it sees the
set of packages the image receives, whatever is installed where the test runs.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

from credexp.utils.paths import INFRA_DIR, ROOT_DIR, SRC_DIR

APP_DIR = SRC_DIR / "app"


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _image_groups() -> set[str]:
    """The groups the `uv sync` of the Dockerfile installs. Comments are skipped: one of them
    names the group the image leaves out."""
    lines = (INFRA_DIR / "api.Dockerfile").read_text(encoding="utf-8").splitlines()
    instructions = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
    return set(re.findall(r"--group\s+([\w-]+)", instructions))


def _installed_by_the_image() -> set[str]:
    """Every distribution `uv sync` puts in the image, through the lock's dependency graph."""
    lock = tomllib.loads((ROOT_DIR / "uv.lock").read_text(encoding="utf-8"))
    packages = {_norm(package["name"]): package for package in lock["package"]}
    project = packages["credexp"]
    pending = list(project.get("dependencies", []))
    for group in _image_groups():
        pending += project.get("dev-dependencies", {}).get(group, [])

    seen: set[tuple[str, str | None]] = set()
    names: set[str] = set()
    while pending:
        dependency = pending.pop()
        name = _norm(dependency["name"])
        for extra in [None, *dependency.get("extra", [])]:
            if (name, extra) in seen:
                continue
            seen.add((name, extra))
            names.add(name)
            package = packages.get(name, {})
            if extra is None:
                pending += package.get("dependencies", [])
            else:
                pending += package.get("optional-dependencies", {}).get(extra, [])
    return names


def _module_file(module: str) -> Path | None:
    """The source file of a `credexp` module, or None for a name that is not one."""
    relative = Path(*module.split(".")[1:])
    for candidate in (SRC_DIR / relative.with_suffix(".py"), SRC_DIR / relative / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _third_party_imports() -> set[str]:
    """Top-level imports of the pages and of every `credexp` module they reach at import time.

    Imports written inside a function are left out, as the loader's import of MLflow is: the
    serving path is built so that they run only when that function is called.
    """
    queue = sorted(APP_DIR.rglob("*.py"))
    visited: set[Path] = set()
    found: set[str] = set()
    while queue:
        path = queue.pop()
        if path in visited:
            continue
        visited.add(path)
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                top = module.split(".")[0]
                if top == "credexp":
                    source = _module_file(module)
                    if source is not None:
                        queue.append(source)
                elif top not in sys.stdlib_module_names and top != "__future__":
                    found.add(top)
    return found


def test_every_package_the_interface_imports_is_installed_in_its_image() -> None:
    owners = packages_distributions()
    imported = _third_party_imports()
    assert "plotly" in imported and "streamlit" in imported, "the scan missed the pages"

    installed = _installed_by_the_image()
    missing = sorted(
        module
        for module in imported
        if not any(_norm(owner) in installed for owner in owners.get(module, [module]))
    )

    assert missing == [], f"imported by credexp.app but absent from the image: {missing}"
