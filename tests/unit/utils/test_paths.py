"""One module answers where the files are, and this is what it answers.

Three of these pin decisions rather than strings. The root is *found*, by walking up to the
marker file, because counting directories works from a source checkout and puts the whole tree
inside `site-packages` from a wheel. Nothing of this project is written under `data/`, for the
reason `credexp.utils.paths` gives. And `ensure_dirs` creates what a run writes into, never
what it reads.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from credexp.utils import paths


def test_the_root_holds_the_marker_it_was_found_by() -> None:
    assert (paths.ROOT_DIR / "pyproject.toml").is_file()


def test_an_explicit_override_wins_over_the_marker(tmp_path: Path, monkeypatch) -> None:
    """A container whose working directory is not the project sets this and is believed."""
    monkeypatch.setenv(paths.ROOT_ENV, str(tmp_path))

    assert paths._find_root() == tmp_path.resolve()


def test_the_environment_variable_is_named_after_the_package() -> None:
    """`CREDEXP_ROOT`, derived from the package name rather than written down."""
    assert paths.PACKAGE == "credexp"
    assert paths.ROOT_ENV == "CREDEXP_ROOT"


def test_the_three_served_artefacts_are_under_models() -> None:
    """What the API loads at startup: a pipeline, a threshold, and a column order."""
    for path in (paths.PIPELINE_PATH, paths.THRESHOLD_PATH, paths.FEATURE_COLUMNS_PATH):
        assert path.parent == paths.MODELS_DIR
        assert path.is_file(), f"{path.name} is tracked and has to be there"


def test_nothing_of_this_project_is_written_under_data() -> None:
    """The Kaggle download and everything derived from it live under `var/`."""
    for path in (paths.RAW_DIR, paths.INTERIM_DIR, paths.PROCESSED_DIR, paths.MLFLOW_DIR):
        assert paths.VAR_DIR in path.parents
    assert not (paths.ROOT_DIR / "data").exists()


def test_the_published_reports_are_under_reports(tmp_path: Path) -> None:
    for path in (
        paths.DECISION_DIR,
        paths.EXPLAINABILITY_DIR,
        paths.MONITORING_DIR,
        paths.PERFORMANCE_DIR,
        paths.TUNING_DIR,
        paths.FIGURES_DIR,
    ):
        assert path.parent == paths.REPORTS_DIR


def test_ensure_dirs_creates_what_a_run_writes_into(tmp_path: Path, monkeypatch) -> None:
    """And nothing it reads: a missing input has to fail where it is read."""
    monkeypatch.setattr(paths, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(paths, "FIGURES_DIR", tmp_path / "reports" / "figures")
    monkeypatch.setattr(paths, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(paths, "VAR_DIR", tmp_path / "var")

    paths.ensure_dirs()

    assert (tmp_path / "reports" / "figures").is_dir()
    assert (tmp_path / "models").is_dir()
    assert (tmp_path / "var").is_dir()
    assert not (tmp_path / "data").exists()


def test_the_root_is_resolved_once_and_read_everywhere(monkeypatch) -> None:
    """`credexp.config` republishes it; it does not compute a second answer."""
    from credexp import config

    assert config.PROJECT_ROOT == paths.ROOT_DIR
    assert config.settings.project_root == paths.ROOT_DIR


@pytest.mark.parametrize(
    "name",
    [
        "SRC_DIR",
        "TESTS_DIR",
        "SCRIPTS_DIR",
        "NOTEBOOKS_DIR",
        "DOCS_DIR",
        "REPORTS_DIR",
        "INFRA_DIR",
    ],
)
def test_every_root_directory_of_the_vocabulary_is_named(name: str) -> None:
    """Every root directory this project is allowed to have, as its own module names them."""
    path = getattr(paths, name)

    assert path.parent == paths.ROOT_DIR or path.parent.parent == paths.ROOT_DIR
    assert os.fspath(path)


def test_load_parquet_logs_a_path_anyone_can_follow() -> None:
    """`load_parquet path=...` ends up read on a machine other than the one that wrote it."""
    assert paths.rel(paths.REPORTS_DIR / "figures" / "roc.png") == "reports/figures/roc.png"


def test_a_table_kept_elsewhere_keeps_its_whole_path(tmp_path) -> None:
    """The Kaggle tables often sit outside the checkout, and only the full path finds them."""
    elsewhere = tmp_path / "application_train.csv"

    assert paths.rel(elsewhere) == elsewhere.resolve().as_posix()
