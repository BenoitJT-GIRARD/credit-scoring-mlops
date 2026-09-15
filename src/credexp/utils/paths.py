"""Where this project's files are. One module answers, and nothing else computes a root.

Every path hangs off :data:`ROOT_DIR`, which is **found**, never assumed. Counting
directories up from a file only works from an editable ``src/`` checkout: installed as a
wheel, the package lands inside ``site-packages``, and the whole tree — models, reports,
figures — would be written there without a word. Finding the marker file instead means the
same code works from a checkout, from a wheel and from a container.

Nothing else in the project resolves a root. No ``sys.path.insert``, no ``Path(__file__)``
walked back three times in a script, no artefact read or written through a path relative to
the working directory: each of those is a second answer to a question that already has one,
and they disagree the day someone runs a script from another directory.

The directory names are a closed vocabulary.
A project adds its own *named artefacts* below — the served model, the published table — and
never a new root directory: a directory outside the vocabulary exists only when
a technical reason imposes it, and that reason is written down.
"""

from __future__ import annotations

import os
from pathlib import Path


def _package_name() -> str:
    """The distribution package this module belongs to, read from the import system.

    Nothing in it names this project, so the module can be copied as it is.
    """
    if __package__:
        return __package__.split(".")[0]
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if candidate.parent.name == "src":
            return candidate.name
    return here.parent.name


#: The package name, and the environment variable that overrides the root: ``<PACKAGE>_ROOT``.
PACKAGE: str = _package_name()
ROOT_ENV: str = f"{PACKAGE.upper()}_ROOT"


def _find_root() -> Path:
    """An explicit override, the marker file, or — installed outside a checkout — the cwd."""
    override = os.environ.get(ROOT_ENV)
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    # Installed outside a checkout. The working directory is the only defensible guess, and
    # it keeps what a run writes where the user is working, and out of site-packages.
    return Path.cwd().resolve()


ROOT_DIR: Path = _find_root()

SRC_DIR: Path = ROOT_DIR / "src" / PACKAGE
TESTS_DIR: Path = ROOT_DIR / "tests"
SCRIPTS_DIR: Path = ROOT_DIR / "scripts"
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"
DOCS_DIR: Path = ROOT_DIR / "docs"
IMAGES_DIR: Path = DOCS_DIR / "images"
REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
DATA_DIR: Path = ROOT_DIR / "data"
MODELS_DIR: Path = ROOT_DIR / "models"
INFRA_DIR: Path = ROOT_DIR / "infra"

#: The only ignored directory: logs, checkpoints, caches, coverage reports, anything a run
#: leaves behind that no reader is meant to open.
VAR_DIR: Path = ROOT_DIR / "var"


def ensure_dirs() -> None:
    """Create the directories a run writes into.

    Reading is never a reason to create a directory: a missing input must fail where it is
    missing, not quietly become an empty folder.
    """
    for path in (REPORTS_DIR, FIGURES_DIR, MODELS_DIR, VAR_DIR):
        path.mkdir(parents=True, exist_ok=True)


# --- What this project reads and writes, by name ----------------------------
#
# The names above are the root directories. The ones below belong to credit scoring, and every
# module that needs one of these files imports it from here.

#: The three files the API loads. They are the deliverable of the training: a fitted
#: pipeline, the threshold chosen on the cost curve, and the column order the pipeline was
#: fitted on. A prediction made with the columns in another order is silently wrong, which
#: is why the order is a file and not a docstring.
PIPELINE_PATH: Path = MODELS_DIR / "pipeline.joblib"
THRESHOLD_PATH: Path = MODELS_DIR / "threshold.json"
FEATURE_COLUMNS_PATH: Path = MODELS_DIR / "feature_columns.json"

#: Where the data lives, and why none of it is under `data/`.
#:
#: Home Credit's terms do not allow redistribution, so this repository ships no row of it.
#: What a reader downloads goes to `var/data/raw/`, and everything derived from it stays
#: beside it: a root directory that exists only to be ignored tells a reader nothing, and
#: `var/` already means « what a run needs and git does not keep ».
DATA_ROOT: Path = VAR_DIR / "data"
RAW_DIR: Path = DATA_ROOT / "raw"
INTERIM_DIR: Path = DATA_ROOT / "interim"
PROCESSED_DIR: Path = DATA_ROOT / "processed"

#: The two frames the published numbers are computed on.
HOLDOUT_PATH: Path = PROCESSED_DIR / "api_holdout.parquet"
DRIFT_REFERENCE_PATH: Path = PROCESSED_DIR / "reference.parquet"

#: MLflow's backing store. A SQLite database and its artefact tree, both run state.
MLFLOW_DIR: Path = VAR_DIR / "mlflow"
MLFLOW_DB_PATH: Path = MLFLOW_DIR / "mlflow.db"
MLFLOW_ARTIFACT_ROOT: Path = MLFLOW_DIR / "artifacts"

#: Published results, by the section of the README that reads them.
DECISION_DIR: Path = REPORTS_DIR / "decision"
EXPLAINABILITY_DIR: Path = REPORTS_DIR / "explainability"
MONITORING_DIR: Path = REPORTS_DIR / "monitoring"
PERFORMANCE_DIR: Path = REPORTS_DIR / "performance"
TUNING_DIR: Path = REPORTS_DIR / "tuning"
