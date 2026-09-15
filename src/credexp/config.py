"""Paths, MLflow locations and the environment overrides, in one settings object.

MLflow keeps its store under `var/mlflow/`, on SQLite and not on the filesystem backend:
that backend is deprecated, and a registry that cannot answer a version query is one the
training cannot read its own hyper-parameters back out of.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from credexp.utils.paths import (
    DATA_ROOT,
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_DB_PATH,
    MLFLOW_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    ROOT_DIR,
)

load_dotenv()

#: Kept under this module's names because the settings object publishes them. The values
#: come from `credexp.utils.paths`, which is the only module in the project that resolves a
#: path; nothing here computes one.
PROJECT_ROOT = ROOT_DIR
DATA_DIR = DATA_ROOT


def _default_tracking_uri() -> str:
    return f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"


def _default_registry_uri() -> str:
    return _default_tracking_uri()


def _default_artifact_root() -> str:
    return MLFLOW_ARTIFACT_ROOT.as_posix()


@dataclass(frozen=True)
class Settings:
    # Environment
    env: str = os.getenv("ENV", "dev")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))

    # Paths
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    models_dir: Path = MODELS_DIR
    mlflow_dir: Path = MLFLOW_DIR

    # MLflow
    mlflow_tracking_uri: str = os.getenv(
        "MLFLOW_TRACKING_URI",
        _default_tracking_uri(),
    )
    mlflow_registry_uri: str = os.getenv(
        "MLFLOW_REGISTRY_URI",
        _default_registry_uri(),
    )
    mlflow_artifact_root: str = os.getenv(
        "MLFLOW_ARTIFACT_ROOT",
        _default_artifact_root(),
    )
    mlflow_experiment_name: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "credexp",
    )

    # API
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    model_uri: str = os.getenv(
        "MODEL_URI",
        "models:/credit_scoring_model/Production",
    )

    # Database
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@localhost:5432/credexp",
    )

    # Monitoring
    evidently_reference_path: str = os.getenv(
        "EVIDENTLY_REFERENCE_PATH",
        str(PROCESSED_DIR / "reference.parquet"),
    )
    prometheus_enabled: bool = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"


settings = Settings()
