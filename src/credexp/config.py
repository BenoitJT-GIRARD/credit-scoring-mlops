"""Paths, MLflow locations and the environment overrides, in one settings object.

MLflow lives under `mlflow/` as a SQLite backend rather than a filesystem store: the
filesystem backend is deprecated, and a registry that cannot answer a version query is a
registry the training cannot read its own hyper-parameters back out of.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MLFLOW_DIR = PROJECT_ROOT / "mlflow"
MLFLOW_DB_PATH = MLFLOW_DIR / "mlflow.db"
MLFLOW_ARTIFACT_ROOT = MLFLOW_DIR / "artifacts"


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
    artifacts_dir: Path = ARTIFACTS_DIR
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
        str(DATA_DIR / "processed" / "reference.parquet"),
    )
    prometheus_enabled: bool = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"


settings = Settings()
