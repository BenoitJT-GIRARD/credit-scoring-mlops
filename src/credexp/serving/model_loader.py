from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

import mlflow
from credexp.config import settings
from credexp.data.io import processed_dir
from mlflow.tracking import MlflowClient


@dataclass(frozen=True)
class ModelBundle:
    pipe: object
    threshold: float
    model_name: str
    model_version: str
    feature_columns: list[str]


def _load_feature_columns() -> list[str]:
    """Load the exact feature column order expected by the model.

    Priority:
    1. FEATURE_COLUMNS_PATH env var / artifacts/models/feature_columns.json
    2. data/processed/api_holdout.parquet
    3. data/processed/features.parquet

    The JSON option is required for remote deployment where processed datasets
    are not necessarily shipped with the API image.
    """
    feature_columns_path = Path(
        os.getenv(
            "FEATURE_COLUMNS_PATH",
            "artifacts/models/feature_columns.json",
        )
    )

    if feature_columns_path.exists():
        return json.loads(feature_columns_path.read_text(encoding="utf-8"))

    holdout_path = processed_dir() / "api_holdout.parquet"
    if holdout_path.exists():
        df = pd.read_parquet(holdout_path)
        return [col for col in df.columns if col != "TARGET"]

    features_path = processed_dir() / "features.parquet"
    if features_path.exists():
        df = pd.read_parquet(features_path)
        return [col for col in df.columns if col != "TARGET"]

    raise FileNotFoundError(
        "Could not load feature columns. Provide either "
        "artifacts/models/feature_columns.json, "
        "data/processed/api_holdout.parquet, or data/processed/features.parquet."
    )


def _load_threshold() -> float:
    """Load the business threshold.

    Priority:
    1. THRESHOLD_PATH env var / artifacts/models/threshold.json
    2. DEFAULT_THRESHOLD env var
    3. 0.5 fallback
    """
    threshold_path = Path(
        os.getenv(
            "THRESHOLD_PATH",
            str(settings.artifacts_dir / "models" / "threshold.json"),
        )
    )

    if threshold_path.exists():
        payload = json.loads(threshold_path.read_text(encoding="utf-8"))
        return float(payload["threshold"])

    return float(os.getenv("DEFAULT_THRESHOLD", "0.5"))


def _load_local_joblib_model() -> object:
    """Load a local joblib pipeline.

    Used for:
    - Hugging Face API Space deployment
    - fallback when MLflow registry is unavailable
    """
    model_path = Path(
        os.getenv(
            "PIPELINE_PATH",
            str(settings.artifacts_dir / "models" / "pipeline.joblib"),
        )
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Local model not found: {model_path}")

    return joblib.load(model_path)


def _resolve_model_version(model_name: str) -> str:
    """Resolve model version from MLflow Registry when available."""
    try:
        client = MlflowClient(
            tracking_uri=settings.mlflow_tracking_uri,
            registry_uri=settings.mlflow_registry_uri,
        )
        versions = client.search_model_versions(f"name='{model_name}'")

        for version in versions:
            aliases = getattr(version, "aliases", [])
            current_stage = getattr(version, "current_stage", "")
            if "Production" in aliases or current_stage == "Production":
                return str(version.version)

        if versions:
            return str(sorted(versions, key=lambda v: int(v.version), reverse=True)[0].version)

    except Exception:
        return "unknown"

    return "unknown"


def _load_mlflow_model(model_name: str) -> tuple[object, str]:
    """Load model from MLflow Registry."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_registry_uri)

    pipe = mlflow.sklearn.load_model(settings.model_uri)
    model_version = _resolve_model_version(model_name)

    return pipe, model_version


def load_model_bundle() -> ModelBundle:
    """Load model, threshold and feature schema.

    Local full-stack mode:
        uses MLflow Registry first, then fallback joblib.

    Remote lightweight mode:
        set USE_LOCAL_MODEL=true and provide:
        - PIPELINE_PATH
        - THRESHOLD_PATH
        - FEATURE_COLUMNS_PATH
    """
    model_name = os.getenv("MODEL_NAME", "credit_scoring_model")
    use_local_model = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"

    threshold = _load_threshold()
    feature_columns = _load_feature_columns()

    if use_local_model:
        pipe = _load_local_joblib_model()
        model_version = os.getenv("MODEL_VERSION", "remote-joblib")
    else:
        try:
            pipe, model_version = _load_mlflow_model(model_name)
        except Exception:
            pipe = _load_local_joblib_model()
            model_version = os.getenv("MODEL_VERSION", "joblib-fallback")

    return ModelBundle(
        pipe=pipe,
        threshold=threshold,
        model_name=model_name,
        model_version=str(model_version),
        feature_columns=feature_columns,
    )
