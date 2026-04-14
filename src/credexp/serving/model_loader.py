from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import pandas as pd
from mlflow.tracking import MlflowClient

import mlflow
from credexp.config import settings
from credexp.data.io import processed_dir


@dataclass(frozen=True)
class ModelBundle:
    pipe: object
    threshold: float
    model_name: str
    model_version: str
    feature_columns: list[str]


def _load_feature_columns() -> list[str]:
    holdout_path = processed_dir() / "api_holdout.parquet"
    df = pd.read_parquet(holdout_path)
    return [col for col in df.columns if col != "TARGET"]


def _load_threshold() -> float:
    threshold_path = settings.artifacts_dir / "models" / "threshold.json"
    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    return float(payload["threshold"])


def _resolve_model_version(model_name: str) -> str:
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
        pass
    return "unknown"


def load_model_bundle() -> ModelBundle:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_registry_uri)

    model_name = "credit_scoring_model"
    model_version = _resolve_model_version(model_name)

    try:
        pipe = mlflow.sklearn.load_model(settings.model_uri)
    except Exception:
        fallback_path = settings.artifacts_dir / "models" / "pipeline.joblib"
        pipe = joblib.load(fallback_path)

    threshold = _load_threshold()
    feature_columns = _load_feature_columns()

    return ModelBundle(
        pipe=pipe,
        threshold=threshold,
        model_name=model_name,
        model_version=model_version,
        feature_columns=feature_columns,
    )
