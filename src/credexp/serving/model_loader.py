from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from credexp.config import settings
from credexp.data.io import processed_dir
from credexp.modeling.manifest import ModelManifest, read_manifest
from credexp.serving.failures import FailureKind
from credexp.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ModelBundle:
    """Container for the loaded scoring model and its serving metadata."""

    pipe: object
    threshold: float
    model_name: str
    model_version: str
    feature_columns: list[str]


def _model_dir() -> Path:
    """Return the default local model artifact directory."""
    return settings.artifacts_dir / "models"


def _first_existing_path(candidates: list[str | Path | None]) -> Path | None:
    """Return the first existing path from a list of optional candidates."""
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return None


def _load_threshold() -> float:
    """Load the business decision threshold.

    Priority:
    1. THRESHOLD_PATH environment variable
    2. artifacts/models/threshold.json
    3. DEFAULT_THRESHOLD environment variable
    4. 0.5 fallback
    """
    threshold_path = _first_existing_path(
        [
            os.getenv("THRESHOLD_PATH"),
            _model_dir() / "threshold.json",
        ]
    )

    if threshold_path is not None:
        payload = json.loads(threshold_path.read_text(encoding="utf-8"))
        return float(payload["threshold"])

    return float(os.getenv("DEFAULT_THRESHOLD", "0.5"))


def _load_feature_columns(pipe: object | None = None) -> list[str]:
    """Load the exact feature column order expected by the model.

    Priority:
    1. FEATURE_COLUMNS_PATH environment variable
    2. artifacts/models/feature_columns.json
    3. data/processed/api_holdout.parquet
    4. data/processed/features.parquet
    5. pipe.feature_names_in_ if available

    The JSON option is preferred for remote deployment, because processed
    datasets are not necessarily shipped with the deployed API image.
    """
    feature_columns_path = _first_existing_path(
        [
            os.getenv("FEATURE_COLUMNS_PATH"),
            _model_dir() / "feature_columns.json",
        ]
    )

    if feature_columns_path is not None:
        columns = json.loads(feature_columns_path.read_text(encoding="utf-8"))
        return [str(col) for col in columns]

    holdout_path = processed_dir() / "api_holdout.parquet"
    if holdout_path.exists():
        df = pd.read_parquet(holdout_path)
        return [col for col in df.columns if col != "TARGET"]

    features_path = processed_dir() / "features.parquet"
    if features_path.exists():
        df = pd.read_parquet(features_path)
        return [col for col in df.columns if col != "TARGET"]

    if pipe is not None and hasattr(pipe, "feature_names_in_"):
        return [str(col) for col in pipe.feature_names_in_]

    raise FileNotFoundError(
        "Could not load feature columns. Provide one of: "
        "FEATURE_COLUMNS_PATH, artifacts/models/feature_columns.json, "
        "data/processed/api_holdout.parquet, data/processed/features.parquet, "
        "or a pipeline exposing feature_names_in_."
    )


def _load_local_joblib_model() -> object:
    """Load a local joblib pipeline.

    Used for:
    - Hugging Face Space deployment
    - local debug without MLflow
    - fallback when MLflow Registry is unavailable
    """
    model_path = _first_existing_path(
        [
            os.getenv("MODEL_JOBLIB_PATH"),
            os.getenv("PIPELINE_PATH"),
            _model_dir() / "pipeline.joblib",
        ]
    )

    if model_path is None:
        raise FileNotFoundError(
            "Local model not found. Expected one of: "
            "MODEL_JOBLIB_PATH, PIPELINE_PATH, artifacts/models/pipeline.joblib."
        )

    return joblib.load(model_path)


def _resolve_model_version(model_name: str) -> str:
    """Resolve model version from MLflow Registry when available.

    This function is best-effort. If the registry is unavailable, it returns
    'unknown' instead of breaking the API startup.
    """
    try:
        client = MlflowClient(
            tracking_uri=settings.mlflow_tracking_uri,
            registry_uri=settings.mlflow_registry_uri,
        )
        versions = list(client.search_model_versions(f"name='{model_name}'"))

        for version in versions:
            aliases = getattr(version, "aliases", []) or []
            current_stage = getattr(version, "current_stage", "")
            if "Production" in aliases or current_stage == "Production":
                return str(version.version)

        if versions:
            latest = sorted(versions, key=lambda item: int(item.version), reverse=True)[0]
            return str(latest.version)

    except Exception:
        return "unknown"

    return "unknown"


def _load_mlflow_model(model_uri: str, model_name: str) -> tuple[object, str]:
    """Load model from MLflow Registry or MLflow model URI."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_registry_uri)

    pipe = mlflow.sklearn.load_model(model_uri)
    model_version = _resolve_model_version(model_name)

    return pipe, model_version


def _should_use_joblib(load_mode: str, model_uri: str) -> bool:
    """Return whether the loader should bypass MLflow and use joblib directly."""
    use_local_model = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"

    return use_local_model or load_mode in {"joblib", "local"} or not model_uri


def _load_manifest() -> ModelManifest | None:
    """Read the model manifest beside the artefact, if the training wrote one.

    A manifest whose schema this code does not read raises rather than being ignored: a
    schema bump means the feature columns mean something else, and serving them anyway
    would produce predictions from inputs the model never saw in that sense.
    """
    path = _first_existing_path(
        [os.getenv("MODEL_MANIFEST_PATH"), _model_dir() / "model_manifest.json"]
    )
    return read_manifest(path) if path else None


def load_model_bundle() -> ModelBundle:
    """Load model, threshold and feature schema.

    Local full-stack mode:
        MODEL_LOAD_MODE=auto
        MODEL_URI=models:/credit_scoring_model/Production

    Remote Hugging Face mode:
        MODEL_LOAD_MODE=joblib
        MODEL_JOBLIB_PATH=/app/artifacts/models/pipeline.joblib
        THRESHOLD_PATH=/app/artifacts/models/threshold.json
        FEATURE_COLUMNS_PATH=/app/artifacts/models/feature_columns.json

    Backward compatibility:
        USE_LOCAL_MODEL=true and PIPELINE_PATH are also supported.
    """
    model_name = os.getenv("MODEL_NAME", "credit_scoring_model")
    model_uri = os.getenv("MODEL_URI", getattr(settings, "model_uri", "")).strip()
    load_mode = os.getenv("MODEL_LOAD_MODE", "auto").lower()

    model_version = os.getenv("MODEL_VERSION", "unknown")

    if _should_use_joblib(load_mode=load_mode, model_uri=model_uri):
        pipe = _load_local_joblib_model()
        model_version = os.getenv("MODEL_VERSION", "joblib")
    else:
        try:
            pipe, model_version = _load_mlflow_model(
                model_uri=model_uri,
                model_name=model_name,
            )
        except Exception as exc:
            # The fallback is deliberate — it is what the lightweight remote deployment
            # runs on. Its silence was not: a misconfigured registry made the API serve a
            # different model than the one it was asked for, and nothing said why. The
            # joblib-fallback version tag records *that* it fell back; this records what
            # it hit.
            log.warning(
                "model_registry_unavailable",
                extra={
                    "model_uri": model_uri,
                    "failure_kind": str(FailureKind.MODEL_UNAVAILABLE),
                    "error": repr(exc),
                },
            )
            pipe = _load_local_joblib_model()
            model_version = os.getenv("MODEL_VERSION", "joblib-fallback")

    manifest = _load_manifest()
    if manifest is not None:
        # The manifest is the single source when it exists: threshold and feature order
        # come from the same file the training wrote, so the two cannot drift apart.
        threshold = manifest.threshold
        feature_columns = list(manifest.feature_columns)
    else:
        # Artefacts trained before the manifest existed. Kept working rather than refused,
        # but the log says which path was taken so a stale artefact is visible.
        log.warning(
            "model_manifest_missing",
            extra={"searched": str(_model_dir() / "model_manifest.json")},
        )
        threshold = _load_threshold()
        feature_columns = _load_feature_columns(pipe=pipe)

    return ModelBundle(
        pipe=pipe,
        threshold=threshold,
        model_name=model_name,
        model_version=str(model_version),
        feature_columns=feature_columns,
    )
