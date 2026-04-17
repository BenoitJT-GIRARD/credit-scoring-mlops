from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from sqlalchemy import create_engine, text

from credexp.config import settings
from credexp.data.io import load_parquet
from credexp.utils.logging import get_logger

log = get_logger(__name__)


def _sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def load_reference_dataframe() -> pd.DataFrame:
    ref_path = Path(settings.evidently_reference_path)

    if not ref_path.is_absolute():
        ref_path = settings.project_root / ref_path

    log.info(f"Loading reference dataframe from {ref_path}")
    df = load_parquet(ref_path)
    return _sanitize_dataframe(df)


def load_current_dataframe(limit: int = 5000) -> pd.DataFrame:
    """
    Charge les dernières requêtes depuis PostgreSQL.
    On suppose que input_payload ressemble à:
    {
      "sk_id_curr": ...,
      "features": { ... }
    }
    """
    engine = create_engine(settings.database_url, pool_pre_ping=True)

    query = text(
        """
        SELECT created_at, input_payload
        FROM predictions
        WHERE status_code = 200
        ORDER BY created_at DESC
        LIMIT :limit
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(query, {"limit": limit}).fetchall()

    if not rows:
        raise ValueError("No production prediction logs found in PostgreSQL.")

    items = []
    for row in rows:
        payload = row.input_payload
        features = payload.get("features", {})
        features["sk_id_curr"] = payload.get("sk_id_curr")
        features["created_at"] = row.created_at
        items.append(features)

    df = pd.DataFrame(items)
    return _sanitize_dataframe(df)


def align_reference_and_current(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aligne les colonnes communes entre reference et current.
    On retire TARGET si présent.
    On retire created_at car absent de la référence et non utile pour le drift de features.
    """
    ref = reference_df.copy()
    cur = current_df.copy()

    for col in ["TARGET", "created_at"]:
        if col in ref.columns:
            ref = ref.drop(columns=[col])
        if col in cur.columns:
            cur = cur.drop(columns=[col])

    common_cols = sorted(set(ref.columns).intersection(set(cur.columns)))
    if not common_cols:
        raise ValueError("No common columns between reference and current datasets.")

    ref = ref[common_cols]
    cur = cur[common_cols]

    return ref, cur


def infer_data_definition(reference_df: pd.DataFrame) -> DataDefinition:
    """
    Définition minimale des types pour Evidently.
    On classe les colonnes object/category/bool en catégorielles,
    le reste en numériques.
    """
    numerical_columns = []
    categorical_columns = []

    for col in reference_df.columns:
        dtype = reference_df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            numerical_columns.append(col)
        else:
            categorical_columns.append(col)

    return DataDefinition(
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
    )


def build_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
):
    data_definition = infer_data_definition(reference_df)

    reference_dataset = Dataset.from_pandas(reference_df, data_definition=data_definition)
    current_dataset = Dataset.from_pandas(current_df, data_definition=data_definition)

    report = Report([DataDriftPreset()])
    result = report.run(reference_data=reference_dataset, current_data=current_dataset)
    return result


def save_drift_report(
    report: Report,
    output_html: Path,
    output_meta_json: Path | None = None,
    *,
    n_reference: int,
    n_current: int,
    n_features: int,
) -> None:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_html))

    if output_meta_json is not None:
        payload = {
            "n_reference": n_reference,
            "n_current": n_current,
            "n_features": n_features,
            "report_path": str(output_html),
        }
        output_meta_json.parent.mkdir(parents=True, exist_ok=True)
        output_meta_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
