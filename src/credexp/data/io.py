"""Where the data lives, and the two formats it is read and written in."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from credexp.utils import INTERIM_DIR, PROCESSED_DIR, RAW_DIR
from credexp.utils.logging import get_logger

log = get_logger(__name__)


def raw_dir() -> Path:
    """The Kaggle CSVs, as downloaded. Nothing here is redistributable, so nothing is tracked."""
    return RAW_DIR


def interim_dir() -> Path:
    """The merged tables, between the raw CSVs and the feature frame."""
    return INTERIM_DIR


def processed_dir() -> Path:
    """The feature frame and the two holdouts the published numbers are computed on."""
    return PROCESSED_DIR


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    log.info(f"read_csv path={path}")
    return pd.read_csv(path, **kwargs)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    log.info(f"save_parquet path={path} shape={df.shape}")
    df.to_parquet(path, index=False)


def load_parquet(path: Path) -> pd.DataFrame:
    log.info(f"load_parquet path={path}")
    return pd.read_parquet(path)
