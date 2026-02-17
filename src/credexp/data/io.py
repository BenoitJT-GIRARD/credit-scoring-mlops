from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from credexp.config import DATA_DIR
from credexp.utils.logging import get_logger

log = get_logger(__name__)


def raw_dir() -> Path:
    """Raw input data (Kaggle CSVs) - NOT tracked in git."""
    return DATA_DIR / "raw"


def interim_dir() -> Path:
    """Intermediate datasets (after merges / partial processing) - NOT tracked in git."""
    return DATA_DIR / "interim"


def processed_dir() -> Path:
    """Processed datasets (features.parquet, holdout samples) - NOT tracked in git."""
    return DATA_DIR / "processed"


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
