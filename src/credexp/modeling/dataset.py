from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from credexp.config import DATA_DIR


@dataclass(frozen=True)
class Dataset:
    df: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series | None
    file_hash: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_features(features_path: Path | None = None) -> Dataset:
    if features_path is None:
        features_path = DATA_DIR / "processed" / "features.parquet"

    df = pd.read_parquet(features_path)
    file_hash = _sha256(features_path)

    train = df[df["TARGET"].notna()].copy()
    test = df[df["TARGET"].isna()].copy()

    y_train = train["TARGET"].astype(int)
    X_train = train.drop(columns=["TARGET"])

    # test Kaggle n'a pas TARGET; pour notre API holdout on aura un autre fichier
    y_test = None
    X_test = test.drop(columns=["TARGET"], errors="ignore")

    return Dataset(
        df=df,
        train=train,
        test=test,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        file_hash=file_hash,
    )
