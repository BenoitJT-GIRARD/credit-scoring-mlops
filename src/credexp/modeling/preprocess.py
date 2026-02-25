from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class InfToNan(BaseEstimator, TransformerMixin):
    """Replace +/-inf with NaN (needed after ratio feature engineering)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Keep pandas if we got pandas
        if isinstance(X, pd.DataFrame):
            return X.replace([np.inf, -np.inf], np.nan)
        # Fallback for numpy arrays
        X = np.asarray(X, dtype=float)
        X[~np.isfinite(X)] = np.nan
        return X


class Clipper(BaseEstimator, TransformerMixin):
    """Clip values to keep MLP stable."""

    def __init__(self, low=-10.0, high=10.0):
        self.low = low
        self.high = high

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.clip(self.low, self.high)
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.low, self.high)
