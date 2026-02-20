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
