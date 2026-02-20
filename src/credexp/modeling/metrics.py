from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvalReport:
    roc_auc: float
    pr_auc: float
    threshold: float
    cm: list[list[int]]  # [[tn, fp],[fn,tp]]


def evaluate_binary(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> EvalReport:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return EvalReport(
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        pr_auc=float(average_precision_score(y_true, y_proba)),
        threshold=float(threshold),
        cm=cm,
    )
