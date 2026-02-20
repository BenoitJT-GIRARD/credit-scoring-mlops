from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from credexp.modeling.preprocess import InfToNan


def make_numeric_steps(scale: bool):
    steps: list[tuple[str, object]] = [
        ("inf_to_nan", InfToNan()),
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale:
        steps.append(("scaler", StandardScaler(with_mean=False)))
    return steps
