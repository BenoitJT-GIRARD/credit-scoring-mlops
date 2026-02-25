from __future__ import annotations

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from credexp.config import settings
from credexp.modeling.pipelines import make_numeric_steps
from credexp.modeling.threshold import business_cost, find_best_threshold

try:
    import lightgbm as lgb
except Exception:
    lgb = None


def tune_lgbm_business_cost(
    X, y, n_trials: int = 30, cv: int = 3, cost_fn: float = 10.0, cost_fp: float = 1.0
):
    if lgb is None:
        raise RuntimeError("lightgbm not installed")

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=settings.random_state)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": 3000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "objective": "binary",
            "class_weight": "balanced",
            "random_state": settings.random_state,
            "n_jobs": -1,
        }

        costs = []
        for tr, va in skf.split(X, y):
            X_tr, X_va = X.iloc[tr], X.iloc[va]
            y_tr, y_va = y.iloc[tr], y.iloc[va]

            model = lgb.LGBMClassifier(**params)
            pipe = Pipeline(
                [
                    *make_numeric_steps(scale=False),
                    ("model", model),
                ]
            )
            pipe.fit(X_tr, y_tr)

            proba = pipe.predict_proba(X_va)[:, 1]
            thr, _ = find_best_threshold(y_va.to_numpy(), proba, cost_fn=cost_fn, cost_fp=cost_fp)
            c = business_cost(y_va.to_numpy(), proba, thr, cost_fn=cost_fn, cost_fp=cost_fp)
            costs.append(c)

        return float(np.mean(costs))  # minimize business cost

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, catch=(ValueError,))
    return study
