"""Optuna, optimising the business cost rather than a ranking metric.

Tuning for AUC and then choosing a threshold optimises two different things in sequence.
The objective here is the cost the threshold will actually be chosen against.
"""

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

    # # --- SANITY BASELINE (debug) ---
    # baseline_params = {
    #     "n_estimators": 2000,
    #     "learning_rate": 0.03,
    #     "num_leaves": 64,
    #     "subsample": 0.8,
    #     "colsample_bytree": 0.8,
    #     "reg_lambda": 1.0,
    #     "objective": "binary",
    #     "class_weight": "balanced",
    #     "random_state": settings.random_state,
    #     "n_jobs": -1,
    #     "verbosity": -1,
    # }

    # baseline_costs = []
    # for tr, va in skf.split(X, y):
    #     X_tr, X_va = X.iloc[tr], X.iloc[va]
    #     y_tr, y_va = y.iloc[tr], y.iloc[va]

    #     model = lgb.LGBMClassifier(**baseline_params)
    #     pipe = Pipeline([*make_numeric_steps(scale=False), ("model", model)])
    #     pipe.fit(X_tr, y_tr)

    #     proba = pipe.predict_proba(X_va)[:, 1]
    #     thr, _ = find_best_threshold(y_va.to_numpy(), proba, cost_fn=cost_fn, cost_fp=cost_fp)
    #     c = business_cost(y_va.to_numpy(), proba, thr, cost_fn=cost_fn, cost_fp=cost_fp)
    #     baseline_costs.append(c)

    # print(f"[SANITY] baseline LGBM mean cost = {float(np.mean(baseline_costs)):.1f}")
    # # --- END SANITY ---

    def objective(trial: optuna.Trial) -> float:
        params = {
            # Core training dynamics
            "n_estimators": trial.suggest_int("n_estimators", 400, 2500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            # Tree structure
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            # Sampling (safe ranges)
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            # Regularization (reasonable)
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            # Fixed stuff
            "objective": "binary",
            "class_weight": "balanced",
            "random_state": settings.random_state,
            "n_jobs": -1,
            # "verbosity": -1,
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

            # Guard 1: model outputs (almost) constant probabilities -> learned nothing
            if float(np.std(proba)) < 1e-6:
                costs.append(1e9)
                continue

            thr, _ = find_best_threshold(
                y_true=y_va.to_numpy(),
                y_proba=proba,
                cost_fn=cost_fn,
                cost_fp=cost_fp,
            )

            c = business_cost(
                y_true=y_va.to_numpy(),
                y_proba=proba,
                threshold=thr,
                cost_fn=cost_fn,
                cost_fp=cost_fp,
            )
            costs.append(c)

        if len(costs) == 0:
            return 1e9
        return float(np.mean(costs))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, catch=(ValueError,))
    return study
