from __future__ import annotations

import optuna
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from credexp.config import settings

try:
    import lightgbm as lgb
except Exception:
    lgb = None


def tune_lgbm_auc(X, y, n_trials: int = 30):
    if lgb is None:
        raise RuntimeError("lightgbm not installed")

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=settings.random_state, stratify=y
    )

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
            "random_state": settings.random_state,
            "n_jobs": -1,
        }

        model = lgb.LGBMClassifier(**params)

        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
        pipe.fit(
            X_tr,
            y_tr,
            model__eval_set=[(X_va, y_va)],
            model__eval_metric="auc",
            model__callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        proba = pipe.predict_proba(X_va)[:, 1]
        return float(roc_auc_score(y_va, proba))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value
