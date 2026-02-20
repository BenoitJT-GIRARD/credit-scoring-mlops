from __future__ import annotations

from dataclasses import asdict, dataclass

import mlflow
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from credexp.config import settings
from credexp.modeling.metrics import evaluate_binary
from credexp.modeling.pipelines import make_numeric_steps
from credexp.modeling.threshold import find_best_threshold
from credexp.utils.logging import get_logger

log = get_logger(__name__)

try:
    import lightgbm as lgb
except Exception:
    lgb = None


@dataclass(frozen=True)
class TrainConfig:
    n_splits: int = 5
    random_state: int = settings.random_state
    use_undersampling: bool = False
    imbalance: str = "balanced"  # "none" | "balanced"
    cost_fn: float = 10.0
    cost_fp: float = 1.0


def _compute_sample_weight(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    n = len(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    w_pos = n / (2 * max(n_pos, 1))
    w_neg = n / (2 * max(n_neg, 1))
    return np.where(y == 1, w_pos, w_neg).astype(float)


def _make_pipeline(model_name: str, activation: str | None, cfg: TrainConfig):
    # Scale only for MLP / LR (trees not needed)
    scale = model_name in {"lr", "mlp"}

    prep_steps = make_numeric_steps(scale=scale)

    steps = []
    steps.extend(prep_steps)

    if model_name == "lr":
        class_weight = "balanced" if cfg.imbalance == "balanced" else None
        model = LogisticRegression(
            max_iter=2000,
            class_weight=class_weight,
            n_jobs=-1,
        )
    elif model_name == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation=activation or "relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=50,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.1,
            random_state=cfg.random_state,
        )
    elif model_name == "lgbm":
        if lgb is None:
            raise RuntimeError("lightgbm not installed")
        class_weight = "balanced" if cfg.imbalance == "balanced" else None

        model = lgb.LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary",
            class_weight=class_weight,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_name={model_name}")

    if cfg.use_undersampling:
        steps.append(("under", RandomUnderSampler(random_state=cfg.random_state)))

    steps.append(("model", model))
    return ImbPipeline(steps)


def run_cv(X, y, model_name: str, activation: str | None, cfg: TrainConfig):
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)

    aucs, pr_aucs, thresholds, costs = [], [], [], []

    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        pipe = _make_pipeline(model_name, activation, cfg)
        pipe.fit(X_tr, y_tr)

        proba = pipe.predict_proba(X_va)[:, 1]

        best_thr, best_cost = find_best_threshold(
            y_true=y_va.to_numpy(),
            y_proba=proba,
            cost_fn=cfg.cost_fn,
            cost_fp=cfg.cost_fp,
        )
        rep = evaluate_binary(y_va.to_numpy(), proba, threshold=best_thr)

        log.info(
            f"fold={fold} model={model_name} auc={rep.roc_auc:.4f} pr_auc={rep.pr_auc:.4f} thr={best_thr:.3f} cost={best_cost:.1f}"
        )

        aucs.append(rep.roc_auc)
        pr_aucs.append(rep.pr_auc)
        thresholds.append(best_thr)
        costs.append(best_cost)

    return {
        "roc_auc_mean": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs)),
        "pr_auc_mean": float(np.mean(pr_aucs)),
        "pr_auc_std": float(np.std(pr_aucs)),
        "best_threshold_mean": float(np.mean(thresholds)),
        "business_cost_mean": float(np.mean(costs)),
    }


def log_run(X, y, model_name: str, activation: str | None, cfg: TrainConfig, dataset_hash: str):
    run_name = model_name if activation is None else f"{model_name}_{activation}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**asdict(cfg), "model": model_name, "activation": activation or ""})
        mlflow.log_param("features_parquet_md5", dataset_hash)
        metrics = run_cv(X, y, model_name=model_name, activation=activation, cfg=cfg)
        mlflow.log_metrics(metrics)
        log.info(f"MLflow logged: {run_name} metrics={metrics}")
