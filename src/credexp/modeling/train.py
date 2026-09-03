"""Cross-validated training, tracked in MLflow.

The threshold is chosen inside the fold, on a split the scoring fold never sees. It used to
be chosen on the fold it was then scored on, and that shortcut is worth 0.0029 per
applicant -- small, real, and free to remove.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mlflow
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from credexp.config import settings
from credexp.modeling.metrics import evaluate_binary
from credexp.modeling.pipelines import make_numeric_steps
from credexp.modeling.threshold import business_cost, find_best_threshold
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

    if model_name == "dummy":
        # here `activation` carries the dummy strategy: "most_frequent" or "stratified"
        model = DummyClassifier(strategy=activation or "most_frequent")
    elif model_name == "lr":
        class_weight = "balanced" if cfg.imbalance == "balanced" else None
        model = LogisticRegression(
            max_iter=5000,
            class_weight=class_weight,
        )
    elif model_name == "mlp":
        lr_init = 1e-3
        if activation in {"tanh", "logistic"}:
            lr_init = 1e-4
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation=activation or "relu",
            alpha=1e-4,
            learning_rate_init=lr_init,
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

    use_under = cfg.use_undersampling and model_name != "dummy"
    if use_under:
        steps.append(("under", RandomUnderSampler(random_state=cfg.random_state)))

    steps.append(("model", model))
    return ImbPipeline(steps)


def run_cv(X, y, model_name: str, activation: str | None, cfg: TrainConfig):
    """Cross-validate a model family and return its selection metrics.

    `business_cost_per_row` is measured **out of sample**: the threshold chosen on fold k is
    applied to fold k+1, so no fold both picks and scores its own threshold.

    It used to do exactly that, and the bias was worth measuring before removing: on the
    holdout, picking and scoring a threshold on the same half costs 0.0029 per applicant
    less than picking it on one half and scoring on the other — about 0.6 % of a cost near
    0.49. Small, but pointing the wrong way, and free to remove. See
    `reports/decision/decision_analysis.json` and `credexp.modeling.threshold.split_threshold_cost`.

    `roc_auc` and `pr_auc` are threshold-free and were never affected.
    """
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)

    aucs, pr_aucs, thresholds = [], [], []
    fold_labels, fold_probas = [], []

    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        pipe = _make_pipeline(model_name, activation, cfg)
        fit_params = {}
        if model_name == "mlp" and cfg.imbalance == "balanced" and not cfg.use_undersampling:
            fit_params["model__sample_weight"] = _compute_sample_weight(y_tr.to_numpy())

        pipe.fit(X_tr, y_tr, **fit_params)

        proba = pipe.predict_proba(X_va)[:, 1]

        if not np.isfinite(proba).all():
            raise ValueError(
                f"Non-finite probabilities detected (nan/inf) for model={model_name} activation={activation} fold={fold}"
            )

        best_thr, _ = find_best_threshold(
            y_true=y_va.to_numpy(),
            y_proba=proba,
            cost_fn=cfg.cost_fn,
            cost_fp=cfg.cost_fp,
        )
        rep = evaluate_binary(y_va.to_numpy(), proba, threshold=best_thr)

        log.info(
            f"fold={fold} model={model_name} auc={rep.roc_auc:.4f} pr_auc={rep.pr_auc:.4f} thr={best_thr:.3f}"
        )

        aucs.append(rep.roc_auc)
        pr_aucs.append(rep.pr_auc)
        thresholds.append(best_thr)
        fold_labels.append(y_va.to_numpy())
        fold_probas.append(proba)

    # Each fold is scored with the threshold its neighbour chose, so no fold both selects
    # and evaluates its own. Cycling keeps every fold scored exactly once.
    n_folds = len(thresholds)
    costs = [
        business_cost(
            fold_labels[i],
            fold_probas[i],
            thresholds[(i - 1) % n_folds],
            cfg.cost_fn,
            cfg.cost_fp,
        )
        / len(fold_labels[i])
        for i in range(n_folds)
    ]

    return {
        "roc_auc_mean": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs)),
        "pr_auc_mean": float(np.mean(pr_aucs)),
        "pr_auc_std": float(np.std(pr_aucs)),
        "best_threshold_mean": float(np.mean(thresholds)),
        "business_cost_per_row": float(np.mean(costs)),
    }


def log_run(X, y, model_name: str, activation: str | None, cfg: TrainConfig, dataset_hash: str):
    run_name = model_name if activation is None else f"{model_name}_{activation}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**asdict(cfg), "model": model_name, "activation": activation or ""})
        mlflow.log_param("features_parquet_sha256", dataset_hash)
        metrics = run_cv(X, y, model_name=model_name, activation=activation, cfg=cfg)
        mlflow.log_metrics(metrics)
        log.info(f"MLflow logged: {run_name} metrics={metrics}")
