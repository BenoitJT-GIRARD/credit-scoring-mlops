from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import mlflow
from credexp.config import ARTIFACTS_DIR, DATA_DIR, settings
from credexp.modeling.pipelines import make_numeric_steps
from credexp.modeling.threshold import business_cost, find_best_threshold
from credexp.utils.logging import get_logger

log = get_logger(__name__)

try:
    import lightgbm as lgb
except Exception as e:
    raise RuntimeError("LightGBM must be installed to run train_final.py") from e


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features", type=str, default=str(DATA_DIR / "processed" / "features.parquet")
    )
    parser.add_argument("--holdout-size", type=float, default=0.10)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--cost-fn", type=float, default=10.0)
    parser.add_argument("--cost-fp", type=float, default=1.0)
    parser.add_argument(
        "--undersample",
        action="store_true",
        help="(optional) not used in this final script for simplicity",
    )
    parser.add_argument("--run-name", type=str, default="lgbm_final")
    args = parser.parse_args()

    features_path = Path(args.features)
    df = pd.read_parquet(features_path)

    df_train = df[df["TARGET"].notna()].copy()
    y = df_train["TARGET"].astype(int)
    X = df_train.drop(columns=["TARGET"])

    # 1) Holdout split (never seen by the model)
    X_dev, X_holdout, y_dev, y_holdout = train_test_split(
        X,
        y,
        test_size=args.holdout_size,
        random_state=settings.random_state,
        stratify=y,
    )

    holdout_path = DATA_DIR / "processed" / "api_holdout.parquet"
    holdout = X_holdout.copy()
    holdout["TARGET"] = y_holdout.values
    holdout.to_parquet(holdout_path, index=False)
    log.info(f"Saved API holdout to {holdout_path} shape={holdout.shape}")

    # 2) Dev split to pick threshold (train/val)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_dev,
        y_dev,
        test_size=args.val_size,
        random_state=settings.random_state,
        stratify=y_dev,
    )

    # Baseline LGBM params (you can tune later via Optuna)
    class_weight = "balanced"
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary",
        class_weight=class_weight,
        random_state=settings.random_state,
        n_jobs=-1,
    )

    pipe = Pipeline(
        [
            *make_numeric_steps(scale=False),
            ("model", model),
        ]
    )

    # MLflow setup
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_registry_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("model_family", "lightgbm")
        mlflow.log_param("holdout_size", args.holdout_size)
        mlflow.log_param("val_size", args.val_size)
        mlflow.log_param("cost_fn", args.cost_fn)
        mlflow.log_param("cost_fp", args.cost_fp)
        mlflow.log_param("class_weight", class_weight)
        mlflow.log_param("features_file", str(features_path))

        # 3) Fit on train split, choose threshold on val split
        pipe.fit(X_tr, y_tr)
        proba_va = pipe.predict_proba(X_va)[:, 1]

        best_thr, best_cost = find_best_threshold(
            y_true=y_va.to_numpy(),
            y_proba=proba_va,
            cost_fn=args.cost_fn,
            cost_fp=args.cost_fp,
        )

        # Val metrics
        val_auc = roc_auc_score(y_va, proba_va)
        val_prauc = average_precision_score(y_va, proba_va)

        mlflow.log_metric("val_roc_auc", float(val_auc))
        mlflow.log_metric("val_pr_auc", float(val_prauc))
        mlflow.log_metric("val_best_threshold", float(best_thr))
        mlflow.log_metric("val_business_cost", float(best_cost))

        log.info(
            f"Val: AUC={val_auc:.4f} PR-AUC={val_prauc:.4f} best_thr={best_thr:.3f} cost={best_cost:.1f}"
        )

        # 4) Refit final model on ALL dev data (train+val)
        pipe.fit(X_dev, y_dev)

        # 5) Evaluate on holdout (never seen)
        proba_hold = pipe.predict_proba(X_holdout)[:, 1]
        hold_auc = roc_auc_score(y_holdout, proba_hold)
        hold_prauc = average_precision_score(y_holdout, proba_hold)
        hold_cost = business_cost(
            y_holdout.to_numpy(), proba_hold, best_thr, args.cost_fn, args.cost_fp
        )

        y_pred_hold = (proba_hold >= best_thr).astype(int)
        cm = confusion_matrix(y_holdout, y_pred_hold).tolist()

        mlflow.log_metric("holdout_roc_auc", float(hold_auc))
        mlflow.log_metric("holdout_pr_auc", float(hold_prauc))
        mlflow.log_metric("holdout_business_cost", float(hold_cost))

        # Log confusion matrix elements (handy for dashboards)
        tn, fp = cm[0]
        fn, tp = cm[1]
        mlflow.log_metric("holdout_tn", tn)
        mlflow.log_metric("holdout_fp", fp)
        mlflow.log_metric("holdout_fn", fn)
        mlflow.log_metric("holdout_tp", tp)

        log.info(
            f"Holdout: AUC={hold_auc:.4f} PR-AUC={hold_prauc:.4f} cost={hold_cost:.1f} cm={cm}"
        )

        # 6) Save local artifacts (joblib + threshold json)
        out_dir = ARTIFACTS_DIR / "models"
        out_dir.mkdir(parents=True, exist_ok=True)

        model_path = out_dir / "pipeline.joblib"
        joblib.dump(pipe, model_path)

        threshold_path = out_dir / "threshold.json"
        threshold_payload = {
            "threshold": float(best_thr),
            "cost_fn": float(args.cost_fn),
            "cost_fp": float(args.cost_fp),
            "selection": "val_optimized_then_refit_on_dev",
        }
        threshold_path.write_text(json.dumps(threshold_payload, indent=2), encoding="utf-8")

        # 7) Log artifacts to MLflow
        mlflow.log_artifact(str(model_path), artifact_path="export")
        mlflow.log_artifact(str(threshold_path), artifact_path="export")
        mlflow.log_artifact(str(holdout_path), artifact_path="export")

        # Also log as MLflow model (appears under run -> Artifacts/model)
        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            registered_model_name="credit_scoring_model",
        )

        log.info("Final model exported and logged to MLflow.")


if __name__ == "__main__":
    main()
