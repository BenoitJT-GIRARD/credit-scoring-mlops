from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from mlflow.tracking import MlflowClient

import mlflow
from credexp.modeling.explainability import (
    ExplainConfig,
    run_explainability,
    split_X_y_from_features,
)


def resolve_project_root() -> Path:
    cwd = Path().resolve()
    root = cwd
    while root != root.parent and not (root / "src").exists():
        root = root.parent
    if not (root / "src").exists():
        raise RuntimeError("Project root not found (missing 'src').")
    return root


def load_latest_registry_model(model_name: str):
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if len(versions) == 0:
        raise RuntimeError(f"No model versions found in registry for: {model_name}")

    latest = sorted(versions, key=lambda v: int(v.version))[-1]
    model_uri = f"models:/{model_name}/{latest.version}"
    pipeline = mlflow.sklearn.load_model(model_uri)
    return pipeline, latest.version, latest.run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="credit_scoring_model")
    parser.add_argument("--features", type=str, default="data/processed/features.parquet")
    parser.add_argument("--out-dir", type=str, default="reports/explainability")
    parser.add_argument("--n-background", type=int, default=5000)
    parser.add_argument("--n-sample", type=int, default=1500)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--log-mlflow", action="store_true", help="Log figures as MLflow artifacts")
    args = parser.parse_args()

    root = resolve_project_root()
    features_path = (root / args.features).resolve()
    out_dir = (root / args.out_dir).resolve()

    # IMPORTANT: use the same DB as your project (sqlite)
    db_path = (root / "mlflow" / "mlflow.db").resolve()
    tracking_uri = f"sqlite:///{db_path.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    # Load data
    df = pd.read_parquet(features_path)
    X, y = split_X_y_from_features(df)

    # Load model from registry (latest version)
    pipeline, version, run_id = load_latest_registry_model(args.model_name)

    cfg = ExplainConfig(
        n_background=args.n_background,
        n_sample=args.n_sample,
        random_state=args.random_state,
    )

    meta = run_explainability(
        pipeline=pipeline,
        X_raw=X,
        config=cfg,
        out_dir=out_dir,
        topn_importance=30,
        max_display_shap=30,
    )

    print("Explainability metadata:")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print("Model:", args.model_name, "| version:", version, "| run_id:", run_id)
    print("Figures in:", out_dir / "figures")

    if args.log_mlflow:
        # attach explainability artifacts to the SAME run that produced the model if possible
        # (we can also create a new run; simplest is new run with linkage)
        with mlflow.start_run(run_name=f"explain_{args.model_name}_v{version}"):
            mlflow.log_param("model_name", args.model_name)
            mlflow.log_param("model_version", int(version))
            mlflow.log_param("model_run_id", run_id)
            mlflow.log_params(
                {
                    "n_background": cfg.n_background,
                    "n_sample": cfg.n_sample,
                    "random_state": cfg.random_state,
                }
            )
            mlflow.log_artifacts(str(out_dir), artifact_path="explainability")


if __name__ == "__main__":
    main()
