from __future__ import annotations

import argparse

import mlflow
from credexp.config import settings
from credexp.modeling.dataset import load_features
from credexp.modeling.tuning import tune_lgbm_business_cost


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--cost-fn", type=float, default=10.0)
    p.add_argument("--cost-fp", type=float, default=1.0)
    p.add_argument("--run-name", type=str, default="optuna_lgbm_business_cost")
    args = p.parse_args()

    ds = load_features()
    X, y = ds.X_train, ds.y_train

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("n_trials", args.trials)
        mlflow.log_param("cv", args.cv)
        mlflow.log_param("cost_fn", args.cost_fn)
        mlflow.log_param("cost_fp", args.cost_fp)
        mlflow.log_param("features_parquet_sha256", ds.file_hash)

        study = tune_lgbm_business_cost(
            X, y, n_trials=args.trials, cv=args.cv, cost_fn=args.cost_fn, cost_fp=args.cost_fp
        )

        mlflow.log_metric("best_business_cost", float(study.best_value))
        for k, v in study.best_params.items():
            mlflow.log_param(f"best_{k}", v)

        # save study trials
        df_trials = study.trials_dataframe()
        out = "optuna_trials.csv"
        df_trials.to_csv(out, index=False)
        mlflow.log_artifact(out)

        print("Best value (business cost):", study.best_value)
        print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
