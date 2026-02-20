from __future__ import annotations

import argparse

import mlflow

from credexp.config import settings
from credexp.modeling.dataset import load_features
from credexp.modeling.train import TrainConfig, log_run
from credexp.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--undersample", action="store_true")
    parser.add_argument("--cost-fn", type=float, default=10.0)
    parser.add_argument("--cost-fp", type=float, default=1.0)
    args = parser.parse_args()

    ds = load_features()
    X, y = ds.X_train, ds.y_train

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    cfg = TrainConfig(
        n_splits=args.cv,
        use_undersampling=args.undersample,
        cost_fn=args.cost_fn,
        cost_fp=args.cost_fp,
    )

    # Baselines
    log_run(
        X, y, model_name="dummy", activation="most_frequent", cfg=cfg, dataset_hash=ds.file_hash
    )
    log_run(X, y, model_name="dummy", activation="stratified", cfg=cfg, dataset_hash=ds.file_hash)

    # LR
    log_run(X, y, model_name="lr", activation=None, cfg=cfg, dataset_hash=ds.file_hash)

    # MLP with 2 activations (requirement)
    log_run(X, y, model_name="mlp", activation="relu", cfg=cfg, dataset_hash=ds.file_hash)
    log_run(X, y, model_name="mlp", activation="logistic", cfg=cfg, dataset_hash=ds.file_hash)

    # LGBM
    log_run(X, y, model_name="lgbm", activation=None, cfg=cfg, dataset_hash=ds.file_hash)

    log.info("Training runs completed.")


if __name__ == "__main__":
    main()
