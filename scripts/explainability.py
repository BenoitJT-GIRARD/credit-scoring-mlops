"""Feature importance and SHAP figures for the model that is actually served.

By default this reads the frozen artefact and the scoring holdout, which is what a reader
has: the MLflow registry is a training-time convenience and is not shipped. Pass
``--from-registry`` to explain the latest registered version instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from credexp.config import settings
from credexp.modeling.explainability import (
    ExplainConfig,
    run_explainability,
    split_X_y_from_features,
)

ROOT = Path(__file__).resolve().parents[1]


def load_from_artifact() -> tuple[object, str]:
    path = settings.artifacts_dir / "models" / "pipeline.joblib"
    if not path.exists():
        raise SystemExit(f"{path} is missing; run scripts/train_final.py first.")
    return joblib.load(path), "local-joblib"


def load_from_registry(model_name: str) -> tuple[object, str]:
    import mlflow
    from mlflow.tracking import MlflowClient

    db_path = (ROOT / "mlflow" / "mlflow.db").resolve()
    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    versions = MlflowClient().search_model_versions(f"name='{model_name}'")
    if not versions:
        raise SystemExit(f"no registered versions for {model_name} in {tracking_uri}")
    latest = sorted(versions, key=lambda v: int(v.version))[-1]
    return mlflow.sklearn.load_model(f"models:/{model_name}/{latest.version}"), str(latest.version)


def load_frame(explicit: Path | None) -> pd.DataFrame:
    """The set the model is explained on: the training matrix when it is available, the
    scoring holdout otherwise. Both describe the same population; the holdout is simply
    what survives without the raw Kaggle tables."""
    candidates = (
        [explicit]
        if explicit
        else [
            settings.data_dir / "processed" / "features.parquet",
            settings.data_dir / "processed" / "api_holdout.parquet",
        ]
    )
    for candidate in candidates:
        if candidate and candidate.exists():
            frame = pd.read_parquet(candidate)
            print(f"explaining on {candidate.name}: {len(frame)} rows")
            return frame
    raise SystemExit(
        "no feature frame found. Rebuild one from the raw Kaggle tables with "
        "scripts/build_features.py."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-registry", action="store_true")
    parser.add_argument("--model-name", default="credit_scoring_model")
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "explainability")
    parser.add_argument("--n-background", type=int, default=5000)
    parser.add_argument("--n-sample", type=int, default=1500)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frame = load_frame(args.features)
    if "TARGET" in frame.columns:
        X, _ = split_X_y_from_features(frame)
    else:
        X = frame

    pipeline, version = (
        load_from_registry(args.model_name) if args.from_registry else load_from_artifact()
    )

    meta = run_explainability(
        pipeline=pipeline,
        X_raw=X,
        config=ExplainConfig(
            n_background=args.n_background,
            n_sample=args.n_sample,
            random_state=args.random_state,
        ),
        out_dir=args.out_dir,
        topn_importance=30,
        max_display_shap=30,
    )

    print(f"model version: {version}")
    for key, value in meta.items():
        print(f"  {key}: {value}")
    print("figures in:", args.out_dir / "figures")


if __name__ == "__main__":
    main()
