from __future__ import annotations

from credexp.config import ARTIFACTS_DIR, DATA_DIR, PROJECT_ROOT, settings


def main() -> None:
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_DIR:     {DATA_DIR}")
    print(f"ARTIFACTS:    {ARTIFACTS_DIR}")
    print(f"ENV:          {settings.env}")
    print(f"MLFLOW URI:   {settings.mlflow_tracking_uri}")
    print(f"EXPERIMENT:   {settings.mlflow_experiment_name}")


if __name__ == "__main__":
    main()
