from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present (safe if missing)
load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


@dataclass(frozen=True)
class Settings:
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", str(PROJECT_ROOT / "mlruns"))
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "credexp")
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))


settings = Settings()
