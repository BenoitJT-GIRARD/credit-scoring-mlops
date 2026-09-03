"""What links the artefact being served to the decisions that produced it.

Four things were true at once and none of them was written down anywhere the serving side
could read: the final hyperparameters were copied by hand out of a tuning run, the feature
contract was a bare list of column names, the imbalance strategy differed between candidate
models without being stated, and nothing said how leakage had been controlled.

They share one cause. The numbers were right and the training was reproducible; what was
missing was the link. A manifest is that link — one file, written when the model is
trained, read when it is loaded.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

__all__ = ["SCHEMA_VERSION", "ModelManifest", "best_trial", "read_manifest", "write_manifest"]

#: Bump this by hand when the *meaning* of a feature column changes, not when the list
#: grows. The loader refuses a manifest it does not know how to read, which turns a silent
#: semantic drift into a startup failure.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ModelManifest:
    """Everything needed to say what a served model is, and where it came from."""

    schema_version: int
    feature_columns: list[str]
    threshold: float
    #: How the threshold was chosen, and how much it moves across resamples. A threshold
    #: without its spread is a point estimate presented as a decision.
    threshold_selection: dict = field(default_factory=dict)
    hyperparameters: dict = field(default_factory=dict)
    #: The tuning run the hyperparameters were read from, so they are never retyped.
    tuning_run: str = ""
    imbalance_strategy: str = ""
    training_data_hash: str = ""
    trained_at: str = ""
    git_revision: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def best_trial(csv_path: Path | str) -> tuple[dict, str]:
    """The winning hyperparameters from a tracked Optuna run, and its trial number.

    The objective is a business cost, so the best trial is the smallest one. Only completed
    trials are eligible: a pruned or failed trial can carry a lower recorded value without
    ever having been evaluated properly.

    Raises rather than returning a default. A final training that guesses its
    hyperparameters does not reproduce, and a silent fallback is how the guess would go
    unnoticed.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"no tuning artefact at {path}. Run scripts/tune_optuna.py first; the final "
            "training reads its hyperparameters from there and will not invent them."
        )

    trials = pd.read_csv(path)
    complete = trials[trials["state"] == "COMPLETE"] if "state" in trials else trials
    if complete.empty:
        raise ValueError(f"{path} holds no completed trial")

    best = complete.loc[complete["value"].idxmin()]
    params = {
        column[len("params_") :]: best[column]
        for column in complete.columns
        if column.startswith("params_")
    }
    integers = {"n_estimators", "num_leaves", "max_depth", "min_child_samples"}
    cleaned = {
        name: int(value) if name in integers else float(value) for name, value in params.items()
    }
    return cleaned, f"trial-{int(best['number'])}"


def write_manifest(path: Path | str, manifest: ModelManifest) -> Path:
    """Write the manifest beside the model artefact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(manifest.to_json(), encoding="utf-8")
    return target


def read_manifest(path: Path | str) -> ModelManifest:
    """Read a manifest, refusing a schema this code does not understand.

    A newer schema means a column list whose meaning has changed since this loader was
    written. Serving it anyway would produce predictions from features that are not what
    the model expects, and nothing downstream would notice.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = int(payload.get("schema_version", 0))
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"model manifest schema {version}, this code reads {SCHEMA_VERSION}. "
            "Retrain, or use the code that matches the artefact."
        )
    return ModelManifest(**payload)


def utc_now() -> str:
    """Timestamp for the manifest, in UTC and to the second."""
    return datetime.now(UTC).isoformat(timespec="seconds")
