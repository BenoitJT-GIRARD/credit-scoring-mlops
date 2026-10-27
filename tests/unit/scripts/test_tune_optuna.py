"""The tuning study writes its trials to the file the final training reads.

`scripts/tune_optuna.py` used to write `optuna_trials.csv` into the working directory, and
`scripts/train_final.py` reads `reports/tuning/optuna_trials.csv`. A new study then left
the final training on the previous trials, without an error.
"""

from __future__ import annotations

import contextlib
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from credexp.modeling.manifest import best_trial


class _Study:
    """What `tune_lgbm_business_cost` returns, reduced to what the script reads."""

    best_value = 0.5
    best_params = {"n_estimators": 400, "learning_rate": 0.05}

    def trials_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "number": [0, 1],
                "value": [0.6, 0.5],
                "params_n_estimators": [800, 400],
                "params_learning_rate": [0.02, 0.05],
                "state": ["COMPLETE", "COMPLETE"],
            }
        )


def _mlflow_double() -> SimpleNamespace:
    """MLflow without a tracking store: every call is accepted and nothing is written."""

    def ignore(*_args, **_kwargs) -> None:
        return None

    return SimpleNamespace(
        set_tracking_uri=ignore,
        set_experiment=ignore,
        start_run=lambda **_kwargs: contextlib.nullcontext(),
        log_param=ignore,
        log_metric=ignore,
        log_artifact=ignore,
    )


def test_the_study_and_the_final_training_name_the_same_file(load_script) -> None:
    tune = load_script("tune_optuna")
    train = load_script("train_final")

    assert tune.TRIALS_PATH == train.TUNING_ARTEFACT


def test_a_study_writes_its_trials_there_and_not_in_the_working_directory(
    load_script, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tune = load_script("tune_optuna")
    target = tmp_path / "reports" / "tuning" / "optuna_trials.csv"
    workdir = tmp_path / "workdir"
    workdir.mkdir()

    monkeypatch.chdir(workdir)
    monkeypatch.setattr(sys, "argv", ["tune_optuna.py", "--trials", "2"])
    # `raising=False`: the script before the fix had no such name, and wrote to the cwd.
    monkeypatch.setattr(tune, "TRIALS_PATH", target, raising=False)
    monkeypatch.setattr(tune, "mlflow", _mlflow_double())
    monkeypatch.setattr(
        tune,
        "load_features",
        lambda: SimpleNamespace(X_train=pd.DataFrame(), y_train=pd.Series(), file_hash="x"),
    )
    monkeypatch.setattr(tune, "tune_lgbm_business_cost", lambda *_args, **_kwargs: _Study())

    tune.main()

    assert not (workdir / "optuna_trials.csv").exists()
    params, trial = best_trial(target)
    assert trial == "trial-1"
    assert params == {"n_estimators": 400, "learning_rate": 0.05}
