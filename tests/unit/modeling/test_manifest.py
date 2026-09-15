"""The manifest that ties a served artefact to the decisions that produced it."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from credexp.modeling.manifest import (
    SCHEMA_VERSION,
    ModelManifest,
    best_trial,
    read_manifest,
    write_manifest,
)


def _trials(tmp_path, rows):
    frame = pd.DataFrame(rows)
    path = tmp_path / "optuna_trials.csv"
    frame.to_csv(path, index=False)
    return path


def test_the_best_trial_is_the_cheapest_completed_one(tmp_path) -> None:
    """The objective is a business cost, so lower wins."""
    path = _trials(
        tmp_path,
        [
            {
                "number": 0,
                "value": 900.0,
                "state": "COMPLETE",
                "params_learning_rate": 0.1,
                "params_num_leaves": 31,
            },
            {
                "number": 1,
                "value": 500.0,
                "state": "COMPLETE",
                "params_learning_rate": 0.02,
                "params_num_leaves": 59,
            },
            {
                "number": 2,
                "value": 700.0,
                "state": "COMPLETE",
                "params_learning_rate": 0.3,
                "params_num_leaves": 12,
            },
        ],
    )

    params, run = best_trial(path)

    assert params == {"learning_rate": 0.02, "num_leaves": 59}
    assert run == "trial-1"


def test_a_pruned_trial_never_wins_however_low_its_value(tmp_path) -> None:
    """A pruned trial can record a low cost without having been evaluated properly."""
    path = _trials(
        tmp_path,
        [
            {"number": 0, "value": 1.0, "state": "PRUNED", "params_num_leaves": 5},
            {"number": 1, "value": 500.0, "state": "COMPLETE", "params_num_leaves": 59},
        ],
    )

    params, run = best_trial(path)

    assert run == "trial-1"
    assert params["num_leaves"] == 59


def test_integer_hyperparameters_come_back_as_integers(tmp_path) -> None:
    """A CSV makes everything a float; LightGBM will not take 59.0 leaves."""
    path = _trials(
        tmp_path,
        [
            {
                "number": 0,
                "value": 1.0,
                "state": "COMPLETE",
                "params_num_leaves": 59.0,
                "params_max_depth": 6.0,
                "params_learning_rate": 0.02,
            }
        ],
    )

    params, _ = best_trial(path)

    assert isinstance(params["num_leaves"], int)
    assert isinstance(params["max_depth"], int)
    assert isinstance(params["learning_rate"], float)


def test_a_missing_tuning_artefact_raises_rather_than_defaulting(tmp_path) -> None:
    """Hyper-parameters read from the study, or a run that stops: `best_trial` raises."""
    with pytest.raises(FileNotFoundError, match="tune_optuna"):
        best_trial(tmp_path / "absent.csv")


def test_a_file_with_no_completed_trial_raises(tmp_path) -> None:
    path = _trials(tmp_path, [{"number": 0, "value": 1.0, "state": "FAIL", "params_a": 1}])

    with pytest.raises(ValueError, match="no completed trial"):
        best_trial(path)


def _manifest(**overrides) -> ModelManifest:
    base = {
        "schema_version": SCHEMA_VERSION,
        "feature_columns": ["EXT_SOURCE_1", "AMT_CREDIT"],
        "threshold": 0.42,
        "threshold_selection": {"split": "val", "criterion": "business_cost"},
        "hyperparameters": {"num_leaves": 59},
        "tuning_run": "trial-1",
        "imbalance_strategy": "class_weight=balanced",
        "training_data_hash": "abc123",
        "trained_at": "2026-09-04T01:00:00+00:00",
        "git_revision": "deadbee",
    }
    return ModelManifest(**{**base, **overrides})


def test_the_manifest_round_trips_every_declared_field(tmp_path) -> None:
    path = write_manifest(tmp_path / "model_manifest.json", _manifest())

    assert read_manifest(path) == _manifest()


def test_an_unknown_schema_version_is_refused(tmp_path) -> None:
    """A newer schema means the columns mean something else. Serving anyway is silent harm."""
    path = tmp_path / "model_manifest.json"
    payload = json.loads(_manifest().to_json())
    payload["schema_version"] = SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        read_manifest(path)


def test_a_manifest_without_a_version_is_refused_too(tmp_path) -> None:
    path = tmp_path / "model_manifest.json"
    payload = json.loads(_manifest().to_json())
    del payload["schema_version"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        read_manifest(path)


def test_the_written_file_is_readable_by_a_human(tmp_path) -> None:
    """It is evidence, and evidence gets read in a diff."""
    path = write_manifest(tmp_path / "model_manifest.json", _manifest())
    text = path.read_text(encoding="utf-8")

    assert text.endswith("\n")
    assert '"threshold": 0.42' in text
    assert text.index('"feature_columns"') < text.index('"threshold"'), "keys are sorted"
