"""The integration tier: two real components of this repository wired together.

Either two real components of the repository face to face, or one real component against a
real external dependency — a database in a container, an index on disk, an engine binary, a
real checkpoint, the real FastAPI or Streamlit application object. A double is admitted only
for a paid or remote service, and the test's docstring says which one and why.

Every file here carries its tier where a reader sees it, at the top of the module:

    pytestmark = pytest.mark.integration

The collection hook below refuses a file that does not. Without it an unmarked file is only
noticed when someone reads it; with it, the suite refuses to run the moment one arrives, and
``-m "not integration"`` keeps meaning what it says.

Fixtures shared by this tier — a session against a container, a temporary index, a loaded
model — belong in this file, so that starting the dependency is written once and skipped
once, with the command that starts it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from credexp.serving import api as api_module
from credexp.serving.api import app

TIER = "integration"

#: This directory. `pytest_collection_modifyitems` is handed EVERY collected item, not only
#: the ones below the conftest that defines it, so the hook filters by path: without this the
#: system tier reports the integration tier's files as unmarked. The marker is read with
#: `get_closest_marker`, never from `item.keywords` — the keywords carry the names of the
#: parent nodes, so the directory called `integration` makes every file in it look marked.
HERE = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    unmarked = sorted(
        {
            str(item.path.relative_to(HERE))
            for item in items
            if getattr(item, "path", None) is not None
            and HERE in item.path.parents
            and item.get_closest_marker(TIER) is None
        }
    )
    if unmarked:
        raise pytest.UsageError(
            f"{len(unmarked)} file(s) under tests/{TIER}/ without "
            f"`pytestmark = pytest.mark.{TIER}`: " + ", ".join(unmarked)
        )


# --- The doubles this tier serves the API with ------------------------------
#
# A fake bundle in place of the shipped model, so that 200 can be demanded and the shape of
# the answer asserted. With a real model these could only ask for « 200 or 503 », which is
# what they used to do. The shipped model has its own tests two files down, and those load it.

FEATURES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "AMT_CREDIT"]
THRESHOLD = 0.42


class FakePipeline:
    """Scores a row by its first feature, so a test can choose the outcome it wants.

    ``calls`` records the size of each ``predict_proba`` call, which is how the batch
    endpoint's whole point gets asserted: one call of five, not five calls of one.
    """

    def __init__(self) -> None:
        self.calls: list[int] = []

    def predict_proba(self, frame):
        self.calls.append(len(frame))
        first = frame[FEATURES[0]].astype(float).fillna(0.0).to_numpy()
        positive = np.clip(first, 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])


class FailingPipeline:
    """Raises, so the failure path can be exercised without breaking a real model."""

    def predict_proba(self, frame):
        raise RuntimeError("the pipeline exploded")


def make_bundle(pipe) -> api_module.ModelBundle:
    return api_module.ModelBundle(
        pipe=pipe,
        threshold=THRESHOLD,
        model_name="credit_scoring_model",
        model_version="test",
        feature_columns=FEATURES,
    )


def payload(value: float, sk_id: int | None = 1) -> dict:
    return {"sk_id_curr": sk_id, "features": {FEATURES[0]: value, FEATURES[1]: 0.1}}


@pytest.fixture()
def loaded(monkeypatch):
    """A bundle that is always present, and a database that is never touched.

    Returns ``(client, pipe, logged)`` — the last being every call the logging path would
    have made, so the tests can assert on it without a database.
    """
    pipe = FakePipeline()
    monkeypatch.setattr(api_module, "BUNDLE", make_bundle(pipe))
    logged: list[dict] = []
    monkeypatch.setattr(
        api_module, "_log_prediction_best_effort", lambda **kwargs: logged.append(kwargs)
    )
    return TestClient(app), pipe, logged


@pytest.fixture()
def failing(monkeypatch):
    """A bundle whose pipeline raises. Returns ``(client, logged)``."""
    monkeypatch.setattr(api_module, "BUNDLE", make_bundle(FailingPipeline()))
    logged: list[dict] = []
    monkeypatch.setattr(
        api_module, "_log_prediction_best_effort", lambda **kwargs: logged.append(kwargs)
    )
    return TestClient(app, raise_server_exceptions=False), logged


# --- The shipped artefacts, loaded the way the container loads them ---------


@pytest.fixture(scope="session")
def shipped_bundle():
    """The three tracked files under `models/`, loaded through the serving path.

    This is the fixture that replaced a skip. Two tests used to be marked
    `skipif(api_module.BUNDLE is None)`, which is a condition no checkout ever met: the
    module-level `BUNDLE` is filled by the application's startup event, and a test client
    that never starts the app leaves it `None`. So the two tests that exercise the shipped
    model never ran anywhere, on any machine, and the suite reported them as skipped rather
    than as absent.

    The artefacts are tracked, so the loading needs no service and no download.
    """
    from credexp.serving.model_loader import load_model_bundle
    from credexp.utils import PIPELINE_PATH

    if not PIPELINE_PATH.is_file():
        pytest.skip(f"run: uv run python scripts/train_final.py  ({PIPELINE_PATH} is missing)")

    with pytest.MonkeyPatch.context() as environment:
        environment.setenv("MODEL_LOAD_MODE", "joblib")
        environment.setenv("MODEL_VERSION", "tracked-artefact")
        return load_model_bundle()


@pytest.fixture()
def served(monkeypatch: pytest.MonkeyPatch, shipped_bundle):
    """A client answering from the shipped model, with the prediction log switched off.

    The log is the one part of the serving path that needs PostgreSQL. What is under test
    here is the model, so the write is replaced and `tests/system/` keeps the database.
    """
    monkeypatch.setattr(api_module, "BUNDLE", shipped_bundle)
    monkeypatch.setattr(api_module, "_log_prediction_best_effort", lambda **kwargs: None)
    return TestClient(app)
