"""Shared doubles for the serving tests.

A fake bundle rather than a real model, so the API's contract can be asserted at all. With
a real model the tests can only ask for "200 or 503", which is what they used to do — an
assertion that holds whether the service works or not.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from credexp.serving import api as api_module
from credexp.serving.api import app

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
