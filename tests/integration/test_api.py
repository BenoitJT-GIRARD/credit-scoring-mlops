"""Endpoints that answer without a model, and one test that needs a real one.

The contract of the API — response shape, batching, failure logging — is tested in
``test_api_contract.py`` against a fake bundle, where 200 can be demanded. What is left
here is what genuinely depends on the deployment: whether a real model loads and scores.

That test is marked ``integration`` and skipped unless a model is actually loaded. It is
skipped with a reason rather than weakened into ``status_code in (200, 503)``, which is
what it used to be — an assertion that holds whether the service works or not, and
therefore checks nothing.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from credexp.serving.api import _build_input_dataframe, app
from credexp.serving.schemas import PredictRequest

pytestmark = pytest.mark.integration

client = TestClient(app)


def test_health_does_not_depend_on_the_model() -> None:
    """Liveness answers even with nothing loaded, which is what an orchestrator asks."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_rejects_a_features_field_that_is_not_a_mapping() -> None:
    response = client.post("/predict", json={"sk_id_curr": 123456, "features": "not_a_dict"})
    assert response.status_code == 422


def test_predict_rejects_unknown_top_level_fields() -> None:
    """extra='forbid' on the schema: a typo in a field name must not be silently ignored."""
    response = client.post(
        "/predict",
        json={"sk_id_curr": 123456, "features": {"EXT_SOURCE_1": 0.5}, "unexpected_field": 1},
    )
    assert response.status_code == 422


def test_a_real_model_scores_a_real_request(served) -> None:
    """The one thing a fake bundle cannot check: that the shipped artefact loads and runs."""
    response = served.post(
        "/predict",
        json={"sk_id_curr": 123456, "features": {"EXT_SOURCE_1": 0.5, "EXT_SOURCE_2": 0.7}},
    )

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["proba_default"] <= 1.0
    assert body["decision"] in (0, 1)
    assert body["model_name"]


def test_model_info_describes_the_loaded_model(served) -> None:
    response = served.get("/model-info")

    assert response.status_code == 200
    assert response.json()["n_features"] > 0


def test_a_partial_payload_is_scored_on_the_full_column_order(served, shipped_bundle) -> None:
    """Two features in, 796 columns out, in the order the pipeline was fitted on.

    `models/feature_columns.json` is what `_build_input_dataframe` reads to place what the
    caller sent and leave the rest to the imputer. The second half of the test is the refusal:
    the same values in another order are a different applicant, and the pipeline says so.
    """
    features = {
        column: float(value) for value, column in enumerate(shipped_bundle.feature_columns[:12])
    }
    straight = served.post("/predict", json={"sk_id_curr": 1, "features": features})
    assert straight.status_code == 200

    frame = _build_input_dataframe(PredictRequest(sk_id_curr=1, features=features), shipped_bundle)
    assert list(frame.columns) == list(shipped_bundle.feature_columns)

    shuffled = frame[list(reversed(frame.columns))]
    with pytest.raises(ValueError, match="same order"):
        shipped_bundle.pipe.predict_proba(shuffled)
