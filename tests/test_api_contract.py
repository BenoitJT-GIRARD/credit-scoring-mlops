"""What the API promises, tested against a model that is always there.

The previous tests asserted ``status_code in (200, 503)``. That passes whether the model
is loaded or not, so it never once checked what the endpoint returns — a green test that
measured nothing. These fix the model in place with a fake bundle and demand 200, which is
the only way the shape of the response can be asserted at all.

The doubles live in ``conftest.py``. The one test that needs a real provisioned service is
in ``test_api.py``, marked ``integration``.
"""

from __future__ import annotations

from conftest import FEATURES, THRESHOLD
from conftest import payload as _payload
from fastapi.testclient import TestClient

from credexp.serving import api as api_module
from credexp.serving.api import app
from credexp.serving.failures import FailureKind


def test_predict_returns_the_documented_shape(loaded) -> None:
    client, _, _ = loaded
    response = client.post("/predict", json=_payload(0.9))

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "proba_default",
        "decision",
        "threshold",
        "model_name",
        "model_version",
        "latency_ms",
    }
    assert isinstance(body["proba_default"], float)
    assert 0.0 <= body["proba_default"] <= 1.0
    assert body["threshold"] == THRESHOLD


def test_the_decision_is_the_threshold_applied_to_the_probability(loaded) -> None:
    """Not a tautology: the two are returned separately and a client will check them."""
    client, _, _ = loaded
    for value in (0.1, 0.41, 0.43, 0.99):
        body = client.post("/predict", json=_payload(value)).json()
        assert body["decision"] == int(body["proba_default"] >= body["threshold"])


def test_the_request_id_is_echoed_back(loaded) -> None:
    client, _, _ = loaded
    response = client.post("/predict", json=_payload(0.5), headers={"x-request-id": "abc-123"})
    assert response.headers["x-request-id"] == "abc-123"


def test_a_batch_is_scored_in_one_call(loaded) -> None:
    """The point of the endpoint. One predict_proba, not one per row."""
    client, pipe, _ = loaded
    items = [_payload(v, sk_id=i) for i, v in enumerate([0.1, 0.5, 0.9, 0.2, 0.8])]

    response = client.post("/predict_batch", json={"items": items})

    assert response.status_code == 200
    assert len(response.json()["results"]) == 5
    assert pipe.calls == [5], "the pipeline was called once, on all five rows"


def test_batch_and_single_scoring_agree(loaded) -> None:
    """predict_proba on a frame and on a row need not take the same numeric path."""
    client, _, _ = loaded
    values = [0.15, 0.55, 0.95]

    singles = [client.post("/predict", json=_payload(v)).json()["proba_default"] for v in values]
    batch = client.post("/predict_batch", json={"items": [_payload(v) for v in values]}).json()[
        "results"
    ]

    for one, many in zip(singles, batch, strict=True):
        assert abs(one - many["proba_default"]) < 1e-9


def test_every_row_of_a_batch_is_logged_with_its_own_derived_id(loaded) -> None:
    client, _, logged = loaded
    logged.clear()

    client.post(
        "/predict_batch",
        json={"items": [_payload(0.2), _payload(0.8)]},
        headers={"x-request-id": "base"},
    )

    assert [entry["request_id"] for entry in logged] == ["base:0", "base:1"]
    assert all(entry["proba"] is not None for entry in logged)


def test_an_inference_failure_is_logged_with_a_closed_failure_kind(failing) -> None:
    """A predictions table holding only successes makes every error rate wrong."""
    client, logged = failing

    response = client.post("/predict", json=_payload(0.5), headers={"x-request-id": "boom"})

    assert response.status_code == 500
    assert len(logged) == 1
    entry = logged[0]
    assert entry["request_id"] == "boom"
    assert entry["failure_kind"] is FailureKind.INFERENCE_ERROR
    assert entry["proba"] is None and entry["decision"] is None
    assert "exploded" in entry["error_message"]


def test_a_failed_batch_logs_one_row_per_client(failing) -> None:
    client, logged = failing

    response = client.post(
        "/predict_batch",
        json={"items": [_payload(0.2), _payload(0.8), _payload(0.5)]},
        headers={"x-request-id": "base"},
    )

    assert response.status_code == 500
    assert [entry["request_id"] for entry in logged] == ["base:0", "base:1", "base:2"]


def test_model_info_answers_when_a_model_is_loaded(loaded) -> None:
    client, _, _ = loaded
    body = client.get("/model-info").json()

    assert body["model_name"] == "credit_scoring_model"
    assert body["threshold"] == THRESHOLD
    assert body["n_features"] == len(FEATURES)


def test_without_a_model_the_endpoints_say_so(monkeypatch) -> None:
    """503 is the right answer here, and it has to be asserted rather than tolerated."""
    monkeypatch.setattr(api_module, "BUNDLE", None)
    client = TestClient(app)

    assert client.post("/predict", json=_payload(0.5)).status_code == 503
    assert client.post("/predict_batch", json={"items": [_payload(0.5)]}).status_code == 503
    assert client.get("/model-info").status_code == 503
    assert client.get("/health").status_code == 200, "liveness does not depend on the model"


def test_a_malformed_payload_is_refused_with_the_error_shape(loaded) -> None:
    client, _, _ = loaded

    assert client.post("/predict", json={"features": "not a dict"}).status_code == 422
    assert client.post("/predict", json={"features": {}}).status_code == 422
    assert client.post("/predict", json={"features": {"a": 1}, "unexpected": 1}).status_code == 422
