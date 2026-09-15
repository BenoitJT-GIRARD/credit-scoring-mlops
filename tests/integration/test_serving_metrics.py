"""The four metrics that describe the model rather than the transport."""

from __future__ import annotations

import pytest
from conftest import FEATURES, THRESHOLD
from prometheus_client import REGISTRY

from credexp.monitoring.serving_metrics import (
    PREDICTION_FAILURES_TOTAL,
    PREDICTIONS_TOTAL,
    THRESHOLD_CROSSINGS_TOTAL,
    observe_failure,
    observe_prediction,
)
from credexp.serving.failures import FailureKind

pytestmark = pytest.mark.integration


def _value(metric, **labels) -> float:
    sample = metric.labels(**labels) if labels else metric
    return float(sample._value.get())


def _histogram_count() -> float:
    return float(REGISTRY.get_sample_value("credexp_default_probability_count") or 0.0)


def test_a_prediction_counts_against_its_model_version() -> None:
    before = _value(PREDICTIONS_TOTAL, model_version="v9", endpoint="predict")

    observe_prediction(model_version="v9", endpoint="predict", proba=0.8, decision=1)

    assert _value(PREDICTIONS_TOTAL, model_version="v9", endpoint="predict") == before + 1


def test_the_probability_lands_in_the_histogram() -> None:
    before = _histogram_count()

    observe_prediction(model_version="v9", endpoint="predict", proba=0.31, decision=0)

    assert _histogram_count() == before + 1


def test_the_two_sides_of_the_threshold_are_counted_apart() -> None:
    """The acceptance rate is what the business side watches, and it needs both sides."""
    before_yes = _value(THRESHOLD_CROSSINGS_TOTAL, decision="1")
    before_no = _value(THRESHOLD_CROSSINGS_TOTAL, decision="0")

    observe_prediction(model_version="v9", endpoint="predict", proba=0.9, decision=1)
    observe_prediction(model_version="v9", endpoint="predict", proba=0.1, decision=0)

    assert _value(THRESHOLD_CROSSINGS_TOTAL, decision="1") == before_yes + 1
    assert _value(THRESHOLD_CROSSINGS_TOTAL, decision="0") == before_no + 1


def test_a_failure_counts_under_its_own_kind() -> None:
    kind = str(FailureKind.INFERENCE_ERROR)
    before = _value(PREDICTION_FAILURES_TOTAL, failure_kind=kind)

    observe_failure(failure_kind=kind)

    assert _value(PREDICTION_FAILURES_TOTAL, failure_kind=kind) == before + 1


def test_the_failure_labels_are_the_same_vocabulary_as_the_database() -> None:
    """One incident has to be followable from a Prometheus spike to the rows behind it."""
    for kind in FailureKind:
        observe_failure(failure_kind=str(kind))
        assert _value(PREDICTION_FAILURES_TOTAL, failure_kind=str(kind)) >= 1


def test_scoring_through_the_api_moves_the_counters(loaded) -> None:
    client, _, _ = loaded
    before = _value(PREDICTIONS_TOTAL, model_version="test", endpoint="predict_batch")

    client.post(
        "/predict_batch",
        json={"items": [{"features": {FEATURES[0]: v}} for v in (0.2, 0.8, 0.9)]},
    )

    after = _value(PREDICTIONS_TOTAL, model_version="test", endpoint="predict_batch")
    assert after == before + 3, "one observation per client, not one per request"


def test_a_failing_batch_counts_one_failure_per_client(failing) -> None:
    client, _ = failing
    kind = str(FailureKind.INFERENCE_ERROR)
    before = _value(PREDICTION_FAILURES_TOTAL, failure_kind=kind)

    client.post("/predict_batch", json={"items": [{"features": {FEATURES[0]: 0.5}}] * 4})

    assert _value(PREDICTION_FAILURES_TOTAL, failure_kind=kind) == before + 4


@pytest.mark.parametrize(
    "name",
    [
        "credexp_predictions_total",
        "credexp_default_probability",
        "credexp_threshold_crossings_total",
        "credexp_prediction_failures_total",
    ],
)
def test_every_metric_is_exposed_on_the_endpoint(name: str, loaded) -> None:
    client, _, _ = loaded
    client.post("/predict", json={"features": {FEATURES[0]: 0.5}})

    body = client.get("/metrics").text
    assert name in body


def test_the_threshold_used_by_the_counter_is_the_bundle_one(loaded) -> None:
    """A decision counted against a different threshold than the one served is a lie."""
    client, _, _ = loaded

    body = client.post("/predict", json={"features": {FEATURES[0]: THRESHOLD + 0.01}}).json()

    assert body["decision"] == 1
    assert body["threshold"] == THRESHOLD
