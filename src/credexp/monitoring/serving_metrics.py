"""Four metrics about the model, beside the ones about the transport.

`prometheus-fastapi-instrumentator` already reports request counts, latencies and status
codes. Those describe the HTTP layer and say nothing about what the model is doing — a
service can be perfectly healthy by every one of them while scoring everyone as a default.

Four, and not thirty. Each answers a question someone would actually ask during an
incident, and a dashboard no one opens is maintenance debt wearing the word observability.

The failure counter shares its label vocabulary with the structured log and the database
column, so one incident can be followed from a Prometheus spike to the rows that caused it.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

__all__ = [
    "DEFAULT_PROBABILITY",
    "PREDICTIONS_TOTAL",
    "PREDICTION_FAILURES_TOTAL",
    "THRESHOLD_CROSSINGS_TOTAL",
    "observe_failure",
    "observe_prediction",
]

#: How many scores were produced, by model version and endpoint. Answers "is anything
#: reaching the model at all", and makes a version rollout visible.
PREDICTIONS_TOTAL = Counter(
    "credexp_predictions_total",
    "Predictions produced, by model version and endpoint.",
    labelnames=("model_version", "endpoint"),
)

#: The distribution of the default probability. A bump that grows at one end of it is
#: visible months before the first outcome of those applications is known.
DEFAULT_PROBABILITY = Histogram(
    "credexp_default_probability",
    "Distribution of the predicted probability of default.",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

#: How the threshold splits the traffic. The acceptance rate is the number the business
#: side watches, and it moves for reasons the probability distribution alone does not show.
THRESHOLD_CROSSINGS_TOTAL = Counter(
    "credexp_threshold_crossings_total",
    "Decisions taken, by side of the business threshold.",
    labelnames=("decision",),
)

#: Failures by kind, using one vocabulary of failure kinds, shared with the log and the table.
PREDICTION_FAILURES_TOTAL = Counter(
    "credexp_prediction_failures_total",
    "Requests that produced no score, by failure kind.",
    labelnames=("failure_kind",),
)


def observe_prediction(*, model_version: str, endpoint: str, proba: float, decision: int) -> None:
    """Record one successful prediction across the three success metrics."""
    PREDICTIONS_TOTAL.labels(model_version=model_version, endpoint=endpoint).inc()
    DEFAULT_PROBABILITY.observe(float(proba))
    THRESHOLD_CROSSINGS_TOTAL.labels(decision=str(int(decision))).inc()


def observe_failure(*, failure_kind: str) -> None:
    """Record one request that produced no score."""
    PREDICTION_FAILURES_TOTAL.labels(failure_kind=str(failure_kind)).inc()
