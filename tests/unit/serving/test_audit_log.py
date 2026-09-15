"""The guard around the prediction log, which is the whole of what this module promises.

Three claims are made about writing a decision down, in the README, in `docs/DB.md` and in
the architecture page: it never blocks an answer, it never raises, and a request that scored
nothing still leaves a row. The system tier proves the first against a closed port. These
prove the other two without a database at all, by handing the writer a session that fails.
"""

from __future__ import annotations

import pytest

from credexp.serving import audit_log
from credexp.serving.failures import FailureKind
from credexp.serving.model_loader import ModelBundle
from credexp.serving.schemas import PredictRequest

BUNDLE = ModelBundle(
    pipe=object(),
    threshold=0.49,
    model_name="credit_scoring_model",
    model_version="test",
    feature_columns=["EXT_SOURCE_1"],
)
REQUEST = PredictRequest(sk_id_curr=1, features={"EXT_SOURCE_1": 0.5})


class _Session:
    """A session that records what it was asked to do, and can be told to fail."""

    def __init__(self, *, explode: bool = False) -> None:
        self.added: list[object] = []
        self.committed = False
        self.rolled_back = False
        self.closed = False
        self._explode = explode

    def add(self, row: object) -> None:
        self.added.append(row)

    def commit(self) -> None:
        if self._explode:
            raise RuntimeError("the database went away")
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def session(monkeypatch: pytest.MonkeyPatch):
    """Replace the session factory, so nothing here opens a connection."""

    def install(*, explode: bool = False) -> _Session:
        made = _Session(explode=explode)
        monkeypatch.setattr(audit_log, "SessionLocal", lambda: made)
        return made

    return install


def test_one_scored_request_becomes_one_row(session) -> None:
    made = session()

    audit_log.log_prediction_best_effort(
        bundle=BUNDLE,
        request_id="r1",
        req=REQUEST,
        response_payload={"proba_default": 0.42, "decision": 0},
        proba=0.42,
        decision=0,
        latency_ms=12.0,
    )

    assert len(made.added) == 1
    assert made.committed
    assert made.closed


def test_a_database_that_fails_costs_the_caller_nothing(session) -> None:
    """The claim three documents make. Here it is, in the only place it can be shown."""
    made = session(explode=True)

    audit_log.log_prediction_best_effort(
        bundle=BUNDLE,
        request_id="r2",
        req=REQUEST,
        response_payload={},
        proba=0.42,
        decision=0,
        latency_ms=12.0,
    )

    assert made.rolled_back
    assert made.closed
    assert not made.committed


def test_a_service_with_no_model_writes_nothing(session) -> None:
    """Before the startup event completes there is no version to attribute a row to."""
    made = session()

    audit_log.log_prediction_best_effort(
        bundle=None,
        request_id="r3",
        req=REQUEST,
        response_payload={},
        proba=None,
        decision=None,
        latency_ms=1.0,
    )

    assert made.added == []
    assert not made.closed, "the session is never even opened"


def test_a_failed_request_leaves_a_row_with_no_score(session) -> None:
    """A table of successes only makes every error rate computed from it wrong."""
    made = session()

    audit_log.log_prediction_best_effort(
        bundle=BUNDLE,
        request_id="r4",
        req=REQUEST,
        response_payload={},
        proba=None,
        decision=None,
        latency_ms=8.0,
        status_code=500,
        failure_kind=FailureKind.INFERENCE_ERROR,
        error_message="the pipeline exploded",
    )

    row = made.added[0]
    assert row.proba_default is None
    assert row.decision is None
    assert row.status_code == 500
    assert row.failure_kind == FailureKind.INFERENCE_ERROR.value


def test_the_row_carries_the_model_that_produced_it(session) -> None:
    made = session()

    audit_log.log_prediction_best_effort(
        bundle=BUNDLE,
        request_id="r5",
        req=REQUEST,
        response_payload={"proba_default": 0.42},
        proba=0.42,
        decision=0,
        latency_ms=5.0,
    )

    row = made.added[0]
    assert (row.model_name, row.model_version, row.threshold) == (
        "credit_scoring_model",
        "test",
        0.49,
    )
