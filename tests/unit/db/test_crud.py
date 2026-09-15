"""The row that records one decision, and the two columns that are allowed to be empty.

`proba_default` and `decision` are nullable because a failed request has no score. Zero is
a score; the absence of one is not, and a 0.0 written for a failure would drag down every
average computed from this table — including the one the log page shows and the one the
drift monitor reads.

Nothing here reaches PostgreSQL. `build_prediction_log` maps a decision onto a row, and
that mapping is what a test can check.
"""

from __future__ import annotations

from credexp.db.crud import build_prediction_log
from credexp.db.models import PredictionLog

SCORED = {
    "request_id": "0d5f",
    "sk_id_curr": 123456,
    "model_name": "credit_scoring_model",
    "model_version": "joblib",
    "threshold": 0.49,
    "proba_default": 0.42,
    "decision": 0,
    "latency_ms": 12.5,
    "status_code": 200,
    "input_payload": {"sk_id_curr": 123456, "features": {"EXT_SOURCE_1": 0.5}},
    "output_payload": {"proba_default": 0.42, "decision": 0},
}


def test_a_scored_request_becomes_a_row_carrying_its_own_answer() -> None:
    row = build_prediction_log(**SCORED)

    assert isinstance(row, PredictionLog)
    assert row.proba_default == 0.42
    assert row.decision == 0
    assert row.threshold == 0.49
    assert row.status_code == 200
    assert row.failure_kind is None


def test_a_failed_request_has_no_score_and_says_so_with_null() -> None:
    """Not 0.0: every rate computed from this table would be wrong by the failure count."""
    row = build_prediction_log(
        **{
            **SCORED,
            "proba_default": None,
            "decision": None,
            "status_code": 500,
            "error_message": "the pipeline exploded",
            "failure_kind": "model_unavailable",
            "output_payload": {},
        }
    )

    assert row.proba_default is None
    assert row.decision is None
    assert row.error_message == "the pipeline exploded"
    assert row.failure_kind == "model_unavailable"


def test_the_payload_is_stored_as_it_was_sent() -> None:
    """Whole, and not projected onto today's columns. `credexp.db.crud` says why."""
    row = build_prediction_log(**SCORED)

    assert row.input_payload == SCORED["input_payload"]
    assert row.output_payload == SCORED["output_payload"]


def test_the_threshold_travels_with_the_probability() -> None:
    """The threshold moves between model versions, and the log is read months later."""
    row = build_prediction_log(**{**SCORED, "threshold": 0.31})

    assert row.threshold == 0.31


def test_the_table_is_the_one_the_log_page_reads() -> None:
    """The page and the model name the same table, and neither invents it."""
    from credexp.app.prediction_log import COLUMNS

    assert PredictionLog.__tablename__ == "predictions"
    assert set(COLUMNS) <= set(PredictionLog.__table__.columns.keys())
