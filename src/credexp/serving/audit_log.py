"""Writing one decision down, and never letting that stop a prediction.

Recording what the service decided is worth a great deal months later and nothing at all in
the second a request is being answered. So every failure path here ends in a warning, and
none of them raises.

It sits beside the routes rather than inside them because the API module is a list of
endpoints, and this is the list of reasons a row may not reach PostgreSQL.
"""

from __future__ import annotations

from credexp.db.crud import build_prediction_log
from credexp.db.session import SessionLocal
from credexp.serving.failures import FailureKind
from credexp.serving.model_loader import ModelBundle
from credexp.serving.schemas import PredictRequest
from credexp.utils.logging import get_logger

log = get_logger(__name__)


def log_prediction_best_effort(
    *,
    bundle: ModelBundle | None,
    request_id: str,
    req: PredictRequest,
    response_payload: dict,
    proba: float | None,
    decision: int | None,
    latency_ms: float,
    status_code: int = 200,
    failure_kind: FailureKind | None = None,
    error_message: str | None = None,
) -> None:
    """Insert one row, and swallow whatever goes wrong doing it.

    The guard is the point. An unreachable database, a table that does not exist yet, a
    connection pool that has gone stale: each of them costs a warning in the service's own
    log, and the applicant still receives their score.
    """
    if bundle is None:
        return

    session = SessionLocal()
    try:
        session.add(
            build_prediction_log(
                request_id=request_id,
                sk_id_curr=req.sk_id_curr,
                model_name=bundle.model_name,
                model_version=bundle.model_version,
                threshold=float(bundle.threshold),
                proba_default=proba,
                decision=decision,
                latency_ms=float(latency_ms),
                status_code=status_code,
                input_payload=req.model_dump(),
                output_payload=response_payload,
                error_message=error_message,
                failure_kind=failure_kind.value if failure_kind else None,
            )
        )
        session.commit()
    except Exception as exc:  # noqa: BLE001 - the caller's answer outranks this row
        session.rollback()
        log.warning("prediction_log_failed", extra={"request_id": request_id, "error": repr(exc)})
    finally:
        session.close()
