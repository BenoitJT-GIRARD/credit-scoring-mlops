from __future__ import annotations

from credexp.db.models import PredictionLog


def build_prediction_log(
    *,
    request_id: str,
    sk_id_curr: int | None,
    model_name: str,
    model_version: str,
    threshold: float,
    proba_default: float,
    decision: int,
    latency_ms: float,
    status_code: int,
    input_payload: dict,
    output_payload: dict,
    error_message: str | None = None,
) -> PredictionLog:
    return PredictionLog(
        request_id=request_id,
        sk_id_curr=sk_id_curr,
        model_name=model_name,
        model_version=model_version,
        threshold=threshold,
        proba_default=proba_default,
        decision=decision,
        latency_ms=latency_ms,
        status_code=status_code,
        input_payload=input_payload,
        output_payload=output_payload,
        error_message=error_message,
    )
