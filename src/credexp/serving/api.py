from __future__ import annotations

import os
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from prometheus_fastapi_instrumentator import Instrumentator

from credexp.config import settings
from credexp.db.crud import build_prediction_log
from credexp.db.init_db import init_db
from credexp.db.session import SessionLocal
from credexp.serving.failures import FailureKind
from credexp.serving.model_loader import ModelBundle, load_model_bundle
from credexp.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from credexp.utils.logging import get_logger

log = get_logger(__name__)

BUNDLE: ModelBundle | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup/shutdown lifecycle.

    The model is loaded once at startup and reused for all requests.

    Database initialization is best-effort: if PostgreSQL/Supabase is unavailable,
    the API can still serve predictions. Prediction logging will also be attempted
    later on each request.
    """
    global BUNDLE

    try:
        init_db()
        log.info("database_initialization_complete")
    except Exception as exc:
        log.warning(
            "database_initialization_failed",
            extra={"error": repr(exc)},
        )

    BUNDLE = load_model_bundle()
    log.info(
        "model_loaded",
        extra={
            "model_name": BUNDLE.model_name,
            "model_version": BUNDLE.model_version,
            "n_features": len(BUNDLE.feature_columns),
        },
    )

    log.info("api_startup_complete")
    yield
    log.info("api_shutdown_complete")


app = FastAPI(
    title="Credit Scoring API",
    version="0.1.0",
    description="API de scoring crédit pour le projet OC - Partie 2",
    root_path=os.getenv("API_ROOT_PATH", ""),
    lifespan=lifespan,
)

if settings.prometheus_enabled:
    Instrumentator().instrument(app).expose(app)


@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    """Attach a request id and log every HTTP request."""
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.request_id = request_id

    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000

    response.headers["x-request-id"] = request_id

    log.info(
        "http_request",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 3),
        },
    )

    return response


@app.get("/health")
def health() -> dict[str, str]:
    """Healthcheck endpoint used by Docker, monitoring and remote demos."""
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Return loaded model metadata."""
    if BUNDLE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    return ModelInfoResponse(
        model_name=BUNDLE.model_name,
        model_version=BUNDLE.model_version,
        threshold=BUNDLE.threshold,
        n_features=len(BUNDLE.feature_columns),
    )


def _build_input_dataframe(
    reqs: PredictRequest | Sequence[PredictRequest], bundle: ModelBundle
) -> pd.DataFrame:
    """Build a frame aligned with the model's feature order, one row per request.

    Missing features are left as None and handled by the trained pipeline's imputer, so
    that a client sending a partial payload gets the same treatment the model was fitted
    under.
    """
    items = [reqs] if isinstance(reqs, PredictRequest) else list(reqs)
    rows = [
        {column: item.features.get(column, None) for column in bundle.feature_columns}
        for item in items
    ]
    return pd.DataFrame(rows, columns=bundle.feature_columns)


def _score_frame(bundle: ModelBundle, frame: pd.DataFrame) -> np.ndarray:
    """Probability of default for every row, in one call.

    The only place the pipeline is invoked. It knows nothing about HTTP, which is what
    lets a batch of a hundred cost one call instead of a hundred — the fixed costs here
    are Python overhead, pandas conversions and scikit-learn's input validation, and none
    of them scales with the number of rows.
    """
    return np.asarray(bundle.pipe.predict_proba(frame))[:, 1].astype(float)


def _log_prediction_best_effort(
    *,
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
    """Persist prediction logs when a database is available.

    Logging must never prevent the API from returning a prediction. This is
    especially important for remote demos where Supabase may be unreachable or
    rate-limited.
    """
    if BUNDLE is None:
        return

    db = SessionLocal()
    try:
        db_log = build_prediction_log(
            request_id=request_id,
            sk_id_curr=req.sk_id_curr,
            model_name=BUNDLE.model_name,
            model_version=BUNDLE.model_version,
            threshold=float(BUNDLE.threshold),
            proba_default=proba,
            decision=decision,
            latency_ms=float(latency_ms),
            status_code=status_code,
            input_payload=req.model_dump(),
            output_payload=response_payload,
            error_message=error_message,
            failure_kind=failure_kind.value if failure_kind else None,
        )
        db.add(db_log)
        db.commit()
    except Exception as exc:
        db.rollback()
        log.warning(
            "prediction_log_failed",
            extra={
                "request_id": request_id,
                "error": repr(exc),
            },
        )
    finally:
        db.close()


def _log_failure(
    *,
    request_id: str,
    req: PredictRequest,
    kind: FailureKind,
    status_code: int,
    exc: Exception,
    latency_ms: float,
) -> None:
    """Record a request that produced no score, in the log and in the database."""
    log.exception(
        str(kind),
        extra={"request_id": request_id, "failure_kind": str(kind), "error": repr(exc)},
    )
    _log_prediction_best_effort(
        request_id=request_id,
        req=req,
        response_payload={},
        proba=None,
        decision=None,
        latency_ms=latency_ms,
        status_code=status_code,
        failure_kind=kind,
        error_message=repr(exc)[:1000],
    )


def _predict_one(req: PredictRequest, request_id: str | None = None) -> PredictResponse:
    """Run one prediction and log it best-effort."""
    if BUNDLE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded yet",
        )

    prediction_request_id = request_id or str(uuid.uuid4())
    start = time.perf_counter()

    frame = _build_input_dataframe(req, BUNDLE)

    try:
        proba = float(_score_frame(BUNDLE, frame)[0])
        decision = int(proba >= BUNDLE.threshold)
    except Exception as exc:
        # The failure leaves a row. A predictions table that holds only successes makes
        # every error rate computed from it wrong by construction, and leaves an incident
        # with nothing to read afterwards.
        _log_failure(
            request_id=prediction_request_id,
            req=req,
            kind=FailureKind.INFERENCE_ERROR,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            exc=exc,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        ) from exc

    latency_ms = (time.perf_counter() - start) * 1000

    response_payload = {
        "proba_default": proba,
        "decision": decision,
        "threshold": float(BUNDLE.threshold),
        "model_name": BUNDLE.model_name,
        "model_version": BUNDLE.model_version,
        "latency_ms": float(latency_ms),
    }

    _log_prediction_best_effort(
        request_id=prediction_request_id,
        req=req,
        response_payload=response_payload,
        proba=proba,
        decision=decision,
        latency_ms=latency_ms,
    )

    return PredictResponse(**response_payload)


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def predict(req: PredictRequest, request: Request) -> PredictResponse:
    """Predict default risk for one client."""
    return _predict_one(
        req=req,
        request_id=getattr(request.state, "request_id", str(uuid.uuid4())),
    )


@app.post(
    "/predict_batch",
    response_model=BatchPredictResponse,
    responses={
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def predict_batch(req: BatchPredictRequest, request: Request) -> BatchPredictResponse:
    """Predict default risk for a batch of clients, in one call to the model.

    The repository presents batching as its retained optimisation, so the endpoint that
    carries the name has to actually batch. One frame, one ``predict_proba``: the costs
    that do not scale with the number of rows — Python overhead, pandas conversions,
    scikit-learn's validation — are paid once instead of once per client.

    Logging stays per row. The database holds one line per prediction, not one per
    request, and each keeps the derived request id it already had.
    """
    if BUNDLE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded yet",
        )

    base_request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start = time.perf_counter()
    frame = _build_input_dataframe(req.items, BUNDLE)

    try:
        probabilities = _score_frame(BUNDLE, frame)
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        for index, item in enumerate(req.items):
            _log_failure(
                request_id=f"{base_request_id}:{index}",
                req=item,
                kind=FailureKind.INFERENCE_ERROR,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                exc=exc,
                latency_ms=latency_ms,
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        ) from exc

    # Divided across the batch, so a row's latency stays comparable to a single
    # prediction's rather than reporting the whole batch against every client.
    latency_ms = (time.perf_counter() - start) * 1000 / max(len(req.items), 1)

    results = []
    for index, (item, proba) in enumerate(zip(req.items, probabilities, strict=True)):
        proba = float(proba)
        decision = int(proba >= BUNDLE.threshold)
        payload = {
            "proba_default": proba,
            "decision": decision,
            "threshold": float(BUNDLE.threshold),
            "model_name": BUNDLE.model_name,
            "model_version": BUNDLE.model_version,
            "latency_ms": float(latency_ms),
        }
        _log_prediction_best_effort(
            request_id=f"{base_request_id}:{index}",
            req=item,
            response_payload=payload,
            proba=proba,
            decision=decision,
            latency_ms=latency_ms,
        )
        results.append(PredictResponse(**payload))

    return BatchPredictResponse(results=results)
