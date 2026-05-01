from __future__ import annotations

import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from prometheus_fastapi_instrumentator import Instrumentator

from credexp.config import settings
from credexp.db.crud import build_prediction_log
from credexp.db.init_db import init_db
from credexp.db.session import SessionLocal
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


def _build_input_dataframe(req: PredictRequest, bundle: ModelBundle) -> pd.DataFrame:
    """Build a one-row dataframe aligned with the model feature order.

    Missing features are filled with None and then handled by the trained pipeline
    imputer/preprocessor.
    """
    row = {column: req.features.get(column, None) for column in bundle.feature_columns}
    return pd.DataFrame([row], columns=bundle.feature_columns)


def _log_prediction_best_effort(
    *,
    request_id: str,
    req: PredictRequest,
    response_payload: dict,
    proba: float,
    decision: int,
    latency_ms: float,
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
            status_code=200,
            input_payload=req.model_dump(),
            output_payload=response_payload,
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


def _predict_one(req: PredictRequest, request_id: str | None = None) -> PredictResponse:
    """Run one prediction and log it best-effort."""
    if BUNDLE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded yet",
        )

    prediction_request_id = request_id or str(uuid.uuid4())
    start = time.perf_counter()

    X = _build_input_dataframe(req=req, bundle=BUNDLE)

    try:
        proba = float(BUNDLE.pipe.predict_proba(X)[:, 1][0])
        decision = int(proba >= BUNDLE.threshold)
    except Exception as exc:
        log.exception(
            "inference_failed",
            extra={
                "request_id": prediction_request_id,
                "error": repr(exc),
            },
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
    """Predict default risk for a batch of clients."""
    base_request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    results = [
        _predict_one(
            req=item,
            request_id=f"{base_request_id}:{index}",
        )
        for index, item in enumerate(req.items)
    ]

    return BatchPredictResponse(results=results)
