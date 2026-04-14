from __future__ import annotations

import time
import uuid

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

app = FastAPI(
    title="Credit Scoring API",
    version="0.1.0",
    description="API de scoring crédit pour le projet OC - Partie 2",
)

if settings.prometheus_enabled:
    Instrumentator().instrument(app).expose(app)

BUNDLE: ModelBundle | None = None


@app.on_event("startup")
def startup_event() -> None:
    global BUNDLE
    init_db()
    BUNDLE = load_model_bundle()
    log.info("API startup complete")


@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    rid = request.headers.get("x-request-id", str(uuid.uuid4()))
    start = time.perf_counter()

    response = await call_next(request)

    latency_ms = (time.perf_counter() - start) * 1000
    response.headers["x-request-id"] = rid

    log.info(
        "http_request",
        extra={
            "request_id": rid,
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 3),
        },
    )
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
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


def _predict_one(req: PredictRequest) -> PredictResponse:
    if BUNDLE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded yet",
        )

    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    row = {c: req.features.get(c, None) for c in BUNDLE.feature_columns}
    X = pd.DataFrame([row], columns=BUNDLE.feature_columns)

    try:
        proba = float(BUNDLE.pipe.predict_proba(X)[:, 1][0])
        decision = int(proba >= BUNDLE.threshold)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        ) from exc

    latency_ms = (time.perf_counter() - t0) * 1000

    response_payload = {
        "proba_default": proba,
        "decision": decision,
        "threshold": float(BUNDLE.threshold),
        "model_name": BUNDLE.model_name,
        "model_version": BUNDLE.model_version,
        "latency_ms": float(latency_ms),
    }

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
    finally:
        db.close()

    return PredictResponse(**response_payload)


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={503: {"model": ErrorResponse}},
)
def predict(req: PredictRequest) -> PredictResponse:
    return _predict_one(req)


@app.post(
    "/predict_batch",
    response_model=BatchPredictResponse,
    responses={503: {"model": ErrorResponse}},
)
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    return BatchPredictResponse(results=[_predict_one(x) for x in req.items])
