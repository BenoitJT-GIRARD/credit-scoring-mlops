"""The routes, the middleware that stamps a request id, and the best-effort decision log.

Best-effort is the deliberate part: a database that is down does not stop the service from
answering. It also means an empty log with a healthy `/predict` is a silent failure, which
is why the failure counter exists beside it.

`/predict_batch` actually batches -- one frame, one `predict_proba` -- because the
repository presents batching as its retained optimisation, and an endpoint that loops would
make that claim false while looking identical from outside.
"""

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
from credexp.db.init_db import init_db
from credexp.monitoring.serving_metrics import observe_failure, observe_prediction
from credexp.serving.audit_log import log_prediction_best_effort
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

#: The database write lives in `credexp.serving.audit_log`. It keeps its old name here
#: because a route calling `_log_prediction_best_effort` reads as one thing whatever module
#: holds it, and because the tests replace it by that name.
_log_prediction_best_effort = log_prediction_best_effort


def _log_failure(
    *,
    bundle: ModelBundle | None,
    request_id: str,
    req: PredictRequest,
    kind: FailureKind,
    status_code: int,
    exc: Exception,
    latency_ms: float,
) -> None:
    """Record a request that produced no score, in three places at once.

    The service's log gets the traceback, the Prometheus counter gets the kind, and the
    database gets a row with a null score. The three are written together because a failure
    counted and not stored, or stored and not counted, is a failure only half visible.
    """
    log.exception(
        str(kind),
        extra={"request_id": request_id, "failure_kind": str(kind), "error": repr(exc)},
    )
    observe_failure(failure_kind=str(kind))
    _log_prediction_best_effort(
        bundle=bundle,
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
    title="Credit scoring API",
    version="0.1.0",
    description=(
        "Estimates the probability that a loan applicant defaults, and turns it into an "
        "accept-or-refuse decision at a threshold chosen by minimising an expected cost "
        "in which one default is worth ten wrongful refusals.\n\n"
        "Every scored request is written to PostgreSQL with the score, the decision, the "
        "threshold and the model version that produced it.\n\n"
        "`proba_default` is a ranking score and not a calibrated probability: the "
        "model is fit with balanced class weights and overstates risk by roughly a factor "
        "of four. Read `decision`."
    ),
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
    lets a batch of a hundred cost one call and not a hundred. The fixed costs here
    are Python overhead, pandas conversions and scikit-learn's input validation, and none
    of them scales with the number of rows.
    """
    return np.asarray(bundle.pipe.predict_proba(frame))[:, 1].astype(float)


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
            bundle=BUNDLE,
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

    observe_prediction(
        model_version=BUNDLE.model_version,
        endpoint="predict",
        proba=proba,
        decision=decision,
    )
    _log_prediction_best_effort(
        bundle=BUNDLE,
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
    carries the name has to actually batch. One frame, one ``predict_proba``, and whatever
    does not grow with the number of rows is paid once for all of them.

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
                bundle=BUNDLE,
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
    # prediction's, so the whole batch is not reported against every client.
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
        observe_prediction(
            model_version=BUNDLE.model_version,
            endpoint="predict_batch",
            proba=proba,
            decision=decision,
        )
        _log_prediction_best_effort(
            bundle=BUNDLE,
            request_id=f"{base_request_id}:{index}",
            req=item,
            response_payload=payload,
            proba=proba,
            decision=decision,
            latency_ms=latency_ms,
        )
        results.append(PredictResponse(**payload))

    return BatchPredictResponse(results=results)
