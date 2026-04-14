from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sk_id_curr: int | None = Field(default=None, description="Optional client identifier")
    features: dict[str, float | int | None] = Field(
        ...,
        description="Feature dictionary aligned with engineered model features",
        min_length=1,
    )


class PredictResponse(BaseModel):
    proba_default: float
    decision: int
    threshold: float
    model_name: str
    model_version: str
    latency_ms: float


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest]


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    threshold: float
    n_features: int


class ErrorResponse(BaseModel):
    detail: str
