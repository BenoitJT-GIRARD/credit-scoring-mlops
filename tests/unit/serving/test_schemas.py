"""The contract the OpenAPI page is generated from, checked where it is declared.

Two of these are decisions rather than types. `extra="forbid"` turns a typo in a field name
into a 422 instead of a silently ignored field, which on a scoring API means an applicant
scored on a payload nobody sent. And `min_length=1` refuses an empty feature map: the
pipeline would impute all 796 columns and return a confident score for no applicant at all.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from credexp.serving.schemas import (
    BatchPredictRequest,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)


def test_a_request_needs_nothing_but_its_features() -> None:
    """The client identifier is optional: a caller scoring a hypothetical has no id."""
    request = PredictRequest(features={"EXT_SOURCE_1": 0.5})

    assert request.sk_id_curr is None
    assert request.features == {"EXT_SOURCE_1": 0.5}


def test_a_misspelled_field_is_refused_rather_than_ignored() -> None:
    """`sk_id` for `sk_id_curr` would otherwise score an applicant with no identifier."""
    with pytest.raises(ValidationError, match="sk_id"):
        PredictRequest(features={"EXT_SOURCE_1": 0.5}, sk_id=1)


def test_an_empty_feature_map_is_refused() -> None:
    """Every column imputed is a score about the population, not about an applicant."""
    with pytest.raises(ValidationError):
        PredictRequest(features={})


def test_a_feature_may_be_missing_but_not_a_word() -> None:
    """`None` is imputed by the pipeline; a string is a client bug worth a 422."""
    assert PredictRequest(features={"EXT_SOURCE_1": None}).features == {"EXT_SOURCE_1": None}

    with pytest.raises(ValidationError):
        PredictRequest(features={"EXT_SOURCE_1": "high"})


def test_the_response_carries_the_threshold_beside_the_probability() -> None:
    """A ranking score without its operating point cannot be read as a decision."""
    response = PredictResponse(
        proba_default=0.42,
        decision=0,
        threshold=0.49,
        model_name="credit_scoring_model",
        model_version="joblib",
        latency_ms=12.0,
    )

    assert response.decision == int(response.proba_default >= response.threshold)


def test_a_batch_is_a_list_of_the_same_requests() -> None:
    """One shape for one applicant and for a hundred, so the two cannot drift apart."""
    batch = BatchPredictRequest(items=[{"features": {"EXT_SOURCE_1": 0.5}}])

    assert isinstance(batch.items[0], PredictRequest)


def test_a_batch_refuses_what_a_single_request_refuses() -> None:
    with pytest.raises(ValidationError):
        BatchPredictRequest(items=[{"features": {}}])


def test_the_model_info_says_how_many_columns_the_pipeline_expects() -> None:
    """The number a caller compares against `models/feature_columns.json`."""
    info = ModelInfoResponse(
        model_name="credit_scoring_model", model_version="joblib", threshold=0.49, n_features=796
    )

    assert info.n_features == 796
