"""Three ways a request from the interface can fail, and one way it can work.

Every one of them used to live inside the page, in a `try` with three `except` branches
and a `st.error` in each, where the only way to exercise them was to open a browser with
the service deliberately broken. They are one function now, and this is that function's
test: no Streamlit, no network, and a transport that answers whatever the case needs.
"""

from __future__ import annotations

import httpx
import pytest

from credexp.app.service import EXAMPLE_PAYLOAD, Scored, parse_payload, score

SCORED = {
    "proba_default": 0.42,
    "decision": 0,
    "threshold": 0.49,
    "model_name": "credit_scoring_model",
    "model_version": "joblib",
    "latency_ms": 12.3,
}


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


@pytest.fixture()
def answering(monkeypatch: pytest.MonkeyPatch):
    """Replace the transport, not the function: the request itself is what is under test."""

    def install(handler) -> None:
        original = httpx.Client

        def client(*args, **kwargs):
            kwargs["transport"] = _transport(handler)
            return original(*args, **kwargs)

        monkeypatch.setattr(httpx, "Client", client)

    return install


# --- Reading what the reader typed ------------------------------------------


def test_a_valid_body_comes_back_as_a_mapping() -> None:
    payload, problem = parse_payload('{"sk_id_curr": 1, "features": {"EXT_SOURCE_1": 0.5}}')

    assert problem is None
    assert payload["sk_id_curr"] == 1


def test_a_syntax_error_is_named_rather_than_raised() -> None:
    """The page shows this sentence; a traceback on a text area helps nobody."""
    payload, problem = parse_payload("{not json")

    assert payload is None
    assert "not valid JSON" in problem


def test_a_bare_list_is_refused_with_what_was_expected() -> None:
    payload, problem = parse_payload("[1, 2, 3]")

    assert payload is None
    assert "JSON object" in problem


def test_the_example_payload_parses_as_it_is_shown() -> None:
    """It is the text the page puts in the box, so it has to be valid on the first click."""
    import json

    payload, problem = parse_payload(json.dumps(EXAMPLE_PAYLOAD))

    assert problem is None
    assert set(payload["features"]) == {"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"}


# --- Asking the service -----------------------------------------------------


def test_a_scored_applicant_comes_back_whole(answering) -> None:
    answering(lambda request: httpx.Response(200, json=SCORED))

    result = score(EXAMPLE_PAYLOAD, base_url="http://service:8000")

    assert result == Scored(ok=True, body=SCORED)


def test_the_request_goes_to_the_predict_route_of_the_given_address(answering) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=SCORED)

    answering(handler)
    score(EXAMPLE_PAYLOAD, base_url="http://elsewhere:9000")

    assert seen == ["http://elsewhere:9000/predict"]


def test_a_refused_request_keeps_the_explanation_the_api_gave(answering) -> None:
    """A 422 says which field is wrong, and a page that hides it makes the reader guess."""
    detail = {"detail": [{"loc": ["body", "features"], "msg": "not a mapping"}]}
    answering(lambda request: httpx.Response(422, json=detail))

    result = score({"features": "not_a_dict"})

    assert not result.ok
    assert "HTTP 422" in result.problem
    assert result.body == detail


def test_a_service_that_is_not_running_is_told_apart_from_one_that_refused(answering) -> None:
    """Different sentence, different fix: start the service, or correct the payload."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    answering(handler)
    result = score(EXAMPLE_PAYLOAD, base_url="http://127.0.0.1:8000")

    assert not result.ok
    assert result.body is None
    assert "did not answer" in result.problem
    assert "http://127.0.0.1:8000" in result.problem


def test_an_error_body_that_is_not_json_does_not_hide_the_status(answering) -> None:
    """A proxy answering 502 in HTML is still an answer the reader has to be told about."""
    answering(lambda request: httpx.Response(502, text="<html>gateway</html>"))

    result = score(EXAMPLE_PAYLOAD)

    assert not result.ok
    assert "HTTP 502" in result.problem
    assert result.body is None
