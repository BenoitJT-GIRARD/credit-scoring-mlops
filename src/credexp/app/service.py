"""Talking to the API from the interface, and saying what happened when it does not answer.

The page used to hold this inline, inside a `try` with three `except` branches and a
`st.error` in each. That is untestable without a browser, and the three failures a reader
actually meets — a payload that is not JSON, a service that refuses the request, a service
that is not running — deserve to be told apart in one place rather than in a page.

Nothing here imports Streamlit. It returns a result; the page decides what that looks like.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx

#: Where the interface looks for the service. The compose file sets it to the service name
#: on its own network; the default is what a reader running both by hand will have.
DEFAULT_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

#: Three external credit-bureau scores. They carry most of the signal in this model, and
#: every feature left out of the payload is imputed, so a short request is a valid one.
EXAMPLE_PAYLOAD: dict[str, Any] = {
    "sk_id_curr": 123456,
    "features": {
        "EXT_SOURCE_1": 0.52,
        "EXT_SOURCE_2": 0.71,
        "EXT_SOURCE_3": 0.41,
    },
}


@dataclass(frozen=True)
class Scored:
    """What came back, or why nothing did.

    `body` carries the API's own answer in both cases, because a 422 explains itself and a
    page that swallows that explanation makes the reader guess at their own payload.
    """

    ok: bool
    body: dict[str, Any] | None = None
    problem: str | None = None


def parse_payload(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Read the request body a reader typed, and name the syntax error if there is one."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"That is not valid JSON — {exc}"
    if not isinstance(payload, dict):
        return None, "The request body has to be a JSON object, not a list or a bare value."
    return payload, None


def score(
    payload: dict[str, Any], base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0
) -> Scored:
    """Send one applicant to `POST /predict` and report what the service said.

    Three outcomes, and each is a different sentence for the reader: it scored, the service
    refused the request and explained why, or nothing answered at that address.
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{base_url}/predict", json=payload)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail: dict[str, Any] | None
        try:
            detail = exc.response.json()
        except ValueError:
            detail = None
        return Scored(
            ok=False,
            body=detail,
            problem=f"The API refused the request — HTTP {exc.response.status_code}",
        )
    except httpx.HTTPError as exc:
        return Scored(
            ok=False,
            problem=f"The API did not answer — {exc}. Is it running at {base_url}?",
        )
    return Scored(ok=True, body=response.json())
