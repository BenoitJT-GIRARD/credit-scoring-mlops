"""The API started as a process, on a port, and questioned over HTTP.

The tier below drives the FastAPI application object in memory. That settles the routing, the
validation and the shape of every response, and none of what the first `docker compose up`
meets: a module path that no longer imports, three artefacts missing from the image, a startup
event that raises, a port already taken. A test client meets none of them.

So this file starts `uvicorn`, waits for `/health`, and asks the running service to score.
Two properties are asserted that only a real process can show. **The service answers with
no database.** Writing each prediction to PostgreSQL is best-effort by design, and the
attrition repository shipped the opposite for months: the insert ran without a guard, so an
unreachable database turned every prediction into a 500. **And the decision follows the
threshold the artefact carries**, which is the one number this repository exists to choose.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest

from credexp.utils import ROOT_DIR

pytestmark = pytest.mark.system

BOOT_TIMEOUT = 120

#: An applicant the shipped model scores. Two features of the 796 are enough: the pipeline
#: imputes the rest, which is the contract `/predict` publishes.
APPLICANT = {"EXT_SOURCE_1": 0.5, "EXT_SOURCE_2": 0.7}


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _get(url: str, timeout: float = 30.0) -> tuple[int, dict]:
    with urllib.request.urlopen(url, timeout=timeout) as answer:  # noqa: S310 - fixed localhost
        return answer.status, json.loads(answer.read().decode("utf-8"))


def _post(url: str, payload: dict, timeout: float = 60.0) -> tuple[int, dict]:
    request = urllib.request.Request(  # noqa: S310 - fixed localhost
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as answer:  # noqa: S310
        return answer.status, json.loads(answer.read().decode("utf-8"))


def _text(url: str, timeout: float = 30.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as answer:  # noqa: S310
        return answer.read().decode("utf-8")


@pytest.fixture(scope="module")
def service(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Start the API on a port of its own, serving the tracked artefacts.

    `DATABASE_URL` points at a port nothing listens on, on purpose: the prediction log is
    best-effort and this tier is where that claim is worth something.
    """
    port = _free_port()
    log = tmp_path_factory.mktemp("service") / "uvicorn.log"
    environment = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "MODEL_LOAD_MODE": "joblib",
        "MODEL_VERSION": "system-test",
        "PROMETHEUS_ENABLED": "true",
        "DATABASE_URL": "postgresql+psycopg://nobody:nobody@127.0.0.1:1/absent",
    }
    # Output to a file: nothing here drains a pipe, and a pipe nobody drains fills up and
    # blocks the server on its own logging.
    with log.open("w", encoding="utf-8") as handle:
        server = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [
                sys.executable,
                "-m",
                "uvicorn",
                "credexp.serving.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=str(ROOT_DIR),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + BOOT_TIMEOUT
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    pytest.fail(
                        f"uvicorn stopped with {server.returncode}:\n"
                        f"{log.read_text(encoding='utf-8', errors='replace')[-4000:]}"
                    )
                try:
                    status, _ = _get(f"{base}/health", timeout=2.0)
                except (urllib.error.URLError, OSError, TimeoutError):
                    time.sleep(0.5)
                    continue
                if status == 200:
                    break
            else:
                pytest.fail(
                    "the service never answered /health:\n"
                    f"{log.read_text(encoding='utf-8', errors='replace')[-4000:]}"
                )
            yield base
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()


def test_the_service_loads_the_tracked_artefact_at_startup(service: str) -> None:
    """`/model-info` answers, which means the startup event found the three files."""
    status, body = _get(f"{service}/model-info")

    assert status == 200
    assert body["n_features"] == 796
    assert body["model_version"] == "system-test"
    assert 0.0 < body["threshold"] < 1.0


def test_an_applicant_is_scored_and_the_decision_follows_the_threshold(service: str) -> None:
    """One request, over HTTP, against the model that is committed here."""
    status, body = _post(f"{service}/predict", {"sk_id_curr": 100002, "features": APPLICANT})

    assert status == 200
    assert 0.0 <= body["proba_default"] <= 1.0
    assert body["decision"] == int(body["proba_default"] >= body["threshold"])
    assert body["latency_ms"] > 0


def test_the_service_answers_with_no_database_behind_it(service: str) -> None:
    """The prediction log is best effort, and this is where that claim is worth something.

    `DATABASE_URL` points at a closed port for the whole module. A service that returned
    500 here would be the defect this tier exists to catch, and it is a defect a sibling
    repository shipped.
    """
    status, body = _post(f"{service}/predict", {"sk_id_curr": 100003, "features": APPLICANT})

    assert status == 200
    assert "proba_default" in body


def test_a_batch_of_five_comes_back_as_five_scores(service: str) -> None:
    """The endpoint that exists so a hundred applicants cost one call and not a hundred."""
    items = [{"sk_id_curr": 200000 + index, "features": APPLICANT} for index in range(5)]
    status, body = _post(f"{service}/predict_batch", {"items": items})

    assert status == 200
    assert len(body["results"]) == 5
    assert all(0.0 <= result["proba_default"] <= 1.0 for result in body["results"])


def test_the_metrics_endpoint_counts_what_the_service_did(service: str) -> None:
    """Prometheus scrapes this page, and the dashboard of the README reads that scrape."""
    scrape = _text(f"{service}/metrics")

    assert "credexp_predictions_total" in scrape
    assert "credexp_default_probability" in scrape
    assert "credexp_threshold_crossings_total" in scrape
    # The HTTP histogram comes from the instrumentator, the four above from the module
    # this repository wrote. A scrape that carried only the first would describe the
    # transport and say nothing about the model.
    assert "http_request_duration_seconds" in scrape
