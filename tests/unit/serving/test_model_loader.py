"""Where the served threshold comes from, and what the log says when the file is missing.

Without `threshold.json` the loader falls back to `DEFAULT_THRESHOLD`, 0.5 unless set, and
the service starts. That fallback changes every decision the service takes, so it is
written to the log.
"""

from __future__ import annotations

import json
import logging

import pytest

from credexp.serving import model_loader


class _Records(logging.Handler):
    """Keeps what the loader logs. The project's loggers do not propagate to the root."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def logged():
    handler = _Records()
    model_loader.log.addHandler(handler)
    try:
        yield handler.records
    finally:
        model_loader.log.removeHandler(handler)


def _warnings(records: list[logging.LogRecord]) -> list[str]:
    return [record.getMessage() for record in records if record.levelno == logging.WARNING]


def test_a_missing_threshold_file_falls_back_and_says_so(
    tmp_path, monkeypatch: pytest.MonkeyPatch, logged
) -> None:
    monkeypatch.setenv("THRESHOLD_PATH", str(tmp_path / "absent.json"))
    monkeypatch.delenv("DEFAULT_THRESHOLD", raising=False)
    monkeypatch.setattr(model_loader, "_model_dir", lambda: tmp_path)

    assert model_loader._load_threshold() == 0.5
    assert _warnings(logged) == ["threshold_file_missing"]


def test_a_threshold_file_is_read_without_a_warning(
    tmp_path, monkeypatch: pytest.MonkeyPatch, logged
) -> None:
    path = tmp_path / "threshold.json"
    path.write_text(json.dumps({"threshold": 0.49}), encoding="utf-8")
    monkeypatch.setenv("THRESHOLD_PATH", str(path))

    assert model_loader._load_threshold() == 0.49
    assert _warnings(logged) == []
