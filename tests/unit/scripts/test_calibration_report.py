"""The calibration figure states the number of applicants its curves are drawn on.

The curves are computed on the evaluation half of the holdout, and the figure used to be
stamped with the size of the whole holdout, twice the number it shows.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression


def _holdout(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"A": rng.normal(size=n), "B": rng.normal(size=n)})
    frame["TARGET"] = (frame["A"] + rng.normal(scale=0.5, size=n) > 1.0).astype(int)
    return frame


def test_the_figure_is_stamped_with_the_evaluated_half(
    load_script, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = load_script("calibration_report")
    holdout = _holdout(200)
    holdout_path = tmp_path / "holdout.parquet"
    holdout.to_parquet(holdout_path, index=False)
    model_path = tmp_path / "pipeline.joblib"
    joblib.dump(
        LogisticRegression().fit(holdout[["A", "B"]], holdout["TARGET"]),
        model_path,
    )
    figures = tmp_path / "figures"

    monkeypatch.setattr(report, "HOLDOUT", holdout_path)
    monkeypatch.setattr(report, "MODEL", model_path)
    monkeypatch.setattr(report, "OUT_JSON", tmp_path / "calibration.json")
    monkeypatch.setattr(report, "OUT_FIGURE", figures / "calibration.png")

    report.main()

    manifest = json.loads((figures / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["images"]["calibration.png"]["n"] == "n(applicants) = 100, n(bins) = 10"
    summary = json.loads((tmp_path / "calibration.json").read_text(encoding="utf-8"))
    assert summary["n_holdout"] == 200
