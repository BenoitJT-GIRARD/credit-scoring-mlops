"""Measure the probability scale and the threshold's spread, then decide on the evidence.

Two questions the repository answered with neither a number nor a caveat: is a returned
probability worth reading as a probability, and is the shipped threshold a stable choice
or one draw among many.

    uv run python scripts/calibration_report.py

Writes `reports/performance/calibration.json` and `reports/figures/calibration.png`.
Nothing is applied to the served model here; the decision follows the measurement and is
recorded in the README.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credexp.config import ARTIFACTS_DIR, DATA_DIR  # noqa: E402
from credexp.modeling.calibration import (  # noqa: E402
    brier,
    calibrate_isotonic,
    expected_calibration_error,
    reliability,
)
from credexp.modeling.threshold import threshold_spread  # noqa: E402
from credexp.utils.logging import get_logger  # noqa: E402

log = get_logger(__name__)

HOLDOUT = DATA_DIR / "processed" / "api_holdout.parquet"
MODEL = ARTIFACTS_DIR / "models" / "pipeline.joblib"
OUT_JSON = ROOT / "reports" / "performance" / "calibration.json"
OUT_FIGURE = ROOT / "reports" / "figures" / "calibration.png"


def _plot(curves: dict[str, pd.DataFrame], summary: dict, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 6))
    axis.plot([0, 1], [0, 1], color="#bbbbbb", linestyle="--", linewidth=1, label="perfect")
    for name, curve in curves.items():
        stats = summary[name]
        axis.plot(
            curve["mean_score"],
            curve["observed"],
            marker="o",
            markersize=4,
            label=f"{name} — Brier {stats['brier']:.4f}, ECE {stats['ece']:.4f}",
        )
    axis.set_xlabel("mean predicted probability (equal-population bins)")
    axis.set_ylabel("observed default rate")
    axis.set_title("Calibration on the holdout")
    axis.legend(fontsize=8, loc="upper left")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    if not HOLDOUT.exists() or not MODEL.exists():
        log.warning(f"need {HOLDOUT} and {MODEL}; run scripts/train_final.py first")
        return

    frame = pd.read_parquet(HOLDOUT)
    y = frame["TARGET"].to_numpy()
    features = frame.drop(columns=["TARGET"])
    pipe = joblib.load(MODEL)
    scores = np.asarray(pipe.predict_proba(features))[:, 1]

    # Half the holdout fits the recalibration, the other half measures it. Fitting on the
    # whole of it and scoring on the same rows would report the fit rather than the gain.
    rng = np.random.default_rng(0)
    order = rng.permutation(len(y))
    fit_idx, test_idx = order[: len(order) // 2], order[len(order) // 2 :]

    recalibrate = calibrate_isotonic(y[fit_idx], scores[fit_idx])
    recalibrated = recalibrate(scores[test_idx])

    summary = {
        "raw": {
            "brier": brier(y[test_idx], scores[test_idx]),
            "ece": expected_calibration_error(y[test_idx], scores[test_idx]),
            "mean_score": float(scores[test_idx].mean()),
        },
        "isotonic": {
            "brier": brier(y[test_idx], recalibrated),
            "ece": expected_calibration_error(y[test_idx], recalibrated),
            "mean_score": float(recalibrated.mean()),
        },
    }
    summary["base_rate"] = float(y[test_idx].mean())
    summary["n_holdout"] = int(len(y))
    summary["threshold_spread"] = threshold_spread(y, scores)

    curves = {
        "raw": reliability(y[test_idx], scores[test_idx]),
        "isotonic": reliability(y[test_idx], recalibrated),
    }
    _plot(curves, summary, OUT_FIGURE)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"[ok] {OUT_JSON}")
    print(f"[ok] {OUT_FIGURE}")


if __name__ == "__main__":
    main()
