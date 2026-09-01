"""Load the shipped model and holdout, run every analysis, write results and figures."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import pandas as pd

from credexp.config import ARTIFACTS_DIR, DATA_DIR
from credexp.modeling.baselines import trivial_baselines
from credexp.modeling.fairness import age_bands, group_report
from credexp.modeling.sensitivity import bootstrap_cost_ci, cost_ratio_sweep
from credexp.modeling.threshold import split_threshold_cost, threshold_shift

matplotlib.use("Agg")  # no display on CI or on a headless machine
import matplotlib.pyplot as plt  # noqa: E402

HOLDOUT = DATA_DIR / "processed" / "api_holdout.parquet"
MODELS = ARTIFACTS_DIR / "models"
OUT_DIR = Path("reports/decision")

# CODE_GENDER is label-encoded by the feature pipeline, so the holdout carries 0 and 1.
# The mapping was verified by joining the holdout to application_train.csv on SK_ID_CURR:
# all 30 751 rows agree, 0 is M and 1 is F. A fairness table of bare integers would be
# unreadable, and guessing the mapping from group sizes would be no better than a coin toss.
GENDER_LABELS = {0: "M", 1: "F"}


def _load():
    if not HOLDOUT.exists():
        raise FileNotFoundError(
            f"Holdout not found at {HOLDOUT}. It is not versioned — Home Credit's terms "
            "do not allow redistribution. Rebuild it with scripts/train_final.py, or "
            "copy it from a run that already produced it."
        )
    df = pd.read_parquet(HOLDOUT)
    pipe = joblib.load(MODELS / "pipeline.joblib")
    threshold = json.loads((MODELS / "threshold.json").read_text(encoding="utf-8"))
    return df, pipe, threshold


def _plot_sensitivity(sweep: list[dict], operating_ratio: float, path: Path) -> None:
    ratios = [r["ratio"] for r in sweep]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ratios, [r["threshold"] for r in sweep], marker="o", color="#2c6e9b")
    ax1.set_xlabel("Cost of a false negative, relative to a false positive")
    ax1.set_ylabel("Optimal threshold", color="#2c6e9b")
    ax2 = ax1.twinx()
    ax2.plot(ratios, [r["cost_per_row"] for r in sweep], marker="s", color="#b8563e")
    ax2.set_ylabel("Cost per applicant", color="#b8563e")
    ax1.axvline(operating_ratio, linestyle="--", color="#8c959f")
    ax1.set_title("The decision rests on an assumed ratio")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_fairness(rows: list[dict], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar([r["group"] for r in rows], [r["refusal_rate"] or 0.0 for r in rows], color="#2c6e9b")
    ax.set_ylabel("Refusal rate at the shipped threshold")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_summary(payload: dict) -> str:
    ci = payload["cost_interval"]
    shift = payload["threshold_shift"]
    lines = [
        "| Question | Answer |",
        "|---|---|",
        f"| Cost per applicant at the shipped threshold | {shift['shipped_cost']:.4f} "
        f"(95% CI {ci['low']:.4f}–{ci['high']:.4f}) |",
        f"| Optimism of tuning the threshold on its own data | "
        f"{payload['optimism']['optimism']:.4f} per applicant |",
        f"| Regret of the shipped threshold vs this sample's optimum | {shift['regret']:.4f} "
        f"(threshold {shift['shipped_threshold']} vs {shift['optimal_threshold']:.2f}) |",
        "",
        "| Baseline | Cost per applicant |",
        "|---|---|",
    ]
    lines += [f"| {b['name']} | {b['cost_per_row']:.4f} |" for b in payload["baselines"]]
    lines += [
        "",
        "| Group | n | Default rate | Refusal rate | FNR | Cost |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload["fairness_age"] + payload["fairness_gender"]:
        fnr = "—" if row["fnr"] is None else f"{row['fnr']:.3f}"
        lines.append(
            f"| {row['group']} | {row['n']} | {row['default_rate']:.3f} | "
            f"{row['refusal_rate']:.3f} | {fnr} | {row['cost_per_row']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def run_decision_analysis() -> dict:
    df, pipe, thr = _load()
    cost_fn, cost_fp, threshold = thr["cost_fn"], thr["cost_fp"], thr["threshold"]

    y = df["TARGET"].astype(int).to_numpy()
    proba = pipe.predict_proba(df.drop(columns=["TARGET"]))[:, 1]

    sweep = cost_ratio_sweep(y, proba)
    payload = {
        "n_holdout": int(len(y)),
        "default_rate": float(y.mean()),
        "shipped_threshold": threshold,
        "cost_ratio": cost_fn / cost_fp,
        "optimism": split_threshold_cost(y, proba, cost_fn, cost_fp),
        "threshold_shift": threshold_shift(y, proba, threshold, cost_fn, cost_fp),
        "cost_interval": bootstrap_cost_ci(y, proba, threshold, cost_fn, cost_fp),
        "sensitivity": sweep,
        "fairness_gender": group_report(
            y,
            proba,
            df["CODE_GENDER"].map(GENDER_LABELS).fillna("unknown").to_numpy(),
            threshold,
            cost_fn,
            cost_fp,
        ),
        "fairness_age": group_report(
            y, proba, age_bands(df["DAYS_BIRTH"]), threshold, cost_fn, cost_fp
        ),
        "baselines": trivial_baselines(y, cost_fn, cost_fp),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "decision_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_sensitivity(sweep, cost_fn / cost_fp, OUT_DIR / "cost_sensitivity.png")
    _plot_fairness(payload["fairness_age"], "Refusal rate by age band", OUT_DIR / "fairness.png")
    (OUT_DIR / "decision_summary.md").write_text(render_summary(payload), encoding="utf-8")
    return payload
