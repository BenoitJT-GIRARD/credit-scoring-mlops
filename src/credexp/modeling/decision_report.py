"""Load the shipped model and holdout, run every analysis, write results and figures."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import joblib
import matplotlib
import pandas as pd

from credexp.figure_style import PALETTE, apply_style, close, reference_line, save_figure
from credexp.modeling.baselines import trivial_baselines
from credexp.modeling.fairness import BAND_LABELS, age_bands, group_report
from credexp.modeling.sensitivity import bootstrap_cost_ci, cost_ratio_sweep
from credexp.modeling.threshold import split_threshold_cost, threshold_shift
from credexp.utils import DECISION_DIR, FIGURES_DIR, HOLDOUT_PATH, MODELS_DIR

SOURCE = "credexp.modeling.decision_report"

matplotlib.use("Agg")  # no display on CI or on a headless machine
import matplotlib.pyplot as plt  # noqa: E402

HOLDOUT = HOLDOUT_PATH
MODELS = MODELS_DIR
OUT_DIR = DECISION_DIR

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


def _plot_sensitivity(
    sweep: list[dict], operating_ratio: float, n_holdout: int, path: Path
) -> None:
    """The threshold and the cost, against the ratio the whole decision rests on.

    Two axes on one figure, which is a choice worth stating: the reader has to see that the
    threshold moves smoothly while the cost has a floor, and the two are in different units.
    The vertical line is where this repository operates.
    """
    apply_style()
    ratios = [r["ratio"] for r in sweep]
    fig, ax1 = plt.subplots()
    ax1.plot(ratios, [r["threshold"] for r in sweep], marker="o", color=PALETTE["primary"])
    ax1.set_xlabel("Cost of a false negative, relative to a false positive")
    ax1.set_ylabel("Optimal threshold", color=PALETTE["primary"])
    ax2 = ax1.twinx()
    ax2.plot(ratios, [r["cost_per_row"] for r in sweep], marker="s", color=PALETTE["secondary"])
    ax2.set_ylabel("Cost per applicant", color=PALETTE["secondary"])
    # The twin shares the x axis and draws none of its own, and the writer asks every axes
    # carrying data what it measures. Answering twice is cheaper than an unlabelled figure.
    ax2.set_xlabel(ax1.get_xlabel())
    reference_line(ax1, x=operating_ratio, label=f"shipped ratio {operating_ratio:g}")
    ax1.set_title("The decision rests on an assumed ratio")
    ax1.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=fig.subplotpars.bottom + 0.08)

    save_figure(
        fig,
        path,
        n={"applicants": n_holdout, "ratios": len(sweep)},
        source=SOURCE,
        note="one threshold optimised per ratio, on the same holdout",
    )
    close(fig)


def _plot_fairness(rows: list[dict], title: str, n_holdout: int, path: Path) -> None:
    """Refusal rate per group, with the population of each written under its bar.

    A rate without its group size is a bar a reader cannot weigh, and these groups differ
    by a factor of five.
    """
    apply_style()
    fig, ax = plt.subplots()
    labels = [f"{r['group']}\nn = {r['n']}" for r in rows]
    ax.bar(labels, [r["refusal_rate"] or 0.0 for r in rows], color=PALETTE["primary"], width=0.6)
    ax.set_xlabel("Group, and how many applicants it holds")
    ax.set_ylabel("Refusal rate at the shipped threshold")
    ax.set_title(title)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=fig.subplotpars.bottom + 0.08)

    save_figure(
        fig,
        path,
        n={"applicants": n_holdout, "groups": len(rows)},
        source=SOURCE,
        note="the same threshold applied to every group; 95 % Wilson interval on each rate",
    )
    close(fig)


#: The caption lives in the script, not in the document: written into the document it would be
#: overwritten the next time this function ran, which is what the header already warns about.
CAPTION = [
    "> **How to read it.** One row per group. Default rate is what the group did, counted and",
    "> never predicted, and every column beside it is read against that. Refusal rate says how",
    "> often the shipped policy said no to the group. FNR (false negative rate) says how many of",
    "> its defaulters slipped past. Cost puts both mistakes on a single scale, under the",
    "> ten-to-one ratio this analysis assumes.",
]


def render_summary(payload: dict) -> str:
    """The published markdown, rendered from the payload beside it and from nothing else.

    The header names the script and the day, because a generated document that does not say
    so is a document someone will eventually edit by hand.
    """
    ci = payload["cost_interval"]
    shift = payload["threshold_shift"]
    lines = [
        f"<!-- Written by scripts/decision_analysis.py on {payload['written']}. "
        "Edits here are overwritten. -->",
        "",
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
    lines += ["", *CAPTION]
    return "\n".join(lines) + "\n"


def run_decision_analysis() -> dict:
    df, pipe, thr = _load()
    cost_fn, cost_fp, threshold = thr["cost_fn"], thr["cost_fp"], thr["threshold"]

    y = df["TARGET"].astype(int).to_numpy()
    proba = pipe.predict_proba(df.drop(columns=["TARGET"]))[:, 1]

    sweep = cost_ratio_sweep(y, proba)
    payload = {
        "written": dt.date.today().isoformat(),
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
            y, proba, age_bands(df["DAYS_BIRTH"]), threshold, cost_fn, cost_fp,
            order=BAND_LABELS,
        ),
        "baselines": trivial_baselines(y, cost_fn, cost_fp),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "decision_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_sensitivity(sweep, cost_fn / cost_fp, len(y), FIGURES_DIR / "cost_sensitivity.png")
    _plot_fairness(
        payload["fairness_age"],
        "Refusal rate by age band",
        len(y),
        FIGURES_DIR / "fairness_age.png",
    )
    _plot_fairness(
        payload["fairness_gender"],
        "Refusal rate by sex",
        len(y),
        FIGURES_DIR / "fairness_gender.png",
    )
    (OUT_DIR / "decision_summary.md").write_text(render_summary(payload), encoding="utf-8")
    return payload
