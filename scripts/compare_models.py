"""Compare the model families on the same folds, and write the table down.

The README used to name this table as missing: the shipped LightGBM was compared against
trivial baselines and against nothing that learns. The comparison lived in an MLflow store
that is not in the repository and is gone, so the numbers it produced could be read in a
notebook output and nowhere else.

This runs it again on the one frame a reader can be shown: the 30 751-applicant holdout, with
the same 796 engineered columns the pipeline was fitted on. It is a tenth of what the shipped
model saw, so the *absolute* scores here are higher than the published ones. Orderings survive
a change of sample, and an ordering is what a family comparison is for.

    uv run python scripts/compare_models.py

Writes `reports/model_comparison.csv`. Every column is defined in `metrics.yaml`.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict

import pandas as pd

from credexp.artifacts import LINE_TERMINATOR
from credexp.modeling.train import TrainConfig, run_cv
from credexp.utils import HOLDOUT_PATH, REPORTS_DIR
from credexp.utils.logging import get_logger

log = get_logger(__name__)

OUTPUT = REPORTS_DIR / "model_comparison.csv"

#: The families, and what each one is here to answer. The two baselines are not models: a
#: comparison whose worst entry still learns something cannot say what learning was worth.
FAMILIES: tuple[tuple[str, str | None, str], ...] = (
    ("dummy", "most_frequent", "refuses nobody: the majority class, every time"),
    ("dummy", "stratified", "a coin weighted by the base rate"),
    ("lr", None, "logistic regression, balanced class weights, scaled features"),
    ("lgbm", None, "gradient-boosted trees, balanced class weights — the shipped family"),
)

FIELDS = (
    "model",
    "variant",
    "n_applicants",
    "n_splits",
    "roc_auc_mean",
    "roc_auc_std",
    "pr_auc_mean",
    "pr_auc_std",
    "best_threshold_mean",
    "business_cost_per_row",
    "seconds",
    "note",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="score fewer applicants, for a quick read of the machinery",
    )
    arguments = parser.parse_args()

    if not HOLDOUT_PATH.is_file():
        raise SystemExit(f"{HOLDOUT_PATH} is missing — run scripts/build_features.py first")

    frame = pd.read_parquet(HOLDOUT_PATH)
    if arguments.rows:
        frame = frame.sample(n=arguments.rows, random_state=0)
    target = frame["TARGET"].astype(int)
    features = frame.drop(columns=["TARGET"])
    log.info(f"comparison on {len(frame)} applicants, default rate {target.mean():.4f}")

    config = TrainConfig(n_splits=arguments.splits)
    rows = []
    for model, variant, note in FAMILIES:
        started = time.monotonic()
        report = run_cv(features, target, model, variant, config)
        rows.append(
            {
                "model": model,
                "variant": variant or "",
                "n_applicants": len(frame),
                "n_splits": arguments.splits,
                **{key: round(value, 4) for key, value in report.items()},
                "seconds": round(time.monotonic() - started, 1),
                "note": note,
            }
        )
        log.info(f"{model} {variant or ''}: {rows[-1]['business_cost_per_row']} per applicant")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS), lineterminator=LINE_TERMINATOR)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[ok] {OUTPUT}")
    print(f"     {len(rows)} families, {arguments.splits} folds, {len(frame)} applicants")
    print(f"     config: {asdict(config)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
