"""One call of fifty rows against fifty calls of one, on the same pipeline.

Batching is the optimisation this repository retained, so the claim gets a number: the
fixed costs -- Python overhead, pandas conversions, scikit-learn's validation -- are paid
once instead of once per applicant.
"""

from __future__ import annotations

import json
import time

import joblib
import pandas as pd

from credexp.data.io import processed_dir
from credexp.utils import PIPELINE_PATH


def load_data():
    holdout_path = processed_dir() / "api_holdout.parquet"
    df = pd.read_parquet(holdout_path)
    if "TARGET" in df.columns:
        df = df.drop(columns=["TARGET"])
    return df.head(100).copy()


def main() -> None:
    pipe = joblib.load(PIPELINE_PATH)
    df = load_data()

    single = df.head(1).copy()
    batch = df.head(50).copy()

    # 50 single-row calls
    t0 = time.perf_counter()
    for _ in range(50):
        _ = pipe.predict_proba(single)
    single_total_ms = (time.perf_counter() - t0) * 1000

    # 1 batch call of 50 rows
    t0 = time.perf_counter()
    _ = pipe.predict_proba(batch)
    batch_total_ms = (time.perf_counter() - t0) * 1000

    result = {
        "single_total_ms_for_50_calls": single_total_ms,
        "single_avg_ms_per_row": single_total_ms / 50,
        "batch_total_ms_for_50_rows": batch_total_ms,
        "batch_avg_ms_per_row": batch_total_ms / 50,
        "speedup_factor_per_row": (single_total_ms / 50) / (batch_total_ms / 50),
    }

    out_path = "reports/performance/batching_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
