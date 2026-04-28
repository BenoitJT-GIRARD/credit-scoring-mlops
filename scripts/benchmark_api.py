from __future__ import annotations

import json
import statistics
import time

import httpx
import numpy as np
import pandas as pd

from credexp.data.io import processed_dir

API_URL = "http://127.0.0.1:8000/predict"


def _sanitize_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, float | np.floating) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def load_payloads(n_rows: int = 50) -> list[dict]:
    holdout_path = processed_dir() / "api_holdout.parquet"
    df = pd.read_parquet(holdout_path)
    if "TARGET" in df.columns:
        df = df.drop(columns=["TARGET"])
    df = df.head(n_rows).copy()

    payloads = []
    for _, row in df.iterrows():
        features = {k: _sanitize_value(v) for k, v in row.to_dict().items()}

        sk_id_curr = features.get("SK_ID_CURR")
        if sk_id_curr is not None:
            try:
                sk_id_curr = int(sk_id_curr)
            except Exception:
                sk_id_curr = None

        payloads.append(
            {
                "sk_id_curr": sk_id_curr,
                "features": features,
            }
        )
    return payloads


def main() -> None:
    payloads = load_payloads(50)
    timings_ms = []
    status_codes = []

    with httpx.Client(timeout=60.0) as client:
        for payload in payloads:
            t0 = time.perf_counter()
            response = client.post(API_URL, json=payload)
            dt = (time.perf_counter() - t0) * 1000

            timings_ms.append(float(dt))
            status_codes.append(int(response.status_code))

    status_counts = pd.Series(status_codes).value_counts().sort_index().to_dict()
    status_counts = {str(int(k)): int(v) for k, v in status_counts.items()}

    result = {
        "n_requests": int(len(payloads)),
        "avg_ms": float(statistics.mean(timings_ms)),
        "median_ms": float(statistics.median(timings_ms)),
        "min_ms": float(min(timings_ms)),
        "max_ms": float(max(timings_ms)),
        "status_codes": status_counts,
    }

    out_path = "reports/performance/api_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
