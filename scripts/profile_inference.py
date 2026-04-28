from __future__ import annotations

import cProfile
import io
import json
import pstats
import time
from pathlib import Path

import joblib
import pandas as pd

from credexp.config import settings
from credexp.data.io import processed_dir


def load_holdout_sample(n_rows: int = 200) -> pd.DataFrame:
    holdout_path = processed_dir() / "api_holdout.parquet"
    df = pd.read_parquet(holdout_path)
    if "TARGET" in df.columns:
        df = df.drop(columns=["TARGET"])
    return df.head(n_rows).copy()


def load_pipeline():
    model_path = settings.artifacts_dir / "models" / "pipeline.joblib"
    return joblib.load(model_path)


def run_inference(pipe, X: pd.DataFrame, n_loops: int = 20) -> list[float]:
    timings_ms = []
    for _ in range(n_loops):
        t0 = time.perf_counter()
        _ = pipe.predict_proba(X)
        dt = (time.perf_counter() - t0) * 1000
        timings_ms.append(dt)
    return timings_ms


def main() -> None:
    X = load_holdout_sample(n_rows=200)
    pipe = load_pipeline()

    profiler = cProfile.Profile()

    with profiler:
        timings_ms = run_inference(pipe, X, n_loops=20)

    avg_ms = sum(timings_ms) / len(timings_ms)
    min_ms = min(timings_ms)
    max_ms = max(timings_ms)

    output_dir = Path("reports/performance")
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / "cprofile_inference.prof"
    txt_path = output_dir / "cprofile_inference_top20.txt"
    metrics_path = output_dir / "inference_benchmark.json"

    profiler.dump_stats(str(stats_path))

    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    stats.print_stats(20)
    txt_path.write_text(s.getvalue(), encoding="utf-8")

    payload = {
        "n_rows": len(X),
        "n_loops": 20,
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "stats_file": str(stats_path),
        "top20_file": str(txt_path),
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print("\nTop 20 cumulative functions:\n")
    print(s.getvalue())


if __name__ == "__main__":
    main()
