"""Profile `predict_proba` over two hundred rows, and write the cumulative-time table.

Both outputs name their files relative to the project, the binary one included: cProfile
keys every function by the absolute file it was read from, and a committed profile that
carries `/home/someone/` or `C:/Users/someone/` says where it was produced and nothing a
second reader can use.
"""

from __future__ import annotations

import cProfile
import io
import json
import marshal
import pstats
import time
from pathlib import Path

import joblib
import pandas as pd

from credexp.data.io import processed_dir
from credexp.utils import PERFORMANCE_DIR, PIPELINE_PATH, ROOT_DIR


def shorten(name: str) -> str:
    """The file a function is defined in, without the machine it was read from.

    A built-in has no file — cProfile names it `<built-in method ...>` — and keeps its name.
    Everything else is answered in this order: inside the project, project-relative; inside
    an installed environment, from the distribution down; otherwise the file name alone.
    """
    if name.startswith("<"):
        return name
    path = Path(name)
    try:
        return path.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        pass
    parts = path.as_posix().split("/")
    for marker in ("site-packages", "Lib", "lib"):
        if marker in parts:
            return "/".join(parts[parts.index(marker) + 1 :])
    return path.name


def shorten_stats(stats: dict) -> dict:
    """The same statistics, keyed by shortened files — callers included, or the table lies."""

    def key(entry):
        filename, lineno, funcname = entry
        return (shorten(filename), lineno, funcname)

    shortened = {}
    for entry, (cc, nc, tt, ct, callers) in stats.items():
        shortened[key(entry)] = (
            cc,
            nc,
            tt,
            ct,
            {key(caller): value for caller, value in callers.items()},
        )
    return shortened


def load_holdout_sample(n_rows: int = 200) -> pd.DataFrame:
    holdout_path = processed_dir() / "api_holdout.parquet"
    df = pd.read_parquet(holdout_path)
    if "TARGET" in df.columns:
        df = df.drop(columns=["TARGET"])
    return df.head(n_rows).copy()


def load_pipeline():
    model_path = PIPELINE_PATH
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

    output_dir = PERFORMANCE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / "cprofile_inference.prof"
    txt_path = output_dir / "cprofile_inference_top20.txt"
    metrics_path = output_dir / "inference_benchmark.json"

    profiler.create_stats()
    profiler.stats = shorten_stats(profiler.stats)
    with stats_path.open("wb") as handle:
        marshal.dump(profiler.stats, handle)

    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    stats.print_stats(20)
    report = s.getvalue()
    txt_path.write_text(report, encoding="utf-8")

    payload = {
        "n_rows": len(X),
        "n_loops": 20,
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "stats_file": stats_path.as_posix(),
        "top20_file": txt_path.as_posix(),
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print("\nTop 20 cumulative functions:\n")
    print(report)


if __name__ == "__main__":
    main()
