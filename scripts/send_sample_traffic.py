"""Send holdout applicants to a running API, so the monitoring has something to show.

A Grafana dashboard on an idle service is a screenshot of empty panels, and the drift
report needs logged requests to compare against the reference. This drives both routes the
way a caller would, with real applicants drawn from the scoring holdout.

    docker compose up -d
    uv run python scripts/send_sample_traffic.py --requests 400
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import httpx
import pandas as pd

from credexp.utils import HOLDOUT_PATH
from credexp.utils.logging import get_logger

log = get_logger(__name__)

HOLDOUT = HOLDOUT_PATH

#: What a caller plausibly holds at application time: the three credit-bureau scores, the
#: amounts, and a few demographics. Anything left out is imputed by the pipeline, so this
#: is a valid request — and keeping it small is what makes the drift report readable.
PAYLOAD_COLUMNS = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_CHILDREN",
    "REGION_POPULATION_RELATIVE",
    "CODE_GENDER",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=300, help="Single-applicant calls.")
    parser.add_argument("--batches", type=int, default=5, help="Calls to /predict_batch.")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--pause", type=float, default=0.05, help="Seconds between calls.")
    parser.add_argument("--holdout", type=Path, default=HOLDOUT)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fix the applicants sent. Left unset on purpose: a fixed seed replays the "
        "same handful of applicants on every run, and the drift report then reads a "
        "repeated sample as a shifted population.",
    )
    return parser.parse_args()


def to_payload(row: pd.Series, columns: list[str]) -> dict:
    features = {}
    for column in columns:
        value = row[column]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        features[column] = float(value)
    return {"sk_id_curr": int(row["SK_ID_CURR"]), "features": features}


def main() -> None:
    args = parse_args()
    if not args.holdout.exists():
        raise SystemExit(
            f"{args.holdout} is missing. Rebuild it from the raw Kaggle tables with "
            "scripts/build_features.py."
        )

    frame = pd.read_parquet(args.holdout)
    columns = [c for c in PAYLOAD_COLUMNS if c in frame.columns]
    missing = sorted(set(PAYLOAD_COLUMNS) - set(columns))
    if missing:
        log.warning("holdout has no %s; sending the %s columns it does have", missing, len(columns))

    needed = args.requests + args.batches * args.batch_size
    rows = frame.sample(n=min(needed, len(frame)), random_state=args.seed).reset_index(drop=True)

    scored = refused = failed = 0
    with httpx.Client(timeout=60.0) as client:
        for i in range(min(args.requests, len(rows))):
            response = client.post(
                f"{args.base_url}/predict", json=to_payload(rows.iloc[i], columns)
            )
            if response.status_code == 200:
                scored += 1
                refused += int(response.json()["decision"])
            else:
                failed += 1
                if failed == 1:
                    log.error("HTTP %s: %s", response.status_code, response.text[:300])
            time.sleep(args.pause)

        for b in range(args.batches):
            start = args.requests + b * args.batch_size
            chunk = rows.iloc[start : start + args.batch_size]
            if chunk.empty:
                break
            body = {"items": [to_payload(chunk.iloc[j], columns) for j in range(len(chunk))]}
            client.post(f"{args.base_url}/predict_batch", json=body)
            time.sleep(args.pause)

    rate = refused / scored if scored else 0.0
    log.info("scored=%s refused=%s (%.1f%%) failed=%s", scored, refused, 100 * rate, failed)


if __name__ == "__main__":
    main()
