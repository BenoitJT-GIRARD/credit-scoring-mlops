"""Write the reference sample the drift report compares live traffic against.

`monitoring_drift.py` reads `var/data/processed/reference.parquet` and nothing in this
repository used to write it: the file had been produced by hand in a notebook, so the
drift report was reproducible only on the machine that happened to still have it.

The reference is a sample of the population the model was fit on. Drift is measured on the
raw API input, so the sample keeps the raw columns and lets `align_reference_and_current`
intersect them with whatever callers actually send.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from credexp.utils import DRIFT_REFERENCE_PATH, HOLDOUT_PATH, PROCESSED_DIR
from credexp.utils.logging import get_logger

log = get_logger(__name__)

#: The engineered training matrix, when it has been rebuilt from the raw Kaggle tables.
FEATURES = PROCESSED_DIR / "features.parquet"
#: The scoring holdout, kept as a fallback so the drift pipeline can be exercised without
#: rebuilding the full feature set. Reference and current then come from the same
#: population, and the report shows tooling rather than drift.
HOLDOUT = HOLDOUT_PATH
DEFAULT_OUTPUT = DRIFT_REFERENCE_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help=f"Parquet to sample from. Default: {FEATURES}, falling back to {HOLDOUT}.",
    )
    parser.add_argument("--rows", type=int, default=20_000, help="Rows to keep in the sample.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve_source(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"{explicit} does not exist")
        return explicit
    if FEATURES.exists():
        return FEATURES
    if HOLDOUT.exists():
        log.warning(
            "%s is missing, sampling %s instead: the reference and the live traffic then "
            "come from the same population, and no drift is expected.",
            FEATURES.name,
            HOLDOUT.name,
        )
        return HOLDOUT
    raise SystemExit(
        "Neither the feature matrix nor the holdout is present. Rebuild them from the raw "
        "Kaggle tables with scripts/build_features.py before running this."
    )


def main() -> None:
    args = parse_args()
    source = resolve_source(args.source)

    df = pd.read_parquet(source)
    if "TARGET" in df.columns:
        # The reference describes the incoming population, not its outcome.
        labelled = df[df["TARGET"].notna()]
        df = (labelled if len(labelled) else df).drop(columns=["TARGET"])

    rows = min(args.rows, len(df))
    sample = df.sample(n=rows, random_state=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(args.output, index=False)
    log.info(
        "reference written: %s rows x %s columns from %s -> %s",
        len(sample),
        sample.shape[1],
        source.name,
        args.output,
    )


if __name__ == "__main__":
    main()
