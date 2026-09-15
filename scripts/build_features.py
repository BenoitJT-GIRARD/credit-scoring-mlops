"""Join the seven raw Kaggle tables into the feature matrix everything else reads.

Needs the raw tables, which are not redistributed here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from credexp.data.build_features import build_and_save
from credexp.utils import PROCESSED_DIR, RAW_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, default=str(RAW_DIR))
    parser.add_argument("--out", type=str, default=str(PROCESSED_DIR / "features.parquet"))
    parser.add_argument(
        "--debug", action="store_true", help="Use only 10k rows per file for quick run"
    )
    args = parser.parse_args()

    build_and_save(raw_path=Path(args.raw_dir), out_path=Path(args.out), debug=args.debug)


if __name__ == "__main__":
    main()
