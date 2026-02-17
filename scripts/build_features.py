from __future__ import annotations

import argparse
from pathlib import Path

from credexp.config import DATA_DIR
from credexp.data.build_features import build_and_save


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, default=str(DATA_DIR / "raw"))
    parser.add_argument("--out", type=str, default=str(DATA_DIR / "processed" / "features.parquet"))
    parser.add_argument(
        "--debug", action="store_true", help="Use only 10k rows per file for quick run"
    )
    args = parser.parse_args()

    build_and_save(raw_path=Path(args.raw_dir), out_path=Path(args.out), debug=args.debug)


if __name__ == "__main__":
    main()
