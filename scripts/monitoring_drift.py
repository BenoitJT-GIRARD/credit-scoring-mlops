"""Compare the logged requests against the reference sample and write the Evidently report.

The HTML is not versioned -- Evidently embeds its data and one report reached 34 MB. The
metadata of each run is.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from credexp.monitoring.drift import (
    align_reference_and_current,
    build_drift_report,
    load_current_dataframe,
    load_reference_dataframe,
    save_drift_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Evidently data drift report.")
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Number of latest production rows to pull from PostgreSQL.",
    )
    parser.add_argument(
        "--output-html",
        type=str,
        default="reports/monitoring/evidently_drift.html",
        help="Output path for the HTML report.",
    )
    parser.add_argument(
        "--output-meta",
        type=str,
        default="reports/monitoring/evidently_drift_meta.json",
        help="Output path for metadata JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    reference_df = load_reference_dataframe()
    current_df = load_current_dataframe(limit=args.limit)

    reference_df, current_df = align_reference_and_current(reference_df, current_df)

    report = build_drift_report(reference_df, current_df)

    output_html = Path(args.output_html)
    output_meta = Path(args.output_meta)

    save_drift_report(
        report,
        output_html=output_html,
        output_meta_json=output_meta,
        n_reference=len(reference_df),
        n_current=len(current_df),
        n_features=len(reference_df.columns),
    )

    print(f"Drift report saved to: {output_html}")
    print(f"Metadata saved to: {output_meta}")
    print(f"Reference rows: {len(reference_df)}")
    print(f"Current rows: {len(current_df)}")
    print(f"Common features: {len(reference_df.columns)}")


if __name__ == "__main__":
    main()
