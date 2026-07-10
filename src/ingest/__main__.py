from __future__ import annotations

import argparse
from datetime import date

from src.ingest.job import run_ingest


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest post-close option chains")
    parser.add_argument("--date", type=date.fromisoformat, default=None,
                        help="quote date (default: today, America/New_York)")
    parser.add_argument("--dry-run", metavar="PATH", default=None,
                        help="write the CSV locally instead of uploading to GCS")
    args = parser.parse_args()

    run_ingest(quote_date=args.date, dry_run_path=args.dry_run)


if __name__ == "__main__":
    main()
