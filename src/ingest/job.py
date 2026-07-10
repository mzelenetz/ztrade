from __future__ import annotations

import logging
import os
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from src.ingest.fetchers import ChainFetcher, fetcher_from_env

log = logging.getLogger("ingest")

DEFAULT_TICKERS = ["AAPL", "AMZN", "LLY", "NVDA", "PLTR", "TSLA"]
MIN_ROWS_PER_TICKER = 100
# Parity coherence: on the nearest expiry, the interquartile range of the
# per-strike implied spots (C−P+K) must be tight relative to their median —
# the same guard that keeps corrupt data out of pricing keeps it out of the
# bucket in the first place.
PARITY_IQR_LIMIT = 0.05


def bucket_name() -> str:
    return os.getenv("GCS_CLOSES_BUCKET", "ztrade-yesterday-closes")


def load_universe() -> list[str]:
    """tickers.txt in the closes bucket → TICKERS env → built-in default."""
    try:
        from google.cloud import storage

        blob = storage.Client().bucket(bucket_name()).blob("tickers.txt")
        text = blob.download_as_text()
        tickers = [
            line.strip().upper()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if tickers:
            log.info("universe from gs://%s/tickers.txt: %s", bucket_name(), tickers)
            return tickers
    except Exception as exc:
        log.warning("could not read tickers.txt (%s); falling back", exc)

    env = os.getenv("TICKERS", "")
    if env.strip():
        return [t.strip().upper() for t in env.split(",") if t.strip()]
    return list(DEFAULT_TICKERS)


def validate_ticker_frame(df: pl.DataFrame, ticker: str) -> str | None:
    """Returns a rejection reason, or None if the data looks publishable."""
    if df.height < MIN_ROWS_PER_TICKER:
        return f"only {df.height} rows"

    spot = float((df["underlying_bid_1545"][0] + df["underlying_ask_1545"][0]) / 2)
    if spot <= 0:
        return "no underlying quote"

    nearest = df["expiration"].min()
    near = df.filter(pl.col("expiration") == nearest)
    calls = {r["strike"]: r["close"] for r in near.filter(pl.col("option_type") == "C").to_dicts()}
    puts = {r["strike"]: r["close"] for r in near.filter(pl.col("option_type") == "P").to_dicts()}
    common = [k for k in calls if k in puts and calls[k] > 0.05 and puts[k] > 0.05]
    if len(common) >= 3:
        implied = [calls[k] - puts[k] + k for k in common]
        med = float(np.median(implied))
        q25, q75 = np.percentile(implied, [25, 75])
        if med <= 0 or (q75 - q25) > PARITY_IQR_LIMIT * med:
            return f"incoherent parity (median implied spot {med:.2f}, IQR {(q75 - q25):.2f})"
        if abs(med / spot - 1) > 0.10:
            return f"parity spot {med:.2f} disagrees with quoted {spot:.2f}"

    return None


def run_ingest(
    quote_date: date | None = None,
    dry_run_path: str | None = None,
    fetcher: ChainFetcher | None = None,
) -> str | None:
    """Fetch the universe, validate, and publish closes-<date>.csv.

    Returns the destination written, or None when skipped (non-trading day).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    today_et = datetime.now(ZoneInfo("America/New_York")).date()
    quote_date = quote_date or today_et
    if quote_date.weekday() >= 5:
        log.info("%s is a weekend — nothing to ingest", quote_date)
        return None

    fetcher = fetcher or fetcher_from_env()
    tickers = load_universe()

    frames: list[pl.DataFrame] = []
    for ticker in tickers:
        try:
            df = fetcher.fetch(ticker, quote_date)
        except Exception as exc:
            log.error("%s: fetch failed: %s", ticker, exc)
            continue

        reason = validate_ticker_frame(df, ticker)
        if reason is not None:
            log.error("%s: rejected — %s", ticker, reason)
            continue

        log.info("%s: %d rows across %d expiries", ticker, df.height, df["expiration"].n_unique())
        frames.append(df)

    if not frames:
        raise RuntimeError("ingest produced no publishable data for any ticker")

    combined = pl.concat(frames)
    blob_name = f"closes-{quote_date:%Y-%m-%d}.csv"

    if dry_run_path:
        combined.write_csv(dry_run_path)
        log.info("dry run: wrote %d rows to %s", combined.height, dry_run_path)
        return dry_run_path

    from google.cloud import storage

    bucket = storage.Client().bucket(bucket_name())
    bucket.blob(blob_name).upload_from_string(combined.write_csv(), content_type="text/csv")
    dest = f"gs://{bucket_name()}/{blob_name}"
    log.info("published %d rows (%d tickers) to %s", combined.height, len(frames), dest)
    return dest
