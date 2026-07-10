from __future__ import annotations

import time
from datetime import date
from typing import Protocol

import polars as pl

# The schema the app's loader (CBOEOptionsData._prep_data) consumes. Every
# fetcher must emit exactly these columns, one row per contract.
SCHEMA_COLUMNS = [
    "underlying_symbol",
    "quote_date",
    "expiration",
    "strike",
    "option_type",
    "close",
    "bid_1545",
    "ask_1545",
    "underlying_bid_1545",
    "underlying_ask_1545",
    "implied_volatility_1545",
    "trade_volume",
    "open_interest",
    "hist_vol_30d",  # trailing 30-trading-day realized vol of the underlying, annualized
]


class ChainFetcher(Protocol):
    """Fetches a full option chain for one ticker in the closes-file schema.

    Swapping in a purchased data feed later means writing one new fetcher
    class and pointing INGEST_SOURCE at it — nothing downstream changes.
    """

    def fetch(self, ticker: str, quote_date: date) -> pl.DataFrame: ...


class YFinanceFetcher:
    """Interim free source. Quotes are ~15-minute delayed and after-hours
    bid/ask can be stale; the app's spread weighting and no-arbitrage hygiene
    already discount bad quotes, and the vendor IV column is informational
    (the app re-inverts IVs from mids under its own carry)."""

    def __init__(self, pause_seconds: float = 0.25):
        self.pause_seconds = pause_seconds

    def fetch(self, ticker: str, quote_date: date) -> pl.DataFrame:
        import yfinance as yf

        tk = yf.Ticker(ticker)

        under_bid, under_ask = self._underlying_quote(tk)
        hist_vol = self._trailing_realized_vol(tk)
        expiries = tk.options
        if not expiries:
            raise RuntimeError(f"{ticker}: no option expiries returned")

        frames: list[pl.DataFrame] = []
        for expiry in expiries:
            try:
                chain = tk.option_chain(expiry)
            except Exception:
                continue  # one bad expiry must not kill the ticker
            for side_df, opt_type in ((chain.calls, "C"), (chain.puts, "P")):
                normalized = self._normalize(
                    side_df, ticker, quote_date, expiry, opt_type, under_bid, under_ask, hist_vol
                )
                if normalized.height:
                    frames.append(normalized)
            time.sleep(self.pause_seconds)

        if not frames:
            raise RuntimeError(f"{ticker}: no option quotes returned")
        return pl.concat(frames)

    @staticmethod
    def _underlying_quote(tk) -> tuple[float, float]:
        info = tk.fast_info
        last = float(info["lastPrice"])
        if last <= 0:
            raise RuntimeError("no underlying price")
        # yfinance has no reliable underlying bid/ask; use last for both sides.
        return last, last

    @staticmethod
    def _trailing_realized_vol(tk, window: int = 30) -> float | None:
        """Annualized std of daily log returns over the last `window` trading days."""
        import numpy as np

        try:
            closes = tk.history(period="4mo")["Close"].dropna()
        except Exception:
            return None
        if len(closes) < window + 1:
            return None
        rets = np.diff(np.log(closes.to_numpy()))[-window:]
        vol = float(np.std(rets, ddof=1) * np.sqrt(252.0))
        return vol if 0.01 < vol < 5.0 else None

    @staticmethod
    def _normalize(
        pdf, ticker: str, quote_date: date, expiry: str, opt_type: str,
        under_bid: float, under_ask: float, hist_vol: float | None = None,
    ) -> pl.DataFrame:
        df = pl.from_pandas(
            pdf[["strike", "lastPrice", "bid", "ask", "impliedVolatility", "volume", "openInterest"]]
        )
        # all-null pandas columns arrive as Object dtype — force numeric first
        df = df.with_columns(
            pl.col(c).cast(pl.Float64, strict=False)
            for c in ("strike", "lastPrice", "bid", "ask", "impliedVolatility", "volume", "openInterest")
        )
        df = df.with_columns(
            pl.col("bid").fill_null(0.0),
            pl.col("ask").fill_null(0.0),
            pl.col("volume").fill_null(0),
            pl.col("openInterest").fill_null(0),
            pl.col("impliedVolatility").fill_null(0.0),
            pl.col("lastPrice").fill_null(0.0),
        ).filter(
            # keep contracts with any price signal at all
            (pl.col("lastPrice") > 0) | (pl.col("bid") > 0) | (pl.col("ask") > 0)
        )

        return df.select(
            pl.lit(ticker).alias("underlying_symbol"),
            pl.lit(quote_date.isoformat()).alias("quote_date"),
            pl.lit(expiry).alias("expiration"),
            pl.col("strike").cast(pl.Float64),
            pl.lit(opt_type).alias("option_type"),
            pl.col("lastPrice").cast(pl.Float64).alias("close"),
            pl.col("bid").cast(pl.Float64).alias("bid_1545"),
            pl.col("ask").cast(pl.Float64).alias("ask_1545"),
            pl.lit(float(under_bid)).alias("underlying_bid_1545"),
            pl.lit(float(under_ask)).alias("underlying_ask_1545"),
            pl.col("impliedVolatility").cast(pl.Float64).alias("implied_volatility_1545"),
            pl.col("volume").cast(pl.Float64).alias("trade_volume"),
            pl.col("openInterest").cast(pl.Float64).alias("open_interest"),
            pl.lit(hist_vol, dtype=pl.Float64).alias("hist_vol_30d"),
        )


def fetcher_from_env() -> ChainFetcher:
    import os

    source = os.getenv("INGEST_SOURCE", "yfinance").lower()
    if source == "yfinance":
        return YFinanceFetcher()
    raise ValueError(f"Unknown INGEST_SOURCE: {source}")
