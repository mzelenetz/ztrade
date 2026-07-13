from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime

import polars as pl

from src.data_sources import GCSClosesDataSource, load_from_env
from src.pricing_utils import CBOEOptionsData, OptionsPrices
from src.services.chain_service import add_greeks, add_probabilities
from src.services.model_inputs import DEFAULT_RATE_CURVE, apply_model_inputs

# Close files are immutable once written, so priced results for an explicit
# close_date can live for hours; only the "latest" (close_date=None) view needs
# a short TTL to pick up a newly landed file.
_CACHE_TTL_DATED = 6 * 3600
_CACHE_TTL_LATEST = 600
_cache: dict[tuple, tuple[float, pl.DataFrame]] = {}
_RAW_TTL_SECONDS = 600
_raw_cache: dict[str | None, tuple[float, pl.DataFrame]] = {}

# Coalesce concurrent identical pricing runs: the dashboard fires several
# endpoints at once with the same model inputs — without a per-key lock each
# would redundantly reprice the ticker (~seconds of CPU apiece).
_key_locks: dict[tuple, threading.Lock] = {}
_key_locks_guard = threading.Lock()


def _lock_for(key: tuple) -> threading.Lock:
    with _key_locks_guard:
        return _key_locks.setdefault(key, threading.Lock())


def _load_raw(close_date: str | None) -> pl.DataFrame:
    now = time.monotonic()
    cached = _raw_cache.get(close_date)
    if cached is not None and now - cached[0] < _RAW_TTL_SECONDS:
        return cached[1]

    source = load_from_env()

    if isinstance(source, GCSClosesDataSource):
        raw = (
            source.load()
            if close_date is None
            else source.load_for_date(datetime.strptime(close_date, "%Y-%m-%d").date())
        )
    else:
        raw = source.load()

    _raw_cache[close_date] = (now, raw)
    return raw


def list_tickers(close_date: str | None) -> list[str]:
    """Ticker list straight from the raw file — no pricing required."""
    return sorted(_load_raw(close_date)["underlying_symbol"].unique().to_list())


def parse_dividends(dividends_json: str) -> dict[str, float]:
    parsed = json.loads(dividends_json) if dividends_json else {}
    return {str(k): float(v) for k, v in parsed.items()}


def parse_dividend_schedule(schedule_json: str) -> dict[str, list[tuple[str, float]]]:
    parsed = json.loads(schedule_json) if schedule_json else {}
    return {
        str(ticker): [(str(ex_date), float(amount)) for ex_date, amount in rows]
        for ticker, rows in parsed.items()
    }


def parse_rate_curve(rate_curve_json: str) -> list[tuple[float, float]]:
    if not rate_curve_json:
        return list(DEFAULT_RATE_CURVE)
    parsed = json.loads(rate_curve_json)
    curve = sorted((float(days), float(rate)) for days, rate in parsed)
    return curve or list(DEFAULT_RATE_CURVE)


def get_priced_data(
    pricing_model: str,
    close_date: str | None,
    vol_mode: str = "surface",
    dividends_json: str = "",
    rate_curve_json: str = "",
    ticker: str | None = None,
    carry_mode: str = "manual",
    dividend_schedule_json: str = "",
) -> pl.DataFrame:
    """Load, price, and cache option data for a full set of model inputs.

    Pricing is per-ticker when one is given — a full-book binomial run takes
    minutes; one ticker takes seconds.
    """
    key = (
        pricing_model, close_date, vol_mode, dividends_json, rate_curve_json,
        ticker, carry_mode, dividend_schedule_json,
    )
    ttl = _CACHE_TTL_DATED if close_date else _CACHE_TTL_LATEST

    cached = _cache.get(key)
    if cached is not None and time.monotonic() - cached[0] < ttl:
        return cached[1]

    with _lock_for(key):
        # Re-check under the lock: a concurrent request may have just priced it.
        cached = _cache.get(key)
        if cached is not None and time.monotonic() - cached[0] < ttl:
            return cached[1]

        raw_df = _load_raw(close_date)
        if ticker is not None:
            raw_df = raw_df.filter(pl.col("underlying_symbol") == ticker)

        default_vol = float(os.getenv("DEFAULT_VOLATILITY", "0.25"))
        use_remote_vol = vol_mode == "historical"

        loader = CBOEOptionsData(
            as_of=close_date,  # None → loader derives the date from the data's quote_date
            default_vol=default_vol,
            use_remote_vol=use_remote_vol,
            dataframe=raw_df,
        )

        df_opts = apply_model_inputs(
            loader.get_data(),
            vol_mode=vol_mode,
            dividends=parse_dividends(dividends_json),
            rate_curve=parse_rate_curve(rate_curve_json),
            default_vol=default_vol,
            carry_mode=carry_mode,
            dividend_schedule=parse_dividend_schedule(dividend_schedule_json),
        )

        try:
            prices = OptionsPrices(df_opts, model=pricing_model)
        except TypeError:
            prices = OptionsPrices(df_opts)
            setattr(prices, "model", pricing_model)

        priced = add_greeks(add_probabilities(prices.price_options()))
        _cache[key] = (time.monotonic(), priced)
        return priced


def list_available_dates() -> list[str]:
    source = load_from_env()
    if not isinstance(source, GCSClosesDataSource):
        return []
    return [d.strftime("%Y-%m-%d") for d in source.list_available_dates()]


def get_vol_history(ticker: str, close_date: str | None, window: int = 30) -> list[dict]:
    """Trailing-30d realized vol as stamped by ingest on each available day.

    Each day's closes file carries the vendor-computed 30d realized vol as of
    that day, so walking the last `window` available dates gives a genuine
    day-by-day series of how that trailing figure has moved — not a replay of
    the 30 daily returns behind a single reading.
    """
    dates = list_available_dates()
    if dates:
        dates = sorted(dates)
        if close_date:
            dates = [d for d in dates if d <= close_date]
        dates = dates[-window:]
    else:
        # No dated history available (e.g. local/dev source) — just the one snapshot.
        dates = [close_date]

    points: list[dict] = []
    for d in dates:
        raw = _load_raw(d)
        if "hist_vol_30d" not in raw.columns:
            continue  # older/foreign closes files predate this column
        raw = raw.filter(pl.col("underlying_symbol") == ticker)
        if raw.height == 0:
            continue
        vol = raw["hist_vol_30d"][0]
        if vol is None:
            continue
        points.append({"date": str(raw["quote_date"][0]), "vol30d": float(vol)})
    return points


def data_source_type() -> str:
    return os.getenv("DATA_SOURCE_TYPE", "local").lower()
