"""Cross-ticker ideas scan on a process pool, plus a background cache warmer.

The per-ticker pipeline (polars prep, carry fitting, pricing, greeks) is
GIL-bound Python — threads can't parallelize it and extra vCPUs go idle.
Worker processes sidestep the GIL, so the scan scales with the CPUs Cloud Run
actually provides. Workers are persistent: each keeps its own priced-data
cache (src.api.data._cache), so repeat scans inside the TTL are near-free.

PricingArgs mirrors get_priced_data's positional signature; SpreadArgs mirrors
build_spreads' tail after the DataFrame.
"""

from __future__ import annotations

import os
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context

import polars as pl

from src.api import data as api_data
from src.services.spreads_service import build_spreads

# (pricing_model, close_date, vol_mode, dividends_json, rate_curve_json,
#  carry_mode, dividend_schedule_json)
PricingArgs = tuple[str, str | None, str, str, str, str, str]
# (metric, delta_min, delta_max, max_contract_ratio, max_straddle_ratio,
#  min_option_price, max_last_price, max_abs_net_delta, max_legs_per_side,
#  max_results, margin_rate, margin_style)
SpreadArgs = tuple

_pool: ProcessPoolExecutor | None = None
_pool_guard = threading.Lock()


def _worker_count() -> int:
    return int(os.getenv("IDEAS_SCAN_WORKERS", str(min(4, os.cpu_count() or 1))))


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    with _pool_guard:
        if _pool is None:
            # spawn, not fork: the server process is multi-threaded (uvicorn,
            # warmer) and forking a threaded process risks inherited-lock
            # deadlocks. Workers are long-lived so the one-time import cost
            # of spawn is paid once.
            _pool = ProcessPoolExecutor(
                max_workers=_worker_count(), mp_context=get_context("spawn")
            )
        return _pool


def _dispose_pool() -> None:
    global _pool
    with _pool_guard:
        if _pool is not None:
            _pool.shutdown(wait=False, cancel_futures=True)
            _pool = None


def scan_ticker(ticker: str, pricing: PricingArgs, spread_args: SpreadArgs) -> list[dict]:
    """Price one ticker and build its candidate spreads. Runs in a worker."""
    model, close_date, vol_mode, dividends, rate_curve, carry_mode, schedule = pricing
    df = api_data.get_priced_data(
        model, close_date, vol_mode, dividends, rate_curve,
        ticker=ticker, carry_mode=carry_mode, dividend_schedule_json=schedule,
    ).filter(pl.col("Ticker") == ticker)
    return build_spreads(df, *spread_args)


def run_scan(
    tickers: list[str], pricing: PricingArgs, spread_args: SpreadArgs
) -> dict[str, list]:
    """Scan every ticker on the process pool; one bad ticker is skipped, a
    broken pool (e.g. an OOM-killed worker) falls back to a sequential
    in-process scan so the request still answers."""
    try:
        pool = _get_pool()
        futures = {pool.submit(scan_ticker, t, pricing, spread_args): t for t in tickers}
        results: dict[str, list] = {}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                results[ticker] = future.result()
            except BrokenProcessPool:
                raise
            except Exception:
                continue  # one bad ticker must not empty the whole board
        return results
    except BrokenProcessPool:
        _dispose_pool()
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = scan_ticker(ticker, pricing, spread_args)
            except Exception:
                continue
        return results


# ---------------------------------------------------------------------------
# Background warmer: keeps the default-settings scan hot so the first user
# request of the morning (or after the pricing cache TTL lapses, or after a
# new closes file lands) doesn't pay the full cold scan.

_WARM_INTERVAL_SECONDS = 600

# Must mirror the /api/ideas router defaults (which mirror the frontend's).
_DEFAULT_SPREAD_ARGS: SpreadArgs = (
    "Last", 5.0, 95.0, 2.5, 1.5, 2.0, 1000.0, 10.0, 50, 100, 0.11325, "reg_t",
)


def _warm_once() -> None:
    dates = api_data.list_available_dates()
    close_date = dates[-1] if dates else None
    pricing: PricingArgs = ("mzpricer", close_date, "surface", "", "", "implied", "")
    tickers = api_data.list_tickers(close_date)
    run_scan(tickers, pricing, _DEFAULT_SPREAD_ARGS)


def _warm_loop() -> None:
    import time

    while True:
        try:
            _warm_once()
        except Exception:
            traceback.print_exc()
        time.sleep(_WARM_INTERVAL_SECONDS)


def start_warmer() -> None:
    threading.Thread(target=_warm_loop, daemon=True, name="ideas-warmer").start()
