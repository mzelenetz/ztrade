from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
from fastapi import APIRouter, Depends, Query

from src.api.data import get_priced_data, list_tickers
from src.api.deps import get_current_username
from src.services.ideas_service import build_ideas
from src.services.spreads_service import build_spreads

router = APIRouter(prefix="/api/ideas", tags=["ideas"])

# Bounded rather than one-thread-per-ticker: pricing releases the GIL (mzpricer
# is a Rust extension, polars ops are native), so this buys real wall-clock
# parallelism without letting a large universe spawn unbounded threads.
MAX_WORKERS = 8


@router.get("")
def get_ideas(
    pricing_model: str = Query("mzpricer"),
    close_date: str | None = Query(None),
    metric: str = Query("Last"),
    delta_min: float = Query(5),
    delta_max: float = Query(95),
    max_contract_ratio: float = Query(2.5),
    max_straddle_ratio: float = Query(1.5),
    min_option_price: float = Query(2.0),
    max_last_price: float = Query(1000.0),
    max_abs_net_delta: float = Query(10.0),
    max_legs_per_side: int = Query(50),
    min_open_interest: float = Query(50, ge=0),
    margin_rate: float = Query(0.11325, ge=0, le=1),
    margin_style: str = Query("reg_t", pattern="^(reg_t|portfolio)$"),
    vol_mode: str = Query("surface", pattern="^(surface|realized_anchor|flat|historical)$"),
    carry_mode: str = Query("implied", pattern="^(implied|manual)$"),
    dividend_schedule: str = Query(""),
    dividends: str = Query(""),
    rate_curve: str = Query(""),
    username: str = Depends(get_current_username),
) -> dict:
    def price_one(ticker: str) -> list:
        df = get_priced_data(
            pricing_model, close_date, vol_mode, dividends, rate_curve,
            ticker=ticker, carry_mode=carry_mode, dividend_schedule_json=dividend_schedule,
        ).filter(pl.col("Ticker") == ticker)
        return build_spreads(
            df, metric, delta_min, delta_max, max_contract_ratio,
            max_straddle_ratio, min_option_price, max_last_price,
            max_abs_net_delta, max_legs_per_side, 100,
            margin_rate, margin_style,
        )

    tickers = list_tickers(close_date)
    spreads_by_ticker: dict[str, list] = {}
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tickers) or 1)) as pool:
        future_to_ticker = {pool.submit(price_one, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                spreads_by_ticker[ticker] = future.result()
            except Exception:
                continue  # one bad ticker must not empty the whole board

    return {"ideas": build_ideas(spreads_by_ticker, min_open_interest)}
