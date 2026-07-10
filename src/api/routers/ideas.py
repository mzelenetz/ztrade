from __future__ import annotations

import polars as pl
from fastapi import APIRouter, Depends, Query

from src.api.data import get_priced_data, list_tickers
from src.api.deps import get_current_username
from src.services.ideas_service import build_ideas
from src.services.spreads_service import build_spreads

router = APIRouter(prefix="/api/ideas", tags=["ideas"])


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
    margin_rate: float = Query(0.11325, ge=0, le=1),
    margin_style: str = Query("reg_t", pattern="^(reg_t|portfolio)$"),
    vol_mode: str = Query("surface", pattern="^(surface|flat|historical)$"),
    carry_mode: str = Query("implied", pattern="^(implied|manual)$"),
    dividend_schedule: str = Query(""),
    dividends: str = Query(""),
    rate_curve: str = Query(""),
    username: str = Depends(get_current_username),
) -> dict:
    spreads_by_ticker: dict[str, list] = {}
    for ticker in list_tickers(close_date):
        try:
            df = get_priced_data(
                pricing_model, close_date, vol_mode, dividends, rate_curve,
                ticker=ticker, carry_mode=carry_mode, dividend_schedule_json=dividend_schedule,
            ).filter(pl.col("Ticker") == ticker)
            spreads_by_ticker[ticker] = build_spreads(
                df, metric, delta_min, delta_max, max_contract_ratio,
                max_straddle_ratio, min_option_price, max_last_price,
                max_abs_net_delta, max_legs_per_side, 100,
                margin_rate, margin_style,
            )
        except Exception:
            continue  # one bad ticker must not empty the whole board

    return {"ideas": build_ideas(spreads_by_ticker)}
