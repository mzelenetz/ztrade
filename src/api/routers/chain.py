from __future__ import annotations

import polars as pl
from fastapi import APIRouter, Depends, Query

from src.api.data import get_priced_data
from src.api.deps import get_current_username
from src.services.chain_service import build_chain

router = APIRouter(prefix="/api/chain", tags=["chain"])


@router.get("")
def get_chain(
    ticker: str = Query(...),
    pricing_model: str = Query("mzpricer"),
    close_date: str | None = Query(None),
    metric: str = Query("Last"),
    delta_min: float = Query(5),
    delta_max: float = Query(95),
    min_price: float = Query(2.0),
    vol_mode: str = Query("surface", pattern="^(surface|flat|historical)$"),
    carry_mode: str = Query("implied", pattern="^(implied|manual)$"),
    dividend_schedule: str = Query(""),
    dividends: str = Query(""),
    rate_curve: str = Query(""),
    username: str = Depends(get_current_username),
) -> dict:
    df = get_priced_data(
        pricing_model, close_date, vol_mode, dividends, rate_curve, ticker=ticker, carry_mode=carry_mode, dividend_schedule_json=dividend_schedule
    ).filter(pl.col("Ticker") == ticker)

    spot = float(df["Spot"][0])
    dividend_yield = float(df["DividendYield"].mean())
    vol30d = float(df["Vol30d"][0])
    valuation_date = df["ValuationTime"].max().strftime("%Y-%m-%d")

    chain = build_chain(df, metric, delta_min, delta_max, min_price)

    return {
        "ticker": ticker,
        "spot": spot,
        "dividendYield": dividend_yield,
        "vol30d": vol30d,
        "valuationDate": valuation_date,
        **chain,
    }
