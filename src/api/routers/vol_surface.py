from __future__ import annotations

import polars as pl
from fastapi import APIRouter, Depends, Query

from src.api.data import get_priced_data
from src.api.deps import get_current_username

router = APIRouter(prefix="/api/vol-surface", tags=["vol-surface"])


@router.get("")
def get_vol_surface(
    ticker: str = Query(...),
    pricing_model: str = Query("mzpricer"),
    close_date: str | None = Query(None),
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

    expiries = []
    for (expiry,), group in df.sort("Expiry").group_by("Expiry", maintain_order=True):
        group = group.sort("Strike")

        points = [
            {
                "strike": row["Strike"],
                "marketIv": row["MarketIV"],
                "type": row["Type"],
            }
            for row in group.to_dicts()
            if row.get("MarketIV") is not None
        ]

        # Only strikes the smile actually fit — wings priced off their own market
        # IV are not part of the model's line.
        fitted = [
            {"strike": row["Strike"], "iv": row["FittedVol"]}
            for row in group.filter(pl.col("VolFromSurface"))
            .unique(subset="Strike", maintain_order=True)
            .to_dicts()
            if row.get("FittedVol") is not None
        ]

        fallback = not fitted

        first = group.row(0, named=True)
        expiries.append(
            {
                "expiry": expiry.strftime("%Y-%m-%d"),
                "spot": float(group["Spot"][0]),
                "points": points,
                "fitted": fitted,
                "fallback": fallback,
                "rate": float(first["Rate"]),
                "divYield": float(first["DividendYield"]),
                "divSource": first["DivSource"],
            }
        )

    return {"ticker": ticker, "expiries": expiries}
