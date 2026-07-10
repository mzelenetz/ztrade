from __future__ import annotations

from math import exp

import polars as pl
from fastapi import APIRouter, Depends, Query

from src.api.data import get_priced_data, parse_dividend_schedule
from src.api.deps import get_current_username
from src.services.model_inputs import dividend_pv, fit_implied_forward

router = APIRouter(prefix="/api/dividends", tags=["dividends"])

SEED_MIN_JUMP = 0.02  # implied PV jumps smaller than this are noise, not dividends


@router.get("")
def get_dividends(
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
        pricing_model, close_date, vol_mode, dividends, rate_curve,
        ticker=ticker, carry_mode=carry_mode, dividend_schedule_json=dividend_schedule,
    ).filter(pl.col("Ticker") == ticker)

    schedule = parse_dividend_schedule(dividend_schedule).get(ticker) or []
    valuation = df["ValuationTime"].max().date()

    expiries = []
    for (expiry,), group in df.sort("Expiry").group_by("Expiry", maintain_order=True):
        first = group.row(0, named=True)
        spot = float(first["Spot"])
        tenor = float(first["T"])
        rate = float(first["Rate"])

        forward = fit_implied_forward(
            strikes=group["Strike"].to_list(),
            types=group["Type"].to_list(),
            mids=group["Mid"].to_list(),
            spot=spot,
            tenor=tenor,
            rate=rate,
        )
        implied_pv = spot - forward * exp(-rate * tenor) if forward is not None else None

        expiries.append(
            {
                "expiry": expiry.strftime("%Y-%m-%d"),
                "tYears": tenor,
                "spot": spot,
                "impliedForward": forward,
                "impliedDivPV": implied_pv,
                "scheduledDivPV": dividend_pv(schedule, valuation, expiry.date(), rate),
                "divSource": first["DivSource"],
            }
        )

    # Seed suggestions: a jump in implied PV between consecutive expiries marks
    # a dividend somewhere in that window.
    seeds = []
    prev_pv = 0.0
    prev_expiry = valuation.strftime("%Y-%m-%d")
    for e in expiries:
        pv = e["impliedDivPV"]
        if pv is None:
            continue
        jump = pv - prev_pv
        if jump > SEED_MIN_JUMP:
            seeds.append(
                {
                    "windowStart": prev_expiry,
                    "windowEnd": e["expiry"],
                    "amount": round(jump, 4),
                }
            )
        prev_pv = pv
        prev_expiry = e["expiry"]

    return {
        "ticker": ticker,
        "valuationDate": valuation.strftime("%Y-%m-%d"),
        "expiries": expiries,
        "seedSuggestions": seeds,
    }
