from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.api.data import list_tickers
from src.api.deps import get_current_username
from src.api.ideas_scan import run_scan
from src.services.ideas_service import build_ideas

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
    pricing = (
        pricing_model, close_date, vol_mode, dividends, rate_curve,
        carry_mode, dividend_schedule,
    )
    spread_args = (
        metric, delta_min, delta_max, max_contract_ratio, max_straddle_ratio,
        min_option_price, max_last_price, max_abs_net_delta, max_legs_per_side,
        100, margin_rate, margin_style,
    )
    spreads_by_ticker = run_scan(list_tickers(close_date), pricing, spread_args)
    return {"ideas": build_ideas(spreads_by_ticker, min_open_interest)}
