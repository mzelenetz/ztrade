from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.api.data import data_source_type, list_available_dates, list_tickers
from src.api.deps import get_current_username
from src.api.schemas import MetaResponse

router = APIRouter(prefix="/api/meta", tags=["meta"])


@router.get("", response_model=MetaResponse)
def get_meta(
    pricing_model: str = Query("mzpricer"),
    close_date: str | None = Query(None),
    username: str = Depends(get_current_username),
) -> MetaResponse:
    available_dates = list_available_dates()
    resolved_date = close_date or (available_dates[-1] if available_dates else None)

    return MetaResponse(
        dataSourceType=data_source_type(),
        availableDates=available_dates,
        tickers=list_tickers(resolved_date),
    )
