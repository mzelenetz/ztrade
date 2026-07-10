from __future__ import annotations

from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MeResponse(BaseModel):
    username: str


class MetaResponse(BaseModel):
    dataSourceType: str
    availableDates: list[str]
    tickers: list[str]
