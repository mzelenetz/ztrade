from __future__ import annotations

from pydantic import BaseModel


class MagicLinkRequest(BaseModel):
    email: str


class MagicLinkResponse(BaseModel):
    # Deliberately generic: never reveals whether the email is on the allowlist.
    sent: bool = True


class VerifyRequest(BaseModel):
    token: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MeResponse(BaseModel):
    username: str


class MetaResponse(BaseModel):
    dataSourceType: str
    availableDates: list[str]
    tickers: list[str]
