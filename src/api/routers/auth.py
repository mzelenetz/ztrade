from __future__ import annotations

import os
import traceback

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.deps import get_current_username
from src.api.schemas import (
    MagicLinkRequest,
    MagicLinkResponse,
    MeResponse,
    TokenResponse,
    VerifyRequest,
)
from src.api.security import create_access_token, create_magic_token, verify_magic_token
from src.auth.allowlist import is_allowed, normalize
from src.auth.email import send_magic_link

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _app_base_url() -> str:
    # Where the SPA lives, used to build the clickable link. Same origin as the
    # API in prod (FastAPI serves the built frontend); the vite dev server in
    # local dev. No trailing slash.
    return os.getenv("APP_BASE_URL", "http://localhost:5173").rstrip("/")


@router.post("/magic/request", response_model=MagicLinkResponse)
def request_magic_link(payload: MagicLinkRequest) -> MagicLinkResponse:
    email = normalize(payload.email)
    # Always answer the same way so the endpoint can't be used to probe which
    # addresses are allowed. Only actually send when the email is on the list.
    if is_allowed(email):
        link = f"{_app_base_url()}/auth/verify?token={create_magic_token(email)}"
        try:
            send_magic_link(email, link)
        except Exception:
            # Don't leak SMTP errors (or the allowlist hit) to the caller.
            traceback.print_exc()
    return MagicLinkResponse(sent=True)


@router.post("/magic/verify", response_model=TokenResponse)
def verify_magic_link(payload: VerifyRequest) -> TokenResponse:
    email = verify_magic_token(payload.token)
    # Re-check the allowlist at verify time so revoking access takes effect even
    # for links already in flight.
    if email is None or not is_allowed(email):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="This sign-in link is invalid or has expired.",
        )
    return TokenResponse(access_token=create_access_token(email))


@router.get("/me", response_model=MeResponse)
def me(username: str = Depends(get_current_username)) -> MeResponse:
    return MeResponse(username=username)
