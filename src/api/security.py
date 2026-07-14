from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone

import jwt

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 12


def _magic_ttl_minutes() -> int:
    return int(os.getenv("MAGIC_LINK_TTL_MINUTES", "15"))


# Falls back to a random per-process secret for local dev; set JWT_SECRET_KEY in
# any environment where tokens need to survive a process restart or be shared
# across instances.
SECRET_KEY = os.getenv("JWT_SECRET_KEY") or secrets.token_urlsafe(32)


def _encode(payload: dict, expire: datetime) -> str:
    return jwt.encode({**payload, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def create_access_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return _encode({"sub": username, "purpose": "access"}, expire)


def decode_access_token(token: str) -> str | None:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None
    # A magic-link token must never be accepted as a session token.
    if payload.get("purpose") != "access":
        return None
    return payload.get("sub")


def create_magic_token(email: str) -> str:
    """Short-lived, single-purpose token embedded in the emailed sign-in link."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=_magic_ttl_minutes())
    return _encode({"sub": email, "purpose": "magic"}, expire)


def verify_magic_token(token: str) -> str | None:
    """Return the email the link was minted for, or None if invalid/expired."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None
    if payload.get("purpose") != "magic":
        return None
    return payload.get("sub")
