from __future__ import annotations

import os

from src.auth.allow_store import load_extra_emails, save_extra_emails


def normalize(email: str) -> str:
    return email.strip().lower()


def admin_emails() -> set[str]:
    """Always-allowed addresses that may also manage the allowed-user list.
    Set ADMIN_EMAILS (comma-separated) on the service; never checked into git."""
    raw = os.getenv("ADMIN_EMAILS", "")
    return {normalize(e) for e in raw.split(",") if e.strip()}


def extra_emails() -> list[str]:
    """Non-admin allowed addresses, managed from the Settings page."""
    return sorted({normalize(e) for e in load_extra_emails()})


def set_extra_emails(emails: list[str]) -> list[str]:
    admins = admin_emails()
    cleaned = sorted(
        {normalize(e) for e in emails if "@" in e.strip()} - admins
    )
    save_extra_emails(cleaned)
    return cleaned


def is_admin(email: str) -> bool:
    return normalize(email) in admin_emails()


def is_allowed(email: str) -> bool:
    e = normalize(email)
    return e in admin_emails() or e in set(extra_emails())
