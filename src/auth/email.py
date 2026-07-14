from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage


class EmailNotConfigured(RuntimeError):
    """Raised when a real send is attempted but no SMTP credentials are set."""


def _smtp_config() -> dict[str, str]:
    return {
        "host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
        "port": os.getenv("SMTP_PORT", "587"),
        "username": os.getenv("SMTP_USERNAME", ""),
        "password": os.getenv("SMTP_PASSWORD", ""),
        # From defaults to the authenticating account (what Gmail expects).
        "from": os.getenv("SMTP_FROM", os.getenv("SMTP_USERNAME", "")),
    }


def _render(email: str, link: str) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = "Your ZTrade sign-in link"
    msg["From"] = _smtp_config()["from"]
    msg["To"] = email
    msg.set_content(
        "Click the link below to sign in to ZTrade. It expires shortly and can "
        f"only be used once for this request.\n\n{link}\n\n"
        "If you didn't request this, you can ignore this email."
    )
    return msg


def send_magic_link(email: str, link: str) -> None:
    """Email the sign-in link. With no SMTP password configured (local dev), the
    link is logged to stdout instead of sent, so the flow is testable offline."""
    cfg = _smtp_config()
    if not cfg["password"]:
        print(f"[magic-link] SMTP not configured; link for {email}: {link}", flush=True)
        return

    msg = _render(email, link)
    context = ssl.create_default_context()
    with smtplib.SMTP(cfg["host"], int(cfg["port"]), timeout=15) as server:
        server.starttls(context=context)
        server.login(cfg["username"], cfg["password"])
        server.send_message(msg)
