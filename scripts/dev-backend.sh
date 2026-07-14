#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export DATA_SOURCE_TYPE=local
# Real (internally consistent) close data; sample_NVDA.csv has shifted dates
export DATA_SOURCE_PATH=tests/fixtures/closes-nvda-2026-01-29.csv
export JWT_SECRET_KEY=dev-only-secret
# Magic-link target is the vite dev server; with no SMTP_PASSWORD set the
# backend logs the link to stdout instead of emailing it.
export APP_BASE_URL="http://localhost:${FRONTEND_PORT:-5173}"
# Non-personal dev admin; real admins are set via ADMIN_EMAILS on the service.
# Extra allowed users are stored in this local file (gitignored) for dev.
export ADMIN_EMAILS="${ADMIN_EMAILS:-dev@ztrade.local}"
export ALLOWED_USERS_FILE="${ALLOWED_USERS_FILE:-.local/allowed_users.json}"
exec .venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port "${BACKEND_PORT:-8001}" --reload
