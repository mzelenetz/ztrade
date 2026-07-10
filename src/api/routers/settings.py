from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.data import data_source_type, list_available_dates
from src.api.deps import get_current_username
from src.ingest.job import bucket_name, load_universe

router = APIRouter(prefix="/api/settings", tags=["settings"])


class TickersPayload(BaseModel):
    tickers: list[str]


def _require_gcs() -> None:
    if data_source_type() not in ("gcs_closes", "gcs", "gcs_latest"):
        raise HTTPException(
            status_code=501,
            detail="Ingest settings need the GCS data source (local dev uses a fixture file)",
        )


@router.get("/ingest")
def get_ingest_settings(username: str = Depends(get_current_username)) -> dict:
    dates = list_available_dates()
    return {
        "tickers": load_universe(),
        "latestCloseDate": dates[-1] if dates else None,
        "bucket": bucket_name(),
        "editable": data_source_type() in ("gcs_closes", "gcs", "gcs_latest"),
    }


@router.put("/ingest/tickers")
def put_tickers(payload: TickersPayload, username: str = Depends(get_current_username)) -> dict:
    _require_gcs()
    tickers = sorted({t.strip().upper() for t in payload.tickers if t.strip()})
    if not tickers:
        raise HTTPException(status_code=422, detail="At least one ticker is required")

    from google.cloud import storage

    blob = storage.Client().bucket(bucket_name()).blob("tickers.txt")
    blob.upload_from_string("\n".join(tickers) + "\n", content_type="text/plain")
    return {"tickers": tickers}


@router.post("/ingest/run")
def run_ingest_job(username: str = Depends(get_current_username)) -> dict:
    """Kick off the ztrade-ingest Cloud Run Job (the same one the scheduler
    triggers). Returns immediately; the new closes file appears in the date
    picker a few minutes later."""
    _require_gcs()

    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    project = os.getenv("GOOGLE_CLOUD_PROJECT", "pcrpal")
    region = os.getenv("INGEST_JOB_REGION", "us-central1")
    job = os.getenv("INGEST_JOB_NAME", "ztrade-ingest")

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    session = AuthorizedSession(credentials)
    resp = session.post(
        f"https://run.googleapis.com/v2/projects/{project}/locations/{region}/jobs/{job}:run",
        json={},
        timeout=30,
    )
    if resp.status_code >= 300:
        raise HTTPException(
            status_code=502,
            detail=f"Could not start ingest job ({resp.status_code}): {resp.text[:200]}",
        )

    execution = resp.json().get("metadata", {}).get("name", "")
    return {"started": True, "execution": execution.rsplit("/", 1)[-1]}
