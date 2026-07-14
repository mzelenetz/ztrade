"""Persistence for the non-admin allowed-user list.

Mirrors how ingest settings are stored: a small blob in the GCS bucket in the
deployed app, a local JSON file for dev (DATA_SOURCE_TYPE=local). Admins are NOT
stored here — they come from the ADMIN_EMAILS env var.
"""

from __future__ import annotations

import json
import os

from src.api.data import data_source_type

_BLOB_NAME = "allowed_users.json"


def _is_gcs() -> bool:
    return data_source_type() in ("gcs_closes", "gcs", "gcs_latest")


def _local_path() -> str:
    return os.getenv("ALLOWED_USERS_FILE", ".local/allowed_users.json")


def load_extra_emails() -> list[str]:
    if _is_gcs():
        from google.cloud import storage
        from google.cloud.exceptions import NotFound

        from src.ingest.job import bucket_name

        blob = storage.Client().bucket(bucket_name()).blob(_BLOB_NAME)
        try:
            return json.loads(blob.download_as_text())
        except NotFound:
            return []

    path = _local_path()
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def save_extra_emails(emails: list[str]) -> None:
    payload = json.dumps(emails)
    if _is_gcs():
        from google.cloud import storage

        from src.ingest.job import bucket_name

        blob = storage.Client().bucket(bucket_name()).blob(_BLOB_NAME)
        blob.upload_from_string(payload, content_type="application/json")
        return

    path = _local_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        f.write(payload)
