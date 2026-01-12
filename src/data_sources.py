from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Protocol

import polars as pl


class DataSource(Protocol):
    """Protocol for loading option data from different backends."""

    def load(self) -> pl.DataFrame:
        ...


@dataclass
class LocalCSVDataSource:
    """Loads option data from a local CSV file."""

    path: str

    def load(self) -> pl.DataFrame:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Local data source not found: {self.path}")
        return pl.read_csv(self.path)


@dataclass
class S3CSVDataSource:
    """Loads option data from an S3 bucket/key pair."""

    bucket: str
    key: str
    client: Optional[object] = None

    def load(self) -> pl.DataFrame:
        import boto3

        client = self.client or boto3.client("s3")
        obj = client.get_object(Bucket=self.bucket, Key=self.key)
        body = obj["Body"].read()
        return pl.read_csv(io.BytesIO(body))


@dataclass
class GCSCSVDataSource:
    """Loads option data from a Google Cloud Storage bucket."""

    bucket: str
    blob_name: str

    def load(self) -> pl.DataFrame:
        client = _gcs_client()
        bucket = client.bucket(self.bucket)
        blob = bucket.blob(self.blob_name)
        data = blob.download_as_bytes()
        return pl.read_csv(io.BytesIO(data))


@dataclass
class GCSClosesDataSource:
    """Loads daily closing data stored as closes-<date>.csv in GCS.

    Picks the latest *date encoded in the filename* (not by upload time).
    """

    bucket: str
    prefix: str = "closes-"
    extension: str = ".csv"

    def _parse_date(self, blob_name: str) -> Optional[datetime.date]:
        pattern = rf"^{re.escape(self.prefix)}(\d{{4}}-\d{{2}}-\d{{2}}){re.escape(self.extension)}$"
        match = re.match(pattern, blob_name)
        if not match:
            return None
        return datetime.strptime(match.group(1), "%Y-%m-%d").date()

    def list_available_dates(self) -> list[datetime.date]:
        client = _gcs_client()
        bucket = client.bucket(self.bucket)

        dates: list[datetime.date] = []
        for blob in client.list_blobs(bucket, prefix=self.prefix):
            parsed = self._parse_date(blob.name)
            if parsed:
                dates.append(parsed)

        return sorted(dates)

    def latest_date(self) -> datetime.date:
        dates = self.list_available_dates()
        if not dates:
            raise FileNotFoundError(
                f"No closing files found in gs://{self.bucket} with prefix '{self.prefix}'"
            )
        return dates[-1]

    def blob_name_for_date(self, date: datetime.date) -> str:
        return f"{self.prefix}{date:%Y-%m-%d}{self.extension}"

    def load_for_date(self, date: datetime.date) -> pl.DataFrame:
        client = _gcs_client()
        bucket = client.bucket(self.bucket)
        blob = bucket.blob(self.blob_name_for_date(date))
        data = blob.download_as_bytes()
        return pl.read_csv(io.BytesIO(data))

    def load(self) -> pl.DataFrame:
        return self.load_for_date(self.latest_date())


@dataclass
class GCSLatestObjectDataSource:
    """Loads the most recently updated object in a GCS bucket (optionally under a prefix).

    This does NOT require providing an explicit object key.
    """

    bucket: str
    prefix: str = ""
    # If True, uses blob.updated; otherwise uses blob.time_created
    use_updated_time: bool = True

    def latest_blob_name(self) -> str:
        client = _gcs_client()
        bucket = client.bucket(self.bucket)

        latest_blob_name: Optional[str] = None
        latest_ts = None

        for blob in client.list_blobs(bucket, prefix=self.prefix):
            ts = blob.updated if self.use_updated_time else blob.time_created
            if ts is None:
                continue
            if latest_ts is None or ts > latest_ts:
                latest_ts = ts
                latest_blob_name = blob.name

        if latest_blob_name is None:
            where = f"gs://{self.bucket}/{self.prefix}" if self.prefix else f"gs://{self.bucket}"
            raise FileNotFoundError(f"No objects found in {where}")

        return latest_blob_name

    def load(self) -> pl.DataFrame:
        client = _gcs_client()
        bucket = client.bucket(self.bucket)

        blob_name = self.latest_blob_name()
        blob = bucket.blob(blob_name)
        data = blob.download_as_bytes()
        return pl.read_csv(io.BytesIO(data))


def _gcs_client():
    try:
        from google.cloud import storage
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "google-cloud-storage is required for GCS support. "
            "Install it with `pip install google-cloud-storage`"
        ) from exc
    return storage.Client()


@dataclass
class DataSourceConfig:
    type: str = "local"
    path: str = "src/data/sample_NVDA.csv"
    s3_bucket: Optional[str] = None
    s3_key: Optional[str] = None
    gcs_bucket: Optional[str] = None
    gcs_blob: Optional[str] = None


def load_from_env() -> DataSource:
    """Build a data source based on environment variables.

    Supported:
      - local
      - s3
      - gcs
      - gcs_closes          (latest date from closes-YYYY-MM-DD.csv)
      - gcs_latest          (most recently updated object, optional prefix)
    """

    source_type = os.getenv("DATA_SOURCE_TYPE", "local").lower()

    if source_type == "s3":
        bucket = os.environ["DATA_SOURCE_BUCKET"]
        key = os.environ["DATA_SOURCE_KEY"]
        return S3CSVDataSource(bucket=bucket, key=key)

    if source_type == "gcs":
        bucket = os.environ["DATA_SOURCE_BUCKET"]
        blob = os.environ["DATA_SOURCE_KEY"]
        return GCSCSVDataSource(bucket=bucket, blob_name=blob)

    if source_type == "gcs_closes":
        bucket = os.getenv("GCS_CLOSES_BUCKET", "ztrade-yesterday-closes")
        prefix = os.getenv("GCS_CLOSES_PREFIX", "closes-")
        extension = os.getenv("GCS_CLOSES_EXTENSION", ".csv")
        return GCSClosesDataSource(bucket=bucket, prefix=prefix, extension=extension)

    if source_type == "gcs_latest":
        bucket = os.environ.get("GCS_LATEST_BUCKET") or os.environ.get("DATA_SOURCE_BUCKET")
        if not bucket:
            raise KeyError(
                "GCS latest requires GCS_LATEST_BUCKET or DATA_SOURCE_BUCKET to be set"
            )
        prefix = os.getenv("GCS_LATEST_PREFIX", "")
        use_updated = os.getenv("GCS_LATEST_USE_UPDATED", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        return GCSLatestObjectDataSource(
            bucket=bucket, prefix=prefix, use_updated_time=use_updated
        )

    path = os.getenv("DATA_SOURCE_PATH", DataSourceConfig().path)
    return LocalCSVDataSource(path=path)
