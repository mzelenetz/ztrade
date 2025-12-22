from __future__ import annotations

import io
import os
from dataclasses import dataclass
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
        try:
            from google.cloud import storage
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "google-cloud-storage is required for GCS support. "
                "Install it with `pip install google-cloud-storage`"
            ) from exc

        client = storage.Client()
        bucket = client.bucket(self.bucket)
        blob = bucket.blob(self.blob_name)
        data = blob.download_as_bytes()
        return pl.read_csv(io.BytesIO(data))


@dataclass
class DataSourceConfig:
    type: str = "local"
    path: str = "options/data/sample_options.csv"
    s3_bucket: Optional[str] = None
    s3_key: Optional[str] = None
    gcs_bucket: Optional[str] = None
    gcs_blob: Optional[str] = None


def load_from_env() -> DataSource:
    """Build a data source based on environment variables."""

    source_type = os.getenv("DATA_SOURCE_TYPE", "local").lower()

    if source_type == "s3":
        bucket = os.environ["DATA_SOURCE_BUCKET"]
        key = os.environ["DATA_SOURCE_KEY"]
        return S3CSVDataSource(bucket=bucket, key=key)

    if source_type == "gcs":
        bucket = os.environ["DATA_SOURCE_BUCKET"]
        blob = os.environ["DATA_SOURCE_KEY"]
        return GCSCSVDataSource(bucket=bucket, blob_name=blob)

    path = os.getenv("DATA_SOURCE_PATH", DataSourceConfig().path)
    return LocalCSVDataSource(path=path)
