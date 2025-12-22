# ZTrade Options Viewer

A Streamlit dashboard for reviewing option chains with paired call/put views, simple authentication, and flexible data loading that can pull from local files or cloud buckets.

## Features
- Authenticated access with environment-configurable users (default `admin` / `demo123`).
- Side-by-side call/put data shown on a single line per strike for each expiry.
- Data-loading abstraction supporting local CSVs, Amazon S3, or Google Cloud Storage.
- Dockerfile for repeatable cloud deployments.

## Getting Started
1. Install dependencies (Python 3.12+):
   ```bash
   pip install --no-cache-dir .
   ```
2. Run the Streamlit UI:
   ```bash
   streamlit run options/ui.py
   ```
3. Log in with the default credentials (`admin` / `demo123`) or configure your own users via environment variables.

### Authentication
- By default the app seeds a single `admin` user with password `demo123`.
- To define your own users, set `APP_USERS` to a JSON object mapping usernames to plaintext passwords, e.g.:
  ```bash
  export APP_USERS='{"alice": "p@ssword", "bob": "hunter2"}'
  ```

### Configuring Data Sources
The app loads option data through a pluggable data source defined by environment variables:
- `DATA_SOURCE_TYPE` (default `local`): one of `local`, `s3`, `gcs`.
- Local CSV: set `DATA_SOURCE_PATH` (defaults to `options/data/sample_options.csv`).
- S3 bucket: set `DATA_SOURCE_TYPE=s3`, `DATA_SOURCE_BUCKET=<bucket>`, `DATA_SOURCE_KEY=<object-key>`.
- GCS bucket: set `DATA_SOURCE_TYPE=gcs`, `DATA_SOURCE_BUCKET=<bucket>`, `DATA_SOURCE_KEY=<blob-name>`, and install `google-cloud-storage`.
- Additional knobs: `DATA_AS_OF_DATE` (YYYY-MM-DD), `DEFAULT_VOLATILITY` (fallback vol), `USE_REMOTE_VOL` (`true`/`false` to toggle yfinance lookups).

### Docker
Build and run the containerized app:
```bash
docker build -t ztrade-app .
docker run --rm -p 8501:8501 -e APP_USERS='{"admin":"demo123"}' ztrade-app
```
Then open http://localhost:8501/ in your browser.

### Sample Data
A small CSV at `options/data/sample_options.csv` is included for offline exploration. Point `DATA_SOURCE_PATH` at your own file or cloud object for production data.

## Development
- The UI is defined in `options/ui.py`.
- Data loading abstractions live in `options/data_sources.py` and option pricing helpers in `options/pricing_utils.py`.
- User management is in `options/auth/users.py`.

