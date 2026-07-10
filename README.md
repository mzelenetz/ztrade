# ZTrade Options Viewer

A React + shadcn/ui dashboard (backed by a FastAPI service) for reviewing option chains with paired call/put views, JWT authentication, and flexible data loading that can pull from local files or cloud buckets. Both the API and the built frontend are served from a single Cloud Run container.

## Features
- Authenticated access with environment-configurable users (default `admin` / `demo123`), JWT-based sessions.
- Side-by-side call/put data in a single AG Grid row per strike for each expiry, grouped by expiry in collapsible sections.
- Data-loading abstraction supporting local CSVs, Amazon S3, or Google Cloud Storage.
- Toggle between mzpricer (default) and QuantLib pricing models.
- Filter expiries by an adjustable call-delta band to focus on desired moneyness.
- A "Spreads" tab that surfaces delta-neutral spread ideas with full leg detail on selection.
- Light/dark theme toggle.
- Dockerfile (multi-stage: Node build + Python runtime) for repeatable cloud deployments.

## Architecture
- **Backend** (`src/api/`): FastAPI app exposing `/api/auth`, `/api/meta`, `/api/chain`, `/api/spreads`. Reuses `src/data_sources.py`, `src/pricing_utils.py`, `src/auth/users.py`, and the pure chain/spread computation logic in `src/services/`.
- **Frontend** (`web/`): Vite + React + TypeScript + Tailwind + shadcn/ui, with AG Grid for the data-dense chain/spreads tables and TanStack Query for data fetching.
- In production, FastAPI serves the built frontend (`web/dist`) directly alongside the `/api/*` routes from the same Cloud Run service/port.

## Getting Started (local development)
1. Install backend dependencies (Python 3.12+), e.g. with `uv`:
   ```bash
   uv sync
   ```
2. Run the API:
   ```bash
   make run
   # or: uvicorn src.api.main:app --reload --port 8000
   ```
3. In a separate terminal, install and run the frontend (proxies `/api` to `localhost:8000`):
   ```bash
   cd web
   npm install
   npm run dev
   ```
4. Open the Vite dev server URL and log in with the default credentials (`admin` / `demo123`) or configure your own users via environment variables.

### Authentication
- By default the app seeds a single `admin` user with password `demo123`.
- To define your own users, set `APP_USERS` to a JSON object mapping usernames to plaintext passwords, e.g.:
  ```bash
  export APP_USERS='{"alice": "p@ssword", "bob": "hunter2"}'
  ```
- Set `JWT_SECRET_KEY` to a stable secret in any environment where tokens should survive a process restart or be shared across instances (otherwise a random per-process secret is used).

### Configuring Data Sources
The app loads option data through a pluggable data source defined by environment variables:
- `DATA_SOURCE_TYPE` (default `local`): one of `local`, `s3`, `gcs`, `gcs_closes`, `gcs_latest`.
- Local CSV: set `DATA_SOURCE_PATH` (defaults to `src/data/sample_NVDA.csv`).
- S3 bucket: set `DATA_SOURCE_TYPE=s3`, `DATA_SOURCE_BUCKET=<bucket>`, `DATA_SOURCE_KEY=<object-key>`.
- GCS bucket: set `DATA_SOURCE_TYPE=gcs`, `DATA_SOURCE_BUCKET=<bucket>`, `DATA_SOURCE_KEY=<blob-name>`, and install `google-cloud-storage`.
- GCS closes: set `DATA_SOURCE_TYPE=gcs_closes`, `GCS_CLOSES_BUCKET`, `GCS_CLOSES_PREFIX`, `GCS_CLOSES_EXTENSION` — the API's close-date picker will list available dates.
- Additional knobs: `DATA_AS_OF_DATE` (YYYY-MM-DD), `DEFAULT_VOLATILITY` (fallback vol), `USE_REMOTE_VOL` (`true`/`false` to toggle yfinance lookups).

### Pricing Models
- Choose the pricing engine from the sidebar select.
- mzpricer is the default model; QuantLib is available as a fallback or for comparison.

### Call-Delta Filter
- Use the sidebar slider to choose the call-delta band (0 = out of the money, 100 = deep in the money).
- Only strikes whose calls fall within the selected band are displayed; their paired puts remain visible for quick comparisons.

### Docker
Build and run the containerized app (single image serves both API and frontend):
```bash
docker build -t ztrade-app .
docker run --rm -p 8080:8080 -e PORT=8080 -e APP_USERS='{"admin":"demo123"}' ztrade-app

# Local, with GCS creds:
# gcloud auth application-default login
# gcloud auth application-default set-quota-project YOUR_PROJECT_ID

docker run --rm -p 8080:8080 -e PORT=8080 -e APP_USERS='{"admin":"demo123"}' -e DATA_SOURCE_TYPE='gcs_closes' -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/adc.json  -v "$HOME/.config/gcloud/application_default_credentials.json:/tmp/adc.json:ro" \
ztrade-app
```
Then open http://localhost:8080/ in your browser.

### Sample Data
A small CSV at `src/data/sample_NVDA.csv` is included for offline exploration (note: its dates are shifted and its quotes are internally inconsistent — use `tests/fixtures/closes-nvda-2026-01-29.csv` for realistic local work). Point `DATA_SOURCE_PATH` at your own file or cloud object for production data.

## Daily ingestion
A Cloud Run Job (`ztrade-ingest`) fetches full option chains after each close and
publishes `closes-YYYY-MM-DD.csv` to `gs://ztrade-yesterday-closes`, so the app's
date picker has fresh data every morning. Cloud Scheduler triggers it weekdays at
4:45pm ET (`ztrade-ingest-daily`).

- **Source**: yfinance for now. The fetcher sits behind the `ChainFetcher`
  protocol in `src/ingest/fetchers.py` — a purchased data feed later is one new
  class plus `INGEST_SOURCE=<name>` on the job.
- **Ticker universe**: edit `gs://ztrade-yesterday-closes/tickers.txt` (one
  symbol per line, `#` comments) — no redeploy needed. Falls back to a `TICKERS`
  env var, then to the built-in six.
- **Validation before publish**: minimum row count, live underlying quote, and a
  put-call-parity coherence check per ticker — corrupt data is rejected rather
  than uploaded.
- Manual run: `make ingest-run` (or `gcloud run jobs execute ztrade-ingest --region us-central1 --wait`).
- Local dry run (no upload): `make ingest-dry`.
- Create/update the job + scheduler: `make deploy-ingest`.

## Development
- Backend routers live in `src/api/routers/` (`auth.py`, `meta.py`, `chain.py`, `spreads.py`); `src/api/main.py` wires them up and serves the built frontend.
- Chain/spread computation logic lives in `src/services/chain_service.py` and `src/services/spreads_service.py`.
- Data loading abstractions live in `src/data_sources.py`; option pricing helpers in `src/pricing_utils.py`.
- User management is in `src/auth/users.py`.
- Frontend source lives in `web/src/` (`pages/`, `components/`, `context/`, `lib/`).

### Cloud deployment
Build and push:
```bash
docker buildx build \
  --platform linux/amd64 \
  -t us-central1-docker.pkg.dev/pcrpal/containers/ztrade:latest \
  --push .
```

Deploy to Cloud Run:
```bash
gcloud run deploy ztrade \
  --image us-central1-docker.pkg.dev/pcrpal/containers/ztrade:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --timeout=900 \
  --set-env-vars=DATA_SOURCE_TYPE=gcs_closes,GCS_CLOSES_BUCKET=ztrade-yesterday-closes,GCS_CLOSES_PREFIX=closes-,GCS_CLOSES_EXTENSION=.csv,GOOGLE_CLOUD_PROJECT=pcrpal
```

## Makefile
Build and deploy all: `make all`
Just mzpricer: `make mz-build mz-install`
Just redeploy without touching mzpricer: `make docker push deploy`
Run backend locally: `make run`
Run frontend locally: `make run-web`
