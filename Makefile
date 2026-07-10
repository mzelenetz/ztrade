PYTHON := python3
PROJECT := pcrpal
SERVICE := ztrade
REGION := us-central1
IMAGE := gcr.io/$(PROJECT)/$(SERVICE)

# ---------- Helpers ----------
.PHONY: help
help:
	@echo "make venv             - create venv"
	@echo "make deps             - install deps"
	@echo "make mz-build         - build mzpricer wheel"
	@echo "make mz-install       - install local mzpricer wheel"
	@echo "make docker           - build docker image"
	@echo "make push             - push image"
	@echo "make deploy           - deploy to Cloud Run"
	@echo "make run              - run backend locally (uvicorn, reload)"
	@echo "make run-web          - run frontend locally (vite dev server)"

# ---------- Python ----------
venv:
	$(PYTHON) -m venv .venv

deps: venv
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

# ---------- mzpricer ----------
mz-build:
	cd mzpricer && maturin build --release --features pyo3/extension-module

mz-install:
	ls mzpricer/target/wheels/*.whl | tail -1 | xargs .venv/bin/pip install -U

# ---------- Runtime ----------
run:
	. .venv/bin/activate && uvicorn src.api.main:app --reload --port 8000

run-web:
	cd web && npm run dev

test:
	uv run pytest -q

# Runs the full suite including the mzpricer cross-check (mzpricer only exists in the image)
test-docker:
	docker run --rm -v "$(PWD)":/repo -w /repo --entrypoint sh ztrade-app \
	  -c "pip install -q pytest && python -m pytest tests/ -q"

# ---------- Ingestion ----------
ingest-dry:
	uv run python -m src.ingest --dry-run /tmp/closes-dry.csv

# One-time (or after changing schedule/env): Cloud Run Job + daily scheduler
deploy-ingest:
	gcloud run jobs deploy ztrade-ingest \
	  --image us-central1-docker.pkg.dev/$(PROJECT)/containers/ztrade:latest \
	  --project $(PROJECT) --region $(REGION) \
	  --command python --args -m,src.ingest \
	  --set-env-vars GCS_CLOSES_BUCKET=ztrade-yesterday-closes \
	  --cpu 1 --memory 512Mi --task-timeout 15m --max-retries 2
	gcloud scheduler jobs create http ztrade-ingest-daily \
	  --project $(PROJECT) --location $(REGION) \
	  --schedule "45 16 * * 1-5" --time-zone "America/New_York" \
	  --uri "https://$(REGION)-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$(PROJECT)/jobs/ztrade-ingest:run" \
	  --http-method POST \
	  --oauth-service-account-email "$(shell gcloud projects describe $(PROJECT) --format='value(projectNumber)')-compute@developer.gserviceaccount.com" \
	  || gcloud scheduler jobs update http ztrade-ingest-daily \
	  --project $(PROJECT) --location $(REGION) \
	  --schedule "45 16 * * 1-5" --time-zone "America/New_York"

ingest-run:
	gcloud run jobs execute ztrade-ingest --project $(PROJECT) --region $(REGION) --wait

# ---------- Docker / Cloud Run ----------
docker:
	docker buildx build \
	--platform linux/amd64 \
	-t $(IMAGE):latest \
	.

push:
	docker push $(IMAGE):latest
	

deploy:
	gcloud run deploy $(SERVICE) \
	  --image $(IMAGE):latest \
	  --project $(PROJECT) \
	  --region $(REGION) \
	  --platform managed \
	  --allow-unauthenticated

all: mz-build mz-install docker push deploy
