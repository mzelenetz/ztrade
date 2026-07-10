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
