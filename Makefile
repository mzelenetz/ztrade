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
	@echo "make run              - run locally"

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
	. .venv/bin/activate && streamlit run src/ui.py

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
