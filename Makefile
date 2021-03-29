PY ?= .venv/bin/python

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt -e .
test:
	$(PY) -m pytest -q
pipeline:
	$(PY) -m borderscore pipeline
coldstart:
	$(PY) -m borderscore coldstart
ui:
	timeout 600 .venv/bin/mlflow server --backend-store-uri sqlite:///artifacts/mlflow.db --default-artifact-root artifacts/mlartifacts --port 5000
.PHONY: setup test pipeline coldstart ui
