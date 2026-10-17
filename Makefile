VENV := .venv/bin

.PHONY: install test lint train-local serve docker-build docker-run

install:
	python3 -m venv .venv && $(VENV)/pip install -q -r requirements-dev.txt

test:
	$(VENV)/pytest -q

lint:
	$(VENV)/ruff check .

# Train from BigQuery (needs gcloud auth application-default login)
train:
	$(VENV)/python -m training.train --source bq://$(PROJECT_ID).mlops.income_training --out artifacts

serve:
	$(VENV)/uvicorn app.main:app --reload --port 8080

docker-build:
	docker build -t income-api:local .

docker-run:
	docker run --rm -p 8080:8080 -v $(PWD)/artifacts:/srv/artifacts:ro income-api:local
