.PHONY: install test lint format clean data models smoke curate

PYTHON ?= python

install:
	uv pip install -r requirements-dev.txt
	uv pip install -e .

test:
	$(PYTHON) -m pytest tests -v --cov=restoration --cov-report=term-missing --cov-report=html

lint:
	ruff check src tests scripts
	mypy src scripts

format:
	black src tests scripts
	isort src tests scripts

data:
	$(PYTHON) scripts/download_data.py

models:
	$(PYTHON) scripts/download_models.py

smoke:
	$(PYTHON) scripts/smoke_models.py

curate:
	$(PYTHON) scripts/curate_and_split.py

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov/
