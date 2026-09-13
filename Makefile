# ============================================================
# EER Makefile
# ============================================================

.PHONY: help install install-dev test test-cov lint format typecheck \
        benchmark smoke-experiment experiments clean clean-all docker

PYTHON ?= python
PYTEST ?= pytest
PIP ?= pip

help:
	@echo "EER Makefile"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install runtime dependencies"
	@echo "  make install-dev    Install with dev + bench extras"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run full test suite"
	@echo "  make test-cov       Run tests with coverage"
	@echo "  make smoke          Run smoke benchmarks"
	@echo ""
	@echo "Quality:"
	@echo "  make lint           Run ruff check"
	@echo "  make format         Run ruff format"
	@echo "  make typecheck      Run mypy"
	@echo ""
	@echo "Experiments:"
	@echo "  make experiments    Reproduce all paper results"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Remove caches"
	@echo "  make clean-all      Remove caches + results"
	@echo ""
	@echo "Docker:"
	@echo "  make docker         Build Docker image"

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev,bench]"

# ------------------------------------------------------------
# Testing
# ------------------------------------------------------------

test:
	$(PYTEST) -v tests/

test-cov:
	$(PYTEST) -v tests/ --cov=eer --cov-report=term-missing --cov-report=html

smoke:
	$(PYTHON) benchmarks/run_benchmarks.py --smoke

# ------------------------------------------------------------
# Quality
# ------------------------------------------------------------

lint:
	ruff check eer/ benchmarks/ tests/

format:
	ruff format eer/ benchmarks/ tests/

typecheck:
	mypy eer/

# ------------------------------------------------------------
# Experiments
# ------------------------------------------------------------

experiments:
	@mkdir -p results figures
	$(PYTHON) benchmarks/run_benchmarks.py \
	    --graphs mve ba er \
	    --n 32 500 10000 \
	    --seeds 0 1 2 \
	    --out results/benchmark.json
	$(PYTHON) benchmarks/format_benchmark.py \
	    --in results/benchmark.json \
	    --stdout > results/benchmark.md

# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml

clean-all: clean
	rm -rf results/*.json results/*.md results/*.csv
	rm -rf figures/*.pdf figures/*.png figures/*.svg

# ------------------------------------------------------------
# Docker
# ------------------------------------------------------------

docker:
	docker build -t eer:latest .
	docker run -it --rm eer:latest pytest -v tests/
