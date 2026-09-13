# Makefile
.PHONY: install test experiments clean

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v

experiments:
	python experiments/run_wallclock.py --all --out results/table_9_1.json

clean:
	rm -rf results/*.json __pycache__ .pytest_cache

all: install test experiments
