#!/usr/bin/env bash
# run_experiments.sh — Reproduce all paper experiments
# Usage: ./run_experiments.sh

set -euo pipefail   # Exit on error, undefined var, pipe failure

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Running smoke tests ==="
python cycle_basis.py
python calibration.py

echo "=== Reproducing Table 9.1 ==="
python experiments/run_wallclock.py --all --out results/table_9_1.json

echo "=== Running unit tests ==="
pytest tests/ -v

echo "=== Done. Results in results/ ==="
