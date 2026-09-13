# 1. Install
pip install numpy scipy networkx numba joblib pyyaml

# 2. Run cycle basis smoke test
python cycle_basis.py
# → Building random BA graph (n=2000)...
# → Total fundamental cycles: ~3994
# → Cycles kept: 1000
# → Build time: 0.8 s

# 3. Run calibration smoke test
python calibration.py
# → Running validation grid search...
# → Best α = 0.5000
# → Best γ = 0.5000

# 4. Reproduce Table 9.1 (BA, n=500)
python experiments/run_wallclock.py --graph ba --n 500

# 5. Full table
python experiments/run_wallclock.py --all

# 6. Save to JSON
python experiments/run_wallclock.py --graph ba --n 10000 --out results/table_9_1.json
