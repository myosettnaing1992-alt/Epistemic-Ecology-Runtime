# Epistemic Ecology Runtime (EER)

[![Tests](https://github.com/myosettnaing1992-alt/Epistemic-Ecology-Runtime/actions/workflows/test.yml/badge.svg)](https://github.com/myosettnaing1992-alt/Epistemic-Ecology-Runtime/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0002--9133--0058-green.svg)](https://orcid.org/0009-0002-9133-0058)

A high-performance Python/Numba numerical framework for simulating belief dynamics,
structural energy minimization, and epistemic graph convergence across dynamic
networks (n = 10^4 to 5 x 10^4).

**Repository**: https://github.com/myosettnaing1992-alt/Epistemic-Ecology-Runtime

---

## Overview

The **Epistemic Ecology Runtime (EER)** implements a strictly convex variational
solver for belief aggregation on directed epistemic graphs. It optimizes belief
vectors `x in [0,1]^n` against prior distributions, contradiction penalties,
strongly connected component (SCC) consensus constraints, cascade regularizers,
and fundamental cycle closures.

EER uses a decoupled edge-buffering graph structure, vectorized block-COO
sparse Extended Hessian assembly, and Numba-JIT coordinate descent solvers
with incremental O(deg) residual updating.

---

## Key Features

- **Memory-efficient graph core (`EpistemicGraph`)**: Decoupled edge-list
  buffering eliminates memory overhead during dynamic graph construction.
- **Vectorized block-COO Hessian assembly**: Constructs

```

H_ext = H_0 + H_SCC + alpha * Q_cascade + gamma * Q_cycle     (eq. 9)

```

  in vectorized O(|C|^2) block-COO matrix format.
- **Structural regularizers**:
  - `Q_cascade`: Depth-bounded DFS path enumeration with memory-efficient
    backtracking over directed derivation cascades (|p| <= L_max, eq. 15).
  - `Q_cycle`: Fundamental cycle basis construction with alternating signed
    incidence `b_sigma(v_k) = (-1)^k` to prevent sign cancellation (eq. 17).
- **Numba-JIT hybrid scheduler**: Mandatory cyclic backbone sweep (M-period)
  combined with aged residual priority updates for fast box-constrained
  convergence.
- **Parallel calibration (`FastGridSearchCalibrator`)**: Surrogate grid
  search with pre-cached structural matrices for fast multi-core parameter
  estimation (alpha, gamma).

---

## Tech Stack

- **Python**: 3.11+
- `numpy >= 1.24.0`
- `scipy >= 1.10.0`
- `numba >= 0.57.0`
- `networkx >= 3.0`
- `joblib >= 1.2.0`
- `pyyaml >= 6.0`

---

## Repository Structure

```text
Epistemic-Ecology-Runtime/
├── .github/
│   └── workflows/
│       └── test.yml
├── eer/
│   ├── __init__.py
│   ├── core_graph.py
│   ├── cycle_basis.py
│   ├── hessian_builder.py
│   ├── schedulers.py
│   ├── calibration.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_core_graph.py
│   ├── test_cycle.py
│   ├── test_hessian.py
│   ├── test_scheduler.py
│   └── test_convergence.py
├── benchmarks/
│   ├── __init__.py
│   ├── run_benchmarks.py
│   ├── format_benchmark.py
│   └── results/
│       ├── .gitkeep
│       ├── README.md
│       ├── sample_benchmark.json
│       └── sample_benchmark.md
├── docs/
│   └── quickstart.md
├── results/
│   ├── .gitkeep
│   └── README.md
├── figures/
│   └── .gitkeep
├── data/
│   ├── .gitkeep
│   └── README.md
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml
├── CHANGELOG.md
├── CITATION.cff
├── CONTRIBUTING.md
├── Dockerfile
├── LICENSE
├── Makefile
├── README.md
├── environment.yml
├── pyproject.toml
└── requirements.txt
```

---

Installation

Clone the repository

```bash
git clone https://github.com/myosettnaing1992-alt/Epistemic-Ecology-Runtime.git
cd Epistemic-Ecology-Runtime
```

pip (recommended)

```bash
pip install -e ".[dev,bench]"
```

Conda

```bash
conda env create -f environment.yml
conda activate eer-env
pip install -e ".[dev,bench]"
```

Docker

```bash
docker build -t eer:latest .
docker run -it --rm eer:latest pytest -v tests/
```

Verify installation

```bash
pytest tests/ -v
```

---

Quickstart

Minimal Example

```python
import numpy as np
from eer import (
    EpistemicGraph,
    assemble_extended_hessian,
    build_cycle_matrix_fundamental,
    run_hybrid_priority_scheduler_optimized,
)

# 1. Initialize graph with custom priors and precision
n = 1000
g = EpistemicGraph(num_nodes=n)
g.b = np.random.uniform(0.0, 1.0, size=n)
g.lambda_vec = np.random.uniform(0.5, 2.0, size=n)

# 2. Add epistemic support edges (sparse graph)
rng = np.random.default_rng(42)
for _ in range(n * 2):
    u, v = rng.integers(0, n, size=2)
    if u != v:
        g.add_support_edge(int(u), int(v), float(rng.uniform(0.5, 1.5)))

# 3. Assemble Extended Hessian with cycle regularizer
Q_cyc = build_cycle_matrix_fundamental(g)
H = assemble_extended_hessian(g, gamma=0.05, Q_cycle=Q_cyc)
H_csc = H.tocsc()

# 4. Solve for equilibrium belief vector x*
x0 = np.full(n, 0.5, dtype=np.float64)
x_star, updates, res = run_hybrid_priority_scheduler_optimized(
    H.indptr, H.indices, H.data,
    H_csc.indptr, H_csc.indices, H_csc.data,
    g.b, x0,
    M=n,           # backbone period = n (Theorem 0.8.4)
    epsilon=1e-3,  # aging rate
    max_sweeps=1000,
    tol=1e-6,
)

print(f"Total Updates: {updates} | Final Residual: {res:.2e}")
```

Parallel Calibration Example

```python
import numpy as np
from eer import EpistemicGraph, FastGridSearchCalibrator

g = EpistemicGraph(num_nodes=500)
# ... add edges, set priors ...

snapshots = [np.random.uniform(0, 1, size=500) for _ in range(5)]

alpha_grid = np.logspace(-2, 0, 8)
gamma_grid = np.logspace(-2, 0, 8)

calibrator = FastGridSearchCalibrator(g, alpha_grid, gamma_grid, L_max=3)
best_alpha, best_gamma, best_mse = calibrator.calibrate(snapshots, n_jobs=-1)

print(f"Optimal Alpha: {best_alpha:.4f}, Gamma: {best_gamma:.4f} (MSE: {best_mse:.6f})")
```

Performance Note: build_cascade_matrix_bounded uses bounded DFS with
backtracking. For dense derivation subgraphs (d > 10), enforce small L_max
(3 <= L_max <= 4) or cap max_paths_per_node.

---

Reproducing the Paper

Every result in the paper is generated by the scripts below. Run them in order:

```bash
# 1. Install
pip install -e ".[dev,bench]"

# 2. Unit tests (verifies Theorems 4.2, 9.1 and Corollary 5.1)
pytest -v tests/

# 3. Table 9.1 — scheduler comparison
python benchmarks/run_benchmarks.py --all --out results/benchmark.json

# 4. Figure 11.1 — residual decay on BA scale-free graph
python benchmarks/run_benchmarks.py --smoke --out results/smoke.json

# 5. Format to markdown
python benchmarks/format_benchmark.py --in results/benchmark.json --stdout

# 6. Full pipeline (Makefile)
make experiments
```

Expected outputs

Script Output Paper reference
run_benchmarks.py --all results/benchmark.json Table 9.1
run_benchmarks.py --smoke results/smoke.json Figure 11.1
format_benchmark.py stdout / README Table 9.1
make experiments results/benchmark.md All tables

---

Testing

Run the full test suite locally:

```bash
pytest -v tests/
```

With coverage:

```bash
pytest --cov=eer --cov-report=term-missing tests/
```

Run a specific theorem test:

```bash
pytest tests/test_convergence.py::TestTheorem42OstrowskiReich -v -s
pytest tests/test_hessian.py::TestCorollary51WellPosedness -v -s
pytest tests/test_scheduler.py::TestTheorem91NoStarvation -v -s
```

CI status is reported by the badge at the top of this README.

---

Theoretical Guarantees

Result Statement Reference
Theorem 3.1 Unique equilibrium under strict convexity Section 3.1
Theorem 4.2 Linear convergence of PCGS without diagonal dominance Section 4.3
Corollary 4.1 Rate bound: rho <= 1 - 1/(2*kappa) Section 4.4
Corollary 5.1 H_ext is positive definite for all alpha, gamma >= 0 Section 5.5
Theorem 9.1 No starvation; waiting time <= M + ceil(R_max / epsilon) Section 9.3

All theorems are verified numerically in tests/.

---

Benchmarks

Reference numbers reproduced via:

```bash
python benchmarks/run_benchmarks.py --all
python benchmarks/format_benchmark.py
```

Averaged over 3 random seeds; timings report mean ± std.

Note on update counts. EER Updates is the total number of coordinate
updates performed by the hybrid scheduler. Each block of n iterations
contains one full cyclic backbone sweep plus n - 1 priority-selected
updates. Wall-clock time is the primary comparison metric; update counts
are not directly comparable to CGS sweeps.

Graph n Method Wall-clock (s) Updates Speedup
MVE 32 Cyclic GS 0.012 ± 0.001 384 1.00x
MVE 32 Random priority 0.008 ± 0.001 478 1.50x
MVE 32 Hybrid priority 0.009 ± 0.001 352 1.33x
BA 500 Cyclic GS 0.245 ± 0.012 15,500 1.00x
BA 500 Random priority 0.198 ± 0.009 14,200 1.24x
BA 500 Hybrid priority 0.062 ± 0.004 1,403 3.95x
BA 10000 Cyclic GS 12.340 ± 0.450 8,420 1.00x
BA 10000 Random priority 13.210 ± 0.510 9,100 0.93x
BA 10000 Hybrid priority 7.890 ± 0.320 5,230 1.56x
ER 10000 Cyclic GS 11.870 ± 0.410 7,890 1.00x
ER 10000 Random priority 12.450 ± 0.480 8,200 0.95x
ER 10000 Hybrid priority 11.920 ± 0.430 7,950 1.00x

Note on the 11x figure. The 11x speedup reported in the paper is specific
to BA scale-free graphs with n = 500 and residual tolerance 1e-6,
measured in coordinate updates (15,500 / 1,403 = 11.0). On larger graphs
(n = 10^4) the update-count speedup is ~1.6x, and on Erdos-Renyi graphs it
vanishes. Wall-clock speedup is smaller than update-count speedup because the
priority queue introduces O(log n) overhead per pop. See
benchmarks/results/sample_benchmark.md for the full breakdown.

---

Hyperparameter Tuning Guide

Parameter Recommended Description
M n (default) Cyclic backbone period. Theorem 0.8.4 assumes M = n.
epsilon 1e-3 Aging factor for residual priority updates (eq. 34).
L_max 3-5 Maximum depth bound for derivation path enumeration.
tol 1e-6 Convergence threshold on `
alpha, gamma log-spaced grid Cascade (alpha) and cycle (gamma) precision weights.

---

Known Limitations

· Sparse regime optimization: Performance gains assume sparse graphs
  (d_avg <= 10). For dense graphs, preconditioned conjugate gradient is preferred.
· Offline surrogate calibration: Parameter selection (alpha_hat, gamma_hat)
  uses surrogate grid search rather than full Bayesian marginalization.
· Discrete graph dynamics: Continuous tracking bounds assume Lipschitz
  continuity; discrete node/edge arrivals use warm-start heuristics.
· Update counts vs wall-clock: Update-count speedup is not directly
  comparable to wall-clock speedup; see the benchmark note above.

---

Citation & License

Distributed under the MIT License. If you use EER in your research, please cite:

```bibtex
@article{naing2026epistemic,
  title   = {Epistemic Ecology Runtime: A Variational Framework for
             Continuous Belief Relaxation on Dynamic Epistemic Graphs},
  author  = {Naing, Myo Sett},
  year    = {2026},
  note    = {ORCID: 0009-0002-9133-0058}
}
```

A machine-readable citation is available in CITATION.cff.

Repository: https://github.com/myosettnaing1992-alt/Epistemic-Ecology-Runtime

---

Contributing

We welcome contributions. Please read CONTRIBUTING.md first.

· Bug reports — open a GitHub issue with a minimal reproduction.

· Feature requests — open an issue describing the use case.

· Pull requests — fork, branch, run pytest tests/ -v, and submit.

Development setup

```bash
git clone https://github.com/myosettnaing1992-alt/Epistemic-Ecology-Runtime.git
cd Epistemic-Ecology-Runtime
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev,bench]"
pre-commit install           # optional
```

Code style

· ruff for linting and formatting (line length 88)

· mypy for type checking (strict on eer/)

· NumPy-style docstrings

---

Acknowledgements

The mathematical structure of EER was inspired by homological methods in
theoretical physics (AKSZ-BV formalism, Deformed Quantum Variance). The
framework presented here is entirely self-contained; no categorical equivalence
is claimed.

---

Contact

· Issues: 
https://github.com/myosettnaing1992-alt/Epistemic-Ecology-Runtime/issues

· ORCID: 0009-0002-9133-0058
```
