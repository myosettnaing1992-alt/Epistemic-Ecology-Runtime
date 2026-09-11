# Epistemic Ecology Runtime (EER)

[![Tests](https://github.com/myosettnaing1992-alt/Epistemic-Ecology-Runtime/actions/workflows/test.yml/badge.svg)](https://github.com/myosettnaing1992-alt/Epistemic-Ecology-Runtime/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0002--9133--0058-green.svg)](https://orcid.org/0009-0002-9133-0058)

A high-performance Python/Numba numerical framework for simulating belief
dynamics, structural energy minimization, and epistemic graph convergence
across dynamic networks ($n = 10^4$ to $5 \times 10^4$).

**Repository**: https://github.com/myosettnaing1992-alt/Epistemic-Ecology-Runtime

---

## 📌 Overview

The **Epistemic Ecology Runtime (EER)** implements a strictly convex
variational solver for belief aggregation on directed epistemic graphs. It
optimizes belief vectors $\mathbf{x} \in [0,1]^n$ against prior
distributions, contradiction penalties, strongly connected component (SCC)
consensus constraints, cascade regularizers, and fundamental cycle closures.

EER uses a decoupled edge-buffering graph structure, vectorized block-COO
sparse Extended Hessian assembly, and Numba-JIT coordinate descent solvers
with incremental $O(\text{deg})$ residual updating.

---

## ✨ Key Features

- **Memory-efficient graph core (`EpistemicGraph`)**: Decoupled edge-list
  buffering eliminates memory overhead during dynamic graph construction.
- **Vectorized block-COO Hessian assembly**: Constructs
  $$H_{\text{ext}} = H_0 + H_{\text{SCC}} + \alpha Q_{\text{cascade}} + \gamma Q_{\text{cycle}}$$
  (eq. 9) in vectorized $O(\vert{}C\vert{}^2)$ block-COO matrix format.
- **Structural regularizers**:
  - $Q_{\text{cascade}}$: Depth-bounded DFS path enumeration with
    memory-efficient backtracking over directed derivation cascades
    ($\vert{}p\vert{} \le L_{\max}$, eq. 15).
  - $Q_{\text{cycle}}$: Fundamental cycle basis construction with
    **alternating signed incidence** $b_\sigma(v_k) = (-1)^k$ to prevent
    sign cancellation (eq. 17).
- **Numba-JIT hybrid scheduler**: Mandatory cyclic backbone sweep
  ($M$-period) combined with aged residual priority updates for fast
  box-constrained convergence.
- **Parallel calibration (`FastGridSearchCalibrator`)**: Surrogate grid
  search with pre-cached structural matrices for fast multi-core parameter
  estimation $(\alpha, \gamma)$.

---

## 🛠️ Tech Stack

- **Python**: 3.11+
- `numpy >= 1.24.0`
- `scipy >= 1.10.0`
- `numba >= 0.57.0`
- `networkx >= 3.0`
- `joblib >= 1.2.0`
- `pyyaml >= 6.0`

---

## 📁 Repository Structure

```text
Epistemic-Ecology-Runtime/
├── .github/workflows/test.yml
├── eer/
│   ├── __init__.py
│   ├── core_graph.py
│   ├── hessian_builder.py
│   ├── schedulers.py
│   ├── calibration.py
│   └── utils.py
├── tests/
│   ├── conftest.py
│   ├── test_core_graph.py
│   ├── test_cycle.py
│   ├── test_hessian.py
│   └── test_scheduler.py
├── benchmarks/
│   ├── run_benchmarks.py
│   └── format_benchmark.py
├── Dockerfile
├── environment.yml
├── pyproject.toml
├── CITATION.cff
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## 🚀 Quickstart

### Clone the repository

```bash
git clone https://github.com/myosettnaing1992-alt/Epistemic-Ecology-Runtime.git
cd Epistemic-Ecology-Runtime
```

### Installation

```bash
pip install -e ".[dev,bench]"
```

Or via Conda:

```bash
conda env create -f environment.yml
conda activate eer-env
pip install -e ".[dev,bench]"
```

### Minimal Example

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

### Parallel Calibration Example

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

> ⚠️ **Performance Note**: `build_cascade_matrix_bounded` uses bounded DFS
> with backtracking. For dense derivation subgraphs ($d > 10$), enforce small
> $L_{\max}$ ($3 \le L_{\max} \le 4$) or cap `max_paths_per_node`.

---

## 🧪 Testing

Run the full test suite locally:

```bash
pytest -v tests/
```

With coverage:

```bash
pytest --cov=eer --cov-report=term-missing tests/
```

CI status is reported by the badge at the top of this README.

---

## 📊 Performance Benchmark

Reference numbers reproduced via:

```bash
python benchmarks/run_benchmarks.py
python benchmarks/format_benchmark.py
```

Averaged over 3 random seeds; timings report mean ± std.

<!-- BENCHMARK_TABLE_START -->
<!-- BENCHMARK_TABLE_END -->

> **Note**: `EER Updates` is the total number of coordinate updates
> performed by the hybrid scheduler. Each *block* of `n` iterations contains
> one full cyclic backbone sweep plus `n − 1` priority-selected updates.
> Wall-clock time is the primary comparison metric; update counts are not
> directly comparable to CGS sweeps.

---

## ⚙️ Hyperparameter Tuning Guide

| Parameter | Recommended | Description |
| :--- | :--- | :--- |
| `M` | `n` (default) | Cyclic backbone period. Theorem 0.8.4 assumes `M = n`. |
| `epsilon` | `1e-3` | Aging factor for residual priority updates (eq. 34). |
| `L_max` | `3–5` | Maximum depth bound for derivation path enumeration. |
| `tol` | `1e-6` | Convergence threshold on $\|H_{\text{ext}}\mathbf{x} - \mathbf{b}\|_\infty$. |
| `alpha`, `gamma` | log-spaced grid | Cascade ($\alpha$) and cycle ($\gamma$) precision weights. |

---

## ⚠️ Known Limitations

- **Sparse regime optimization**: Performance gains assume sparse graphs
  ($d_{\text{avg}} \le 10$). For dense graphs, preconditioned conjugate
  gradient is preferred.
- **Offline surrogate calibration**: Parameter selection
  $(\hat{\alpha}, \hat{\gamma})$ uses surrogate grid search rather than full
  Bayesian marginalization.
- **Discrete graph dynamics**: Continuous tracking bounds assume Lipschitz
  continuity; discrete node/edge arrivals use warm-start heuristics.

---

## 📄 Citation & License

Distributed under the MIT License. If you use EER in your research, please
cite:

```bibtex
@article{naing2026epistemic,
  title={A Strictly Convex Variational Framework for Belief Aggregation on
         Static Directed Epistemic Graphs},
  author={Naing, Myo Sett},
  year={2026},
  note={ORCID: 0009-0002-9133-0058}
}
```

**Repository**: https://github.com/myosettnaing1992-alt/Epistemic-Ecology-Runtimevariational solver for belief aggregation on directed epistemic graphs. It
optimizes belief vectors $\mathbf{x} \in [0,1]^n$ against prior
distributions, contradiction penalties, strongly connected component (SCC)
consensus constraints, cascade regularizers, and fundamental cycle closures.

EER uses a decoupled edge-buffering graph structure, vectorized block-COO
sparse Extended Hessian assembly, and Numba-JIT coordinate descent solvers
with incremental O(deg) residual updating.

---

## ✨ Key Features

- **Memory-efficient graph core (`EpistemicGraph`)**: decoupled edge-list
  buffering eliminates DOK memory overhead during graph construction.
- **Vectorized block-COO Hessian assembly**: constructs
  $H_{\text{ext}} = H_0 + H_{\text{SCC}} + \alpha Q_{\text{cascade}} + \gamma Q_{\text{cycle}}$
  (eq. 9) in $O(|C|^2)$ block-COO format.
- **Structural regularizers**:
  - $Q_{\text{cascade}}$: depth-bounded DFS path enumeration over directed
    derivation cascades ($|p| \le L_{\max}$, eq. 15).
  - $Q_{\text{cycle}}$: fundamental cycle basis construction with
    **alternating signed incidence** $b_\sigma(v_k) = (-1)^k$ to prevent
    cancellation (eq. 17).
- **Numba-JIT hybrid scheduler (Algorithm 3)**: mandatory cyclic backbone
  sweep ($M$-period) combined with aged residual priority updates.
- **Parallel calibration**: `FastGridSearchCalibrator` with pre-cached
  structural regularizers.

---

## 🛠️ Tech Stack

- **Python**: 3.11+
- `numpy >= 1.24.0`
- `scipy >= 1.10.0`
- `numba >= 0.57.0`
- `networkx >= 3.0`
- `joblib >= 1.2.0`
- `pyyaml >= 6.0`

---

## 📁 Repository Structure

```text
epistemic-ecology-runtime/
├── .github/workflows/test.yml
├── eer/
│   ├── __init__.py
│   ├── core_graph.py
│   ├── hessian_builder.py
│   ├── schedulers.py
│   ├── calibration.py
│   └── utils.py
├── tests/
│   ├── conftest.py
│   ├── test_core_graph.py
│   ├── test_cycle.py
│   ├── test_hessian.py
│   └── test_scheduler.py
├── benchmarks/
│   ├── run_benchmarks.py
│   └── format_benchmark.py
├── Dockerfile
├── environment.yml
├── pyproject.toml
├── CITATION.cff
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## 🚀 Quickstart

### Installation

```bash
pip install -e ".[dev,bench]"
```

Or via Conda:

```bash
conda env create -f environment.yml
conda activate eer-env
pip install -e ".[dev,bench]"
```

### Minimal Example

```python
import numpy as np
from eer import (
    EpistemicGraph,
    assemble_extended_hessian,
    build_cycle_matrix_fundamental,
    run_hybrid_priority_scheduler_optimized,
)

n = 1000
g = EpistemicGraph(num_nodes=n)
for u, v in [(0, 1), (1, 2), (2, 3), (3, 0)]:
    g.add_support_edge(u, v, w=1.0)

Q_cyc = build_cycle_matrix_fundamental(g)
H = assemble_extended_hessian(g, gamma=0.05, Q_cycle=Q_cyc)
H_csc = H.tocsc()

x0 = np.full(n, 0.5, dtype=np.float64)
x_star, updates, res = run_hybrid_priority_scheduler_optimized(
    H.indptr, H.indices, H.data,
    H_csc.indptr, H_csc.indices, H_csc.data,
    g.b, x0, M=n, epsilon=1e-3, max_sweeps=1000, tol=1e-6,
)
print(f"updates={updates}  residual={res:.2e}")
```

> ⚠️ **Performance Note**: `build_cascade_matrix_bounded` scales as
> $O(n \cdot d^{L_{\max}})$. For dense derivation subgraphs ($d > 10$),
> enforce path constraints or cap `max_paths_per_node`.

---

## 📊 Performance Benchmark

Reference numbers reproduced via:

```bash
python benchmarks/run_benchmarks.py
python benchmarks/format_benchmark.py
```

Averaged over 3 random seeds; timings report mean ± std.

<!-- BENCHMARK_TABLE_START -->
<!-- BENCHMARK_TABLE_END -->

> **Note**: EER's `EER Updates` count differs from CGS `Sweeps` because
> each EER block contains one full cyclic backbone sweep plus `n − 1`
> priority-selected coordinate updates. Wall-clock time is the primary
> comparison metric.

---

## ⚙️ Hyperparameter Tuning Guide

| Parameter | Recommended | Description |
| :--- | :--- | :--- |
| `M` | `n` (default) | Cyclic backbone period. Theorem 0.8.4 assumes `M = n`. |
| `epsilon` | `1e-3` | Aging rate for residual priority updates (eq. 34). |
| `L_max` | `3–5` | Maximum depth for derivation path enumeration. |
| `tol` | `1e-6` | Convergence threshold on $\|H_{\text{ext}}\mathbf{x} - \mathbf{b}\|_\infty$. |
| `alpha`, `gamma` | log-spaced grid | Cascade and cycle precision parameters. |

---

## ⚠️ Known Limitations

- **Sparse regime optimization**: performance gains assume sparse graphs
  ($d_{\text{avg}} \le 10$). For dense graphs, preconditioned conjugate
  gradient is preferred.
- **Offline surrogate calibration**: parameter selection
  $(\hat{\alpha}, \hat{\gamma})$ uses grid search rather than full Bayesian
  marginalization.
- **Discrete graph dynamics**: continuous tracking bounds assume Lipschitz
  continuity; discrete node/edge arrivals use warm-start heuristics.

---

## 📄 Citation & License

Distributed under the MIT License. If you use EER in your research, please
cite:

``bibtex

@ article{naing2026epistemic,
  title={A Strictly Convex Variational Framework for Belief Aggregation on
         Static Directed Epistemic Graphs},
  author={Naing, Myo Sett},
  year={2026},
  note={ORCID: 0009-0002-9133-0058}
}
``
