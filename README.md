bounded DFS path enumeration with memory-efficient
    backtracking over directed derivation cascades ($\vert{}p\vert{} \le L_{\max}$, eq. 15).
  - $Q_{\text{cycle}}$: Fundamental cycle basis construction with
    **alternating signed incidence** $b_\sigma(v_k) = (-1)^k$ to prevent
    sign cancellation (eq. 17).
- **Numba-JIT hybrid scheduler**: Mandatory cyclic backbone
  sweep ($M$-period) combined with aged residual priority updates for fast box-constrained convergence.
- **Parallel calibration (`FastGridSearchCalibrator`)**: Surrogate grid search 
  with pre-cached structural matrices for fast multi-core parameter estimation $(\alpha, \gamma)$.

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

---

## 📌 Overview

The **Epistemic Ecology Runtime (EER)** implements a strictly convex
variational solver for belief aggregation on directed epistemic graphs. It
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
@article{naing2026epistemic,
  title={A Strictly Convex Variational Framework for Belief Aggregation on Static Directed Epistemic Graphs}, author={Naing, Myo Sett}, year={2026}, note={ORCID: 0009-0002-9133-0058}}
 
 `` * Parallel Hyperparameter Calibration: Multi-core grid-search calibrator (GridSearchCalibrator) using joblib over validation state snapshots.
🛠️ Tech Stack & Requirements
 * Python: 3.11+
 * Core Computational Stack:
   * numpy \ge 1.24.0
   * scipy \ge 1.10.0 (Sparse linear algebra, CGS, CSGraph, connected components)
   * numba \ge 0.57.0 (JIT compilation for coordinate steps and priority queues)
   * networkx \ge 3.0 (Cycle basis enumeration)
 * Parallelism & Reproducibility:
   * joblib \ge 1.2.0 (Embarrassingly parallel grid search)
   * pyyaml (Configuration management)
     
📁 Repository Structure
epistemic-ecology-runtime/
├── eer/
│   ├── __init__.py
│   ├── core_graph.py         # EpistemicGraph class & edge buffers
│   ├── hessian_builder.py    # Vectorized H_ext, SCC, Cascade & Cycle matrices
│   ├── schedulers.py         # Numba JIT hybrid priority & projected solvers
│   └── calibration.py        # Parallel grid-search surrogate calibration
├── tests/
│   ├── test_hessian.py       # Symmetry, positive-definiteness & unit tests
│   └── test_scheduler.py     # Convergence and monotonicity smoke tests
├── Dockerfile                # Isolated reproducible execution environment
├── environment.yml           # Conda environment definition
├── main.py                   # CLI entry point for experiments
└── README.md

🚀 Quickstart
1. Installation
Using Conda:
conda env create -f environment.yml
conda activate eer-env

Using Docker:
docker build -t eer-runtime .
docker run -it eer-runtime

2. Basic Usage Example
import numpy as np
from eer.core_graph import EpistemicGraph
from eer.hessian_builder import (
    assemble_extended_hessian,
    build_cascade_matrix_bounded,
    build_cycle_matrix_fundamental,
)
from eer.schedulers import run_hybrid_priority_scheduler_numba

# 1. Initialize Epistemic Graph
n = 1000
graph = EpistemicGraph(num_nodes=n)

# 2. Add structural relationships
graph.add_support_edge(0, 1, w=1.0)
graph.add_contradiction_edge(1, 2, w=0.8)
graph.add_derivation_edge(2, 3, w=1.2)

# 3. Pre-compute structural regularizers
Q_cas = build_cascade_matrix_bounded(graph, L_max=4)
Q_cyc = build_cycle_matrix_fundamental(graph)

# 4. Assemble Extended Hessian Matrix
H_ext = assemble_extended_hessian(
    graph, 
    alpha=0.1, 
    gamma=0.05, 
    Q_cascade=Q_cas, 
    Q_cycle=Q_cyc
)

# 5. Run Numba-Accelerated Hybrid Priority Solver
x0 = np.full(n, 0.5, dtype=np.float64)
x_star, total_updates, final_res = run_hybrid_priority_scheduler_numba(
    H_ext.indptr, H_ext.indices, H_ext.data, graph.b, x0,
    M=n, epsilon=1e-3, max_sweeps=1000, tol=1e-6
)

print(f"Convergence reached in {total_updates} coordinate updates.")
print(f"Final residual inf-norm: {final_res:.2e}")

🔬 Calibration & Experiments
To run hyperparameter calibration across \alpha \in [0.01, 1.0] and \gamma \in [0.01, 1.0] over empirical validation snapshots:
from eer.calibration import GridSearchCalibrator

alpha_grid = np.linspace(0.01, 1.0, 10)
gamma_grid = np.linspace(0.01, 1.0, 10)

calibrator = GridSearchCalibrator(graph, alpha_grid, gamma_grid)
best_alpha, best_gamma, min_mse = calibrator.calibrate(validation_snapshots, n_jobs=-1)

print(f"Optimal Alpha: {best_alpha:.4f} | Optimal Gamma: {best_gamma:.4f} | Min MSE: {min_mse:.6f}")

🧪 Testing
Run unit and Testing
ststest tests/

📄 Citation & License

This project is licensed under the MIT License. If you use this implementation in your research, please cite:
@article{myo2026epistemic,
  title={Epistemic Ecology Runtime: A Variational Topology and Dynamic Quantum Vacuum Framework for Knowledge Graphs},
  author={Nain, Myo Set},
  year={2026},
  note={ORCID: 0009-0002-9133-0058}
}

# Epistemic Ecology Runtime (EER)

[![Tests](https://github.com/myosettnaing/epistemic-ecology-runtime/actions/workflows/test.yml/badge.svg)](https://github.com/myosettnaing/epistemic-ecology-runtime/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0002--9133--0058-green.svg)](https://orcid.org/0009-0002-9133-0058)

A high-performance Python/Numba numerical framework for simulating belief
dynamics, structural energy minimization, and epistemic graph convergence
across dynamic networks ($n = 10^4$ to $5 \times 10^4$).

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
  - $Q_{\text{cascade}}$: Depth-bounded DFS path enumeration with memory-efficient
    backtracking over directed derivation cascades ($\vert{}p\vert{} \le L_{\max}$, eq. 15).
  - $Q_{\text{cycle}}$: Fundamental cycle basis construction with
    **alternating signed incidence** $b_\sigma(v_k) = (-1)^k$ to prevent
    sign cancellation (eq. 17).
- **Numba-JIT hybrid scheduler**: Mandatory cyclic backbone
  sweep ($M$-period) combined with aged residual priority updates for fast box-constrained convergence.
- **Parallel calibration (`FastGridSearchCalibrator`)**: Surrogate grid search 
  with pre-cached structural matrices for fast multi-core parameter estimation $(\alpha, \gamma)$.

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
