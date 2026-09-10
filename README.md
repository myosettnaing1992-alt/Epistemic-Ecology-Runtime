Epistemic Ecology Runtime (EER)

A high-performance Python/Numba numerical framework for simulating belief dynamics, structural energy minimization, and epistemic graph convergence across large-scale dynamic networks (n = 10^4 \text{ to } 5 \times 10^4).
📌 Overview
The Epistemic Ecology Runtime (EER) provides an optimized solver and calibration pipeline for evaluating belief propagation, contradiction penalties, strongly connected component (SCC) consensus, cascade regularizers, and fundamental cycle constraints.
Designed for computational efficiency and mathematical rigor, EER leverages a decoupled edge-buffering graph structure, vectorized block-COO sparse Hessian assembly, and Numba JIT-compiled coordinate descent schedulers to achieve order-of-magnitude speedups over pure Python implementations.
✨ Key Features
 * Memory-Efficient Graph Core (EpistemicGraph): Uses decoupled edge-list buffers to eliminate DOK matrix memory overhead during dynamic graph building.
 * Vectorized Block-COO Hessian Assembly: Constructs the Extended Hessian Matrix H_{\text{ext}} = H_0 + H_{\text{SCC}} + \alpha Q_{\text{cascade}} + \gamma Q_{\text{cycle}} in O(\vert{}C\vert{}^2) block COO format without nested element assignments.
 * Bounded Path & Cycle Regularizers:
   * Q_{\text{cascade}}: Depth-bounded DFS path enumeration for directed derivation cascades (\vert{}p\vert{} \le L_{\text{max}}).
   * Q_{\text{cycle}}: Fundamental cycle basis construction via NetworkX spanning forests.
 * Numba JIT Hybrid Scheduler (Algorithm 3): Combines a mandatory cyclic backbone sweep (M-period) with aged residual priority updates executed entirely in JIT-compiled native memory space.
 * Parallel Hyperparameter Calibration: Multi-core grid-search calibrator (GridSearchCalibrator) using joblib over validation state snapshots.
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
Run unit and integration smoke tests:
pytest tests/

📄 Citation & License
This project is licensed under the MIT License. If you use this implementation in your research, please cite:
@article{myo2026epistemic,
  title={Epistemic Ecology Runtime: A Variational Topology and Dynamic Quantum Vacuum Framework for Knowledge Graphs},
  author={Nain, Myo Set},
  year={2026},
  note={ORCID: 0009-0002-9133-0058}
}

