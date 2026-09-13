"""
Hyperparameter calibration via parallel surrogate grid search.

Distinguishes:
- The canonical Type-II ML objective (Section 5.4)
- The validation-based surrogate actually used in practice
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from .core_graph import EpistemicGraph
from .cycle_basis import build_cycle_matrix_fundamental
from .hessian_builder import (
    assemble_extended_hessian,
    build_cascade_matrix_bounded,
)


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------

@dataclass
class CalibrationResult:
    alpha: float
    gamma: float
    mse: float
    n_evaluations: int
    method: str = "validation_grid"


# ----------------------------------------------------------------------
# FastGridSearchCalibrator
# ----------------------------------------------------------------------

class FastGridSearchCalibrator:
    """
    Parallel validation grid search for (alpha, gamma).

    Pre-caches structural matrices (H_0, Q_cascade, Q_cycle) so that the
    inner loop reduces to solving a linear system for each (alpha, gamma)
    pair.
    """

    def __init__(
        self,
        graph: EpistemicGraph,
        alpha_grid: np.ndarray,
        gamma_grid: np.ndarray,
        L_max: int = 3,
        max_paths_per_node: int = 500,
    ):
        self.graph = graph
        self.alpha_grid = np.asarray(alpha_grid, dtype=np.float64)
        self.gamma_grid = np.asarray(gamma_grid, dtype=np.float64)

        # Pre-cache structural matrices
        n = graph.num_nodes
        from scipy.sparse import diags

        Lambda = diags(graph.lambda_vec, 0, format="csr")
        L_S = graph.support_laplacian()
        L_D = graph.derived_from_laplacian()
        self.H0 = (Lambda + L_S + L_D).tocsr()

        self.Q_cascade = build_cascade_matrix_bounded(
            graph, L_max=L_max, max_paths_per_node=max_paths_per_node
        )
        self.Q_cycle = build_cycle_matrix_fundamental(graph)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _solve_at(self, alpha: float, gamma: float) -> np.ndarray:
        H = self.H0 + alpha * self.Q_cascade + gamma * self.Q_cycle
        H = 0
