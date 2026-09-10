"""Parallel hyperparameter calibrator with pre-cached structural regularizers."""

from typing import List, Tuple
import numpy as np
from joblib import Parallel, delayed

from eer.hessian_builder import (
    assemble_extended_hessian,
    build_cascade_matrix_bounded,
    build_cycle_matrix_fundamental,
)
from eer.schedulers import run_hybrid_priority_scheduler_optimized


class FastGridSearchCalibrator:
    """Pre-cached surrogate grid search for (alpha, gamma).

    Pre-computes H_0 + H_SCC, Q_cascade, and Q_cycle once; evaluates each
    (alpha, gamma) candidate by cheap matrix addition.
    """

    def __init__(
        self,
        graph,
        alpha_grid: np.ndarray,
        gamma_grid: np.ndarray,
        L_max: int = 4,
    ):
        self.graph = graph
        self.alpha_grid = np.asarray(alpha_grid, dtype=np.float64)
        self.gamma_grid = np.asarray(gamma_grid, dtype=np.float64)

        # Pre-compute base Hessian H_base = H_0 + H_SCC (alpha = gamma = 0)
        self.H_base = assemble_extended_hessian(graph, alpha=0.0, gamma=0.0)
        self.Q_cascade = build_cascade_matrix_bounded(graph, L_max=L_max)
        self.Q_cycle = build_cycle_matrix_fundamental(graph)

    def _build_H(self, alpha: float, gamma: float):
        H = self.H_base
        if alpha > 0 and self.Q_cascade.nnz > 0:
            H = H + alpha * self.Q_cascade
        if gamma > 0 and self.Q_cycle.nnz > 0:
            H = H + gamma * self.Q_cycle
        return H.tocsr()

    def _evaluate_candidate(
        self,
        alpha: float,
        gamma: float,
        snapshots: List[np.ndarray],
        tol: float,
    ) -> Tuple[float, float, float]:
        H_csr = self._build_H(alpha, gamma)
        H_csc = H_csr.tocsc()
        n = self.graph.n
        x0 = np.full(n, 0.5, dtype=np.float64)

        x_star, _, _ = run_hybrid_priority_scheduler_optimized(
            H_csr.indptr,
            H_csr.indices,
            H_csr.data,
            H_csc.indptr,
            H_csc.indices,
            H_csc.data,
            self.graph.b,
            x0,
            M=n,
            epsilon=1e-3,
            max_sweeps=1000,
            tol=tol,
        )
        mse = float(np.mean([np.mean((s - x_star) ** 2) for s in snapshots]))
        return float(alpha), float(gamma), mse

    def calibrate(
        self,
        validation_snapshots: List[np.ndarray],
        n_jobs: int = -1,
        tol: float = 1e-5,
    ) -> Tuple[float, float, float]:
        params = [(a, g) for a in self.alpha_grid for g in self.gamma_grid]
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._evaluate_candidate)(a, g, validation_snapshots, tol)
            for a, g in params
        )
        return min(results, key=lambda x: x[2])
