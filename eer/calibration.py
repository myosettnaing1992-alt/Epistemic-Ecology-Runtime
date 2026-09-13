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
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

from .core_graph import EpistemicGraph
from .cycle_basis import build_cycle_matrix_fundamental
from .hessian_builder import build_cascade_matrix_bounded


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
        H = 0.5 * (H + H.T).tocsc()
        try:
            return spsolve(H, self.graph.b)
        except Exception:
            return np.full(self.graph.num_nodes, np.nan)

    def _eval_one(
        self, alpha: float, gamma: float, snapshots: np.ndarray
    ) -> tuple[float, float, float]:
        x_star = self._solve_at(alpha, gamma)
        if not np.all(np.isfinite(x_star)):
            return alpha, gamma, np.inf
        resid = snapshots - x_star[np.newaxis, :]
        mse = float(np.mean(np.sum(resid ** 2, axis=1)))
        return alpha, gamma, mse

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(
        self,
        snapshots: list[np.ndarray] | np.ndarray,
        n_jobs: int = -1,
        verbose: bool = False,
    ) -> tuple[float, float, float]:
        """
        Run the grid search.

        Parameters
        ----------
        snapshots : (m, n) array of validation belief vectors
        n_jobs : int, joblib convention (-1 = all cores)
        verbose : bool

        Returns
        -------
        (best_alpha, best_gamma, best_mse)
        """
        X = np.asarray(snapshots, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("snapshots must be 2-D (m, n).")

        jobs = [
            (float(a), float(g))
            for a in self.alpha_grid
            for g in self.gamma_grid
        ]

        results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(self._eval_one)(a, g, X) for a, g in jobs
        )

        best_alpha, best_gamma, best_mse = None, None, np.inf
        for a, g, m in results:
            if m < best_mse:
                best_alpha, best_gamma, best_mse = a, g, m

        return float(best_alpha), float(best_gamma), float(best_mse)

    def calibrate_full(
        self,
        snapshots: list[np.ndarray] | np.ndarray,
        n_jobs: int = -1,
    ) -> CalibrationResult:
        a, g, m = self.calibrate(snapshots, n_jobs=n_jobs)
        return CalibrationResult(
            alpha=a, gamma=g, mse=m,
            n_evaluations=len(self.alpha_grid) * len(self.gamma_grid),
            method="validation_grid",
        )


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import networkx as nx

    n = 200
    G_nx = nx.barabasi_albert_graph(n, 3, seed=0)
    g = EpistemicGraph(num_nodes=n)
    g.b = np.random.uniform(0, 1, size=n)
    for u, v in G_nx.edges():
        g.add_support_edge(int(u), int(v), 1.0)

    snapshots = np.random.uniform(0, 1, size=(5, n))

    cal = FastGridSearchCalibrator(
        g,
        alpha_grid=np.logspace(-2, 0, 5),
        gamma_grid=np.logspace(-2, 0, 5),
        L_max=3,
    )
    best_a, best_g, best_mse = cal.calibrate(snapshots, n_jobs=1)
    print(f"Best alpha = {best_a:.4f}, gamma = {best_g:.4f}, MSE = {best_mse:.6e}")
